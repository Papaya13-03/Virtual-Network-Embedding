import os
import random
import torch
import numpy as np
import yaml
from typing import List, Dict, Tuple
from algorithms.mp_dqn_vne.global_controller import GlobalController
from problem.substrate_network import SubstrateNetwork, SubstrateNode
from problem.virtual_network import VirtualNetwork, VirtualNode
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution
from collections import deque
import torch.nn as nn
import torch.optim as optim

# --- Neural Network Architecture ---
class NN(nn.Module):
    def __init__(self, state_size, action_size):
        super(NN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = NN(state_size, action_size)
        self.target_model = NN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state)
        q_values = self.model(state_t)
        return torch.argmax(q_values).item()

# --- Swarm RL Manager ---
class DQN_PSO_Agent:
    def __init__(self, state_size, action_size, num_agents=4):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.agents = [DQNAgent(state_size, action_size) for _ in range(num_agents)]
        self.memory = deque(maxlen=2000)
        self.batch_size = 64
        self.gamma = 0.95
        
        # PSO terms
        self.personal_best = [NN(state_size, action_size) for _ in range(num_agents)]
        self.global_best = NN(state_size, action_size)
        self.E = [-float('inf')] * num_agents
        self.E_p = [-float('inf')] * num_agents
        self.E_G = -float('inf')
        self.BETA = 0.1  # Influence from pBest
        self.DELTA = 0.1 # Influence from gBest

        for i in range(num_agents):
            self.personal_best[i].load_state_dict(self.agents[i].model.state_dict())
        self.global_best.load_state_dict(self.agents[0].model.state_dict())

    def update_best(self, scores):
        for i in range(self.num_agents):
            self.E[i] = scores[i]
            if self.E[i] > self.E_p[i]:
                self.E_p[i] = self.E[i]
                self.personal_best[i].load_state_dict(self.agents[i].model.state_dict())
            if self.E[i] > self.E_G:
                self.E_G = self.E[i]
                self.global_best.load_state_dict(self.agents[i].model.state_dict())

    def update_q(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([m[0] for m in batch]))
        actions = np.array([m[1] for m in batch])
        rewards = torch.FloatTensor(np.array([m[2] for m in batch]))
        next_states = torch.FloatTensor(np.array([m[3] for m in batch]))
        dones = torch.FloatTensor(np.array([m[4] for m in batch]))

        for j in range(self.num_agents):
            # Normal DQN Target
            with torch.no_grad():
                next_q = self.agents[j].target_model(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q
                
                # PSO-influenced Target tweak as per solution.md
                p_best_q = self.personal_best[j](states).max(1)[0]
                g_best_q = self.global_best(states).max(1)[0]
                
                curr_q = self.agents[j].model(states).gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze()
                target_q += self.BETA * (p_best_q - curr_q) + self.DELTA * (g_best_q - curr_q)

            # Training step
            predicted_q = self.agents[j].model(states).gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze()
            loss = self.agents[j].criterion(predicted_q, target_q)
            self.agents[j].optimizer.zero_grad()
            loss.backward()
            self.agents[j].optimizer.step()
            
            if self.agents[j].epsilon > self.agents[j].epsilon_min:
                self.agents[j].epsilon *= self.agents[j].epsilon_decay

class MPDQNVNE:
    """
    Improved MP-VNE using DQN for PreCost (Fitness) estimation.
    Node selection: PSO guided by DQN values.
    Link selection: Multi-path allocation.
    """
    def __init__(self, state_size=6, num_agents=4):
        self.name = "MP-DQN-VNE"
        self.state_size = state_size
        self.num_agents = num_agents
        self.pso_dqn = None
        self.global_controller = None

    def solve(self, substrate: SubstrateNetwork, virtual_request: VirtualNetworkRequest) -> EmbeddingSolution:
        if self.global_controller is None:
            self.global_controller = GlobalController(substrate)
            self.action_size = 20 # Can expand if needed
            self.pso_dqn = DQN_PSO_Agent(self.state_size, self.action_size, self.num_agents)

        vnetwork = virtual_request.virtual_network
        vnodes = list(vnetwork.nodes.values())
        
        # 1. Candidate Selection
        candidate_nodes = self.global_controller.process_request(vnetwork)
        if any(not c for c in candidate_nodes):
            return EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)

        # 2. PSO Search with DQN Fitness
        best_particle_idx = self.pso(candidate_nodes, vnetwork)
        
        # 3. Build mapping
        best_mapping = {
            vnodes[i].id: candidate_nodes[i][idx].id
            for i, idx in enumerate(best_particle_idx)
        }
        
        # 4. Final Commitment (Multi-Path)
        try:
            vlink_paths = self.global_controller.commit_mapping(best_mapping, vnetwork)
            
            # Calculate Revenue and Cost for reward
            revenue = sum(vn.cpu_demand for vn in vnetwork.nodes.values()) + sum(vl.bandwidth_demand for vl in vnetwork.links.values())
            cost = sum(vnetwork.nodes[v_id].cpu_demand * getattr(self.global_controller._find_snode(s_id)[1], 'cpu_price', 1.0) for v_id, s_id in best_mapping.items())
            
            # Multi-path cost (BW * number of paths or hops)
            total_bw_cost = 0
            formatted_link_mapping = {}
            for (v_src, v_dst), allocated_paths in vlink_paths.items():
                formatted_paths = []
                for path_links, allocated_bw in allocated_paths:
                    link_tuples = [(l.source, l.target) for l in path_links]
                    formatted_paths.append((link_tuples, allocated_bw))
                    total_bw_cost += allocated_bw * len(path_links)
                formatted_link_mapping[(v_src, v_dst)] = formatted_paths
            
            total_cost = cost + total_bw_cost
            reward = 100.0 * (revenue / (total_cost + 1e-6))
            
            # Update Swarm DQN
            self.pso_dqn.update_best([reward for _ in range(self.num_agents)])
            self.pso_dqn.update_q()
            
            return EmbeddingSolution(
                vnr_id=virtual_request.id,
                is_successful=True,
                node_mapping=best_mapping,
                link_mapping=formatted_link_mapping,
                embedding_cost=total_cost
            )
        except Exception:
            return EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)

    def pso(self, candidates: List[List[SubstrateNode]], vnetwork: VirtualNetwork) -> List[int]:
        num_particles = 15
        num_iterations = 10
        w, c1, c2 = 0.7, 1.5, 1.5
        num_vnodes = len(candidates)
        
        vnodes = list(vnetwork.nodes.values())
        
        population = [[random.randint(0, len(candidates[j]) - 1) for j in range(num_vnodes)] for _ in range(num_particles)]
        velocities = [[0.0 for _ in range(num_vnodes)] for _ in range(num_particles)]
        
        pbest = [p[:] for p in population]
        pbest_score = [self.fitness(p, candidates, vnodes) for p in population]
        
        gbest = pbest[pbest_score.index(min(pbest_score))][:]
        gbest_score = pbest_score[pbest_score.index(min(pbest_score))]
        
        for it in range(num_iterations):
            for i in range(num_particles):
                for j in range(num_vnodes):
                    r1, r2 = random.random(), random.random()
                    velocities[i][j] = (w * velocities[i][j] + 
                                       c1 * r1 * (pbest[i][j] - population[i][j]) + 
                                       c2 * r2 * (gbest[j] - population[i][j]))
                    population[i][j] = int(round(population[i][j] + velocities[i][j])) % len(candidates[j])
                    
                score = self.fitness(population[i], candidates, vnodes, 1.0)
                if score < pbest_score[i]:
                    pbest[i] = population[i][:]
                    pbest_score[i] = score
                    
            if min(pbest_score) < gbest_score:
                gbest = pbest[pbest_score.index(min(pbest_score))][:]
                gbest_score = min(pbest_score)
        
        return gbest

    def fitness(self, particle_idx: List[int], candidates: List[List[SubstrateNode]], vnodes: List[VirtualNode], progress: float = 0.0) -> float:
        mapping = [candidates[i][idx] for i, idx in enumerate(particle_idx)]
        snode_ids = {s.id for s in mapping}
        if len(snode_ids) != len(mapping): return float('inf')
        
        # Base Node Cost
        node_cost = sum(v.cpu_demand * getattr(s, 'cpu_price', 1.0) for v, s in zip(vnodes, mapping))
        
        # DQN-based estimated quality (PreCost)
        dqn_penalty = 0.0
        for i, snode in enumerate(mapping):
            state = self.global_controller.extract_node_features(snode, vnodes[i], progress)
            with torch.no_grad():
                # We use the Global Best DQN to estimate the value of this node mapping
                # Higher Q-value is better, so we subtract it from fitness cost
                q_values = self.pso_dqn.global_best(torch.FloatTensor(state))
                # Sum of q-values or max q-value as a proxy for "goodness"
                dqn_penalty -= q_values.max().item() * 10.0 # Scaling factor

        return node_cost + dqn_penalty
