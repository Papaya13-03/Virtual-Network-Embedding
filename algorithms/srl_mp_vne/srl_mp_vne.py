import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Dict, Tuple
from .global_controller import GlobalController
from problem.substrate_network import SubstrateNode
from problem.virtual_network import VirtualNode, VirtualNetwork
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution

# --- Neural Network ---
class NN(nn.Module):
    def __init__(self, state_size, action_size):
        super(NN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_size)
        )
    def forward(self, x): return self.fc(x)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size, self.action_size = state_size, action_size
        self.model = NN(state_size, action_size)
        self.target_model = NN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon, self.epsilon_min, self.epsilon_decay = 1.0, 0.05, 0.995

    def get_action(self, state):
        if np.random.rand() <= self.epsilon: return random.randrange(self.action_size)
        with torch.no_grad(): return torch.argmax(self.model(torch.FloatTensor(state))).item()

# --- Swarm RL Manager ---
class DQN_PSO_Agent:
    def __init__(self, state_size, action_size, num_agents=4):
        self.num_agents, self.state_size, self.action_size = num_agents, state_size, action_size
        self.agents = [DQNAgent(state_size, action_size) for _ in range(num_agents)]
        self.memory = deque(maxlen=2000)
        self.batch_size, self.gamma = 64, 0.95
        self.personal_best = [NN(state_size, action_size) for _ in range(num_agents)]
        self.global_best = NN(state_size, action_size)
        self.E, self.E_p, self.E_G = [-float('inf')] * num_agents, [-float('inf')] * num_agents, -float('inf')
        self.BETA, self.DELTA = 0.1, 0.1
        for i in range(num_agents): self.personal_best[i].load_state_dict(self.agents[i].model.state_dict())
        self.global_best.load_state_dict(self.agents[0].model.state_dict())

    def update_best(self, scores):
        for i in range(self.num_agents):
            self.E[i] = scores[i]
            if self.E[i] > self.E_p[i]:
                self.E_p[i] = self.E[i]; self.personal_best[i].load_state_dict(self.agents[i].model.state_dict())
            if self.E[i] > self.E_G:
                self.E_G = self.E[i]; self.global_best.load_state_dict(self.agents[i].model.state_dict())

    def update_q(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        s = torch.FloatTensor(np.array([m[0] for m in batch]))
        a = np.array([m[1] for m in batch])
        r = torch.FloatTensor(np.array([m[2] for m in batch]))
        ns = torch.FloatTensor(np.array([m[3] for m in batch]))
        d = torch.FloatTensor(np.array([m[4] for m in batch]))
        for j in range(self.num_agents):
            with torch.no_grad():
                nx_q = self.agents[j].target_model(ns).max(1)[0]
                tar_q = r + (1 - d) * self.gamma * nx_q
                pb_q = self.personal_best[j](s).max(1)[0]
                gb_q = self.global_best(s).max(1)[0]
                cur_q = self.agents[j].model(s).gather(1, torch.LongTensor(a).unsqueeze(1)).squeeze()
                tar_q += self.BETA * (pb_q - cur_q) + self.DELTA * (gb_q - cur_q)
            pred_q = self.agents[j].model(s).gather(1, torch.LongTensor(a).unsqueeze(1)).squeeze()
            loss = self.agents[j].criterion(pred_q, tar_q)
            self.agents[j].optimizer.zero_grad(); loss.backward(); self.agents[j].optimizer.step()
            if self.agents[j].epsilon > self.agents[j].epsilon_min: self.agents[j].epsilon *= self.agents[j].epsilon_decay

# --- Main Algorithm ---
class SRLMPVNE:
    def __init__(self, state_size=6, num_agents=4):
        self.name = "SRL-MP-VNE"
        self.state_size, self.num_agents = state_size, num_agents
        self.global_controller = None
        self.swarm_rl = None

    def solve(self, substrate, vnr: VirtualNetworkRequest):
        if self.global_controller is None:
            self.global_controller = GlobalController(substrate)
            self.swarm_rl = DQN_PSO_Agent(self.state_size, 1, self.num_agents)

        vnetwork = vnr.virtual_network
        vnodes = list(vnetwork.nodes.values())
        candidates = self.global_controller.find_all_candidates(vnetwork)
        if any(not c for c in candidates): return EmbeddingSolution(vnetwork.id, False)

        # PSO Node Selection guided by Swarm RL
        best_idx = self.pso(candidates, vnodes)
        mapping = {vnodes[i].id: candidates[i][best_idx[i]].id for i in range(len(vnodes))}
        
        node_cost = sum(v.cpu_demand * getattr(candidates[j][best_idx[j]], 'cpu_price', 1.0) for j, v in enumerate(vnodes))
        
        try:
            vlink_paths_raw = self.global_controller.commit_mapping(mapping, vnetwork)
            
            # Format link mapping for serialization
            vlink_mapping = {}
            total_bw_cost = 0
            for (v_src, v_dst), allocated_paths in vlink_paths_raw.items():
                formatted_paths = []
                for path_links, allocated_bw in allocated_paths:
                    link_tuples = [(l.source, l.target) for l in path_links]
                    formatted_paths.append((link_tuples, allocated_bw))
                    total_bw_cost += allocated_bw * len(path_links)
                vlink_mapping[(v_src, v_dst)] = formatted_paths
            
            total_cost = node_cost + total_bw_cost
            reward = 100.0 # Success
            success = True
        except ValueError:
            reward = -50.0 # Failure
            total_cost = 0.0
            vlink_mapping = {}
            success = False

        # Train Swarm RL with Multi-Path result
        final_score = reward
        self.swarm_rl.update_best([final_score] * self.num_agents)
        
        # Extract experience for training
        for i, vnode in enumerate(vnodes):
            state = self.global_controller.extract_node_features(candidates[i][best_idx[i]], vnode, 1.0)
            next_state = state # Simplified
            self.swarm_rl.memory.append((state, 0, reward, next_state, 1 if success else 0))
        
        self.swarm_rl.update_q()
        self.global_controller.clear_caches()
        
        return EmbeddingSolution(
            vnr_id=vnr.id,
            is_successful=success,
            node_mapping=mapping,
            link_mapping=vlink_mapping,
            embedding_cost=total_cost
        )

    def pso(self, candidates, vnodes):
        num_particles, max_iter = 10, 50
        dim = len(vnodes)
        population = [[random.randint(0, len(candidates[j]) - 1) for j in range(dim)] for _ in range(num_particles)]
        velocities = [[0 for _ in range(dim)] for _ in range(num_particles)]
        pbest, pbest_score = [p[:] for p in population], [float('inf')] * num_particles
        gbest, gbest_score = None, float('inf')
        
        c1, c2, w = 1.5, 1.5, 0.7
        for _ in range(max_iter):
            for i in range(num_particles):
                score = self.fitness(population[i], candidates, vnodes)
                if score < pbest_score[i]:
                    pbest[i], pbest_score[i] = population[i][:], score
                    if score < gbest_score: gbest, gbest_score = population[i][:], score
            
            for i in range(num_particles):
                for j in range(dim):
                    r1, r2 = random.random(), random.random()
                    velocities[i][j] = w * velocities[i][j] + c1*r1*(pbest[i][j] - population[i][j]) + c2*r2*(gbest[j] - population[i][j])
                    population[i][j] = int(round(population[i][j] + velocities[i][j])) % len(candidates[j])
        return gbest

    def fitness(self, particle_idx, candidates, vnodes):
        mapping = [candidates[i][idx] for i, idx in enumerate(particle_idx)]
        if len({s.id for s in mapping}) != len(mapping): return float('inf')
        
        cost = sum(v.cpu_demand * getattr(s, 'cpu_price', 1.0) for v, s in zip(vnodes, mapping))
        dqn_penalty = 0.0
        for i, snode in enumerate(mapping):
            state = self.global_controller.extract_node_features(snode, vnodes[i], 1.0)
            with torch.no_grad():
                q_vals = self.swarm_rl.global_best(torch.FloatTensor(state))
                dqn_penalty -= q_vals.max().item() * 5.0
        return cost + dqn_penalty
