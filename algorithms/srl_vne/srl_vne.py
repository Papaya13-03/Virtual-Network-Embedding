import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, OrderedDict
from typing import List, Dict, Tuple
from algorithms.srl_vne.global_controller import GlobalController
from problem.substrate_network import SubstrateNetwork, SubstrateNode, SubstrateLink
from problem.virtual_network import VirtualNetwork, VirtualNode, VirtualLink
from problem.request import VirtualNetworkRequest
from problem.embedding_solution import EmbeddingSolution

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
                # Q_new = Q + α*(R + γmaxQ' - Q) + β*(pBest - Q) + δ*(gBest - Q)
                # In DQN, this translates to adding β and δ terms to the target
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

# --- Main SRL-VNE Wrapper ---
class SRLVNE:
    def __init__(self, state_size=6, num_agents=4):
        self.global_controller = None
        self.state_size = state_size
        self.num_agents = num_agents
        self.pso_dqn = None
        self._active_mappings: Dict[str, Dict] = OrderedDict()

    def _release_expired(self, current_time: float) -> None:
        expired_ids = [rid for rid, data in self._active_mappings.items()
                       if data["expire_time"] <= current_time]
        for expired_id in expired_ids:
            data = self._active_mappings.pop(expired_id)
            self.global_controller.release_mapping(
                data["mapping"], data["vnetwork"], data["vlink_paths"]
            )

    def solve(self, substrate: SubstrateNetwork, virtual_request: VirtualNetworkRequest) -> EmbeddingSolution:
        if self.global_controller is None:
            self.global_controller = GlobalController(substrate)
            # Find max candidates across any node to set action_size
            self.action_size = 20 # Discretized or capped candidate selection
            self.pso_dqn = DQN_PSO_Agent(self.state_size, self.action_size, self.num_agents)

        self._release_expired(virtual_request.arrival_time)
        vnetwork = virtual_request.virtual_network
        vnodes = sorted(vnetwork.nodes.values(), key=lambda n: n.cpu_demand, reverse=True)
        
        # Map each vnode to a candidate substrate node
        mapping = {}
        used_snodes = set()
        agent_scores = [0.0] * self.num_agents
        
        # We'll use the "Global Best" agent to pick the final mapping, 
        # but in training we'd sample from all agents.
        # For a single 'solve' call, we explore with one then update.
        agent_idx = 0 # Picking agent 0 for this request
        
        history = [] # (state, action, reward, next_state, done)
        
        total_cpu_demand = sum(vn.cpu_demand for vn in vnodes)
        
        for vnode in vnodes:
            candidates = self.global_controller.process_request_for_node(vnode)
            candidates = [c for c in candidates if c.id not in used_snodes]
            if not candidates: break
            
            # Action space limited to available candidates
            curr_action_size = min(len(candidates), self.action_size)
            # State: [v_cpu, s_cpu, s_bw, s_degree, progress, bias]
            progress = len(mapping) / len(vnodes)
            state = self._extract_state(vnode, candidates[0], progress)
            
            action = self.pso_dqn.agents[agent_idx].get_action(state)
            action = action % len(candidates) # wrap around
            
            mapping[vnode.id] = candidates[action].id
            used_snodes.add(candidates[action].id)
            
            # Intermediate "reward" is 0 until commitment
            history.append((state, action, 0.0, None, False))

        if len(mapping) < len(vnodes):
            self._finalize_history(history, -10.0, None)
            return EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)

        try:
            vlink_paths = self.global_controller.commit_mapping(mapping, vnetwork)
            
            # Calculate Revenue and Cost
            revenue = sum(vn.cpu_demand for vn in vnetwork.nodes.values()) + sum(vl.bandwidth_demand for vl in vnetwork.links.values())
            cost = sum(vnetwork.nodes[v_id].cpu_demand * getattr(self.global_controller._find_snode(s_id)[1], 'cpu_price', 1.0) for v_id, s_id in mapping.items())
            cost += sum(vnetwork.links[vlink_key].bandwidth_demand * len(path) for vlink_key, path in vlink_paths.items())
            
            reward = 100.0 * (revenue / (cost + 1e-6))
            self._finalize_history(history, reward, None)
            self.pso_dqn.update_best([reward if i == agent_idx else -10.0 for i in range(self.num_agents)])
            self.pso_dqn.update_q()
            
            final_link_mapping = {}
            for k, p in vlink_paths.items():
                final_link_mapping[k] = [([(l.source, l.target) for l in p], vnetwork.links[k].bandwidth_demand)]
                
            self._active_mappings[virtual_request.id] = {
                "mapping": mapping,
                "vnetwork": vnetwork,
                "vlink_paths": vlink_paths,
                "expire_time": virtual_request.arrival_time + virtual_request.lifetime,
            }

            return EmbeddingSolution(vnr_id=virtual_request.id, is_successful=True, node_mapping=mapping, link_mapping=final_link_mapping, embedding_cost=cost)
        except Exception:
            self._finalize_history(history, -50.0, None)
            self.pso_dqn.update_q()
            return EmbeddingSolution(vnr_id=virtual_request.id, is_successful=False)

    def _extract_state(self, vnode, snode, progress):
        # State: [v_cpu, s_cpu, s_bw_avg, s_degree, progress, bias]
        total_links = 0
        s_avail_bw = 0
        
        # Intra-domain logic
        for lc in self.global_controller.local_controllers:
            total_links += len(lc.domain.network.links)
            for (u, v), link in lc.domain.network.links.items():
                if u == snode.id or v == snode.id:
                    s_avail_bw += getattr(link, 'available_bw', link.bandwidth_capacity)
                    
        # Inter-domain logic
        total_links += len(self.global_controller.snetwork.inter_domain_links)
        for (u, v), link in self.global_controller.snetwork.inter_domain_links.items():
            if u == snode.id or v == snode.id:
                s_avail_bw += getattr(link, 'available_bw', link.bandwidth_capacity)
                
        return np.array([
            vnode.cpu_demand / 20.0, 
            snode.available_cpu / 100.0, 
            s_avail_bw / 500.0,
            total_links / 1000.0,
            progress,
            0.5
        ])

    def _finalize_history(self, history, reward, next_state):
        for i, (s, a, _, _, _) in enumerate(history):
            done = (i == len(history) - 1)
            ns = history[i+1][0] if not done else np.zeros(self.state_size)
            self.pso_dqn.memory.append((s, a, reward, ns, done))
