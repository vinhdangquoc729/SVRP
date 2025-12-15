from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Literal

import torch
from torch import Tensor
import numpy as np
import random
import copy

from env.scenario import ScenarioConfig
from env.state import SVRPState
from models.policy import PolicyNetwork


class InferenceStrategy(ABC):
    def __init__(
        self,
        policy: PolicyNetwork,
        scenario: ScenarioConfig,
        device: str = "cpu",
        max_steps: Optional[int] = None,
    ):
        self.policy = policy.to(device)
        self.scenario = scenario
        self.device = device
        self.max_steps = max_steps or scenario.max_horizon
        self.policy.eval()

    @abstractmethod
    def solve_one(
        self,
        env,
    ) -> Tuple[List[List[int]], float, Dict[str, Any]]:
        raise NotImplementedError

class GreedyInference(InferenceStrategy):
    @torch.no_grad()
    def solve_one(
        self,
        env,
    ) -> Tuple[List[List[int]], float, Dict[str, Any]]:
        state: SVRPState = env.reset(batch_size=1)

        B = 1
        K = self.scenario.num_vehicles
        done = torch.zeros(B, dtype=torch.bool, device=self.device)

        routes: List[List[int]] = [[0] for _ in range(K)]

        lstm_hidden = None
        
        total_reward = 0.0
        steps = 0

        while (not done.all()) and steps < self.max_steps:
            action_mask = env.get_action_mask()  # [1,K,N]
            
            logits, next_lstm_hidden = self.policy(state, action_mask, lstm_hidden)
            lstm_hidden = next_lstm_hidden
            
            logits = logits.masked_fill(~action_mask, float("-inf"))

            actions = torch.argmax(logits, dim=-1)          # [1,K]
            actions_cpu = actions[0].tolist()               # list length K

            for v in range(K):
                routes[v].append(actions_cpu[v])

            next_state, reward, done_step, _ = env.step(actions)
            reward = reward.to(self.device)       # [1]
            done_step = done_step.to(self.device) # [1]

            total_reward += reward.item()
            done = done | done_step
            state = next_state

            steps += 1

        for v in range(K):
            if routes[v][-1] != 0:
                routes[v].append(0)

        total_cost = -total_reward 

        info = {
            "steps": steps,
            "total_reward": total_reward,
        }
        return routes, total_cost, info

class SamplingInference(InferenceStrategy):
    def __init__(
        self,
        policy: PolicyNetwork,
        scenario: ScenarioConfig,
        device: str = "cpu",
        max_steps: Optional[int] = None,
        num_samples: int = 16,
    ):
        super().__init__(policy, scenario, device, max_steps)
        self.num_samples = num_samples

    @torch.no_grad()
    def solve_one(
        self,
        env,
    ) -> Tuple[List[List[int]], float, Dict[str, Any]]:
        best_cost: float = float("inf")
        best_routes: Optional[List[List[int]]] = None

        all_costs: List[float] = []

        initial_state = env.state.clone()

        for _ in range(self.num_samples):
            env.reset_by_state(initial_state)
            
            routes, cost, info = self._run_single_rollout(env)
            all_costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_routes = routes

        info = {
            "best_cost": best_cost,
            "all_costs": all_costs,
        }
        return best_routes, best_cost, info

    def _run_single_rollout(
        self,
        env,
    ) -> Tuple[List[List[int]], float, Dict[str, Any]]:
        state = env.state

        B = 1
        K = self.scenario.num_vehicles
        done = torch.zeros(B, dtype=torch.bool, device=self.device)

        routes: List[List[int]] = [[0] for _ in range(K)]
        
        lstm_hidden = None
        
        total_reward = 0.0
        steps = 0

        while (not done.all()) and steps < self.max_steps:
            action_mask = env.get_action_mask()

            logits, next_lstm_hidden = self.policy(state, action_mask, lstm_hidden)
            lstm_hidden = next_lstm_hidden
            
            logits = logits.masked_fill(~action_mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1)

            actions_list = []
            for v in range(K):
                p_v = probs[0, v]
                if torch.all(p_v <= 0):
                    action_v = torch.tensor(0, device=self.device)
                else:
                    action_v = torch.multinomial(p_v, 1)[0]
                actions_list.append(action_v)

            actions = torch.stack(actions_list, dim=0).unsqueeze(0)
            actions_cpu = actions[0].tolist()
            for v in range(K):
                routes[v].append(actions_cpu[v])

            next_state, reward, done_step, _ = env.step(actions)
            reward = reward.to(self.device)
            done_step = done_step.to(self.device)

            total_reward += reward.item()
            done = done | done_step
            state = next_state

            steps += 1

        for v in range(K):
            if routes[v][-1] != 0:
                routes[v].append(0)

        total_cost = -total_reward
        info = {
            "steps": steps,
            "total_reward": total_reward,
        }
        return routes, total_cost, info

class HybridEvolutionaryInference(InferenceStrategy):
    # ... (__init__ giữ nguyên) ...
    def __init__(
        self,
        policy: PolicyNetwork,
        scenario: ScenarioConfig,
        device: str = "cpu",
        num_samples_init: int = 30,
        generations: int = 30,
        mutation_rate: float = 0.1,
        init_method: Literal["rl", "random"] = "rl",
        strategy: Literal["ga", "mpeax", "als", "special_hybrid"] = "ga"
    ):
        super().__init__(policy, scenario, device)
        self.sampler = SamplingInference(policy, scenario, device, num_samples=num_samples_init)
        self.pop_size = num_samples_init
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.init_method = init_method
        self.strategy = strategy

    def solve_one(self, env) -> Tuple[List[List[int]], float, Dict[str, Any]]:
        initial_state = env.state.clone()
        
        state = env.state
        demands = state.customers.demand[0].cpu().numpy()
        dist_matrix = state.customers.travel_cost[0].cpu().numpy() 
        capacity = env.cfg.capacity
        
        N_nodes = demands.shape[0] 
        num_customers = N_nodes - 1
        
        population = []
        if self.init_method == "rl":
            for _ in range(self.pop_size):
                env.reset_by_state(initial_state)
                
                routes, _, _ = self.sampler._run_single_rollout(env)
                chrom = []
                for r in routes:
                    for node in r:
                        if node != 0:
                            chrom.append(node)
                population.append(chrom)
        
        elif self.init_method == "random":
            base_chrom = list(range(1, N_nodes))
            for _ in range(self.pop_size):
                chrom = copy.deepcopy(base_chrom)
                random.shuffle(chrom)
                population.append(chrom)
        
        valid_population = [p for p in population if len(p) == num_customers]
        if len(valid_population) > 0:
            population = valid_population
        
        while len(population) < self.pop_size and len(population) > 0:
            population.append(copy.deepcopy(random.choice(population)))
            
        if len(population) == 0:
             population = [list(range(1, N_nodes)) for _ in range(self.pop_size)]

        best_chrom = population[0]
        best_fitness = float('inf')
        
        fitnesses = []
        for chrom in population:
            cost = self._evaluate_chromosome(chrom, demands, dist_matrix, capacity)
            fitnesses.append(cost)
            if cost < best_fitness:
                best_fitness = cost
                best_chrom = chrom

        special_ind = copy.deepcopy(best_chrom)
        special_fitness = best_fitness
        stagnation_counter = 0
        STAGNATION_LIMIT = 5
        
        for gen in range(self.generations):
            new_population = []
            
            best_idx = np.argmin(fitnesses)
            new_population.append(population[best_idx])

            while len(new_population) < self.pop_size:
                if self.strategy == "mpeax":
                    if random.random() < 0.8:
                         parents = [self._tournament_selection(population, fitnesses) for _ in range(4)]
                         child = self._multi_parent_edge_assembly_crossover(parents)
                    else:
                         p1 = self._tournament_selection(population, fitnesses)
                         p2 = self._tournament_selection(population, fitnesses)
                         child = self._ordered_crossover(p1, p2)
                else:
                    p1 = self._tournament_selection(population, fitnesses)
                    p2 = self._tournament_selection(population, fitnesses)
                    child = self._ordered_crossover(p1, p2)
                
                if random.random() < self.mutation_rate:
                    child = self._swap_mutation(child)
                
                if self.strategy == "als":
                     child = self._local_search_2opt(child, demands, dist_matrix, capacity, limit=20)

                new_population.append(child)
            
            population = new_population
            
            fitnesses = []
            current_gen_best_fitness = float('inf')
            current_gen_best_chrom = None

            for i, chrom in enumerate(population):
                cost = self._evaluate_chromosome(chrom, demands, dist_matrix, capacity)
                fitnesses.append(cost)
                if cost < current_gen_best_fitness:
                    current_gen_best_fitness = cost
                    current_gen_best_chrom = chrom
            
            if current_gen_best_fitness < best_fitness:
                best_fitness = current_gen_best_fitness
                best_chrom = current_gen_best_chrom

            if self.strategy == "special_hybrid":
                prev_special_fitness = special_fitness
                
                special_ind = self._local_search_2opt(special_ind, demands, dist_matrix, capacity, limit=100)
                special_fitness = self._evaluate_chromosome(special_ind, demands, dist_matrix, capacity)
                
                if special_fitness < best_fitness:
                    best_fitness = special_fitness
                    best_chrom = copy.deepcopy(special_ind)
                
                if special_fitness >= prev_special_fitness - 1e-5:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                
                if stagnation_counter >= STAGNATION_LIMIT:
                    challenger = copy.deepcopy(random.choice(population))
                    challenger = self._local_search_2opt(challenger, demands, dist_matrix, capacity, limit=100)
                    chall_fitness = self._evaluate_chromosome(challenger, demands, dist_matrix, capacity)
                    
                    if chall_fitness < special_fitness:
                        special_ind = challenger
                        special_fitness = chall_fitness
                        stagnation_counter = 0
                        if special_fitness < best_fitness:
                            best_fitness = special_fitness
                            best_chrom = copy.deepcopy(special_ind)

        formatted_routes = [[0] + best_chrom + [0]]

        return formatted_routes, best_fitness, {"generations": self.generations}
    
    def _evaluate_chromosome(self, chrom, demands, dist_mat, capacity, time_windows=None):
        current_node = 0
        current_load = capacity
        total_dist = 0.0
        
        current_time_step = 0 
        
        TW_PENALTY_VAL = 0.25
        
        for next_node in chrom:
            dist_move = dist_mat[current_node][next_node]
            total_dist += dist_move
            
            dem = demands[next_node]
            
            if current_load < dem:
                dist_fail_return = dist_mat[next_node][0]
                dist_refill_back = dist_mat[0][next_node]
                total_dist += (dist_fail_return + dist_refill_back)
                current_load = capacity
                
                current_time_step += 2 
            
            current_load -= dem
            current_node = next_node
            
            if time_windows is not None:
                max_horizon = time_windows.shape[1]
                
                t_idx = min(current_time_step, max_horizon - 1)
                
                if time_windows[next_node, t_idx] < 0.5:
                    total_dist += TW_PENALTY_VAL

            current_time_step += 1
            
        total_dist += dist_mat[current_node][0]
        return total_dist

    def _tournament_selection(self, pop, fits, k=3):
        indices = random.sample(range(len(pop)), k)
        best_idx = min(indices, key=lambda i: fits[i])
        return pop[best_idx]

    def _ordered_crossover(self, p1, p2):
        size = len(p1)
        if size < 2: return p1 
        
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = p1[start:end]
        
        current_p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while current_p2_idx < size and p2[current_p2_idx] in child:
                    current_p2_idx += 1
                if current_p2_idx < size:
                    child[i] = p2[current_p2_idx]
        
        remaining = [x for x in p2 if x not in child and x is not None]
        for i in range(size):
            if child[i] is None and remaining:
                child[i] = remaining.pop(0)
                
        return child

    def _multi_parent_edge_assembly_crossover(self, parents):
        if not parents: return []
        size = len(parents[0])
        if size < 2: return parents[0]

        adj = {i: [] for i in range(1, size + 2)} 
        
        for p in parents:
            for i in range(size - 1):
                u, v = p[i], p[i+1]
                adj.setdefault(u, []).append(v)
                adj.setdefault(v, []).append(u)
        
        child = []
        current_node = parents[0][0]
        child.append(current_node)
        visited = {current_node}
        
        while len(child) < size:
            candidates = []
            if current_node in adj:
                for neighbor in adj[current_node]:
                    if neighbor not in visited:
                        candidates.append(neighbor)
            
            if candidates:
                counts = {}
                for c in candidates:
                    counts[c] = counts.get(c, 0) + 1
                best_c = max(counts, key=counts.get)
                next_node = best_c
            else:
                unvisited = [n for n in parents[0] if n not in visited]
                if not unvisited: break 
                next_node = random.choice(unvisited)
            
            child.append(next_node)
            visited.add(next_node)
            current_node = next_node
            
        return child

    def _local_search_2opt(self, chrom, demands, dist_mat, capacity, limit=50):
        best_chrom = chrom
        best_cost = self._evaluate_chromosome(chrom, demands, dist_mat, capacity)
        
        improved = True
        count = 0
        
        while improved and count < limit:
            improved = False
            count += 1
            for i in range(len(chrom) - 1):
                for j in range(i + 1, len(chrom)):
                    if j - i == 1: continue 
                    
                    new_chrom = chrom[:i] + chrom[i:j][::-1] + chrom[j:]
                    cost = self._evaluate_chromosome(new_chrom, demands, dist_mat, capacity)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_chrom = new_chrom
                        improved = True
                        break 
                if improved: break
        
        return best_chrom

    def _swap_mutation(self, chrom):
        if len(chrom) < 2: return chrom
        idx1, idx2 = random.sample(range(len(chrom)), 2)
        chrom[idx1], chrom[idx2] = chrom[idx2], chrom[idx1]
        return chrom