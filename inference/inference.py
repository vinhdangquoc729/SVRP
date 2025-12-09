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

        for _ in range(self.num_samples):
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
        state: SVRPState = env.reset(batch_size=1)

        B = 1
        K = self.scenario.num_vehicles
        done = torch.zeros(B, dtype=torch.bool, device=self.device)

        routes: List[List[int]] = [[0] for _ in range(K)]
        
        # --- FIX: KHỞI TẠO HIDDEN STATE ---
        lstm_hidden = None
        
        total_reward = 0.0
        steps = 0

        while (not done.all()) and steps < self.max_steps:
            action_mask = env.get_action_mask()  # [1,K,N]

            # --- FIX: GỌI POLICY VỚI HIDDEN VÀ UNPACK KẾT QUẢ ---
            logits, next_lstm_hidden = self.policy(state, action_mask, lstm_hidden)
            lstm_hidden = next_lstm_hidden
            
            logits = logits.masked_fill(~action_mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1)  # [1,K,N]

            actions_list = []
            for v in range(K):
                p_v = probs[0, v]  # [N]
                if torch.all(p_v <= 0):
                    action_v = torch.tensor(0, device=self.device)
                else:
                    action_v = torch.multinomial(p_v, 1)[0]
                actions_list.append(action_v)

            actions = torch.stack(actions_list, dim=0).unsqueeze(0)  # [1,K]
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
    def __init__(
        self,
        policy: PolicyNetwork,
        scenario: ScenarioConfig,
        device: str = "cpu",
        num_samples_init: int = 30,
        generations: int = 30,
        mutation_rate: float = 0.1,
        init_method: Literal["rl", "random"] = "rl"  # <--- THÊM THAM SỐ NÀY
    ):
        super().__init__(policy, scenario, device)
        self.sampler = SamplingInference(policy, scenario, device, num_samples=num_samples_init)
        self.pop_size = num_samples_init
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.init_method = init_method # Lưu lại phương pháp khởi tạo

    def solve_one(self, env) -> Tuple[List[List[int]], float, Dict[str, Any]]:
        state = env.state
        demands = state.customers.demand[0].cpu().numpy()
        coords = state.customers.coords[0].cpu().numpy()
        capacity = env.cfg.capacity
        
        N_nodes = coords.shape[0] 
        num_customers = N_nodes - 1
        
        dist_matrix = np.zeros((N_nodes, N_nodes))
        for i in range(N_nodes):
            diff = coords[i] - coords
            dist_matrix[i] = np.sqrt(np.sum(diff**2, axis=1))

        population = []
        
        if self.init_method == "rl":
            for _ in range(self.pop_size):
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

        for gen in range(self.generations):
            fitnesses = []
            for chrom in population:
                cost = self._evaluate_chromosome(chrom, demands, dist_matrix, capacity)
                fitnesses.append(cost)
                if cost < best_fitness:
                    best_fitness = cost
                    best_chrom = chrom

            new_population = []
            best_idx = np.argmin(fitnesses)
            new_population.append(population[best_idx])

            while len(new_population) < self.pop_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                child = self._ordered_crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = self._swap_mutation(child)
                new_population.append(child)
            population = new_population

        final_routes = self._split_chromosome(best_chrom, demands, capacity)
        formatted_routes = []
        for r in final_routes:
            formatted_routes.append([0] + r + [0])
        while len(formatted_routes) < self.scenario.num_vehicles:
            formatted_routes.append([0])

        return formatted_routes, best_fitness, {"generations": self.generations}

    def _evaluate_chromosome(self, chrom, demands, dist_mat, capacity):
        routes = self._split_chromosome(chrom, demands, capacity)
        total_dist = 0
        for route in routes:
            full_route = [0] + route + [0]
            for i in range(len(full_route) - 1):
                u, v = full_route[i], full_route[i+1]
                total_dist += dist_mat[u][v]
        return total_dist

    def _split_chromosome(self, chrom, demands, capacity):
        routes = []
        current_route = []
        current_load = 0
        
        for customer in chrom:
            dem = demands[customer]
            if current_load + dem <= capacity:
                current_route.append(customer)
                current_load += dem
            else:
                routes.append(current_route)
                current_route = [customer]
                current_load = dem
        if current_route:
            routes.append(current_route)
        return routes

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

    def _swap_mutation(self, chrom):
        if len(chrom) < 2: return chrom
        idx1, idx2 = random.sample(range(len(chrom)), 2)
        chrom[idx1], chrom[idx2] = chrom[idx2], chrom[idx1]
        return chrom