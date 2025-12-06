from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch import Tensor

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
        """
        Chạy giải 1 instance SVRP.

        Args:
            env: SVRPEnvironment (framework mới, có reset/step/get_action_mask)

        Returns:
            routes: List[num_vehicles][route_length] các node index (0 = depot)
            total_cost: tổng chi phí (>=0) = - tổng reward
            info: dict phụ (nếu cần thêm metric)
        """
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

        total_reward = 0.0
        steps = 0

        while (not done.all()) and steps < self.max_steps:
            action_mask: Tensor = env.get_action_mask()  # [1,K,N]
            logits: Tensor = self.policy(state, action_mask)  # [1,K,N]
            logits = logits.masked_fill(~action_mask, float("-inf"))

            actions: Tensor = torch.argmax(logits, dim=-1)  # [1,K]
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

        total_cost = -total_reward  # vì reward = -cost

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
        total_reward = 0.0
        steps = 0

        while (not done.all()) and steps < self.max_steps:
            action_mask = env.get_action_mask()  # [1,K,N]

            logits = self.policy(state, action_mask)  # [1,K,N]
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