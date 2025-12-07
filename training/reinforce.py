from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from env.scenario import ScenarioConfig
from env.state import SVRPState
from models.policy import PolicyNetwork

class PtValueNet(nn.Module):
    def __init__(self, scenario: ScenarioConfig, hidden_dim: int = 128):
        super().__init__()
        self.K = scenario.num_vehicles
        self.N = scenario.num_nodes

        in_dim = self.K * self.N  # flatten P_{t+1} -> vector
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pt: Tensor) -> Tensor:
        B, K, N = pt.shape
        x = pt.view(B, K * N)     # [B, K*N]
        v = self.net(x).squeeze(-1)  # [B]
        return v


class StateValueNet(nn.Module):
    def __init__(self, scenario: ScenarioConfig, hidden_dim: int = 128):
        super().__init__()
        self.cfg = scenario

        # 4 đặc trưng scalar -> hidden -> 1
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: SVRPState) -> Tensor:
        customers = state.customers
        vehicles = state.vehicles

        demand = customers.demand.detach()
        loads = vehicles.loads.detach()
        time = vehicles.time.detach()

        rem_demand = demand[:, 1:]  # [B, N-1]
        demand_sum = rem_demand.sum(dim=1, keepdim=True) / float(self.cfg.capacity)
        demand_max = rem_demand.max(dim=1, keepdim=True).values / float(self.cfg.capacity)

        load_mean = loads.mean(dim=1, keepdim=True) / float(self.cfg.capacity)

        t_norm = (time.float() / float(max(1, self.cfg.max_horizon))).unsqueeze(-1)

        features = torch.cat([demand_sum, demand_max, load_mean, t_norm], dim=-1)  # [B,4]
        values = self.net(features).squeeze(-1)  # [B]
        return values


@dataclass
class TrainStats:
    mean_reward: float
    policy_loss: float
    value_loss: float
    episode_length: int


class ReinforceTrainer:
    """
    ∇_θ J(θ) ≈ 1/B ∑_b ∑_t (R_t^b - V(s_t^b)) ∇_θ log π_θ(a_t^b | s_t^b)
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        scenario: ScenarioConfig,
        lr: float = 1e-4,
        value_lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_weight: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.scenario = scenario
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.value_net = PtValueNet(scenario).to(device)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optim = optim.Adam(self.value_net.parameters(), lr=value_lr)

    def train_batch(
        self,
        env,
        batch_size: int = 32,
        max_steps: Optional[int] = None,
    ) -> TrainStats:
        """
        Chạy 1 batch episode song song (batch_size environment),
        rồi update policy + value mạng.

        Args:
            env: SVRPEnvironment mới
            batch_size: số episode song song
            max_steps: giới hạn số bước (mặc định = scenario.max_horizon)

        Returns:
            TrainStats
        """
        if max_steps is None:
            max_steps = self.scenario.max_horizon

        trajectories = self._rollout(env, batch_size, max_steps)

        returns, advantages = self._compute_returns_and_advantages(
            trajectories["rewards"],
            trajectories["values"],
            trajectories["masks"],
        )

        policy_loss = self._update_policy(
            trajectories["log_probs"],
            trajectories["entropies"],
            advantages,
            trajectories["masks"],
        )
        value_loss = self._update_value(
            trajectories["values"],
            returns,
            trajectories["masks"],
        )

        mean_reward = trajectories["rewards"].sum(dim=0).mean().item()

        stats = TrainStats(
            mean_reward=mean_reward,
            policy_loss=policy_loss,
            value_loss=value_loss,
            episode_length=trajectories["rewards"].size(0),
        )
        return stats

    def _rollout(
        self,
        env,
        batch_size: int,
        max_steps: int,
    ) -> Dict[str, Tensor]:
        
        self.policy.train()
        self.value_net.train()

        state: SVRPState = env.reset(batch_size)
        B = batch_size
        done = torch.zeros(B, dtype=torch.bool, device=self.device)

        lstm_hidden = None 

        log_probs_list = []
        rewards_list = []
        values_list = []
        entropies_list = []
        masks_list = []

        for t in range(max_steps):
            alive = (~done).float()
            action_mask = env.get_action_mask().to(self.device)

            logits, next_lstm_hidden = self.policy(state, action_mask, lstm_hidden)
            
            lstm_hidden = next_lstm_hidden 

            # Masking logits
            logits = logits.masked_fill(~action_mask, -1e9)
            probs = torch.softmax(logits, dim=-1)
            log_probs_all = torch.log_softmax(logits, dim=-1)

            # Sampling Action
            B_, K, N = probs.shape
            probs_flat = probs.view(B_ * K, N)
            actions_flat = torch.multinomial(probs_flat, 1)
            actions = actions_flat.view(B_, K)

            log_prob_actions = self.policy.log_prob_of_actions(logits, actions)
            log_prob_sum = log_prob_actions.sum(dim=1)

            entropy_per_veh = -(probs * log_probs_all).sum(dim=-1)
            entropy = entropy_per_veh.mean(dim=1)

            values = self.value_net(probs.detach()) 

            next_state, reward, done_step, _ = env.step(actions)
            reward = reward.to(self.device)
            done_step = done_step.to(self.device)

            done = done | done_step
            mask = (~done).float()

            rewards_list.append(reward * alive)
            masks_list.append(alive)
            log_probs_list.append(log_prob_sum)
            values_list.append(values)
            entropies_list.append(entropy)

            state = next_state

            if done.all():
                break

        log_probs = torch.stack(log_probs_list, dim=0)
        rewards = torch.stack(rewards_list, dim=0)
        values = torch.stack(values_list, dim=0)
        entropies = torch.stack(entropies_list, dim=0)
        masks = torch.stack(masks_list, dim=0)

        return {
            "log_probs": log_probs,
            "rewards": rewards,
            "values": values,
            "entropies": entropies,
            "masks": masks,
        }


    def _compute_returns_and_advantages(
        self,
        rewards: Tensor,   # [T,B]
        values: Tensor,    # [T,B]
        masks: Tensor,     # [T,B]
    ) -> Tuple[Tensor, Tensor]:
        """
        Tính discounted returns + advantages = R_t - V(s_t)
        """
        T, B = rewards.shape
        device = rewards.device

        returns = torch.zeros_like(rewards, device=device)
        G = torch.zeros(B, device=device)

        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G * masks[t]
            returns[t] = G

        advantages = returns - values
        return returns, advantages


    def _update_policy(
        self,
        log_probs: Tensor,   # [T,B]
        entropies: Tensor,   # [T,B]
        advantages: Tensor,  # [T,B]
        masks: Tensor,       # [T,B]
    ) -> float:
        """
        L(θ) = - E[ advantage * log_prob ] - λ * H
        """
        # mask out steps sau khi done
        mask = masks

        # policy gradient term
        pg_term = -log_probs * advantages.detach() * mask  # [T,B]
        policy_loss = pg_term.sum(dim=0).mean()

        # entropy term
        entropy_term = entropies * mask
        entropy_loss = -self.entropy_weight * entropy_term.sum(dim=0).mean()

        loss = policy_loss + entropy_loss

        self.policy_optim.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optim.step()

        return loss.item()


    def _update_value(
        self,
        values: Tensor,   # [T,B]
        returns: Tensor,  # [T,B]
        masks: Tensor,    # [T,B]
    ) -> float:
        """
        L_v = MSE( V(s_t), R_t )
        """
        mask = masks

        mse = (values - returns) ** 2
        masked_mse = mse * mask
        value_loss = masked_mse.sum(dim=0).mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        return value_loss.item()

    def save(self, path_prefix: str):
        torch.save(self.policy.state_dict(), f"{path_prefix}_policy.pt")
        torch.save(self.value_net.state_dict(), f"{path_prefix}_value.pt")

    def load(self, path_prefix: str):
        self.policy.load_state_dict(torch.load(f"{path_prefix}_policy.pt", map_location=self.device))
        self.value_net.load_state_dict(torch.load(f"{path_prefix}_value.pt", map_location=self.device))
