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
    Mixed Strategy Trainer: Hỗ trợ cả RL (Reinforce) và Imitation Learning.
    Gradient được tích lũy thủ công, runner cần gọi optimizer.step() ở ngoài.
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
        Chạy 1 batch RL.
        LƯU Ý: Hàm này KHÔNG gọi optimizer.step(). Nó chỉ backward loss (tích lũy gradient).
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

    def backward_imitation(
        self,
        env,
        demonstration_list: List[Tuple[int, List[int]]],
        lambda_il: float = 0.5
    ) -> float:
        """
        Tính toán Loss Imitation (CrossEntropy) dựa trên demonstrations và gọi backward()
        để cộng dồn gradient vào mạng Policy.
        Args:
            demonstration_list: List các tuple (index_trong_batch, route_mẫu)
        """
        if not demonstration_list:
            return 0.0
            
        self.policy.train()
        
        # Lấy danh sách index cần train imitation
        indices = [x[0] for x in demonstration_list]
        
        # State phải là state ban đầu (được reset từ bên ngoài trước khi gọi hàm này)
        state = env.state
        
        criterion = nn.CrossEntropyLoss(reduction='none') # Để tự mask
        total_loss = 0
        
        # Tìm max length của các demo routes
        max_len = max([len(x[1]) for x in demonstration_list])
        
        lstm_hidden = None
        
        batch_size = env.state.customers.demand.size(0)
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # Mask đánh dấu những sample nào có demo
        has_demo_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        has_demo_mask[indices] = True
        
        # Forward loop
        for t in range(max_len - 1):
            action_mask = env.get_action_mask().to(self.device)
            
            logits, next_lstm_hidden = self.policy(state, action_mask, lstm_hidden)
            lstm_hidden = next_lstm_hidden
            
            # Mask action không hợp lệ
            logits = logits.masked_fill(~action_mask, -1e9)
            
            # --- FIX: Squeeze dimension xe (K) nếu có ---
            # Input [B, K, N] -> [B, N] để khớp CrossEntropyLoss
            if logits.dim() == 3:
                logits_squeezed = logits.squeeze(1)
            else:
                logits_squeezed = logits
            
            target_actions = []
            valid_mask = []
            
            for b in range(batch_size):
                if b in indices:
                    # Tìm route tương ứng
                    route = next(r for i, r in demonstration_list if i == b)
                    if t + 1 < len(route):
                        target_actions.append(route[t+1])
                        valid_mask.append(True)
                    else:
                        target_actions.append(0) # Padding
                        valid_mask.append(False)
                else:
                    target_actions.append(0)
                    valid_mask.append(False)
            
            target_tensor = torch.tensor(target_actions, device=self.device, dtype=torch.long)
            step_mask = torch.tensor(valid_mask, device=self.device, dtype=torch.bool)
            
            # Chỉ tính loss nếu bước này có mẫu demo nào valid
            if step_mask.any():
                loss = criterion(logits_squeezed, target_tensor)
                # Chỉ lấy loss của những mẫu valid
                masked_loss = (loss * step_mask.float()).sum() / (step_mask.sum() + 1e-6)
                total_loss += masked_loss

            # Teacher Forcing: Step môi trường theo Target
            # Cần unsqueeze để trả về shape [B, K] cho env (giả sử K=1)
            actions = target_tensor.unsqueeze(1)
            
            next_state, _, done_step, _ = env.step(actions)
            state = next_state
            
            done = done | done_step
            if done.all():
                break
        
        # Scale loss và nhân với trọng số lambda
        final_il_loss = (total_loss / float(max_len)) * lambda_il
        
        # Backward tích lũy gradient
        final_il_loss.backward()
        
        if self.max_grad_norm is not None:
             nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

        return final_il_loss.item()

    def _rollout(
        self,
        env,
        batch_size: int,
        max_steps: int,
    ) -> Dict[str, Tensor]:
        
        self.policy.train()
        self.value_net.train()

        # Lưu ý: env đã được reset ở bên ngoài (trong runner)
        state = env.state
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

            logits = logits.masked_fill(~action_mask, -1e9)
            probs = torch.softmax(logits, dim=-1)
            log_probs_all = torch.log_softmax(logits, dim=-1)

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
        rewards: Tensor,
        values: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Tensor]:
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
        log_probs: Tensor,
        entropies: Tensor,
        advantages: Tensor,
        masks: Tensor,
    ) -> float:
        mask = masks
        pg_term = -log_probs * advantages.detach() * mask
        policy_loss = pg_term.sum(dim=0).mean()

        entropy_term = entropies * mask
        entropy_loss = -self.entropy_weight * entropy_term.sum(dim=0).mean()

        loss = policy_loss + entropy_loss

        # --- MODIFIED: Không zero_grad và không step tại đây ---
        # self.policy_optim.zero_grad() 
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        # self.policy_optim.step()

        return loss.item()

    def _update_value(
        self,
        values: Tensor,
        returns: Tensor,
        masks: Tensor,
    ) -> float:
        mask = masks
        mse = (values - returns) ** 2
        masked_mse = mse * mask
        value_loss = masked_mse.sum(dim=0).mean()

        # --- MODIFIED: Không zero_grad và không step tại đây ---
        # self.value_optim.zero_grad()
        value_loss.backward()
        # self.value_optim.step()

        return value_loss.item()

    def step_optimizers(self):
        """Helper để gọi step ở ngoài vòng lặp"""
        self.policy_optim.step()
        self.value_optim.step()
        self.policy_optim.zero_grad()
        self.value_optim.zero_grad()

    def save(self, path_prefix: str):
        torch.save(self.policy.state_dict(), f"{path_prefix}_policy.pt")
        torch.save(self.value_net.state_dict(), f"{path_prefix}_value.pt")

    def load(self, path_prefix: str):
        self.policy.load_state_dict(torch.load(f"{path_prefix}_policy.pt", map_location=self.device))
        self.value_net.load_state_dict(torch.load(f"{path_prefix}_value.pt", map_location=self.device))