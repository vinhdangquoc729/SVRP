# svrp_rl/embedding.py

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from env.scenario import ScenarioConfig
from env.state import SVRPState


class CustomerEmbedding(nn.Module):
    """
    Node embedding bao gồm:
      - weather (global) expanded -> [B, N, W]
      - demand -> [B, N, 1]
      - tw_open (current timestep) -> [B, N, 1]
      - customer_type embedding -> [B, N, type_emb_dim] (nếu có)
    Trả về node_emb: [B, N, d_model]
    """
    def __init__(self, scenario: ScenarioConfig, d_model: int, type_emb_dim: int = 8):
        super().__init__()
        self.cfg = scenario
        self.d_model = d_model
        self.type_emb_dim = type_emb_dim
        self.num_customer_types = getattr(scenario, "num_customer_types", 3)

        self.type_emb = nn.Embedding(self.num_customer_types, self.type_emb_dim)

        self.in_dim = scenario.weather_dim + 1 + 1 + self.type_emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

    def forward(
        self,
        weather: Tensor,               # [B, W]
        demand: Tensor,                # [B, N]
        time_windows: Optional[Tensor],# [B, N, H] or None
        time: Tensor,                  # [B] current timestep
        customer_type: Optional[Tensor] = None  # [B, N] long or None
    ) -> Tensor:
        B, N = demand.shape
        device = demand.device

        # weather expand
        w = weather.unsqueeze(1).expand(-1, N, -1)  # [B, N, W]

        # demand
        d = demand.unsqueeze(-1)                   # [B, N, 1]

        # time-window open flag at current time
        if time_windows is None:
            tw_open = torch.ones(B, N, 1, device=device)  # nếu không có TW => luôn mở
        else:
            # time: [B], time_windows: [B, N, H]
            H = time_windows.size(-1)
            t_idx = time.clamp(max=H - 1).long()               # [B]
            # expand to [B, N, 1] indices for gather
            t_idx_exp = t_idx.view(B, 1, 1).expand(-1, N, 1)  # [B, N, 1]
            tw_gather = time_windows.gather(dim=2, index=t_idx_exp)  # [B, N, 1]
            tw_open = (tw_gather > 0.5).float()                # [B, N, 1]

        if customer_type is None:
            type_feat = torch.zeros(B, N, self.type_emb_dim, device=device)
        else:
            ct = customer_type.long()
            type_feat = self.type_emb(ct)  # [B, N, type_emb_dim]

        feat = torch.cat([w, d, tw_open, type_feat], dim=-1)  # [B, N, in_dim]
        node_emb = self.mlp(feat)                             # [B, N, d_model]
        return node_emb


class VehicleEmbedding(nn.Module):
    """
    Embedding cho mỗi vehicle:
      - position one-hot [B, K, N]
      - normalized load [B, K, 1]
      - current time one-hot [B, K, H]
    Sau đó dùng LSTM (seq len = 1) để có embedding [B, K, d_model]
    """
    def __init__(self, scenario: ScenarioConfig, d_model: int):
        super().__init__()
        self.cfg = scenario
        self.d_model = d_model
        self.num_nodes = scenario.num_nodes
        self.time_dim = scenario.max_horizon

        in_dim = self.num_nodes + 1 + self.time_dim  # pos_oh + load + time_oh

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        positions: Tensor,   # [B, K] long
        loads: Tensor,       # [B, K] float
        time: Tensor,        # [B] long (current timestep)
        hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, K = positions.shape
        device = positions.device

        pos_oh = F.one_hot(positions.long(), num_classes=self.num_nodes).float()  # [B, K, N]

        load_norm = (loads / float(self.cfg.capacity)).unsqueeze(-1)  # [B, K, 1]

        t_idx = torch.clamp(time.long(), min=0, max=self.time_dim - 1).view(B, 1)
        t_oh = F.one_hot(t_idx, num_classes=self.time_dim).float().squeeze(1)  # [B, H]
        t_oh = t_oh.unsqueeze(1).expand(-1, K, -1)                            # [B, K, H]

        feat = torch.cat([pos_oh, load_norm, t_oh], dim=-1)  # [B, K, in_dim]

        seq = feat.view(B * K, 1, -1)
        output, next_hidden = self.lstm(seq, hidden)  # output: [B*K, 1, d_model]

        veh_emb = output.view(B, K, self.d_model)  # [B, K, d_model]
        veh_emb = self.norm(veh_emb)

        return veh_emb, next_hidden


@dataclass
class EmbeddingInputs:
    node_emb: Tensor      # [B, N, d_model]
    vehicle_emb: Tensor   # [B, K, d_model]


def build_embeddings(
    state: SVRPState,
    cust_emb: CustomerEmbedding,
    veh_emb: VehicleEmbedding,
    lstm_hidden: Optional[Tuple[Tensor, Tensor]] = None
) -> Tuple[EmbeddingInputs, Tuple[Tensor, Tensor]]:
    """
    Helper: xây dựng embedding từ state dùng cho policy.
    Trả EmbeddingInputs và next_hidden.
    """
    customers = state.customers
    vehicles = state.vehicles

    node_emb = cust_emb(
        customers.weather,
        customers.demand,
        getattr(customers, "time_windows", None),
        vehicles.time,
        getattr(customers, "customer_type", None),
    )

    vehicle_emb, next_hidden = veh_emb(
        vehicles.positions,
        vehicles.loads,
        vehicles.time,
        lstm_hidden,
    )

    return EmbeddingInputs(node_emb=node_emb, vehicle_emb=vehicle_emb), next_hidden
