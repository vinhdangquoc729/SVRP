from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from env.scenario import ScenarioConfig
from env.state import SVRPState


class CustomerEmbedding(nn.Module):
    def __init__(self, scenario: ScenarioConfig, d_model: int):
        super().__init__()
        self.cfg = scenario
        self.d_model = d_model
        self.in_dim = scenario.weather_dim + 1 + scenario.num_nodes
        self.conv = nn.Sequential(
            nn.Conv1d(self.in_dim, d_model, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, weather: Tensor, demand: Tensor, travel_cost: Tensor) -> Tensor:
        weather = weather.detach()
        demand = demand.detach()
        travel_cost = travel_cost.detach()

        B, N = demand.shape
        w_expanded = weather.unsqueeze(1).expand(-1, N, -1)         # [B,N,W]
        d_expanded = demand.unsqueeze(-1)                           # [B,N,1]
        tc_row = travel_cost                                        # [B,N,N]

        features = torch.cat([w_expanded, d_expanded, tc_row], dim=-1)
        x = features.permute(0, 2, 1)                              # [B, C_in, N]
        x = self.conv(x)                                           # [B, d_model, N]
        node_emb = x.permute(0, 2, 1).contiguous()                 # [B, N, d_model]
        return node_emb


class VehicleEmbedding(nn.Module):
    def __init__(self, scenario: ScenarioConfig, d_model: int):
        super().__init__()
        self.cfg = scenario
        self.d_model = d_model
        self.num_nodes = scenario.num_nodes
        self.time_dim = scenario.max_horizon

        in_dim = self.num_nodes + 1 + self.time_dim
        
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, 
        positions: Tensor, 
        loads: Tensor, 
        time: Tensor, 
        hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Trả về:
            veh_emb: [B, K, d_model]
            new_hidden: Tuple((h, c)) - trạng thái ẩn mới để dùng cho step sau
        """
        positions = positions.detach().clone()
        loads = loads.detach()
        time = time.detach()

        B, K = positions.shape

        # Tạo features
        pos_oh = F.one_hot(positions, num_classes=self.num_nodes).float()  # [B,K,N]
        load_norm = (loads / float(self.cfg.capacity)).unsqueeze(-1)       # [B,K,1]

        t_idx = torch.clamp(time.long(), min=0, max=self.time_dim - 1)
        t_oh = F.one_hot(t_idx, num_classes=self.time_dim).float()         # [B,time_dim]
        t_oh = t_oh.unsqueeze(1).expand(-1, K, -1)                         # [B,K,time_dim]

        feat = torch.cat([pos_oh, load_norm, t_oh], dim=-1)                # [B,K,in_dim]

        seq = feat.view(B * K, 1, -1)
        output, next_hidden = self.lstm(seq, hidden)

        veh_emb = output.view(B, K, self.d_model) # [B, K, d_model]
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
    
    customers = state.customers
    vehicles = state.vehicles

    node_emb = cust_emb(customers.weather, customers.demand, customers.travel_cost)
    
    vehicle_emb, next_hidden = veh_emb(
        vehicles.positions, vehicles.loads, vehicles.time, lstm_hidden
    )

    return EmbeddingInputs(node_emb=node_emb, vehicle_emb=vehicle_emb), next_hidden