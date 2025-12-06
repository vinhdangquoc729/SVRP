from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from env.scenario import ScenarioConfig
from env.state import SVRPState
from .embedding import CustomerEmbedding, VehicleEmbedding, build_embeddings


class PolicyNetwork(nn.Module):
    def __init__(self, scenario: ScenarioConfig, d_model: int = 128, d_k: Optional[int] = None):
        super().__init__()
        self.cfg = scenario
        self.d_model = d_model
        self.d_k = d_k or d_model

        self.customer_embedding = CustomerEmbedding(scenario, d_model)
        self.vehicle_embedding = VehicleEmbedding(scenario, d_model)

        self.key_proj = nn.Linear(d_model, self.d_k)
        self.query_proj = nn.Linear(d_model, self.d_k)

    def forward(
        self,
        state: SVRPState,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        emb = build_embeddings(state, self.customer_embedding, self.vehicle_embedding)
        node_emb = emb.node_emb        # [B, N, d_model]
        vehicle_emb = emb.vehicle_emb  # [B, K, d_model]

        K = self.key_proj(node_emb)        # [B, N, d_k]
        Q = self.query_proj(vehicle_emb)   # [B, K, d_k]

        logits = torch.einsum("bkd,bnd->bkn", Q, K) / math.sqrt(self.d_k)  # [B,K,N]

        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)

        return logits

    @torch.no_grad()
    def greedy_action(self, state: SVRPState, mask: Optional[Tensor] = None) -> Tensor:
        logits = self.forward(state, mask)         # [B,K,N]
        actions = torch.argmax(logits, dim=-1)     # [B,K]
        return actions

    def log_prob_of_actions(self, logits: Tensor, actions: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=-1)  # [B,K,N]

        B, K, N = logits.shape
        device = logits.device

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
        veh_idx = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)

        selected_log_probs = log_probs[batch_idx, veh_idx, actions]  # [B,K]
        return selected_log_probs
