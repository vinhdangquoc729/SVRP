from typing import Tuple
import torch
from torch import Tensor

from .scenario import ScenarioConfig


class StochasticInstanceGenerator:
    def __init__(self, scenario: ScenarioConfig):
        self.scenario = scenario
        self.device = scenario.device

        # Nếu fixed_customers, tạo sẵn toạ độ
        self._fixed_coords = None
        if scenario.fixed_customers:
            self._fixed_coords = self._generate_coords()

    def _generate_coords(self) -> Tensor:
        N = self.scenario.num_nodes
        coords = torch.rand(N, 2)
        coords[0] = torch.tensor([0.5, 0.5])  # depot ở giữa
        return coords

    def _compute_distance_matrix(self, coords: Tensor) -> Tensor:
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)   # [N, N, 2]
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-9)      # [N, N]
        dist.fill_diagonal_(0.0)
        return dist

    def _generate_time_windows(self, batch_size: int) -> Tensor:
        cfg = self.scenario
        N = cfg.num_nodes
        H = cfg.max_horizon

        time_windows = torch.zeros(batch_size, N, H, device=self.device)

        for b in range(batch_size):
            for n in range(1, N): 
                start = torch.randint(0, H // 2, (1,)).item()
                end = torch.randint(H // 2, H, (1,)).item()
                time_windows[b, n, start:end] = 1.0

            time_windows[b, 0, :] = 1.0

        return time_windows

    def generate(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None, 
    ) -> Tuple[Tensor, Tensor, Tensor]:
        device = self.device
        cfg = self.scenario
        N = cfg.num_nodes

        weather = torch.randn(batch_size, cfg.weather_dim, device=device)

        if cfg.fixed_customers and self._fixed_coords is not None:
            coords = self._fixed_coords.to(device)
            base_dist = self._compute_distance_matrix(coords)  # [N, N]
            base_dist = base_dist.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N]
        else:
            coords = torch.rand(batch_size, N, 2, device=device)
            diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, N, N, 2]
            base_dist = torch.sqrt((diff ** 2).sum(-1) + 1e-9)  # [B, N, N]
            base_dist[:, torch.arange(N), torch.arange(N)] = 0.0

        weather_scale = 0.1 * weather.mean(dim=-1, keepdim=True)  # [B, 1]
        travel_cost = base_dist * (1.0 + weather_scale.unsqueeze(-1))

        base_demand = 5 + 10 * torch.rand(batch_size, N, device=device)
        base_demand[:, 0] = 0.0

        w_scalar = weather.mean(dim=-1, keepdim=True)  # [B, 1]
        weather_term = w_scalar.expand(-1, N)          # [B, N]

        noise = torch.randn(batch_size, N, device=device)

        a, b, g = cfg.a_ratio, cfg.b_ratio, cfg.gamma_ratio
        demand = a * base_demand + b * weather_term + g * noise

        demand = demand.clamp_min(0.0)
        demand[:, 0] = 0.0

        if cfg.time_windows:
            time_windows = self._generate_time_windows(batch_size)
        else:
            time_windows = None

        return weather, demand, travel_cost, coords, time_windows
