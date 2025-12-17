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

    def _generate_time_windows(
        self,
        batch_size: int,
        customer_type: Tensor,
    ) -> Tensor:
        cfg = self.scenario
        device = self.device
        N = cfg.num_nodes
        H = cfg.max_horizon
        jitter = cfg.tw_jitter

        time_windows = torch.zeros(batch_size, N, H, device=device)

        centers = torch.tensor(
            [
                int(0.3 * H),  # type 0 – morning
                int(0.55 * H), # type 1 – afternoon
                int(0.8 * H),  # type 2 – evening
            ],
            device=device,
        )

        for b in range(batch_size):
            for n in range(1, N):
                t = customer_type[b, n]
                center = centers[t]

                start = max(0, center - jitter - torch.randint(0, jitter + 1, (1,)).item())
                end = min(H, center + jitter + torch.randint(0, jitter + 1, (1,)).item())

                time_windows[b, n, start:end] = 1.0

            # depot luôn mở
            time_windows[b, 0, :] = 1.0

        return time_windows

    def _generate_customer_types(self, batch_size: int) -> Tensor:
        cfg = self.scenario
        N = cfg.num_nodes

        customer_type = torch.zeros(batch_size, N, dtype=torch.long, device=self.device)

        customer_type[:, 1:] = torch.randint(
            0, cfg.num_customer_types, (batch_size, N - 1), device=self.device
        )

        return customer_type

    def generate(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        device = self.device
        cfg = self.scenario
        N = cfg.num_nodes

        weather = torch.randn(batch_size, cfg.weather_dim, device=device)

        if cfg.fixed_customers and self._fixed_coords is not None:
            coords = self._fixed_coords.to(device)
            base_dist = self._compute_distance_matrix(coords)
            base_dist = base_dist.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            coords = torch.rand(batch_size, N, 2, device=device)
            diff = coords.unsqueeze(2) - coords.unsqueeze(1)
            base_dist = torch.sqrt((diff ** 2).sum(-1) + 1e-9)
            base_dist[:, torch.arange(N), torch.arange(N)] = 0.0

        weather_scale = 0.1 * weather.mean(dim=-1, keepdim=True)
        travel_cost = base_dist * (1.0 + weather_scale.unsqueeze(-1))

        base_demand = 5 + 10 * torch.rand(batch_size, N, device=device)
        base_demand[:, 0] = 0.0

        w_scalar = weather.mean(dim=-1, keepdim=True)
        weather_term = w_scalar.expand(-1, N)
        noise = torch.randn(batch_size, N, device=device)

        a, b, g = cfg.a_ratio, cfg.b_ratio, cfg.gamma_ratio
        demand = a * base_demand + b * weather_term + g * noise
        demand = demand.clamp_min(0.0)
        demand[:, 0] = 0.0

        customer_type = self._generate_customer_types(batch_size)

        if cfg.time_windows:
            time_windows = self._generate_time_windows(batch_size, customer_type)
        else:
            time_windows = None

        return (weather,demand,travel_cost,coords,time_windows,customer_type,)

if __name__ == "__main__":
    torch.manual_seed(42)
    scenario = ScenarioConfig(
        num_customers=10,
        num_vehicles=1,
        capacity=50.0,
        fixed_customers=True,
        time_windows=True,
    )
    generator = StochasticInstanceGenerator(scenario)
    weather, demand, travel_cost, coords, time_windows, customer_type = generator.generate(batch_size=3)
    print("Weather:", weather)
    print("Demand:", demand)
    print("Travel Cost:", travel_cost)
    print("Coords:", coords)
    print("Time Windows:", time_windows)
    print("Customer Type:", customer_type)