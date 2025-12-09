# svrp_rl/env/svrp_env.py

from typing import Dict, Any, Tuple
import torch
from torch import Tensor

from .scenario import ScenarioConfig
from .state import CustomerState, VehicleState, SVRPState
from .stochastic_generator import StochasticInstanceGenerator


class SVRPEnvironment:
    def __init__(self, scenario: ScenarioConfig, device: str = "cpu"):
        self.cfg = scenario
        self.device = device
        self.generator = StochasticInstanceGenerator(scenario)

        self.batch_size: int | None = None
        self.state: SVRPState | None = None
        self.timestep: int = 0

    @property
    def num_nodes(self) -> int:
        return self.cfg.num_nodes

    @property
    def num_vehicles(self) -> int:
        return self.cfg.num_vehicles

    @property
    def capacity(self) -> float:
        return self.cfg.capacity

    def reset(self, batch_size: int) -> SVRPState:
        """
        Khởi tạo episode mới.
        """
        self.batch_size = batch_size
        self.timestep = 0

        weather, demand, travel_cost, coords, time_windows = self.generator.generate(batch_size)

        customers = CustomerState(
            weather=weather,          # [B, W]
            demand=demand,            # [B, N]
            travel_cost=travel_cost,  # [B, N, N]
            coords=coords,            # [B, N, 2]
            time_windows=time_windows # [B, N, H] hoặc None
        )

        B = batch_size
        K = self.cfg.num_vehicles
        device = self.device

        positions = torch.zeros(B, K, dtype=torch.long, device=device)
        loads = torch.full((B, K), float(self.cfg.capacity), device=device) 
        time = torch.zeros(B, dtype=torch.long, device=device)

        vehicles = VehicleState(
            positions=positions,
            loads=loads,
            time=time,
        )

        self.state = SVRPState(customers=customers, vehicles=vehicles)
        return self.state

    def step(self, actions: Tensor) -> Tuple[SVRPState, Tensor, Tensor, Dict[str, Any]]:
        """
        Thực hiện 1 bước:
        Returns:
            next_state: SVRPState
            rewards: [B] (negative cost)
            done: [B] bool
            info: dict (có thể chứa cost_step, total_unserved, v.v.)
        """
        # assert self.state is not None, "Call reset() before step()."
        # assert actions.dim() == 2, "actions shape must be [B, K]"

        state = self.state
        B, K = actions.shape
        # assert B == self.batch_size, "Batch size mismatch in step()."

        device = self.device

        demand = state.customers.demand
        travel_cost = state.customers.travel_cost
        positions = state.vehicles.positions
        loads = state.vehicles.loads
        time = state.vehicles.time

        step_cost = torch.zeros(B, device=device)

        for v in range(K):
            current_pos = positions[:, v]     # [B]
            next_pos = actions[:, v]          # [B]

            move_cost = travel_cost[torch.arange(B, device=device), current_pos, next_pos]  # [B]
            step_cost += move_cost

            is_idle = (current_pos == next_pos)
            step_cost[is_idle] += 10.0

            is_depot = (next_pos == 0)

            # case depot: refill
            loads[is_depot, v] = self.capacity

            # case customer
            is_customer = ~is_depot
            if is_customer.any():
                idx = torch.nonzero(is_customer, as_tuple=False).squeeze(1)
                b_idx = idx
                i_idx = next_pos[idx] 

                node_demand = demand[b_idx, i_idx]
                vehicle_load = loads[b_idx, v]

                delivered = torch.minimum(node_demand, vehicle_load)

                demand[b_idx, i_idx] = node_demand - delivered
                loads[b_idx, v] = vehicle_load - delivered

            positions[:, v] = next_pos

        time = time + 1
        self.timestep += 1

        self.state = SVRPState(
            customers=CustomerState(
                weather=state.customers.weather,
                demand=demand,
                travel_cost=travel_cost,
                coords=state.customers.coords,
                time_windows=state.customers.time_windows,
            ),
            vehicles=VehicleState(
                positions=positions,
                loads=loads,
                time=time,
            ),
        )

        # reward = -cost
        rewards = -step_cost  # [B]

        unserved = demand[:, 1:].sum(dim=1)  # [B]
        finished = (unserved <= 1e-6)
        timeout = (time >= self.cfg.max_horizon)
        done = finished | timeout

        info: Dict[str, Any] = {
            "step_travel_cost": step_cost.detach().cpu(),
            "unserved_demand": unserved.detach().cpu(),
            "finished": finished.detach().cpu(),
            "timeout": timeout.detach().cpu(),
        }

        return self.state, rewards, done, info

    def get_action_mask(self) -> Tensor:
            """
            Sinh mask hợp lệ cho action:
            Returns:
                mask: [B, K, N] với:
                - True: action hợp lệ
                - False: action không hợp lệ
            Logic cập nhật:
            - Node khách (i > 0): Chỉ hợp lệ nếu còn demand.
            - Depot (0): 
                + Luôn hợp lệ nếu xe HẾT hàng (load = 0) để về refill.
                + Luôn hợp lệ nếu HẾT sạch khách (finished) để về kết thúc.
                + BỊ CẤM (False) nếu xe CÒN hàng VÀ CÒN khách (để ép agent đi khách tiếp).
            """
            # assert self.state is not None, "Call reset() before get_action_mask()."

            state = self.state
            B = self.batch_size
            K = self.cfg.num_vehicles
            N = self.cfg.num_nodes
            device = self.device

            demand = state.customers.demand  # [B, N]
            loads = state.vehicles.loads     # [B, K]

            # 1. Khởi tạo mask: Mặc định True hết
            mask = torch.ones(B, K, N, dtype=torch.bool, device=device)

            # 2. Mask Customer: Cấm đi đến khách đã hết demand
            no_demand = (demand <= 1e-6)
            
            mask = mask & ~no_demand.unsqueeze(1).expand(-1, K, -1)

            # 3. Mask Depot (Node 0) - Logic "Ép đi khách" của bạn
            # Điều kiện để về depot:
            # a. Xe hết hàng (cần về nạp): loads <= epsilon
            is_empty_load = (loads <= 1e-6) # [B, K]
            
            # b. Đã hết sạch khách trên bản đồ: sum(demand) ~ 0
            total_demand = demand.sum(dim=1) # [B]
            is_all_served = (total_demand <= 1e-6) # [B]
            is_all_served = is_all_served.unsqueeze(1).expand(-1, K) # [B, K]
            
            # Ngược lại: Nếu (Còn hàng) VÀ (Còn khách) -> Cấm về 0
            can_visit_depot = is_empty_load | is_all_served
            
            mask[:, :, 0] = can_visit_depot

            mask[:, :, 0] = can_visit_depot

            return mask