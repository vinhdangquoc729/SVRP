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

        result = self.generator.generate(batch_size)
        if len(result) == 6:
            weather, demand, travel_cost, coords, time_windows, customer_type = result
        else:
            weather, demand, travel_cost, coords, time_windows = result
            customer_type = None

        customers = CustomerState(
            weather=weather,          # [B, W]
            demand=demand,            # [B, N]
            travel_cost=travel_cost,  # [B, N, N]
            coords=coords,            # [B, N, 2]
            time_windows=time_windows # [B, N, H] hoặc None
        )
        if customer_type is not None:
            setattr(customers, "customer_type", customer_type)

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
        time = state.vehicles.time  # [B], current timestep BEFORE increment

        step_cost = torch.zeros(B, device=device)

        for v in range(K):
            current_pos = positions[:, v]     # [B]
            next_pos = actions[:, v]          # [B]

            # travel cost
            move_cost = travel_cost[torch.arange(B, device=device), current_pos, next_pos]  # [B]
            step_cost += move_cost

            # idle penalty (nếu không di chuyển)
            is_idle = (current_pos == next_pos)
            step_cost[is_idle] += 10.0

            is_depot = (next_pos == 0)

            # case depot: refill
            loads[is_depot, v] = self.capacity

            is_customer = ~is_depot
            if is_customer.any():
                idx = torch.nonzero(is_customer, as_tuple=False).squeeze(1)
                b_idx = idx
                i_idx = next_pos[idx]

                node_demand = demand[b_idx, i_idx]
                vehicle_load = loads[b_idx, v]

                need_refill = (vehicle_load < node_demand)

                if need_refill.any():
                    b_need = b_idx[need_refill]
                    i_need = i_idx[need_refill]

                    step_cost[b_need] += (
                        travel_cost[b_need, i_need, 0] +
                        travel_cost[b_need, 0, i_need]
                    )

                    time[b_need] += 2
                    loads[b_need, v] = self.capacity

                delivered = demand[b_idx, i_idx]

                demand[b_idx, i_idx] = 0.0
                loads[b_idx, v] = loads[b_idx, v] - delivered

                tw = state.customers.time_windows
                if tw is not None:
                    H = tw.size(-1)
                    t_idx = time[b_idx].clamp(max=H - 1)

                    tw_open = tw[b_idx, i_idx, t_idx] > 0.5
                    closed = ~tw_open
                    if closed.any():
                        penalty_val = float(getattr(self.cfg, "closed_tw_penalty", 0.25))
                        step_cost[b_idx[closed]] += penalty_val

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
                customer_types=state.customers.customer_types
            ),
            vehicles=VehicleState(
                positions=positions,
                loads=loads,
                time=time,
            ),
        )
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
        NOTE: Không cấm do time windows ở đây — chúng ta chỉ phạt khi đến sai giờ (soft penalty).
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

        no_demand = (demand <= 1e-6)  # [B, N]
        mask = mask & ~no_demand.unsqueeze(1).expand(-1, K, -1)

        is_empty_load = (loads <= 1e-6)  # [B, K]

        total_demand = demand.sum(dim=1)  # [B]
        is_all_served = (total_demand <= 1e-6)  # [B]
        is_all_served = is_all_served.unsqueeze(1).expand(-1, K)  # [B, K]

        can_visit_depot = is_empty_load | is_all_served  # [B, K]

        mask[:, :, 0] = can_visit_depot

        return mask
    
    def reset_by_state(self, state: SVRPState) -> SVRPState:
        """
        Nạp một trạng thái cụ thể (đã lưu từ trước) vào môi trường.
        """
        self.state = state.clone() # Clone để không làm hỏng dữ liệu gốc trong list
        self.batch_size = state.customers.demand.size(0)
        self.timestep = 0 # Reset thời gian về 0
        return self.state