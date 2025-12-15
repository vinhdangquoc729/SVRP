from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor


@dataclass
class CustomerState:
    weather: Tensor
    demand: Tensor
    travel_cost: Tensor
    coords: Tensor
    time_windows: Optional[Tensor] = None
    customer_types: Optional[Tensor] = None

@dataclass
class VehicleState:
    positions: Tensor
    loads: Tensor
    time: Tensor

@dataclass
class SVRPState:
    customers: CustomerState
    vehicles: VehicleState

    def clone(self) -> "SVRPState":
        return SVRPState(
            customers=CustomerState(
                weather=self.customers.weather.clone(),
                demand=self.customers.demand.clone(),
                travel_cost=self.customers.travel_cost.clone(),
                coords=self.customers.coords.clone(),
                time_windows=self.customers.time_windows.clone() if self.customers.time_windows is not None else None,
                customer_types=self.customers.customer_types.clone() if self.customers.customer_types is not None else None,
            ),
            vehicles=VehicleState(
                positions=self.vehicles.positions.clone(),
                loads=self.vehicles.loads.clone(),
                time=self.vehicles.time.clone(),
            ),
        )
