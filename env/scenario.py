from dataclasses import dataclass

@dataclass
class ScenarioConfig:
    """
    Basic configuration for a SVRP scenario.
    """
    num_customers: int 
    num_vehicles: int
    capacity: float
    
    weather_dim: int = 3
    a_ratio: float = 0.6 
    b_ratio: float = 0.2
    gamma_ratio: float = 0.2

    max_horizon: int = 100
    fixed_customers: bool = False
    time_windows: bool = True 

    num_customer_types: int = 3
    tw_jitter: int = 3

    device: str = "cpu"

    @property
    def num_nodes(self) -> int:
        return self.num_customers + 1
