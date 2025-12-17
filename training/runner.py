from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, List, Tuple
import random
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from env.scenario import ScenarioConfig
from env.state import SVRPState
from env.svrp_env import SVRPEnvironment
from models.policy import PolicyNetwork
from training.reinforce import ReinforceTrainer
from inference import GreedyInference, SamplingInference, HybridEvolutionaryInference
import matplotlib.pyplot as plt
import math
from env.rendering import plot_instance

@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    baseline_lr: float = 1e-3
    entropy_weight: float = 1e-2
    max_steps: int = 100
    log_interval: int = 10
    eval_interval: int = 20
    val_episodes: int = 20    
    test_episodes: int = 100  
    save_dir: str = "checkpoints"
    device: str = "cpu"
    seed: int = 42
    d_model: int = 128

InferenceName = Literal["greedy", "sampling"]

class ExperimentRunner:
    def __init__(
        self,
        scenario: ScenarioConfig,
        train_cfg: TrainConfig,
        inference_name: InferenceName = "greedy",
        num_samples_sampling: int = 16,
        log_dir: Optional[str] = "runs/svrp_rl",
    ):
        self.scenario = scenario
        self.cfg = train_cfg

        torch.manual_seed(train_cfg.seed)
        np.random.seed(train_cfg.seed)
        random.seed(train_cfg.seed)
        self.device = torch.device(train_cfg.device)
        self.env = SVRPEnvironment(scenario, device=self.device)

        self.policy = PolicyNetwork(
            scenario=scenario,
            d_model=train_cfg.d_model,
        ).to(self.device)

        # Lưu ý: ReinforceTrainer phải là phiên bản đã sửa (ko có zero_grad bên trong)
        self.trainer = ReinforceTrainer(
            policy=self.policy,
            scenario=scenario,
            lr=train_cfg.lr,
            entropy_weight=train_cfg.entropy_weight,
            device=self.device,
        )

        # --- INFERENCE STRATEGIES (Cho Evaluation) ---
        self.inference_rl = SamplingInference(
            self.policy, scenario, device=self.device, num_samples=20
        )
        self.inference_ea = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=20, generations=30,
            init_method="rl", strategy="ga"
        )
        self.inference_pure_ga = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=20, generations=30, 
            init_method="random", strategy="ga"
        )
        self.inference_rl_mpeax = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=20, generations=30, 
            init_method="rl", strategy="mpeax"
        )
        self.inference_pure_mpeax = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=20, generations=30, 
            init_method="random", strategy="mpeax"
        )
        self.inference_rl_special = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=20, generations=30, 
            init_method="rl", strategy="special_hybrid"
        )
        self.inference_pure_special = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=20, generations=30, 
            init_method="random", strategy="special_hybrid"
        )
        
        # --- FAST GA FOR TRAINING (Dùng để sinh demo nhanh trong 20 epoch đầu) ---
        self.inference_fast_ga = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=10, generations=10, 
            init_method="random", strategy="mpeax"
        )

        self.save_dir = Path(train_cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
        self.best_eval_cost: float = float("inf")

        print(f"Generating FIXED datasets with seed {train_cfg.seed}...")
        
        # 1. Tạo Validation Set (dùng để eval trong lúc train)
        self.validation_set = []
        for _ in range(train_cfg.val_episodes):
            state = self.env.reset(batch_size=1)
            self.validation_set.append(state.clone())
        print(f"  -> Validation set: {len(self.validation_set)} episodes generated.")

        # 2. Tạo Test Set (dùng để test sau khi train xong)
        self.test_set = []
        for _ in range(train_cfg.test_episodes): # Hoặc một số lượng khác nếu muốn
            state = self.env.reset(batch_size=1)
            self.test_set.append(state.clone())
        print(f"  -> Test set: {len(self.test_set)} episodes generated.")
    def train(self):
        # Cấu hình Chiến thuật
        WARMUP_EPOCHS = 200     # Chỉ dùng GA hướng dẫn trong 20 epoch đầu
        LAMBDA_IL = 0.8    # Trọng số Imitation Learning
        NUM_DEMO_PER_BATCH = 16 # Số lượng mẫu chạy GA

        print(f"Starting Training: {self.cfg.epochs} epochs")
        print(f" - Phase 1 (Epoch 1-{WARMUP_EPOCHS}): Mixed Strategy (RL + GA Guidance)")
        print(f" - Phase 2 (Epoch {WARMUP_EPOCHS+1}+): Pure RL (Reinforce)")

        history =  {
            "RL": [],
            "RL_GA": [], "Pure_GA": [],
            "RL_MPEAX": [], "Pure_MPEAX": [],
            "RL_Special": [], "Pure_Special": [],
        }

        for epoch in range(1, self.cfg.epochs + 1):
            
            # Reset Env & Lấy State đầu vào cho cả Epoch
            state = self.env.reset(batch_size=self.cfg.batch_size)
            initial_state_clone = state.clone() 
            
            # Reset Gradients chung
            self.trainer.policy_optim.zero_grad()
            self.trainer.value_optim.zero_grad()
            
            loss_il = 0.0
            phase_name = "RL"

            # ===============================================
            # KIỂM TRA GIAI ĐOẠN (PHASE)
            # ===============================================
            if epoch <= WARMUP_EPOCHS:
                phase_name = "Mixed"
                
                # --- PHASE 1: CHẠY GA ĐỂ LẤY DEMO ---
                demos = []
                indices_to_solve = list(range(self.cfg.batch_size))[:NUM_DEMO_PER_BATCH]
                
                ga_env = SVRPEnvironment(self.scenario, device=self.device)
                
                for idx in indices_to_solve:
                    ga_env.reset(batch_size=1)
                    # Copy data state thủ công
                    with torch.no_grad():
                        ga_env.state.customers.demand.data.copy_(initial_state_clone.customers.demand[idx].unsqueeze(0))
                        ga_env.state.customers.travel_cost.data.copy_(initial_state_clone.customers.travel_cost[idx].unsqueeze(0))
                        if hasattr(ga_env.state.customers, 'loc'):
                             ga_env.state.customers.loc.data.copy_(initial_state_clone.customers.loc[idx].unsqueeze(0))
                        if hasattr(ga_env.state.vehicles, 'loc'):
                             ga_env.state.vehicles.loc.data.copy_(initial_state_clone.vehicles.loc[idx].unsqueeze(0))

                    # Chạy GA nhanh
                    routes, _, _ = self.inference_fast_ga.solve_one(ga_env)
                    route_flat = routes[0]
                    demos.append((idx, route_flat))
                
                # Tính Gradient Imitation (Cộng dồn vào Policy)
                self.env.reset_by_state(initial_state_clone)
                loss_il = self.trainer.backward_imitation(self.env, demos, lambda_il=LAMBDA_IL)
            
            else:
                phase_name = "PureRL"
                # --- PHASE 2: KHÔNG LÀM GÌ CẢ (SKIP GA) ---
                loss_il = 0.0

            # ===============================================
            # CHẠY RL (LUÔN CHẠY TRONG CẢ 2 PHASE)
            # ===============================================
            self.env.reset_by_state(initial_state_clone)
            
            # Chạy RL Rollout và Backward tích lũy
            stats = self.trainer.train_batch(
                self.env, 
                batch_size=self.cfg.batch_size,
                max_steps=self.cfg.max_steps
            )
            
            # ===============================================
            # CẬP NHẬT TRỌNG SỐ
            # ===============================================
            self.trainer.step_optimizers()

            # Logging
            print(
                f"[Epoch {epoch:03d}/{self.cfg.epochs}] ({phase_name}) "
                f"Rw: {stats.mean_reward:.2f} | "
                f"L_RL: {stats.policy_loss:.2f} | "
                f"L_IL: {loss_il:.4f}"
            )

            if self.writer is not None:
                self.writer.add_scalar("train/reward", stats.mean_reward, epoch)
                self.writer.add_scalar("train/policy_loss", stats.policy_loss, epoch)
                self.writer.add_scalar("train/imitation_loss", loss_il, epoch)
                self.writer.add_scalar("train/baseline_loss", stats.value_loss, epoch)

            # --- EVALUATION ---
            if epoch % self.cfg.eval_interval == 0:
                print(f"--- Evaluating on VALIDATION set (Epoch {epoch}) ---")
                eval_mean_cost = self.evaluate(self.cfg.test_episodes, dataset=self.validation_set)
                history["RL"].append(eval_mean_cost["RL"])
                history["RL_GA"].append(eval_mean_cost["RL_GA"])
                history["Pure_GA"].append(eval_mean_cost["Pure_GA"])
                history["RL_MPEAX"].append(eval_mean_cost["RL_MPEAX"])
                history["Pure_MPEAX"].append(eval_mean_cost["Pure_MPEAX"])
                history["RL_Special"].append(eval_mean_cost["RL_Special"])
                history["Pure_Special"].append(eval_mean_cost["Pure_Special"])

                if self.writer is not None:
                    self.writer.add_scalar("eval/mean_cost", eval_mean_cost["RL_Special"], epoch)

                if eval_mean_cost["RL_Special"] < self.best_eval_cost:
                    self.best_eval_cost = eval_mean_cost["RL_Special"]
                    best_path = self.save_dir / "model_best"
                    self._save(best_path)
                    print(f"  -> New best eval cost {eval_mean_cost['RL_Special']:.4f}, saved to {best_path}")

            if epoch % self.cfg.log_interval == 0:
                ckpt_path = self.save_dir / f"model_epoch_{epoch}"
                self._save(ckpt_path)

        final_path = self.save_dir / "model_final"
        self._save(final_path)
        print(f"Training done, final model saved to {final_path}")

        # Calculate win rate (percentage of times winning) matrix of each pair algorithms like ("RL", "RL_GA"), ("RL", "Pure_GA"), ...
        algorithms = ["RL", "RL_GA", "Pure_GA", "RL_MPEAX", "Pure_MPEAX", "RL_Special", "Pure_Special"]
        win_rate = {}

        for algo1 in algorithms:
            win_rate[algo1] = {}
            for algo2 in algorithms:
                if algo1 == algo2:
                    win_rate[algo1][algo2] = 0.5
                else:
                    win_rate[algo1][algo2] = (history[algo1] < history[algo2]).sum() / len(history[algo1])

        # Print win rate matrix
        print("\nWin rate matrix:")
        for algo1 in algorithms:
            print(" ".join(f"{win_rate[algo1][algo2]:.2f}" for algo2 in algorithms))

        # Save win rate matrix to file
        with open(self.save_dir / "win_rate_matrix.txt", "w") as f:
            for algo1 in algorithms:
                f.write(" ".join(f"{win_rate[algo1][algo2]:.2f}" for algo2 in algorithms) + "\n")

    def evaluate(self, num_instances: int, dataset: list[SVRPState] = None) -> float:
        """
        Nếu có dataset, chạy eval trên dataset đó.
        Nếu không có dataset, tự sinh ngẫu nhiên num_instances (logic cũ).
        """
        self.policy.eval()
        costs = {
            "RL": 0.0,
            "RL_GA": 0.0, "Pure_GA": 0.0,
            "RL_MPEAX": 0.0, "Pure_MPEAX": 0.0,
            "RL_Special": 0.0, "Pure_Special": 0.0,
        }

        # Xác định nguồn dữ liệu để chạy
        if dataset is not None:
            # Nếu truyền dataset cố định vào thì dùng nó
            iterator = dataset
            actual_num = len(dataset)
        else:
            # Nếu không thì sinh ngẫu nhiên (như cũ)
            iterator = [self.env.reset(batch_size=1) for _ in range(num_instances)]
            actual_num = num_instances

        for i, state_base in enumerate(iterator):
            # Không gọi self.env.reset() ở đây nữa mà dùng state từ list
            fixed_state = state_base.clone()
            
            def run_strat(name, inf_obj, save_img=False):
                self.env.reset_by_state(fixed_state)
                routes, cost, _ = inf_obj.solve_one(self.env)
                costs[name] += cost
                if save_img:
                     plot_instance(fixed_state, 0, routes, title=f"{name} (Cost {cost:.2f})", 
                                   save_path=str(self.save_dir / f"inst_{i}_{name}.png"))
                return cost
            
            save = (i == 0) 
            
            # Chạy các chiến thuật
            run_strat("RL", self.inference_rl, save)
            run_strat("RL_GA", self.inference_ea, save)
            run_strat("Pure_GA", self.inference_pure_ga, save)
            run_strat("RL_MPEAX", self.inference_rl_mpeax, save)
            run_strat("Pure_MPEAX", self.inference_pure_mpeax, save)
            run_strat("RL_Special", self.inference_rl_special, save)
            run_strat("Pure_Special", self.inference_pure_special, save)

        means = {k: v / actual_num for k, v in costs.items()}
        print(f"===> Eval Results (N={actual_num}):")
        print(f"  1. RL              : {means['RL']:.4f}")
        print(f"  2. RL + GA         : {means['RL_GA']:.4f}")
        print(f"  3. Pure GA         : {means['Pure_GA']:.4f}")
        print(f"  4. RL + MPEAX      : {means['RL_MPEAX']:.4f}")
        print(f"  5. Pure MPEAX      : {means['Pure_MPEAX']:.4f}")
        print(f"  6. RL + Special    : {means['RL_Special']:.4f}")
        print(f"  7. Pure Special    : {means['Pure_Special']:.4f}")
        
        return means

    def _save(self, path_prefix: Path):
        path_prefix = Path(path_prefix)
        self.trainer.save(str(path_prefix))

    def load(self, path_prefix: str):
        self.trainer.load(path_prefix)