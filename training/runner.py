from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type

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
    test_episodes: int = 100
    save_dir: str = "checkpoints"
    device: str = "cpu"
    seed: int = 42
    d_model: int = 128  # embedding dim cho policy

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
        self.device = torch.device(train_cfg.device)
        self.env = SVRPEnvironment(scenario, device=self.device)

        self.policy = PolicyNetwork(
            scenario=scenario,
            d_model=train_cfg.d_model,
        ).to(self.device)

        self.trainer = ReinforceTrainer(
            policy=self.policy,
            scenario=scenario,
            lr=train_cfg.lr,
            entropy_weight=train_cfg.entropy_weight,
            device=self.device,
        )

        # 1. RL (Baseline)
        self.inference_rl = SamplingInference(
            self.policy, scenario, device=self.device, num_samples=20
        )
        
        # 2. RL + GA (Original)
        self.inference_ea = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=20, generations=30,
            init_method="rl", strategy="ga"
        )

        # 3. Pure GA (Original)
        self.inference_pure_ga = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=20, generations=30, 
            init_method="random", strategy="ga"
        )

        # 4. RL + MPEAX
        self.inference_rl_mpeax = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=20, generations=30, 
            init_method="rl", strategy="mpeax"
        )

        # 5. Pure MPEAX
        self.inference_pure_mpeax = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=20, generations=30, 
            init_method="random", strategy="mpeax"
        )

        # 6. RL + Special Hybrid
        self.inference_rl_special = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=20, generations=30, 
            init_method="rl", strategy="special_hybrid"
        )
        
        # 7. Pure Special Hybrid
        self.inference_pure_special = HybridEvolutionaryInference(
            self.policy, scenario, device=self.device,
            num_samples_init=20, generations=30, 
            init_method="random", strategy="special_hybrid"
        )

        self.save_dir = Path(train_cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None

        self.best_eval_cost: float = float("inf")

        print(f"Generating fixed validation set of {train_cfg.test_episodes} episodes...")
        self.validation_set = []
        for _ in range(train_cfg.test_episodes):
            state = self.env.reset(batch_size=1)
            self.validation_set.append(state.clone())
        print("Validation set generated and stored.")
    def train(self):
        for epoch in range(1, self.cfg.epochs + 1):
            stats = self.trainer.train_batch(
                env=self.env,
                batch_size=self.cfg.batch_size,
                max_steps=self.cfg.max_steps,
            )

            mean_reward = stats.mean_reward
            policy_loss = stats.policy_loss
            baseline_loss = stats.value_loss 

            # Logging console
            print(
                f"[Epoch {epoch:03d}/{self.cfg.epochs}] "
                f"reward (train) = {mean_reward:.4f} | "
                f"policy_loss = {policy_loss:.4f} | "
                f"baseline_loss = {baseline_loss:.4f}"
            )

            if self.writer is not None:
                self.writer.add_scalar("train/reward", mean_reward, epoch)
                self.writer.add_scalar("train/policy_loss", policy_loss, epoch)
                self.writer.add_scalar("train/baseline_loss", baseline_loss, epoch)

            if epoch % self.cfg.eval_interval == 0:
                eval_mean_cost = self.evaluate(self.cfg.test_episodes)
                if self.writer is not None:
                    self.writer.add_scalar("eval/mean_cost", eval_mean_cost, epoch)

                if eval_mean_cost < self.best_eval_cost:
                    self.best_eval_cost = eval_mean_cost
                    best_path = self.save_dir / "model_best"
                    self._save(best_path)
                    print(
                        f"  -> New best eval cost {eval_mean_cost:.4f}, "
                        f"saved to {best_path}"
                    )

            if epoch % self.cfg.log_interval == 0:
                ckpt_path = self.save_dir / f"model_epoch_{epoch}"
                self._save(ckpt_path)

        final_path = self.save_dir / "model_final"
        self._save(final_path)
        print(f"Training done, final model saved to {final_path}")

    def evaluate(self, num_instances: int) -> float:
        self.policy.eval()
        costs = {
            "RL": 0.0,
            "RL_GA": 0.0, "Pure_GA": 0.0,
            "RL_MPEAX": 0.0, "Pure_MPEAX": 0.0,
            "RL_Special": 0.0, "Pure_Special": 0.0,
        }

        for i in range(num_instances):
            state_base = self.env.reset(batch_size=1)
            fixed_state = state_base.clone()
            def run_strat(name, inf_obj, save_img=False):
                self.env.reset_by_state(fixed_state)
                
                routes, cost, _ = inf_obj.solve_one(self.env)
                costs[name] += cost
                if save_img:
                     plot_instance(fixed_state, 0, routes, title=f"{name} (Cost {cost:.2f})", 
                                   save_path=str(self.save_dir / f"inst_{i}_{name}.png"))
                return cost
            
            save = (i == 0) # Save image for 1st instance
            
            # 1. RL
            run_strat("RL", self.inference_rl, save)
            
            # 2. RL + GA
            run_strat("RL_GA", self.inference_ea, save)
            
            # 3. Pure GA
            run_strat("Pure_GA", self.inference_pure_ga, save)
            
            # 4. RL + MPEAX
            run_strat("RL_MPEAX", self.inference_rl_mpeax, save)
            
            # 5. Pure MPEAX
            run_strat("Pure_MPEAX", self.inference_pure_mpeax, save)
            
            # 6. RL + Special Hybrid
            run_strat("RL_Special", self.inference_rl_special, save)
            
            # 7. Pure Special Hybrid
            run_strat("Pure_Special", self.inference_pure_special, save)

        means = {k: v / num_instances for k, v in costs.items()}
        print(f"===> Eval Results (N={num_instances}):")
        print(f"  1. RL              : {means['RL']:.4f}")
        print(f"  2. RL + GA         : {means['RL_GA']:.4f}")
        print(f"  3. Pure GA         : {means['Pure_GA']:.4f}")
        print(f"  4. RL + MPEAX      : {means['RL_MPEAX']:.4f}")
        print(f"  5. Pure MPEAX      : {means['Pure_MPEAX']:.4f}")
        print(f"  6. RL + Special    : {means['RL_Special']:.4f}")
        print(f"  7. Pure Special    : {means['Pure_Special']:.4f}")
        
        return means["RL_Special"]
    def _save(self, path_prefix: Path):
        path_prefix = Path(path_prefix)
        self.trainer.save(str(path_prefix))

    def load(self, path_prefix: str):
        self.trainer.load(path_prefix)