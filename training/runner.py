from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type

import torch
from torch.utils.tensorboard import SummaryWriter

from env.scenario import ScenarioConfig
from env.svrp_env import SVRPEnvironment
from models.policy import PolicyNetwork
from training.reinforce import ReinforceTrainer
from inference import GreedyInference, SamplingInference
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
            # embedding_dim=train_cfg.d_model,
            lr=train_cfg.lr,
            # baseline_lr=train_cfg.baseline_lr,
            entropy_weight=train_cfg.entropy_weight,
            device=self.device,
        )

        if inference_name == "greedy":
            self.inference = GreedyInference(self.policy, scenario, device=self.device)
        elif inference_name == "sampling":
            self.inference = SamplingInference(
                self.policy,
                scenario,
                device=self.device,
                num_samples=num_samples_sampling,
            )
        else:
            raise ValueError(f"Unknown inference strategy: {inference_name}")

        self.save_dir = Path(train_cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None

        self.best_eval_cost: float = float("inf")

    def train(self):
        for epoch in range(1, self.cfg.epochs + 1):
            stats = self.trainer.train_batch(
                env=self.env,
                batch_size=self.cfg.batch_size,
                max_steps=self.cfg.max_steps,
            )

            mean_reward = stats.mean_reward
            policy_loss = stats.policy_loss
            baseline_loss = stats.value_loss  # value_net loss

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
        total_cost = 0.0

        for i in range(num_instances):
            routes, cost, info = self.inference.solve_one(self.env)
            total_cost += cost

            # print(
            #     f"[Eval {i+1}/{num_instances}] cost = {cost:.4f} | "
            #     f"routes = {routes}"
            # )

            self.env.reset(batch_size=1)

            # Save image for the first 3 instances
            if i < 3:
                save_path = self.save_dir / f"instance_{i+1}.png"
                plot_instance(self.env.state, batch_idx=0, routes=routes, save_path=str(save_path))

        mean_cost = total_cost / num_instances
        print(f"===> Eval mean cost over {num_instances} instances: {mean_cost:.4f}")
        return mean_cost

    def _save(self, path_prefix: Path):
        path_prefix = Path(path_prefix)
        self.trainer.save(str(path_prefix))

    def load(self, path_prefix: str):
        self.trainer.load(path_prefix)

