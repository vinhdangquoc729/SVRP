import argparse
import torch

from env.scenario import ScenarioConfig
from training.runner import TrainConfig, ExperimentRunner


def build_scenario_from_args(args) -> ScenarioConfig:
    return ScenarioConfig(
        num_customers=args.num_customers,
        num_vehicles=args.num_vehicles,
        capacity=args.capacity,
        max_horizon=args.max_horizon,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    # Scenario
    parser.add_argument("--num_customers", type=int, default=20)
    parser.add_argument("--num_vehicles", type=int, default=1)
    parser.add_argument("--capacity", type=float, default=5000.0)
    parser.add_argument("--max_horizon", type=int, default=100)

    # Train config
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--baseline_lr", type=float, default=1e-3)
    parser.add_argument("--entropy_weight", type=float, default=1e-2)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--test_episodes", type=int, default=10)
    # parser.add_argument("--max_horizon", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="use cuda if available",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
    )
    parser.add_argument(
        "--inference",
        type=str,
        choices=["greedy", "sampling"],
        default="greedy",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="path prefix to load model from (for test mode)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="num samples for sampling inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    scenario = build_scenario_from_args(args)

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        baseline_lr=args.baseline_lr,
        entropy_weight=args.entropy_weight,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        test_episodes=args.test_episodes,
        save_dir=args.save_dir,
        device=device,
        seed=args.seed,
        d_model=args.d_model,
    )

    runner = ExperimentRunner(
        scenario=scenario,
        train_cfg=train_cfg,
        inference_name=args.inference,
        num_samples_sampling=args.num_samples,
    )

    if args.mode == "train":
        runner.train()
    else:
        if args.load is None:
            raise ValueError("--mode test cần --load path_prefix")
        runner.load(args.load)
        runner.evaluate(train_cfg.test_episodes)


if __name__ == "__main__":
    main()
