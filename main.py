import argparse
import torch

from env.scenario import ScenarioConfig
from training.runner import TrainConfig, ExperimentRunner


def build_scenario_from_args(args) -> ScenarioConfig:
    return ScenarioConfig(
        num_customers=args.num_customers,
        num_vehicles=args.num_vehicles,
        capacity=args.capacity,
        fixed_customers=True,
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--baseline_lr", type=float, default=1e-3)
    parser.add_argument("--entropy_weight", type=float, default=1e-2)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--val_episodes", type=int, default=20, help="Số lượng mẫu để validate mỗi khi eval")
    parser.add_argument("--test_episodes", type=int, default=100, help="Số lượng mẫu để test cuối cùng")
    # parser.add_argument("--max_horizon", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument(
        "--value_update_freq",
        type=int,
        default=5,
        help="Update value network every N epochs (default: 5)",
    )
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
        val_episodes=args.val_episodes,
        test_episodes=args.test_episodes,
        save_dir=args.save_dir,
        device=device,
        seed=args.seed,
        d_model=args.d_model,
        value_update_freq=args.value_update_freq,
    )

    runner = ExperimentRunner(
        scenario=scenario,
        train_cfg=train_cfg,
        inference_name=args.inference,
        num_samples_sampling=args.num_samples,
    )

    if args.mode == "train":
        runner.train()
        print("\n=== Training Finished. Running Final Test on TEST SET ===")
        # Get per-instance costs for win-rate calculation
        mean_costs, per_instance_costs = runner.evaluate(
            num_instances=0, 
            dataset=runner.test_set,
            return_per_instance=True
        )
        
        # Calculate and display win-rate matrix
        runner.calculate_and_display_winrate(per_instance_costs)
        
    else:  # mode == "test"
        if args.load is None:
            raise ValueError("--mode test cần --load path_prefix")
        runner.load(args.load)
        
        print(f"\n=== Running Evaluation on TEST SET (Seed {args.seed}) ===")
        # Get per-instance costs for win-rate calculation
        mean_costs, per_instance_costs = runner.evaluate(
            num_instances=0, 
            dataset=runner.test_set,
            return_per_instance=True
        )
        
        # Calculate and display win-rate matrix
        runner.calculate_and_display_winrate(per_instance_costs)


if __name__ == "__main__":
    main()
