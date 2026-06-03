#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dp_macadam.config import ExperimentConfig
from dp_macadam.data import get_mnist_loaders
from dp_macadam.model import AdaClipNet
from dp_macadam.optimizers import run_dpmacadam
from dp_macadam.privacy import estimate_epsilon
from dp_macadam.utils import format_sigma, free_memory, save_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run DP-MACAdam MNIST experiments.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.5])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--results-prefix", default="mnist_results")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def make_config(args):
    return ExperimentConfig(
        seeds=args.seeds,
        sigmas=args.sigmas,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        eta=args.eta,
        max_grad_norm=args.max_grad_norm,
        delta=args.delta,
        data_root=args.data_root,
        results_dir=args.results_dir,
        results_prefix=args.results_prefix,
        num_workers=args.num_workers,
    )


def print_summary(results_path):
    if not os.path.exists(results_path):
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    print(f"\n{results_path}")
    print(f"{'Algorithm':<25} {'Final Acc (mean +/- std)':>25}")
    print("-" * 52)

    for algo, all_seeds in results.items():
        final_accs = [seed_accs[-1] for seed_accs in all_seeds]
        mean = np.mean(final_accs)
        std = np.std(final_accs)
        print(f"{algo:<25} {mean:.4f} +/- {std:.4f}")


def main():
    args = parse_args()
    config = make_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset, test_dataset, train_loader, test_loader = get_mnist_loaders(
        batch_size=config.batch_size,
        data_root=config.data_root,
        num_workers=config.num_workers,
    )

    print(f"\nTrain size:    {len(train_dataset):,}")
    print(f"Test size:     {len(test_dataset):,}")
    print(f"Batch size:    {config.batch_size}")
    print(f"Batches/epoch: {len(train_loader)}")
    print(f"Seeds:         {config.seeds}")
    print(f"Epochs:        {config.num_epochs}")
    print(f"Model params:  {sum(p.numel() for p in AdaClipNet().parameters()):,}")

    loss_fn = nn.CrossEntropyLoss()
    algorithms = {
        "dp-macadam": run_dpmacadam,
    }

    for sigma in config.sigmas:
        sample_rate = config.batch_size / len(train_dataset)
        total_steps = config.num_epochs * len(train_loader)
        epsilon = estimate_epsilon(sigma, sample_rate, total_steps, config.delta)

        print(f"\nNoise scale: {sigma}")
        print(f"Privacy: epsilon ~= {epsilon:.2f}, delta = {config.delta} (PRV)")

        results_path = os.path.join(
            config.results_dir,
            f"{config.results_prefix}_sigma{format_sigma(sigma)}.json",
        )

        for algo_name, algo_fn in algorithms.items():
            print(f"\n{'=' * 60}")
            print(f"Algorithm: {algo_name}")
            print(f"{'=' * 60}")

            all_accs = []
            for seed in config.seeds:
                print(f"\n--- Seed {seed} ---")
                accs = algo_fn(
                    seed=seed,
                    sigma=sigma,
                    config=config,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    loss_fn=loss_fn,
                    device=device,
                )
                all_accs.append(accs)

            save_results({algo_name: all_accs}, results_path)
            free_memory(device)

        print("\n" + "=" * 60)
        print("All experiments complete.")
        print(f"Results saved to: {results_path}")
        print("=" * 60)
        print_summary(results_path)


if __name__ == "__main__":
    main()

