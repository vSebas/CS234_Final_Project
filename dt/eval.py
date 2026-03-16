#!/usr/bin/env python3
"""
Evaluation script for Decision Transformer.

Computes action-prediction metrics on held-out data.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from dt.model import DecisionTransformer, DTConfig
from dt.dataset import TrajectoryDataset, DatasetStats


def evaluate_model(
    model: DecisionTransformer,
    dataloader: DataLoader,
    stats: DatasetStats,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate DT model on a dataset.

    Returns:
        Dict with action prediction metrics.
    """
    model.eval()

    all_action_errors = []
    all_action_preds = []
    all_action_targets = []

    with torch.no_grad():
        for batch in dataloader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["rtg"].to(device)
            timesteps = batch["timesteps"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            action_preds, _ = model(states, actions, rtg, timesteps, attention_mask)

            # Mask for valid tokens
            mask = attention_mask.bool()

            # Action errors
            action_error = (action_preds - actions) ** 2
            for b in range(states.shape[0]):
                valid_mask = mask[b]
                if valid_mask.sum() > 0:
                    all_action_errors.append(action_error[b, valid_mask].cpu().numpy())
                    all_action_preds.append(action_preds[b, valid_mask].cpu().numpy())
                    all_action_targets.append(actions[b, valid_mask].cpu().numpy())

    # Concatenate all errors
    all_action_errors = np.concatenate(all_action_errors, axis=0)
    all_action_preds = np.concatenate(all_action_preds, axis=0)
    all_action_targets = np.concatenate(all_action_targets, axis=0)

    # Denormalize for interpretable metrics
    if stats is not None:
        action_preds_denorm = all_action_preds * stats.action_std + stats.action_mean
        action_targets_denorm = all_action_targets * stats.action_std + stats.action_mean
        action_errors_denorm = (action_preds_denorm - action_targets_denorm) ** 2
    else:
        action_errors_denorm = all_action_errors

    # Compute metrics
    metrics = {
        # Normalized MSE (what the model optimizes)
        "action_mse_normalized": float(np.mean(all_action_errors)),

        # Per-dimension action errors (normalized)
        "action_delta_mse": float(np.mean(all_action_errors[:, 0])),
        "action_fx_mse": float(np.mean(all_action_errors[:, 1])),

        # Denormalized action errors (interpretable units)
        "action_delta_rmse_rad": float(np.sqrt(np.mean(action_errors_denorm[:, 0]))),
        "action_fx_rmse_kn": float(np.sqrt(np.mean(action_errors_denorm[:, 1]))),

        # Sample counts
        "n_samples": len(all_action_errors),
    }

    return metrics


def load_model_and_stats(checkpoint_path: Path) -> tuple:
    """Load model and stats from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model config
    model_config_dict = checkpoint.get("model_config", {})
    model_config = DTConfig(**model_config_dict)

    # Build and load model
    model = DecisionTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Load stats
    stats_path = checkpoint_path.parent / "dataset_stats.npz"
    if stats_path.exists():
        stats = DatasetStats.load(stats_path)
    else:
        stats = None

    return model, stats, model_config, device


def main():
    parser = argparse.ArgumentParser(description="Evaluate Decision Transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help=(
            "Dataset shard directory, comma-separated list of shard directories, "
            "or a root directory containing shard subdirectories."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit episodes for quick eval")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("Decision Transformer Evaluation")
    print("=" * 60)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    model, stats, model_config, device = load_model_and_stats(checkpoint_path)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {device}")

    # Load dataset
    print(f"\nLoading dataset source: {args.data_dir}")
    dataset = TrajectoryDataset(
        args.data_dir,
        context_length=model_config.context_length,
        normalize=True,
        stats=stats,
        max_episodes=args.max_episodes,
    )
    print(f"Dataset: {len(dataset)} samples from {len(dataset.episodes)} episodes")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_model(model, dataloader, stats, device)

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"\nAction Prediction:")
    print(f"  MSE (normalized):     {metrics['action_mse_normalized']:.6f}")
    print(f"  Delta RMSE:           {metrics['action_delta_rmse_rad']:.6f} rad ({np.degrees(metrics['action_delta_rmse_rad']):.3f} deg)")
    print(f"  Fx RMSE:              {metrics['action_fx_rmse_kn']:.6f} kN")

    print(f"\nSamples evaluated: {metrics['n_samples']}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
