#!/usr/bin/env python3
"""
Plot DT training curves from metrics.jsonl.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DT training curves")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training run directory containing metrics.jsonl",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Could not find metrics log: {metrics_path}")

    train_steps = []
    action_loss = []

    epochs = []
    val_action = []

    with open(metrics_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            event = rec.get("event")
            if event == "train_step":
                train_steps.append(rec["global_step"])
                action_loss.append(rec["action_loss"])
            elif event == "epoch_end":
                epochs.append(rec["epoch"] + 1)
                val_action.append(rec["val_action_loss"])

    if not train_steps:
        raise ValueError(f"No train_step records found in {metrics_path}")

    train_plot_path = run_dir / "loss_curves.png"
    plt.figure(figsize=(9, 5))
    plt.plot(train_steps, action_loss, label="train action loss", linewidth=1.2)
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.title("Training Action Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(train_plot_path, dpi=180)
    plt.close()

    if epochs:
        val_plot_path = run_dir / "val_loss_curves.png"
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, val_action, marker="o", label="val action loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Action Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(val_plot_path, dpi=180)
        plt.close()
        print(val_plot_path)

    print(train_plot_path)


if __name__ == "__main__":
    main()
