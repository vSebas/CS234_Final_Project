#!/usr/bin/env python3
"""
Training script for Decision Transformer.

Based on PLAN.md Section 3:
- Two-head outputs: action prediction + state prediction
- Loss: MSE on actions + lambda_x * MSE on state predictions
- AdamW optimizer with warmup
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from dt.model import DecisionTransformer, DTConfig
from dt.dataset import TrajectoryDataset, DatasetStats, create_dataloaders


@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    data_dir: str = "data/datasets"
    output_dir: str = "dt/checkpoints"

    # Model
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.1
    context_length: int = 30
    max_ep_len: int = 300

    # Loss
    lambda_x: float = 0.5  # Weight on state prediction loss

    # Optimization
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    # Schedule
    num_epochs: int = 100
    warmup_steps: int = 2000
    eval_every: int = 1
    save_every: int = 10
    log_every: int = 100

    # Data loading
    num_workers: int = 4
    train_ratio: float = 0.9

    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class DTTrainer:
    """Trainer for Decision Transformer."""

    def __init__(
        self,
        model: DecisionTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        stats: DatasetStats,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.stats = stats

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Logging
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.output_dir / "logs")

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _create_scheduler(self):
        """Create learning rate scheduler with linear warmup and cosine decay."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return 1.0  # Constant after warmup (could add decay)
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        self.model.train()

        # Move batch to device
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rtg = batch["rtg"].to(self.device)
        timesteps = batch["timesteps"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Forward pass
        action_preds, state_preds = self.model(
            states, actions, rtg, timesteps, attention_mask
        )

        # Compute losses (only on valid tokens)
        mask = attention_mask.unsqueeze(-1).float()

        # Action loss (predict current action from current state)
        action_loss = self.loss_fn(action_preds * mask, actions * mask)

        # State prediction loss (predict next state observation)
        # Target: next state observation (shifted by 1)
        # Only use first state_dim features (vehicle obs, not track/obstacle)
        state_dim = self.model.config.state_dim
        next_states = torch.cat([
            states[:, 1:, :state_dim],
            states[:, -1:, :state_dim],  # Pad last
        ], dim=1)
        state_loss = self.loss_fn(state_preds * mask, next_states * mask)

        # Total loss
        loss = action_loss + self.config.lambda_x * state_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": loss.item(),
            "action_loss": action_loss.item(),
            "state_loss": state_loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_action_loss = 0.0
        total_state_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            states = batch["states"].to(self.device)
            actions = batch["actions"].to(self.device)
            rtg = batch["rtg"].to(self.device)
            timesteps = batch["timesteps"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            action_preds, state_preds = self.model(
                states, actions, rtg, timesteps, attention_mask
            )

            mask = attention_mask.unsqueeze(-1).float()
            action_loss = self.loss_fn(action_preds * mask, actions * mask)

            state_dim = self.model.config.state_dim
            next_states = torch.cat([
                states[:, 1:, :state_dim],
                states[:, -1:, :state_dim],
            ], dim=1)
            state_loss = self.loss_fn(state_preds * mask, next_states * mask)

            loss = action_loss + self.config.lambda_x * state_loss

            total_loss += loss.item()
            total_action_loss += action_loss.item()
            total_state_loss += state_loss.item()
            n_batches += 1

        return {
            "val_loss": total_loss / max(1, n_batches),
            "val_action_loss": total_action_loss / max(1, n_batches),
            "val_state_loss": total_state_loss / max(1, n_batches),
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "model_config": asdict(self.model.config),
        }

        path = self.output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")

        # Save stats alongside
        stats_path = self.output_dir / "dataset_stats.npz"
        self.stats.save(stats_path)

    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint and return starting epoch."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        return checkpoint["epoch"] + 1

    def train(self, resume_from: Optional[Path] = None) -> None:
        """Main training loop."""
        start_epoch = 0
        if resume_from and resume_from.exists():
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resumed from epoch {start_epoch}")

        # Save config
        self.config.save(self.output_dir / "config.json")

        print(f"\nStarting training:")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print()

        for epoch in range(start_epoch, self.config.num_epochs):
            t_epoch_start = time.time()

            # Training
            epoch_losses = []
            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step(batch)
                epoch_losses.append(metrics["loss"])
                self.global_step += 1

                # Log
                if self.global_step % self.config.log_every == 0:
                    self.writer.add_scalar("train/loss", metrics["loss"], self.global_step)
                    self.writer.add_scalar("train/action_loss", metrics["action_loss"], self.global_step)
                    self.writer.add_scalar("train/state_loss", metrics["state_loss"], self.global_step)
                    self.writer.add_scalar("train/lr", metrics["lr"], self.global_step)

            train_loss = np.mean(epoch_losses)

            # Validation
            if (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self.evaluate()
                val_loss = val_metrics["val_loss"]

                self.writer.add_scalar("val/loss", val_loss, self.global_step)
                self.writer.add_scalar("val/action_loss", val_metrics["val_action_loss"], self.global_step)
                self.writer.add_scalar("val/state_loss", val_metrics["val_state_loss"], self.global_step)

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                t_epoch = time.time() - t_epoch_start
                print(
                    f"Epoch {epoch + 1:4d} | "
                    f"train_loss: {train_loss:.6f} | "
                    f"val_loss: {val_loss:.6f} | "
                    f"best: {self.best_val_loss:.6f} | "
                    f"time: {t_epoch:.1f}s"
                )

                # Save checkpoint
                if (epoch + 1) % self.config.save_every == 0 or is_best:
                    self.save_checkpoint(epoch, is_best=is_best)

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.6f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Decision Transformer")

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/datasets",
        help=(
            "Dataset shard directory, comma-separated list of shard directories, "
            "or a root directory containing shard subdirectories."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="dt/checkpoints")

    # Model
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Loss
    parser.add_argument("--lambda-x", type=float, default=0.5, help="State prediction loss weight")

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=2000)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create config
    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        context_length=args.context_length,
        dropout=args.dropout,
        lambda_x=args.lambda_x,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )

    print("=" * 60)
    print("Decision Transformer Training")
    print("=" * 60)
    print(f"Data source: {config.data_dir}")
    print(f"Output directory: {config.output_dir}")
    print()

    # Create dataloaders
    train_loader, val_loader, stats = create_dataloaders(
        config.data_dir,
        context_length=config.context_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_ratio=config.train_ratio,
        seed=config.seed,
    )

    # Create model
    sample_batch = next(iter(train_loader))
    state_aug_dim = sample_batch["states"].shape[-1]
    act_dim = sample_batch["actions"].shape[-1]

    # Parse dimensions from state_aug_dim
    # state_aug_dim = state_dim + track_dim + obstacle_slots * obstacle_feat_dim
    # = 8 + 2 + 8 * 3 = 34
    state_dim = 8
    track_dim = 2
    obstacle_slots = 8
    obstacle_feat_dim = 3

    model_config = DTConfig(
        state_dim=state_dim,
        track_dim=track_dim,
        obstacle_slots=obstacle_slots,
        obstacle_feat_dim=obstacle_feat_dim,
        act_dim=act_dim,
        n_layer=config.n_layer,
        n_head=config.n_head,
        d_model=config.d_model,
        d_ff=config.d_ff,
        dropout=config.dropout,
        context_length=config.context_length,
        max_ep_len=config.max_ep_len,
    )

    model = DecisionTransformer(model_config)
    print(f"Model config: {model_config}")
    print(f"State aug dim: {state_aug_dim}")
    print(f"Action dim: {act_dim}")

    # Create trainer
    trainer = DTTrainer(model, train_loader, val_loader, config, stats)

    # Resume if specified
    resume_path = Path(args.resume) if args.resume else None

    # Train
    trainer.train(resume_from=resume_path)


if __name__ == "__main__":
    main()
