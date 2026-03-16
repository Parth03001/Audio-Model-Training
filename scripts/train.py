"""
BSR Audio Classification — Training Script
===========================================
Two-phase training:
  Phase 1 (freeze_epochs): Backbone frozen, only head trains at high LR
  Phase 2 (remaining):     Entire model fine-tuned at low LR

Usage:
    # Quick start (uses config.yaml defaults):
    python scripts/train.py

    # With explicit paths:
    python scripts/train.py \
        --manifest  data/segments/manifest.csv \
        --config    configs/config.yaml \
        --backbone  CNN14 \
        --epochs    25 \
        --batch_size 32

    # Resume from checkpoint:
    python scripts/train.py --resume checkpoints/epoch_010.pth
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset      import build_dataloaders
from src.models       import build_model
from src.augmentation import TrainTransform, mixup_batch
from src.utils        import compute_metrics, print_eval_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Train BSR classifier")
    p.add_argument("--manifest",   default="data/segments/manifest.csv")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--backbone",   default=None,  help="Override config backbone")
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--resume",     default=None,  help="Path to checkpoint to resume from")
    p.add_argument("--no_mixup",   action="store_true")
    p.add_argument("--output_dir", default="checkpoints")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Confidence-weighted BCE loss
# ─────────────────────────────────────────────

def weighted_bce_loss(
    logits:      torch.Tensor,  # (B,)
    labels:      torch.Tensor,  # (B,)
    confidences: torch.Tensor,  # (B,)  per-sample weights
) -> torch.Tensor:
    """
    Binary cross-entropy weighted by annotator confidence.
    Uncertain samples (e.g. labeled 'Unsure') contribute less to the loss.
    """
    bce = nn.BCEWithLogitsLoss(reduction="none")(logits, labels)
    return (bce * confidences).mean()


# ─────────────────────────────────────────────
# Training & evaluation loops
# ─────────────────────────────────────────────

def train_epoch(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    use_mixup:   bool = True,
    mixup_alpha: float = 0.4,
) -> dict:
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for specs, labels, confidences in tqdm(loader, desc="  train", leave=False):
        specs       = specs.to(device)
        labels      = labels.to(device)
        confidences = confidences.to(device)

        if use_mixup:
            specs, labels, confidences = mixup_batch(
                specs, labels, confidences, alpha=mixup_alpha
            )

        optimizer.zero_grad()
        logits = model(specs)
        loss   = weighted_bce_loss(logits, labels, confidences)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).round().long()  # round mixup soft labels

    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def eval_epoch(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for specs, labels, confidences in tqdm(loader, desc="  eval ", leave=False):
        specs       = specs.to(device)
        labels      = labels.to(device)
        confidences = confidences.to(device)

        logits = model(specs)
        loss   = weighted_bce_loss(logits, labels, confidences)

        total_loss += loss.item() * specs.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────

def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    metrics:   dict,
    path:      str,
) -> None:
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "metrics":    metrics,
    }, path)


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt  = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from epoch {ckpt['epoch']}  (val_f1={ckpt['metrics'].get('f1', '?'):.4f})")
    return ckpt["epoch"]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config)

    # CLI args override config
    backbone   = args.backbone   or config["model"]["backbone"]
    epochs     = args.epochs     or config["training"]["epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    seed       = config["training"]["seed"]

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── DataLoaders ────────────────────────────
    train_transform = TrainTransform(
        time_mask_param = config["augmentation"]["spec_time_mask_param"],
        freq_mask_param = config["augmentation"]["spec_freq_mask_param"],
    ) if config["augmentation"]["enabled"] else None

    loaders = build_dataloaders(
        manifest_csv   = args.manifest,
        batch_size     = batch_size,
        num_workers    = config["training"]["num_workers"],
        train_transform = train_transform,
        val_split      = config["training"]["val_split"],
        test_split     = config["training"]["test_split"],
        seed           = seed,
    )

    # ── Model ──────────────────────────────────
    model = build_model(
        backbone_name   = backbone,
        pretrained      = config["model"]["pretrained"],
        freeze_backbone = True,
        dropout         = config["model"]["dropout"],
    ).to(device)

    # ── Optimizers (two phases) ─────────────────
    head_lr     = config["model"]["head_lr"]
    backbone_lr = config["model"]["unfreeze_lr"]
    freeze_epochs = config["model"]["freeze_epochs"]

    optimizer = torch.optim.AdamW(
        model.get_param_groups(head_lr=head_lr, backbone_lr=backbone_lr),
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - freeze_epochs
    )

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer)

    # ── Logging ────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=config["training"]["log_dir"])

    best_val_f1  = 0.0
    no_improve   = 0
    patience     = config["training"]["early_stopping_patience"]
    use_mixup    = config["augmentation"]["use_mixup"] and not args.no_mixup

    # ── Training loop ──────────────────────────
    print(f"\nPhase 1: Training head only for {freeze_epochs} epochs")
    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1

        # Phase transition: unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs:
            print(f"\nPhase 2: Unfreezing backbone — lr={backbone_lr}")
            model.unfreeze_backbone()
            # Re-init optimizer with differential LRs
            optimizer = torch.optim.AdamW(
                model.get_param_groups(head_lr=head_lr, backbone_lr=backbone_lr),
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs
            )

        train_metrics = train_epoch(
            model, loaders["train"], optimizer, device,
            use_mixup=use_mixup,
            mixup_alpha=config["augmentation"]["mixup_alpha"],
        )
        val_metrics = eval_epoch(model, loaders["val"], device)

        # LR scheduler step (only in phase 2)
        if epoch >= freeze_epochs:
            scheduler.step()

        # Logging
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, current_epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, current_epoch)

        print(
            f"Epoch {current_epoch:3d}/{epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_f1={train_metrics['f1']:.4f}  │  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f}  "
            f"val_auc={val_metrics.get('auc', float('nan')):.4f}"
        )

        # Save best checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            no_improve  = 0
            ckpt_path   = str(output_dir / "best_model.pth")
            save_checkpoint(model, optimizer, current_epoch, val_metrics, ckpt_path)
            print(f"  ✓ New best val_f1={best_val_f1:.4f} → saved {ckpt_path}")
        else:
            no_improve += 1

        # Periodic checkpoint every 5 epochs
        if current_epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, current_epoch, val_metrics,
                str(output_dir / f"epoch_{current_epoch:03d}.pth")
            )

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break

    writer.close()

    # ── Final test evaluation ───────────────────
    print("\nLoading best model for test evaluation...")
    best_ckpt = torch.load(str(output_dir / "best_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model"])

    all_logits = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for specs, labels, _ in loaders["test"]:
            logits = model(specs.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    print_eval_report(
        torch.cat(all_logits),
        torch.cat(all_labels),
        split_name="Test",
    )

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    print(f"Checkpoints saved in: {output_dir}")


if __name__ == "__main__":
    main()
