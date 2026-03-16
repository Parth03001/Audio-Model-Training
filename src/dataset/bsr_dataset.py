"""
BSR Audio Dataset
=================
PyTorch Dataset that reads the manifest.csv produced by AudioSegmenter,
loads pre-computed mel spectrograms, applies augmentation, and returns
(spectrogram_tensor, label, confidence_weight) triples.

Critically uses FILE-LEVEL group splits to prevent data leakage —
clips from the same audio file never appear in both train and val.
"""

import csv
import logging
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit

log = logging.getLogger(__name__)


class BSRDataset(Dataset):
    """
    Args:
        manifest_csv:   Path to manifest.csv from AudioSegmenter
        split:          "train", "val", or "test"
        val_split:      Fraction for validation (default 0.15)
        test_split:     Fraction for test     (default 0.15)
        transform:      Optional callable applied to the (n_mels, T) numpy array
                        — use this for SpecAugment or normalization
        seed:           Random seed for reproducible splits
        min_confidence: Clips with confidence below this are excluded
        normalize:      Standardize spectrogram to zero mean / unit variance
    """

    def __init__(
        self,
        manifest_csv: str,
        split: str = "train",
        val_split: float = 0.15,
        test_split: float = 0.15,
        transform: Optional[Callable] = None,
        seed: int = 42,
        min_confidence: float = 0.0,
        normalize: bool = True,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be train/val/test, got {split}"

        self.transform       = transform
        self.normalize       = normalize
        self.min_confidence  = min_confidence

        all_rows = self._load_manifest(manifest_csv, min_confidence)
        self.samples = self._split(all_rows, split, val_split, test_split, seed)

        labels = [r["label"] for r in self.samples]
        n_ok   = labels.count(0)
        n_nok  = labels.count(1)
        print(f"[{split:5s}]  OK={n_ok}  NOK={n_nok}  "
              f"total={len(self.samples)}")

    # ─────────────────────────────────────────
    # Dataset interface
    # ─────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]

        mel = np.load(row["path"])  # (n_mels, T) float32

        if self.normalize:
            mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        if self.transform is not None:
            mel = self.transform(mel)

        # Add channel dim → (1, n_mels, T)  for CNN input
        tensor = torch.from_numpy(mel).unsqueeze(0)

        label      = torch.tensor(row["label"],      dtype=torch.float32)
        confidence = torch.tensor(row["confidence"], dtype=torch.float32)

        return tensor, label, confidence

    # ─────────────────────────────────────────
    # Split helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _load_manifest(path: str, min_confidence: float) -> list[dict]:
        rows = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                conf = float(row["confidence"])
                if conf < min_confidence:
                    continue
                rows.append({
                    "path":       row["path"],
                    "label":      int(row["label"]),
                    "confidence": conf,
                    "file_id":    row["file_id"],
                })
        return rows

    @staticmethod
    def _split(
        rows: list[dict],
        split: str,
        val_split: float,
        test_split: float,
        seed: int,
    ) -> list[dict]:
        """
        Group-aware split: all clips from the same file go to the same partition.
        This prevents the model from memorising per-file acoustic signatures.
        """
        file_ids = [r["file_id"] for r in rows]
        labels   = [r["label"]   for r in rows]

        # First: carve out test set
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(rows, labels, groups=file_ids))

        train_val_rows    = [rows[i]     for i in train_val_idx]
        train_val_groups  = [file_ids[i] for i in train_val_idx]
        train_val_labels  = [labels[i]   for i in train_val_idx]

        # Then: carve validation from remaining train_val
        adjusted_val = val_split / (1.0 - test_split)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val, random_state=seed)
        train_idx, val_idx = next(gss_val.split(
            train_val_rows, train_val_labels, groups=train_val_groups
        ))

        splits = {
            "train": [train_val_rows[i] for i in train_idx],
            "val":   [train_val_rows[i] for i in val_idx],
            "test":  [rows[i]           for i in test_idx],
        }
        return splits[split]

    # ─────────────────────────────────────────
    # Balanced sampler
    # ─────────────────────────────────────────

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Returns a WeightedRandomSampler that oversamples the minority class
        so the model sees a balanced OK/NOK distribution each epoch.
        """
        labels = [r["label"] for r in self.samples]
        class_counts = {0: labels.count(0), 1: labels.count(1)}

        # Weight per sample = 1 / count_of_its_class
        weights = [
            1.0 / class_counts[label]
            for label in labels
        ]
        return WeightedRandomSampler(
            weights    = torch.tensor(weights),
            num_samples = len(weights),
            replacement = True,
        )


# ─────────────────────────────────────────────────────────────
# Factory — build train/val/test DataLoaders in one call
# ─────────────────────────────────────────────────────────────

def build_dataloaders(
    manifest_csv: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    min_confidence: float = 0.0,
) -> dict[str, DataLoader]:
    """
    Returns {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    Train loader uses WeightedRandomSampler for class balance.
    """
    common = dict(
        manifest_csv    = manifest_csv,
        val_split       = val_split,
        test_split      = test_split,
        seed            = seed,
        min_confidence  = min_confidence,
    )

    train_ds = BSRDataset(split="train", transform=train_transform, **common)
    val_ds   = BSRDataset(split="val",   **common)
    test_ds  = BSRDataset(split="test",  **common)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = train_ds.get_weighted_sampler(),
        num_workers = num_workers,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
