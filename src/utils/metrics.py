"""
Metrics for BSR binary classification.
F1-score is the primary metric — more informative than accuracy
when OK/NOK distribution may be skewed.
"""

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Compute binary classification metrics from raw logits.

    Args:
        logits:    (N,) raw model outputs (before sigmoid)
        labels:    (N,) ground-truth {0, 1}
        threshold: Decision threshold (default 0.5)

    Returns:
        dict with f1, precision, recall, auc, accuracy
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    y     = labels.cpu().numpy().astype(int)

    results = {
        "f1":        f1_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "recall":    recall_score(y, preds, zero_division=0),
        "accuracy":  (preds == y).mean(),
    }

    try:
        results["auc"] = roc_auc_score(y, probs)
    except ValueError:
        results["auc"] = float("nan")   # only 1 class present in batch

    return results


def print_eval_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    split_name: str = "Test",
) -> None:
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    y     = labels.cpu().numpy().astype(int)

    print(f"\n{'─'*50}")
    print(f"{split_name} Evaluation Report")
    print(f"{'─'*50}")
    print(classification_report(y, preds, target_names=["OK", "NOK"], digits=4))

    cm = confusion_matrix(y, preds)
    print("Confusion Matrix:")
    print(f"           Pred OK  Pred NOK")
    print(f"Actual OK   {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"Actual NOK  {cm[1,0]:5d}    {cm[1,1]:5d}")

    try:
        auc = roc_auc_score(y, probs)
        print(f"\nAUC-ROC: {auc:.4f}")
    except ValueError:
        pass
    print(f"{'─'*50}")
