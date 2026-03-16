"""
Inference / Prediction Script
==============================
Run BSR classification on new audio files (unlabeled).

Produces:
  - Per-clip predictions with timestamps and confidence scores
  - Per-file summary (OK / NOK / ratio)
  - Optional: timeline plot showing where NOK events occur in each file

Usage:
    # Single file
    python scripts/predict.py \
        --checkpoint checkpoints/best_model.pth \
        --input      path/to/new_recording.wav

    # Directory of files
    python scripts/predict.py \
        --checkpoint checkpoints/best_model.pth \
        --input      data/new_recordings/ \
        --output     predictions/

    # With timeline plots
    python scripts/predict.py \
        --checkpoint checkpoints/best_model.pth \
        --input      data/new_recordings/ \
        --plot
"""

import sys
import argparse
import json
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import build_model, BSRClassifier


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEFAULT_SR          = 22050
DEFAULT_WINDOW_SEC  = 2.0
DEFAULT_HOP_SEC     = 1.0
DEFAULT_N_MELS      = 128
DEFAULT_N_FFT       = 2048
DEFAULT_HOP_LENGTH  = 512
DEFAULT_THRESHOLD   = 0.5


def parse_args():
    p = argparse.ArgumentParser(description="BSR audio classifier inference")
    p.add_argument("--checkpoint",  required=True, help="Path to best_model.pth")
    p.add_argument("--input",       required=True, help="Audio file or directory")
    p.add_argument("--output",      default="predictions", help="Output directory")
    p.add_argument("--threshold",   type=float, default=DEFAULT_THRESHOLD,
                   help="Probability threshold for NOK (default 0.5)")
    p.add_argument("--backbone",    default="CNN14")
    p.add_argument("--plot",        action="store_true", help="Save timeline plots")
    p.add_argument("--config",      default="configs/config.yaml")
    return p.parse_args()


# ─────────────────────────────────────────────
# Audio → clips → mel specs
# ─────────────────────────────────────────────

def load_and_segment(
    file_path: str,
    sr: int       = DEFAULT_SR,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float    = DEFAULT_HOP_SEC,
    n_mels: int   = DEFAULT_N_MELS,
    n_fft: int    = DEFAULT_N_FFT,
    hop_length: int   = DEFAULT_HOP_LENGTH,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """
    Load audio file and return list of (mel_spec, (start_sec, end_sec)) pairs.
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)

    window_samples = int(window_sec * sr)
    hop_samples    = int(hop_sec    * sr)

    clips     = []
    timestamps = []

    for start in range(0, len(y) - window_samples, hop_samples):
        end  = start + window_samples
        clip = y[start:end]

        mel = librosa.feature.melspectrogram(
            y=clip, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        clips.append(log_mel)
        timestamps.append((start / sr, end / sr))

    return clips, timestamps


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

@torch.no_grad()
def predict_file(
    model:       BSRClassifier,
    file_path:   str,
    device:      torch.device,
    threshold:   float = DEFAULT_THRESHOLD,
    batch_size:  int   = 64,
) -> list[dict]:
    """
    Run inference on all clips from one audio file.

    Returns list of dicts:
        {start_sec, end_sec, prob_nok, pred_label, pred_str}
    """
    clips, timestamps = load_and_segment(file_path)

    results = []
    model.eval()

    for i in range(0, len(clips), batch_size):
        batch_clips = clips[i:i + batch_size]
        batch_ts    = timestamps[i:i + batch_size]

        # Stack → (B, 1, n_mels, T)
        batch_tensor = torch.from_numpy(
            np.stack(batch_clips)
        ).unsqueeze(1).to(device)

        logits = model(batch_tensor)               # (B,)
        probs  = torch.sigmoid(logits).cpu().numpy()

        for prob, (start, end) in zip(probs, batch_ts):
            pred = 1 if prob >= threshold else 0
            results.append({
                "start_sec":  round(float(start), 3),
                "end_sec":    round(float(end),   3),
                "prob_nok":   round(float(prob),  4),
                "pred_label": int(pred),
                "pred_str":   "NOK" if pred == 1 else "OK",
            })

    return results


def summarize_file(results: list[dict]) -> dict:
    """Aggregate per-clip predictions to a file-level summary."""
    n_total = len(results)
    n_nok   = sum(1 for r in results if r["pred_label"] == 1)
    n_ok    = n_total - n_nok
    max_prob_nok  = max(r["prob_nok"] for r in results) if results else 0.0
    mean_prob_nok = np.mean([r["prob_nok"] for r in results]) if results else 0.0

    # File is NOK if any window exceeds threshold (conservative: any defect = fail)
    file_verdict = "NOK" if n_nok > 0 else "OK"

    return {
        "verdict":        file_verdict,
        "n_clips":        n_total,
        "n_nok_clips":    n_nok,
        "n_ok_clips":     n_ok,
        "nok_ratio":      round(n_nok / n_total, 4) if n_total > 0 else 0.0,
        "max_prob_nok":   round(float(max_prob_nok),  4),
        "mean_prob_nok":  round(float(mean_prob_nok), 4),
    }


# ─────────────────────────────────────────────
# Timeline plot
# ─────────────────────────────────────────────

def plot_timeline(
    results:   list[dict],
    file_path: str,
    output_path: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """
    Save a timeline plot showing NOK probability over the full audio duration.
    Green regions = OK, Red regions = NOK.
    """
    times = [r["start_sec"] for r in results]
    probs = [r["prob_nok"]  for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    fig.suptitle(f"BSR Classification: {Path(file_path).name}", fontsize=12)

    # Probability timeline
    ax1.plot(times, probs, color="#2c7be5", linewidth=1.0, label="P(NOK)")
    ax1.axhline(threshold, color="#e74c3c", linewidth=1.0, linestyle="--",
                label=f"threshold={threshold}")
    ax1.fill_between(times, probs, alpha=0.15, color="#2c7be5")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("P(NOK)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    # Binary verdict bar
    colors = ["#e74c3c" if p >= threshold else "#27ae60" for p in probs]
    ax2.bar(times, [1.0] * len(times), width=1.0, color=colors, align="edge")
    ax2.set_yticks([])
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Verdict")

    # Add legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27ae60", label="OK"),
        Patch(facecolor="#e74c3c", label="NOK"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(
        backbone_name   = args.backbone,
        pretrained      = False,   # we load our own checkpoint
        freeze_backbone = False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Collect audio files
    input_path = Path(args.input)
    if input_path.is_file():
        audio_files = [input_path]
    else:
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        audio_files = [f for f in input_path.rglob("*") if f.suffix.lower() in exts]
        audio_files.sort()

    if not audio_files:
        print(f"No audio files found in: {args.input}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Predict each file ────────────────────────
    summary_rows = []

    for audio_file in audio_files:
        print(f"\nProcessing: {audio_file.name}")
        results = predict_file(model, str(audio_file), device, threshold=args.threshold)
        summary = summarize_file(results)

        print(f"  Verdict:     {summary['verdict']}")
        print(f"  NOK clips:   {summary['n_nok_clips']}/{summary['n_clips']}  "
              f"({100*summary['nok_ratio']:.1f}%)")
        print(f"  Max P(NOK):  {summary['max_prob_nok']:.4f}")

        # Save per-clip CSV
        clip_csv = output_dir / f"{audio_file.stem}_clips.csv"
        with open(clip_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

        # Save timeline plot
        if args.plot:
            plot_path = output_dir / f"{audio_file.stem}_timeline.png"
            plot_timeline(results, str(audio_file), str(plot_path), args.threshold)
            print(f"  Plot saved:  {plot_path}")

        summary_rows.append({
            "file":            audio_file.name,
            "verdict":         summary["verdict"],
            "n_clips":         summary["n_clips"],
            "n_nok_clips":     summary["n_nok_clips"],
            "nok_ratio":       summary["nok_ratio"],
            "max_prob_nok":    summary["max_prob_nok"],
            "mean_prob_nok":   summary["mean_prob_nok"],
        })

    # Save summary CSV
    if summary_rows:
        summary_path = output_dir / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary saved: {summary_path}")

        # Print final table
        print(f"\n{'─'*60}")
        print(f"{'File':<30} {'Verdict':<8} {'NOK%':>6}  {'MaxP':>6}")
        print(f"{'─'*60}")
        for row in summary_rows:
            print(f"{row['file']:<30} {row['verdict']:<8} "
                  f"{100*row['nok_ratio']:>5.1f}%  {row['max_prob_nok']:>6.4f}")
        print(f"{'─'*60}")


if __name__ == "__main__":
    main()
