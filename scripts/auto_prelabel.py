"""
Acoustic Event Detector — Auto Pre-Labeling for Label Studio
=============================================================
Problem:
  Whole-file labels (OK/NOK based on filename/folder) are NOISY:
    - "OK" files can contain brief rattles (mislabeled or undetected)
    - "NOK" files contain long stretches of clean driving between defect events
  Training on whole-file labels → model learns file identity, not acoustic defects.

Solution:
  1. Detect anomalous acoustic regions within every file using per-file
     baseline-normalized features (kurtosis + HF energy ratio + spectral flux).
  2. Upload candidate regions as pre-annotations to Label Studio.
  3. Annotator verifies/corrects in ~3-5 hrs instead of labeling from scratch.

This script outputs:
  - data/annotations/prelabels.json   ← import into Label Studio
  - data/annotations/prelabels.csv    ← human-readable summary

Usage:
    python scripts/auto_prelabel.py \
        --audio_dir test_data \
        --output    data/annotations/prelabels.json \
        --threshold 3.5      ← anomaly score multiplier above file baseline
        --min_event_sec 0.5  ← minimum event duration after merging
        --pad_sec 0.3        ← context padding around each event

Then import prelabels.json into Label Studio:
    Project → Import → JSON (Label Studio format)
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import soundfile as sf


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir",    default="test_data",
                   help="Root dir — scans recursively for audio files")
    p.add_argument("--output",       default="data/annotations/prelabels.json")
    p.add_argument("--threshold",    type=float, default=3.5,
                   help="Frames above threshold × file-baseline are flagged as candidate events")
    p.add_argument("--min_event_sec",type=float, default=0.3,
                   help="Merge events within this gap; discard events shorter than this")
    p.add_argument("--pad_sec",      type=float, default=0.5,
                   help="Seconds of context added before/after each detected event")
    p.add_argument("--sr",           type=int,   default=22050)
    p.add_argument("--hop_ms",       type=int,   default=100,
                   help="Analysis frame hop in milliseconds")
    return p.parse_args()


# ─────────────────────────────────────────────
# Anomaly score per frame
# ─────────────────────────────────────────────

def compute_anomaly_score(
    y:          np.ndarray,
    sr:         int,
    hop_length: int,
    kurtosis_win_sec: float = 0.5,
) -> np.ndarray:
    """
    Compute a per-frame anomaly score using three complementary features:

      1. Spectral flux (onset strength) — detects sudden broadband events
         → rattles and squeaks show as onset spikes
      2. High-frequency energy ratio (>2 kHz) — BSR and glass events are
         broadband; road noise is mostly low-frequency
      3. Kurtosis — impulsive events (clicks, rattles) produce high kurtosis;
         Gaussian road noise has kurtosis ≈ 3

    All features are normalized by their own file median so the road noise
    baseline of the file becomes 1.0 — the score is FILE-RELATIVE, not absolute.
    This is critical for mixed-condition files recorded at different speeds/roads.
    """
    # 1. Spectral flux
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    flux_norm = flux / (np.median(flux) + 1e-9)

    # 2. High-frequency energy ratio (>2 kHz)
    S     = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    hf    = S[freqs >= 2000].sum(0) / (S.sum(0) + 1e-9)
    hf_norm = hf / (np.median(hf) + 1e-9)

    # 3. Kurtosis per short window
    win   = int(kurtosis_win_sec * sr)
    step  = hop_length
    kurt  = []
    for i in range(0, len(y) - win, step):
        chunk = y[i:i + win]
        m2    = np.mean(chunk**2)
        m4    = np.mean(chunk**4)
        kurt.append(m4 / (m2**2 + 1e-12))
    kurt      = np.array(kurt, dtype=np.float32)
    kurt_norm = kurt / (np.median(kurt) + 1e-9)

    # Align lengths (STFT may produce slightly different frame count)
    n = min(len(flux_norm), len(hf_norm), len(kurt_norm))
    anomaly = (
        flux_norm[:n] * 0.4 +
        hf_norm[:n]   * 0.3 +
        kurt_norm[:n] * 0.3
    )
    return anomaly


# ─────────────────────────────────────────────
# Frame indices → time regions
# ─────────────────────────────────────────────

def frames_to_regions(
    anomaly:       np.ndarray,
    threshold:     float,
    hop_sec:       float,
    min_event_sec: float,
    pad_sec:       float,
    duration_sec:  float,
) -> list[tuple[float, float]]:
    """
    Convert anomalous frame indices into merged time regions.

    Returns list of (start_sec, end_sec) tuples.
    """
    flagged = np.where(anomaly >= threshold)[0]
    if len(flagged) == 0:
        return []

    # Group consecutive flagged frames into contiguous events
    events = []
    start  = flagged[0]
    prev   = flagged[0]

    gap_frames = max(1, int(min_event_sec / hop_sec))

    for idx in flagged[1:]:
        if idx - prev <= gap_frames:
            prev = idx
        else:
            events.append((start, prev))
            start = idx
            prev  = idx
    events.append((start, prev))

    # Convert to seconds + add padding + filter short events
    regions = []
    for s_frame, e_frame in events:
        s_sec = max(0.0,          s_frame * hop_sec - pad_sec)
        e_sec = min(duration_sec, e_frame * hop_sec + pad_sec)
        if (e_sec - s_sec) >= min_event_sec:
            regions.append((round(s_sec, 2), round(e_sec, 2)))

    return regions


# ─────────────────────────────────────────────
# Label Studio pre-annotation format
# ─────────────────────────────────────────────

def build_ls_task(
    file_path:    Path,
    file_label:   str,
    regions:      list[tuple[float, float]],
    duration_sec: float,
) -> dict:
    """
    Build a Label Studio task dict with pre-annotated regions.
    The 'was_cancelled': False marks it as a pre-annotation (not a final label)
    so the annotator sees regions highlighted but knows they need verification.

    Regions are stored as fractions (0.0–1.0) of total duration, which is
    what Label Studio's audio widget expects.
    """
    result = []

    for i, (start_sec, end_sec) in enumerate(regions):
        start_frac = start_sec / duration_sec
        end_frac   = end_sec   / duration_sec

        # Suggest a label — NOK files get NOK_BSR as default suggestion,
        # OK files get "Unsure" so annotator explicitly decides
        suggested = "NOK_BSR" if file_label == "NOK" else "Unsure"

        result.append({
            "id":        f"region_{i}",
            "type":      "labels",
            "from_name": "label",
            "to_name":   "audio",
            "value": {
                "start":  round(start_frac, 5),
                "end":    round(end_frac,   5),
                "labels": [suggested],
            },
            "origin": "prediction",   # marks as auto-generated (shown differently in UI)
        })

    # Add overall quality prediction
    result.append({
        "id":        "overall",
        "type":      "choices",
        "from_name": "overall_quality",
        "to_name":   "audio",
        "value": {
            "choices": [file_label]
        },
        "origin": "prediction",
    })

    return {
        "data": {
            "audio":    f"/data/local-files/?d={str(file_path.resolve()).replace(chr(92), '/')}",
            "file_name": file_path.name,
            "file_id":   file_path.stem,
        },
        "predictions": [{
            "model_version": "acoustic_anomaly_detector_v1",
            "score":         0.0,    # no confidence since rule-based
            "result":        result,
        }],
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args    = parse_args()
    hop_len = int(args.sr * args.hop_ms / 1000)   # samples per frame
    hop_sec = hop_len / args.sr                    # seconds per frame

    audio_dir = Path(args.audio_dir)
    files     = sorted(f for f in audio_dir.rglob("*")
                       if f.suffix.lower() in AUDIO_EXTENSIONS)

    if not files:
        print(f"No audio files found in {audio_dir}")
        return

    tasks    = []
    csv_rows = []

    print(f"\nAnalyzing {len(files)} files  "
          f"(threshold={args.threshold}x baseline, "
          f"min_event={args.min_event_sec}s, "
          f"pad={args.pad_sec}s)\n")
    print(f"{'File':<45} {'FileLbl':<8} {'Events':>7} {'Total flagged sec':>18}")
    print("─" * 82)

    for f in files:
        # Determine file-level label hint from folder name
        folder_lower = f.parent.name.lower()
        if "not_ok" in folder_lower or "nok" in folder_lower:
            file_label = "NOK"
        elif "ok" in folder_lower:
            file_label = "OK"
        else:
            file_label = "UNKNOWN"

        try:
            y, _  = librosa.load(str(f), sr=args.sr, mono=True)
            info  = sf.info(str(f))
            dur   = info.duration
        except Exception as e:
            print(f"  ERROR loading {f.name}: {e}")
            continue

        anomaly = compute_anomaly_score(y, args.sr, hop_len)
        regions = frames_to_regions(
            anomaly,
            threshold     = args.threshold,
            hop_sec       = hop_sec,
            min_event_sec = args.min_event_sec,
            pad_sec       = args.pad_sec,
            duration_sec  = dur,
        )

        total_flagged = sum(e - s for s, e in regions)
        print(f"{f.name:<45} {file_label:<8} {len(regions):>7}    {total_flagged:>8.1f}s")

        task = build_ls_task(f, file_label, regions, dur)
        tasks.append(task)

        for s, e in regions:
            csv_rows.append({
                "file_id":      f.stem,
                "file_name":    f.name,
                "file_label":   file_label,
                "start_sec":    s,
                "end_sec":      e,
                "duration_sec": round(e - s, 2),
                "auto_label":   "NOK_BSR" if file_label == "NOK" else "Unsure",
                "needs_review": True,
            })

    # ── Write outputs ────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"\nLabel Studio import file → {output_path}")

    csv_path = output_path.with_suffix(".csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Review summary CSV     → {csv_path}")

    total_events = sum(len(t["predictions"][0]["result"]) - 1 for t in tasks)
    print(f"\nTotal candidate regions to review: {total_events}")
    print(f"\nNext: Import {output_path} into Label Studio")
    print(f"      Project → Import → select prelabels.json")
    print(f"      Then verify each highlighted region — accept, reject, or relabel")
    print(f"      Use hotkeys: 1=OK  2=NOK_BSR  3=NOK_IP  4=NOK_Sunroof  5=Unsure")


if __name__ == "__main__":
    main()
