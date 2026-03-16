"""
Generate Annotations CSV from Folder Structure
===============================================
!! WARNING — DO NOT USE FOR TRAINING DATA !!

This script assigns whole-file labels (OK/NOK) based purely on folder name.
That approach creates biased training data because:

  1. "OK" files often contain brief rattles or road events that should be NOK
  2. "NOK" files contain long stretches of clean driving between defect events

Training on these noisy whole-file labels teaches the model to recognise
PER-FILE acoustic identity (recording conditions, driver behaviour, road surface)
rather than the actual BSR defect signature.

USE THIS INSTEAD:
  Step 1 — Auto-detect candidate NOK events acoustically:
      python scripts/auto_prelabel.py --audio_dir test_data

  Step 2 — Import prelabels.json into Label Studio and verify each region.

  Step 3 — Export verified annotations:
      python scripts/setup_label_studio.py --action export --token YOUR_TOKEN

This script is kept only as a LAST RESORT baseline for quick experiments
when you have no time to annotate.  Expect lower model quality.
"""

import sys
print("=" * 70)
print("WARNING: generate_annotations_from_folder.py produces BIASED labels.")
print("Use auto_prelabel.py + Label Studio verification instead.")
print("Run with --force to proceed anyway (not recommended for final model).")
print("=" * 70)

if "--force" not in sys.argv:
    sys.exit(1)
"""

import csv
import argparse
from pathlib import Path

import librosa


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ok_dir",   default=None, help="Directory of OK audio files")
    p.add_argument("--nok_dir",  default=None, help="Directory of NOK audio files")
    p.add_argument("--root_dir", default=None,
                   help="Root directory containing ok/ and not_ok/ subdirs")
    p.add_argument("--output",   default="data/annotations/annotations.csv")
    return p.parse_args()


def collect_files(directory: str, label: str) -> list[dict]:
    rows = []
    for f in sorted(Path(directory).iterdir()):
        if f.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        # Determine sub-class from filename keywords
        name_lower = f.stem.lower()
        if label == "NOK":
            if "ip" in name_lower or "ipnoise" in name_lower:
                detailed_label = "NOK_IP"
            elif "glass" in name_lower or "sunroof" in name_lower:
                detailed_label = "NOK_Sunroof"
            elif "steering" in name_lower or "jog" in name_lower or "rear" in name_lower:
                detailed_label = "NOK_BSR"
            else:
                detailed_label = "NOK_BSR"   # default NOK → BSR
        else:
            detailed_label = "OK"

        # Use full file duration as a single region (whole-file annotation)
        try:
            duration = librosa.get_duration(path=str(f))
        except Exception:
            duration = 0.0

        rows.append({
            "file_id":         f.stem,
            "file_name":       f.name,
            "start_frac":      0.0,
            "end_frac":        1.0,
            "label":           detailed_label,
            "overall_quality": label,
            "notes":           f"auto-labeled from folder structure",
            "duration_sec":    round(duration, 2),
        })
        print(f"  {label:<4}  {detailed_label:<14}  {duration:6.1f}s  {f.name}")

    return rows


def main():
    args = parse_args()

    ok_dir  = args.ok_dir
    nok_dir = args.nok_dir

    if args.root_dir:
        root = Path(args.root_dir)
        # Accept either ok/not_ok or ok/nok subfolder naming
        for candidate in ["ok", "OK"]:
            if (root / candidate).exists():
                ok_dir = str(root / candidate)
                break
        for candidate in ["not_ok", "nok", "NOK", "NOT_OK"]:
            if (root / candidate).exists():
                nok_dir = str(root / candidate)
                break

    if not ok_dir or not nok_dir:
        print("Provide --ok_dir and --nok_dir, or --root_dir containing ok/ and not_ok/")
        return

    print(f"\nCollecting OK files from:  {ok_dir}")
    ok_rows  = collect_files(ok_dir,  "OK")
    print(f"\nCollecting NOK files from: {nok_dir}")
    nok_rows = collect_files(nok_dir, "NOK")

    all_rows = ok_rows + nok_rows

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["file_id", "file_name", "start_frac", "end_frac",
                  "label", "overall_quality", "notes", "duration_sec"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} annotation rows → {output_path}")
    print(f"  OK : {len(ok_rows)} files")
    print(f"  NOK: {len(nok_rows)} files")
    print(f"\nNext step:")
    print(f"  python scripts/segment_data.py --annotations {output_path} --audio_dir test_data")


if __name__ == "__main__":
    main()
