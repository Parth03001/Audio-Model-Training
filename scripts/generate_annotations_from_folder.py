"""
Generate Annotations CSV from Folder Structure
===============================================
Use this when audio files are already sorted into OK/ and NOK/ subfolders
(as in the test_data/ directory) — skips Label Studio entirely for those files.

Produces data/annotations/annotations.csv in exactly the same format that
setup_label_studio.py export produces, so segment_data.py + train.py
work without any changes.

Usage:
    python scripts/generate_annotations_from_folder.py \
        --ok_dir    test_data/ok \
        --nok_dir   test_data/not_ok \
        --output    data/annotations/annotations.csv

    # Or point at a single root dir that contains ok/ and not_ok/ subdirs:
    python scripts/generate_annotations_from_folder.py \
        --root_dir  test_data \
        --output    data/annotations/annotations.csv
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
