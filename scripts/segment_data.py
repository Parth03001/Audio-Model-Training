"""
Data Segmentation Script
========================
Run this AFTER exporting annotations from Label Studio.
Slices all labeled audio files into mel-spectrogram clips and
writes manifest.csv.

Usage:
    python scripts/segment_data.py \
        --annotations data/annotations/annotations.csv \
        --audio_dir   data/raw \
        --output_dir  data/segments \
        --config      configs/config.yaml
"""

import sys
import argparse
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import AnnotationReader, AudioSegmenter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", default="data/annotations/annotations.csv")
    p.add_argument("--audio_dir",   default="data/raw")
    p.add_argument("--output_dir",  default="data/segments")
    p.add_argument("--config",      default="configs/config.yaml")
    p.add_argument("--overwrite",   action="store_true",
                   help="Delete existing segments and re-generate")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)["data"]

    reader = AnnotationReader(
        annotations_csv = args.annotations,
        audio_dir       = args.audio_dir,
        exclude_unsure  = True,
    )

    segmenter = AudioSegmenter(
        annotation_reader = reader,
        output_dir        = args.output_dir,
        sample_rate       = cfg["sample_rate"],
        window_sec        = cfg["window_sec"],
        hop_sec           = cfg["hop_sec"],
        n_mels            = cfg["n_mels"],
        n_fft             = cfg["n_fft"],
        hop_length        = cfg["hop_length"],
        nok_threshold     = cfg["nok_overlap_threshold"],
        overwrite         = args.overwrite,
    )

    manifest = segmenter.run()
    print(f"\nReady. Run training with:")
    print(f"  python scripts/train.py --manifest {manifest}")


if __name__ == "__main__":
    main()
