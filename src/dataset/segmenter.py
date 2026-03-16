"""
Audio Segmenter
===============
Slices 4-5 minute audio files into short overlapping clips,
assigns labels from annotation regions, and saves them to disk.

Converts each clip to a log-mel spectrogram (numpy .npy file)
so training reads fast pre-processed tensors — no real-time decode.

Output structure:
    data/segments/
        manifest.csv           ← (path, label, confidence, file_id, split)
        OK/
            file01_0000.npy
            file01_0001.npy
            ...
        NOK/
            file03_0120.npy
            ...
"""

import os
import csv
import shutil
import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from src.dataset.annotation_reader import (
    AnnotationReader,
    FileAnnotation,
    get_clip_label,
)

log = logging.getLogger(__name__)


class AudioSegmenter:
    """
    Segments all annotated audio files into mel-spectrogram clips.

    Args:
        annotation_reader: Loaded AnnotationReader instance
        output_dir:        Root directory for segments
        sample_rate:       Target sample rate (default 22050 Hz)
        window_sec:        Clip length in seconds (default 2.0)
        hop_sec:           Hop size between clips (default 1.0)
        n_mels:            Number of mel filterbanks (default 128)
        n_fft:             FFT size (default 2048)
        hop_length:        STFT hop length in samples (default 512)
        nok_threshold:     Fraction of clip that must be NOK to label as NOK
        overwrite:         Re-segment even if output already exists
    """

    def __init__(
        self,
        annotation_reader: AnnotationReader,
        output_dir: str = "data/segments",
        sample_rate: int = 22050,
        window_sec: float = 2.0,
        hop_sec: float = 1.0,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        nok_threshold: float = 0.5,
        overwrite: bool = False,
    ):
        self.reader       = annotation_reader
        self.output_dir   = Path(output_dir)
        self.sr           = sample_rate
        self.window_sec   = window_sec
        self.hop_sec      = hop_sec
        self.n_mels       = n_mels
        self.n_fft        = n_fft
        self.hop_length   = hop_length
        self.nok_threshold = nok_threshold
        self.overwrite    = overwrite

        self.window_samples = int(window_sec * sample_rate)
        self.hop_samples    = int(hop_sec    * sample_rate)

    def run(self) -> Path:
        """
        Segment all files in the annotation reader.
        Returns path to manifest.csv.
        """
        file_annotations = self.reader.load()

        if self.overwrite and self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        (self.output_dir / "OK").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "NOK").mkdir(parents=True, exist_ok=True)

        manifest_rows = []
        stats = {"OK": 0, "NOK": 0, "skipped": 0}

        for file_id, fa in tqdm(file_annotations.items(), desc="Segmenting files"):
            if fa.file_path is None:
                log.warning(f"Audio file not found for {file_id}, skipping")
                stats["skipped"] += 1
                continue

            rows = self._segment_file(fa, stats)
            manifest_rows.extend(rows)

        manifest_path = self._write_manifest(manifest_rows)

        print(f"\nSegmentation complete:")
        print(f"  OK  clips : {stats['OK']}")
        print(f"  NOK clips : {stats['NOK']}")
        print(f"  Skipped   : {stats['skipped']}")
        print(f"  Manifest  : {manifest_path}")
        return manifest_path

    def _segment_file(self, fa: FileAnnotation, stats: dict) -> list[dict]:
        """Load one audio file, slice it, compute mel specs, save .npy files."""
        try:
            y, _ = librosa.load(str(fa.file_path), sr=self.sr, mono=True)
        except Exception as e:
            log.error(f"Failed to load {fa.file_path}: {e}")
            stats["skipped"] += 1
            return []

        rows = []
        clip_idx = 0
        n_samples = len(y)

        for start in range(0, n_samples - self.window_samples, self.hop_samples):
            end   = start + self.window_samples
            clip  = y[start:end]

            start_sec = start / self.sr
            end_sec   = end   / self.sr

            binary_label, confidence = get_clip_label(
                start_sec, end_sec, fa.regions, self.nok_threshold
            )

            if binary_label == -1:
                continue  # ambiguous clip — skip

            label_str = "OK" if binary_label == 0 else "NOK"
            mel_spec  = self._compute_melspec(clip)

            fname     = f"{fa.file_id}_{clip_idx:04d}.npy"
            save_path = self.output_dir / label_str / fname
            np.save(str(save_path), mel_spec)

            rows.append({
                "path":       str(save_path),
                "label":      binary_label,
                "label_str":  label_str,
                "confidence": confidence,
                "file_id":    fa.file_id,
                "start_sec":  round(start_sec, 3),
                "end_sec":    round(end_sec,   3),
            })

            stats[label_str] += 1
            clip_idx += 1

        return rows

    def _compute_melspec(self, clip: np.ndarray) -> np.ndarray:
        """Convert raw waveform clip to log-mel spectrogram (H×W float32)."""
        mel = librosa.feature.melspectrogram(
            y=clip,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        return log_mel  # shape: (n_mels, time_frames)

    def _write_manifest(self, rows: list[dict]) -> Path:
        manifest_path = self.output_dir / "manifest.csv"
        fieldnames = ["path", "label", "label_str", "confidence",
                      "file_id", "start_sec", "end_sec"]
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return manifest_path
