"""
Annotation Reader
=================
Reads Label Studio CSV exports and resolves fractional timestamps
to absolute seconds using actual audio file durations.

Handles three annotation patterns:
  1. Whole-file label (no regions drawn — one row, start=0, end=1)
  2. Temporal regions  (annotator drew regions on waveform)
  3. Mixed files with both OK and NOK segments
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np


# Binary label mapping used everywhere downstream
LABEL_MAP = {
    "OK":          0,
    "NOK_BSR":     1,
    "NOK_IP":      1,
    "NOK_Sunroof": 1,
    "NOK":         1,   # fallback from overall_quality field
    "Unsure":      -1,  # excluded from training by default
    "UNKNOWN":     -1,
}

# Sub-class label map (for multi-class experiments)
SUBCLASS_MAP = {
    "OK":          0,
    "NOK_BSR":     1,
    "NOK_IP":      2,
    "NOK_Sunroof": 3,
    "NOK":         1,
    "Unsure":      -1,
    "UNKNOWN":     -1,
}


@dataclass
class AnnotationRegion:
    """A single labeled time region within one audio file."""
    file_id:    str
    file_name:  str
    start_sec:  float
    end_sec:    float
    label:      str          # raw string label
    binary_label: int        # 0=OK, 1=NOK, -1=unsure
    subclass:   int          # fine-grained class
    confidence: float = 1.0  # 1.0=certain, 0.7=likely, 0.3=unsure
    overall_quality: str = "UNKNOWN"
    notes: str = ""


@dataclass
class FileAnnotation:
    """All annotations for one audio file."""
    file_id:   str
    file_name: str
    file_path: Optional[Path]
    duration_sec: float
    overall_quality: str
    regions: list = field(default_factory=list)

    @property
    def has_temporal_regions(self) -> bool:
        return any(not (r.start_sec == 0 and r.end_sec == self.duration_sec)
                   for r in self.regions)


class AnnotationReader:
    """
    Loads annotation CSV produced by setup_label_studio.py export
    and resolves timestamps using actual audio durations.
    """

    def __init__(
        self,
        annotations_csv: str,
        audio_dir: str,
        exclude_unsure: bool = True,
        confidence_map: Optional[dict] = None,
    ):
        self.annotations_csv = Path(annotations_csv)
        self.audio_dir       = Path(audio_dir)
        self.exclude_unsure  = exclude_unsure
        self.confidence_map  = confidence_map or {
            "OK":          1.0,
            "NOK_BSR":     1.0,
            "NOK_IP":      1.0,
            "NOK_Sunroof": 1.0,
            "NOK":         0.9,
            "Unsure":      0.3,
        }
        self._file_annotations: dict[str, FileAnnotation] = {}

    def load(self) -> dict[str, FileAnnotation]:
        """
        Parse annotation CSV and resolve fractional timestamps to seconds.
        Returns dict keyed by file_id.
        """
        rows = self._read_csv()
        file_durations = self._get_all_durations(rows)

        for row in rows:
            file_id   = row["file_id"]
            file_name = row["file_name"]
            duration  = file_durations.get(file_id, 0.0)

            label   = row["label"].strip()
            binary  = LABEL_MAP.get(label, -1)
            subclass = SUBCLASS_MAP.get(label, -1)

            if self.exclude_unsure and binary == -1:
                continue

            # Resolve fractions → seconds
            start_frac = float(row.get("start_frac", 0.0))
            end_frac   = float(row.get("end_frac",   1.0))
            start_sec  = start_frac * duration
            end_sec    = end_frac   * duration

            region = AnnotationRegion(
                file_id         = file_id,
                file_name       = file_name,
                start_sec       = start_sec,
                end_sec         = end_sec,
                label           = label,
                binary_label    = binary,
                subclass        = subclass,
                confidence      = self.confidence_map.get(label, 0.5),
                overall_quality = row.get("overall_quality", "UNKNOWN"),
                notes           = row.get("notes", ""),
            )

            if file_id not in self._file_annotations:
                file_path = self._find_audio_file(file_name)
                self._file_annotations[file_id] = FileAnnotation(
                    file_id         = file_id,
                    file_name       = file_name,
                    file_path       = file_path,
                    duration_sec    = duration,
                    overall_quality = row.get("overall_quality", "UNKNOWN"),
                    regions         = [],
                )

            self._file_annotations[file_id].regions.append(region)

        print(f"Loaded annotations for {len(self._file_annotations)} files")
        self._print_stats()
        return self._file_annotations

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _read_csv(self) -> list[dict]:
        with open(self.annotations_csv, newline="") as f:
            return list(csv.DictReader(f))

    def _get_all_durations(self, rows: list[dict]) -> dict[str, float]:
        """Resolve audio duration for each unique file_id."""
        seen = {}
        for row in rows:
            file_id   = row["file_id"]
            file_name = row["file_name"]
            if file_id in seen:
                continue
            path = self._find_audio_file(file_name)
            if path:
                try:
                    dur = librosa.get_duration(path=str(path))
                    seen[file_id] = dur
                except Exception:
                    seen[file_id] = 0.0
            else:
                seen[file_id] = 0.0
        return seen

    def _find_audio_file(self, file_name: str) -> Optional[Path]:
        """Search audio_dir for a file matching file_name."""
        path = self.audio_dir / file_name
        if path.exists():
            return path
        # Try without extension
        stem = Path(file_name).stem
        for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            p = self.audio_dir / (stem + ext)
            if p.exists():
                return p
        return None

    def _print_stats(self) -> None:
        total_ok  = sum(
            1 for fa in self._file_annotations.values()
            for r in fa.regions if r.binary_label == 0
        )
        total_nok = sum(
            1 for fa in self._file_annotations.values()
            for r in fa.regions if r.binary_label == 1
        )
        total_unsure = sum(
            1 for fa in self._file_annotations.values()
            for r in fa.regions if r.binary_label == -1
        )
        print(f"  Regions — OK: {total_ok}  NOK: {total_nok}  Unsure: {total_unsure}")


def get_clip_label(
    clip_start: float,
    clip_end:   float,
    regions:    list[AnnotationRegion],
    threshold:  float = 0.5,
) -> tuple[int, float]:
    """
    Assign a label to a clip by checking how much it overlaps with NOK regions.

    Returns:
        (binary_label, confidence_weight)
        binary_label: 0=OK, 1=NOK, -1=unsure
    """
    clip_duration = clip_end - clip_start

    nok_overlap   = 0.0
    ok_overlap    = 0.0
    min_confidence = 1.0

    for r in regions:
        overlap_start = max(clip_start, r.start_sec)
        overlap_end   = min(clip_end,   r.end_sec)
        overlap       = max(0.0, overlap_end - overlap_start)

        if r.binary_label == 1:
            nok_overlap  += overlap
        elif r.binary_label == 0:
            ok_overlap   += overlap

        if overlap > 0:
            min_confidence = min(min_confidence, r.confidence)

    nok_fraction = nok_overlap / clip_duration if clip_duration > 0 else 0.0

    if nok_fraction >= threshold:
        return 1, min_confidence
    elif ok_overlap / clip_duration >= threshold:
        return 0, min_confidence
    else:
        return -1, 0.3   # ambiguous clip
