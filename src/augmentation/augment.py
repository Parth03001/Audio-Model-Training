"""
Augmentation Pipeline
=====================
Two-stage augmentation for audio classification with limited data:

  Stage 1 — Waveform augmentation  (applied in AudioSegmenter or on-the-fly)
  Stage 2 — Spectrogram augmentation (applied in Dataset __getitem__)

SpecAugment (stage 2) is the most important technique for audio models —
it masks random time and frequency bands, forcing the model to be robust
to partial information loss, which mirrors real BSR track conditions where
road noise or wind partially masks defect sounds.

Mixup (stage 2) blends two spectrogram clips from the same batch with
weighted labels — highly effective for small datasets.
"""

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Stage 1: Waveform Augmentation
# ─────────────────────────────────────────────

class WaveformAugmenter:
    """
    Applies random transformations directly to raw waveforms (numpy float32).
    Use this during offline segmentation to generate augmented .npy variants,
    or wrap BSRDataset to apply on-the-fly by loading .wav files instead of .npy.

    Args:
        sample_rate:          Audio sample rate
        time_stretch_range:   (min, max) stretch factor, e.g. (0.8, 1.2)
        pitch_shift_steps:    Max semitone shift (±)
        noise_factor:         Gaussian noise std as fraction of signal amplitude
        p_stretch:            Probability of applying time stretch
        p_pitch:              Probability of applying pitch shift
        p_noise:              Probability of adding noise
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        time_stretch_range: tuple = (0.85, 1.15),
        pitch_shift_steps: int = 2,
        noise_factor: float = 0.005,
        p_stretch: float = 0.5,
        p_pitch:   float = 0.4,
        p_noise:   float = 0.5,
    ):
        self.sr                  = sample_rate
        self.time_stretch_range  = time_stretch_range
        self.pitch_shift_steps   = pitch_shift_steps
        self.noise_factor        = noise_factor
        self.p_stretch           = p_stretch
        self.p_pitch             = p_pitch
        self.p_noise             = p_noise

    def __call__(self, y: np.ndarray) -> np.ndarray:
        import librosa

        if random.random() < self.p_stretch:
            rate = random.uniform(*self.time_stretch_range)
            y = librosa.effects.time_stretch(y, rate=rate)

        if random.random() < self.p_pitch:
            steps = random.randint(-self.pitch_shift_steps, self.pitch_shift_steps)
            if steps != 0:
                y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=steps)

        if random.random() < self.p_noise:
            noise = np.random.normal(0, self.noise_factor * np.max(np.abs(y)), len(y))
            y = y + noise.astype(y.dtype)

        return y


# ─────────────────────────────────────────────
# Stage 2: Spectrogram Augmentation (SpecAugment)
# ─────────────────────────────────────────────

class SpecAugment:
    """
    SpecAugment applied to a log-mel spectrogram numpy array (n_mels × T).

    Args:
        time_mask_param:  Maximum width of time mask (in frames)
        freq_mask_param:  Maximum width of frequency mask (in mel bins)
        n_time_masks:     Number of time masks to apply
        n_freq_masks:     Number of frequency masks to apply
        mask_value:       Value to fill masked regions (default 0.0)
    """

    def __init__(
        self,
        time_mask_param: int = 30,
        freq_mask_param: int = 20,
        n_time_masks: int = 2,
        n_freq_masks: int = 2,
        mask_value: float = 0.0,
    ):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.n_time_masks    = n_time_masks
        self.n_freq_masks    = n_freq_masks
        self.mask_value      = mask_value

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """
        mel: (n_mels, T) numpy array
        Returns augmented (n_mels, T) array
        """
        mel = mel.copy()
        n_mels, T = mel.shape

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, n_mels - f))
            mel[f0:f0 + f, :] = self.mask_value

        # Time masking
        for _ in range(self.n_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(0, T - t))
            mel[:, t0:t0 + t] = self.mask_value

        return mel


class TrainTransform:
    """
    Composed transform applied in BSRDataset __getitem__ during training.
    Applies SpecAugment then returns the augmented numpy array.
    """

    def __init__(
        self,
        time_mask_param: int = 30,
        freq_mask_param: int = 20,
        p_specaugment: float = 0.8,
    ):
        self.spec_augment   = SpecAugment(time_mask_param, freq_mask_param)
        self.p_specaugment  = p_specaugment

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        if random.random() < self.p_specaugment:
            mel = self.spec_augment(mel)
        return mel


# ─────────────────────────────────────────────
# Mixup (applied at batch level in training loop)
# ─────────────────────────────────────────────

def mixup_batch(
    specs: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply Mixup to a batch of spectrograms.

    Args:
        specs:       (B, 1, n_mels, T) float tensor
        labels:      (B,) float tensor  {0.0, 1.0}
        confidences: (B,) float tensor  per-sample weights
        alpha:       Beta distribution parameter (higher → more mixing)

    Returns:
        mixed_specs, mixed_labels, mixed_confidences
    """
    if alpha <= 0:
        return specs, labels, confidences

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # keep dominant sample > 50%

    batch_size = specs.size(0)
    index = torch.randperm(batch_size, device=specs.device)

    mixed_specs       = lam * specs       + (1 - lam) * specs[index]
    mixed_labels      = lam * labels      + (1 - lam) * labels[index]
    mixed_confidences = lam * confidences + (1 - lam) * confidences[index]

    return mixed_specs, mixed_labels, mixed_confidences
