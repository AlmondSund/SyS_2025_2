"""Feature extraction from audio segments."""

from __future__ import annotations

from typing import Dict, Tuple

import librosa
import numpy as np

from .config import SpectralConfig
from .audio import ensure_local_audio


def load_audio_segment(path_or_url: str, cfg: SpectralConfig) -> Tuple[np.ndarray, str]:
    """Load a mono audio snippet limited to cfg.duration seconds."""
    local_path = ensure_local_audio(path_or_url, cfg)
    samples, _ = librosa.load(
        local_path,
        sr=cfg.sr,
        mono=True,
        duration=cfg.duration,
        res_type=cfg.res_type,
    )
    expected = int(round(cfg.duration * cfg.sr))
    if samples.size < expected:
        samples = np.pad(samples, (0, expected - samples.size))
    elif samples.size > expected:
        samples = samples[:expected]
    return samples.astype(np.float32), local_path


def magnitude_spectrum(samples: np.ndarray, cfg: SpectralConfig) -> np.ndarray:
    stft = librosa.stft(
        samples,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window="hann",
        center=cfg.center_stft,
    )
    mag = np.abs(stft)
    feat = mag.mean(axis=1)
    if cfg.l2_normalize:
        norm = np.linalg.norm(feat) + cfg.eps
        feat = feat / norm
    return feat.astype(np.float32)


def extract_feature_vector(path_or_url: str, cfg: SpectralConfig) -> Tuple[np.ndarray, Dict[str, str]]:
    samples, resolved_path = load_audio_segment(path_or_url, cfg)
    spectrum = magnitude_spectrum(samples, cfg)
    metadata = {"resolved_path": resolved_path, "source": path_or_url}
    return spectrum, metadata
