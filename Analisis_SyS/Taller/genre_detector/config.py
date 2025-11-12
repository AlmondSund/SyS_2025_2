"""Configuration helpers for the spectral k-NN genre detector."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(slots=True)
class SpectralConfig:
    """Hyper-parameters governing audio loading and STFT processing."""

    sr: int = 22050
    duration: float = 5.0
    n_fft: int = 4096
    hop_length: int = 2048
    k: int = 5
    res_type: str = "soxr_vhq"
    l2_normalize: bool = True
    center_stft: bool = False
    n_jobs: int = -1
    eps: float = 1e-12
    audio_cache_dir: str = "Analisis_SyS/Taller/audio_cache"
    ffmpeg_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
