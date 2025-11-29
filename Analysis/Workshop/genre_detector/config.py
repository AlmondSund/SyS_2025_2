"""Configuration helpers for the spectral k-NN genre detector."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent.parent if len(BASE_DIR.parents) >= 2 else BASE_DIR.parent
PACKAGE_ANCHOR = BASE_DIR.parent.name


def resolve_path(path_like: Union[str, Path]) -> Path:
    """Return an absolute path anchored to the Taller assets when possible."""
    original = str(path_like)
    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate

    normalized = original.replace("\\", "/")
    parts = candidate.parts
    if normalized.startswith(("./", "../")) or (parts and parts[0] in (".", "..")):
        anchor = Path.cwd()
    elif parts and parts[0] == PACKAGE_ANCHOR:
        anchor = PROJECT_ROOT
    else:
        anchor = BASE_DIR
    return (anchor / candidate).resolve()


DEFAULT_AUDIO_CACHE = str(resolve_path("audio_cache"))


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
    audio_cache_dir: str = DEFAULT_AUDIO_CACHE
    ffmpeg_path: Optional[str] = None

    def __post_init__(self) -> None:
        cache_dir = resolve_path(self.audio_cache_dir)
        object.__setattr__(self, "audio_cache_dir", str(cache_dir))

    def to_dict(self) -> dict:
        return asdict(self)
