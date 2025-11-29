"""Audio ingestion helpers (local files + YouTube URLs)."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .config import SpectralConfig, resolve_path

try:
    import yt_dlp
except ImportError:  # pragma: no cover - optional dependency when not using URLs
    yt_dlp = None

try:  # pragma: no cover - optional dependency
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - fallback to system ffmpeg
    imageio_ffmpeg = None


_AUDIO_EXT_PRIORITY = (".wav", ".flac", ".ogg", ".m4a", ".mp3", ".opus", ".webm")
_NON_AUDIO_SUFFIXES = {".json", ".part", ".ytdl", ".info.json", ".jpg", ".jpeg", ".png"}
_SAFE_NATIVE_EXTS = {".wav", ".flac", ".ogg"}
_FFMPEG_HINT = (
    "Instala ffmpeg en el sistema o agrega la dependencia 'imageio-ffmpeg' (pip install "
    "imageio-ffmpeg) y vuelve a ejecutar."
)
_AUTO_FFMPEG_PATH: Optional[str] = None


def looks_like_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _pick_cached_audio(cache_dir: Path, key: str) -> Optional[Path]:
    for ext in _AUDIO_EXT_PRIORITY:
        candidate = cache_dir / f"{key}{ext}"
        if candidate.exists():
            return candidate
    for ext in _AUDIO_EXT_PRIORITY:
        for candidate in cache_dir.glob(f"**/{key}{ext}"):
            if candidate.exists():
                return candidate
    for candidate in cache_dir.glob(f"**/{key}.*"):
        if candidate.suffix.lower() not in _NON_AUDIO_SUFFIXES and candidate.exists():
            return candidate
    return None


def _auto_ffmpeg_path() -> Optional[str]:
    """Return an ffmpeg binary bundled via imageio-ffmpeg when available."""
    global _AUTO_FFMPEG_PATH
    if _AUTO_FFMPEG_PATH is not None:
        return _AUTO_FFMPEG_PATH
    if imageio_ffmpeg is None:
        return None
    try:  # pragma: no cover - depends on wheel contents
        _AUTO_FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        _AUTO_FFMPEG_PATH = None
    return _AUTO_FFMPEG_PATH


def _resolve_ffmpeg_path(cfg: SpectralConfig) -> Optional[str]:
    """Pick the best ffmpeg executable available (cfg -> PATH -> bundled)."""
    candidate = cfg.ffmpeg_path
    if candidate:
        cand_path = Path(candidate)
        if cand_path.is_dir():
            cand_path = cand_path / "ffmpeg"
        if cand_path.is_file():
            return str(cand_path)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    auto_path = _auto_ffmpeg_path()
    if auto_path and Path(auto_path).exists():
        return auto_path

    return None


def _convert_to_wav(src: Path, dst: Path, ffmpeg_path: str) -> Path:
    """Convert any audio format supported by ffmpeg into a single cached WAV."""
    if dst.exists():
        return dst

    tmp_fd, tmp_name = tempfile.mkstemp(dir=dst.parent, suffix=".wav")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        tmp_path.replace(dst)
        return dst
    except FileNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"No se encontró ffmpeg en {ffmpeg_path!r}. {_FFMPEG_HINT}") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on data
        raise RuntimeError(f"ffmpeg falló al convertir {src.name}: {exc}. {_FFMPEG_HINT}") from exc
    finally:
        if tmp_path.exists() and not dst.exists():
            tmp_path.unlink(missing_ok=True)


def _ensure_wav_asset(cache_dir: Path, cache_key: str, ffmpeg_path: Optional[str]) -> Path:
    """Ensure we have a WAV version of the cached audio (converts when needed)."""
    wav_target = cache_dir / f"{cache_key}.wav"
    if wav_target.exists():
        return wav_target

    candidate = _pick_cached_audio(cache_dir, cache_key)
    if candidate is None:
        raise RuntimeError("No se encontró el audio descargado en el caché.")

    if candidate.suffix.lower() in _SAFE_NATIVE_EXTS:
        return candidate

    if ffmpeg_path is None:
        raise RuntimeError(
            f"El audio se descargó como '{candidate.suffix}' y se requiere ffmpeg para convertirlo. {_FFMPEG_HINT}"
        )

    return _convert_to_wav(candidate, wav_target, ffmpeg_path)


def ensure_local_audio(path_or_url: str, cfg: SpectralConfig) -> str:
    """Return a local audio path; download from YouTube when necessary."""
    if os.path.exists(path_or_url):
        return path_or_url

    if not looks_like_url(path_or_url):
        resolved = resolve_path(path_or_url)
        if Path(resolved).exists():
            return str(resolved)
        raise FileNotFoundError(
            f"No existe el archivo ni es una URL válida: {path_or_url} (resuelto como {resolved})."
        )

    if yt_dlp is None:
        raise RuntimeError("Se requiere 'yt_dlp' para descargar audio desde enlaces.")

    cache_dir = Path(cfg.audio_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = hashlib.sha1(path_or_url.encode("utf-8")).hexdigest()
    cached = _pick_cached_audio(cache_dir, cache_key)
    if cached is not None and cached.suffix.lower() in _SAFE_NATIVE_EXTS:
        return str(cached)

    ffmpeg_path = _resolve_ffmpeg_path(cfg)

    if cached is not None:
        return str(_ensure_wav_asset(cache_dir, cache_key, ffmpeg_path))

    base_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(cache_dir / f"{cache_key}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "retries": 5,
        "fragment_retries": 5,
        "overwrites": False,
        "socket_timeout": 20,
    }
    if ffmpeg_path:
        base_opts["ffmpeg_location"] = ffmpeg_path

    try:
        with yt_dlp.YoutubeDL(base_opts) as dl:
            dl.download([path_or_url])
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"No se pudo descargar el audio: {exc}") from exc

    cached = _pick_cached_audio(cache_dir, cache_key)
    if cached is None:
        raise RuntimeError("La descarga terminó sin generar un archivo de audio.")

    return str(_ensure_wav_asset(cache_dir, cache_key, ffmpeg_path))
