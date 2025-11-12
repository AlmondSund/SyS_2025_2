"""High-level helpers for fitting and running the detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import pandas as pd

from .config import BASE_DIR, SpectralConfig, resolve_path
from .knn import KNNSpectralClassifier


DEFAULT_MODEL_PATH = str((BASE_DIR / "artifacts" / "genre_knn.joblib").resolve())
DEFAULT_CACHE_DESC = str((BASE_DIR / "audio_cache").resolve())


def _looks_like_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def auto_fit_and_save(
    csv_path: str,
    out_model_path: str,
    path_col: Optional[str] = None,
    label_col: Optional[str] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    csv_path = str(resolve_path(csv_path))
    out_model_path = str(resolve_path(out_model_path))
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"No se encontró el dataset en {csv_path}")
    cfg_kwargs = dict(cfg_overrides or {})
    cfg = SpectralConfig(**cfg_kwargs)
    df = pd.read_csv(csv_path)
    clf = KNNSpectralClassifier(cfg)
    clf.fit_from_dataframe(df, path_col=path_col, label_col=label_col)
    Path(out_model_path).parent.mkdir(parents=True, exist_ok=True)
    clf.save(out_model_path)
    return out_model_path


def classify_with_model(model_path: str, audio_path_or_url: str, k: Optional[int] = None) -> Dict[str, Any]:
    model_path = str(resolve_path(model_path))
    clf = KNNSpectralClassifier().load(model_path)
    pred, neighbors, probs, resolved_path = clf.predict(audio_path_or_url, k=k)
    return {
        "predicted_label": pred,
        "neighbors": neighbors,
        "vote_probs": probs,
        "resolved_audio_path": resolved_path,
        "config": clf.meta.get("config", {}),
    }


def cli():
    import argparse

    parser = argparse.ArgumentParser(description="Detector k-NN de género musical basado en espectro.")
    parser.add_argument("--dataset", type=str, help="Ruta al CSV con rutas/URLs y su género.")
    parser.add_argument("--path-col", type=str, default=None)
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument(
        "--out-model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Ruta de salida del modelo (por defecto {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--hop-length", type=int, default=2048)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--center-stft", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=f"Directorio para cachear audios descargados (por defecto {DEFAULT_CACHE_DESC}).",
    )
    parser.add_argument("--ffmpeg", type=str, default=None)
    parser.add_argument("--classify", type=str, default=None, help="Audio o URL para clasificar.")
    parser.add_argument("--k-infer", type=int, default=None)

    args = parser.parse_args()
    cfg_over = dict(
        sr=args.sr,
        duration=args.duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        k=args.k,
        center_stft=args.center_stft,
        n_jobs=args.n_jobs,
    )
    if args.cache_dir:
        cfg_over["audio_cache_dir"] = str(resolve_path(args.cache_dir))
    if args.ffmpeg:
        cfg_over["ffmpeg_path"] = args.ffmpeg

    dataset_path = str(resolve_path(args.dataset)) if args.dataset else None
    out_model_path = str(resolve_path(args.out_model))

    if dataset_path:
        model_path = auto_fit_and_save(
            csv_path=dataset_path,
            out_model_path=out_model_path,
            path_col=args.path_col,
            label_col=args.label_col,
            cfg_overrides=cfg_over,
        )
        print(f"[ok] Modelo guardado en: {model_path}")

    if args.classify:
        classify_target = args.classify
        if not _looks_like_url(classify_target):
            classify_target = str(resolve_path(classify_target))
        if not Path(out_model_path).exists():
            raise SystemExit(f"Modelo no encontrado: {out_model_path}")
        result = classify_with_model(out_model_path, classify_target, k=args.k_infer)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
