"""High-level helpers for fitting and running the detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .config import SpectralConfig
from .knn import KNNSpectralClassifier


def auto_fit_and_save(
    csv_path: str,
    out_model_path: str,
    path_col: Optional[str] = None,
    label_col: Optional[str] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    cfg_kwargs = dict(cfg_overrides or {})
    cfg = SpectralConfig(**cfg_kwargs)
    df = pd.read_csv(csv_path)
    clf = KNNSpectralClassifier(cfg)
    clf.fit_from_dataframe(df, path_col=path_col, label_col=label_col)
    Path(out_model_path).parent.mkdir(parents=True, exist_ok=True)
    clf.save(out_model_path)
    return out_model_path


def classify_with_model(model_path: str, audio_path_or_url: str, k: Optional[int] = None) -> Dict[str, Any]:
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
    parser.add_argument("--out-model", type=str, default="Analisis_SyS/Taller/artifacts/knn_model.joblib")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--hop-length", type=int, default=2048)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--center-stft", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cache-dir", type=str, default=None)
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
        cfg_over["audio_cache_dir"] = args.cache_dir
    if args.ffmpeg:
        cfg_over["ffmpeg_path"] = args.ffmpeg

    if args.dataset:
        model_path = auto_fit_and_save(
            csv_path=args.dataset,
            out_model_path=args.out_model,
            path_col=args.path_col,
            label_col=args.label_col,
            cfg_overrides=cfg_over,
        )
        print(f"[ok] Modelo guardado en: {model_path}")

    if args.classify:
        if not Path(args.out_model).exists():
            raise SystemExit(f"Modelo no encontrado: {args.out_model}")
        result = classify_with_model(args.out_model, args.classify, k=args.k_infer)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
