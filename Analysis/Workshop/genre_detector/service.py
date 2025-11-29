"""Opinionated helpers bound to the Taller assets (dataset.csv, artifacts, cache)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .knn import KNNSpectralClassifier
from .pipeline import auto_fit_and_save, classify_with_model

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_CSV = BASE_DIR / "dataset.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DEFAULT_MODEL = ARTIFACTS_DIR / "genre_knn.joblib"
DEFAULT_CACHE = BASE_DIR / "audio_cache"


def _default_cfg_overrides() -> Dict[str, Any]:
    return {
        "audio_cache_dir": str(DEFAULT_CACHE),
        "n_jobs": -1,
    }


def ensure_default_model(force_retrain: bool = False, cfg_overrides: Optional[Dict[str, Any]] = None) -> Path:
    if DEFAULT_MODEL.exists() and not force_retrain:
        return DEFAULT_MODEL

    overrides = _default_cfg_overrides()
    overrides.update(cfg_overrides or {})
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"No se encontró el dataset CSV en {DATASET_CSV}")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    auto_fit_and_save(
        csv_path=str(DATASET_CSV),
        out_model_path=str(DEFAULT_MODEL),
        path_col="link",
        label_col="genre",
        cfg_overrides=overrides,
    )
    return DEFAULT_MODEL


def predict_from_link(
    link: str,
    *,
    k: Optional[int] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    model_path = ensure_default_model(force_retrain=force_retrain, cfg_overrides=cfg_overrides)
    return classify_with_model(str(model_path), link, k=k)


def load_default_classifier(
    *,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    force_retrain: bool = False,
) -> KNNSpectralClassifier:
    """Ensure the default artifacts exist and return a loaded classifier."""
    model_path = ensure_default_model(force_retrain=force_retrain, cfg_overrides=cfg_overrides)
    clf = KNNSpectralClassifier()
    clf.load(str(model_path))
    return clf
