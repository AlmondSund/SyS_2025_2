"""Reusable components for the Taller 2 genre detector."""

from .config import SpectralConfig
from .knn import KNNSpectralClassifier
from .pipeline import auto_fit_and_save, classify_with_model
from .service import ensure_default_model, predict_from_link, load_default_classifier
from .viz import project_knn_scene

__all__ = [
    "SpectralConfig",
    "KNNSpectralClassifier",
    "auto_fit_and_save",
    "classify_with_model",
    "ensure_default_model",
    "predict_from_link",
    "load_default_classifier",
    "project_knn_scene",
]
