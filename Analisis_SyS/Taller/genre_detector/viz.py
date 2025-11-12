"""Visualization helpers for the spectral k-NN detector."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from .knn import KNNSpectralClassifier


def project_knn_scene(
    clf: KNNSpectralClassifier,
    query_feature: np.ndarray,
    neighbor_indices: Sequence[int] | Iterable[int],
    dims: int = 3,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Project the training set + query sample onto the first `dims` principal axes."""
    if clf.X is None:
        raise RuntimeError("El clasificador no ha sido entrenado; no hay espectros para proyectar.")

    X = clf.X
    dims = max(1, min(dims, X.shape[1]))
    mean_vec = X.mean(axis=0, keepdims=True)
    centered = X - mean_vec
    # SVD para obtener los componentes principales sin depender de librerías extra.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:dims].T

    training_proj = centered @ components
    query_proj = (query_feature - mean_vec.squeeze()) @ components

    columns = [f"pc{i+1}" for i in range(dims)]
    data = pd.DataFrame(training_proj, columns=columns)
    # Garantiza columnas pc1..pc3 para facilitar graficación 3D.
    for extra_dim in range(dims, 3):
        col = f"pc{extra_dim+1}"
        data[col] = 0.0
    padded_query = np.zeros(3, dtype=float)
    padded_query[: dims if dims <= 3 else 3] = query_proj[: min(dims, 3)]

    data["label"] = clf.labels
    data["source"] = clf.sources
    data["path"] = clf.paths

    neighbor_mask = np.zeros(len(data), dtype=bool)
    for idx in neighbor_indices:
        if 0 <= idx < len(data):
            neighbor_mask[int(idx)] = True
    data["is_neighbor"] = neighbor_mask
    return data, padded_query
