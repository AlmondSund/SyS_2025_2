"""k-NN classifier built on top of spectral magnitude features."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load, Parallel, delayed

from .config import SpectralConfig
from .features import extract_feature_vector


def _detect_columns(df: pd.DataFrame, path_col: Optional[str], label_col: Optional[str]) -> Tuple[str, str]:
    if path_col and label_col:
        return path_col, label_col
    cols = {c.lower(): c for c in df.columns}
    path_candidates = ["path", "filepath", "file", "audio", "filename", "wav", "mp3", "ogg", "uri", "link"]
    label_candidates = ["label", "genre", "class", "target", "y", "tag"]
    resolved_path = path_col or next((cols[c] for c in path_candidates if c in cols), None)
    resolved_label = label_col or next((cols[c] for c in label_candidates if c in cols), None)
    if resolved_path is None or resolved_label is None:
        raise ValueError(
            "No pude detectar columnas de ruta y etiqueta. Incluye ('path'/'filepath'/...) y ('label'/'genre'/...)."
        )
    return resolved_path, resolved_label


class KNNSpectralClassifier:
    def __init__(self, cfg: Optional[SpectralConfig] = None):
        self.cfg = cfg or SpectralConfig()
        self.X: Optional[np.ndarray] = None
        self.labels: List[str] = []
        self.paths: List[str] = []
        self.sources: List[str] = []
        self.meta: Dict[str, Any] = {}

    def _extract_feature(self, path_or_url: str, label: str) -> Tuple[np.ndarray, str, str]:
        feat, meta = extract_feature_vector(path_or_url, self.cfg)
        return feat, meta["resolved_path"], path_or_url

    def fit_from_dataframe(self, df: pd.DataFrame, path_col: Optional[str] = None, label_col: Optional[str] = None) -> "KNNSpectralClassifier":
        path_col, label_col = _detect_columns(df, path_col, label_col)
        subset = df[[path_col, label_col]].dropna()
        jobs = (
            delayed(self._extract_feature)(row[path_col], row[label_col])
            for _, row in subset.iterrows()
        )
        records = Parallel(n_jobs=self.cfg.n_jobs)(jobs)
        if not records:
            raise RuntimeError("El dataframe no contiene registros válidos para entrenar.")

        features, paths, sources = zip(*records)
        self.X = np.vstack(features)
        self.labels = list(subset[label_col].values)
        self.paths = list(paths)
        self.sources = list(sources)
        self.meta = {"config": self.cfg.to_dict(), "n_samples": len(self.labels)}
        return self

    def fit(self, paths: Sequence[str], labels: Sequence[str]) -> "KNNSpectralClassifier":
        if len(paths) != len(labels):
            raise ValueError("paths y labels deben tener la misma longitud.")
        jobs = (delayed(self._extract_feature)(p, l) for p, l in zip(paths, labels))
        records = Parallel(n_jobs=self.cfg.n_jobs)(jobs)
        if not records:
            raise RuntimeError("No hay registros para entrenar.")
        feats, resolved_paths, _ = zip(*records)
        self.X = np.vstack(feats)
        self.paths = list(resolved_paths)
        self.sources = list(paths)
        self.labels = list(labels)
        self.meta = {"config": self.cfg.to_dict(), "n_samples": len(self.labels)}
        return self

    def _ensure_ready(self) -> None:
        if self.X is None or not self.labels:
            raise RuntimeError("El clasificador no ha sido entrenado.")

    def predict(
        self,
        path_or_url: str,
        k: Optional[int] = None,
        *,
        return_context: bool = False,
    ) -> Tuple[Any, ...]:
        self._ensure_ready()
        k = k or self.cfg.k
        feat, meta = extract_feature_vector(path_or_url, self.cfg)
        resolved_path = meta["resolved_path"]

        assert self.X is not None  # ayuda a mypy
        dists = np.linalg.norm(self.X - feat, axis=1)
        k = max(1, min(k, len(dists)))
        idx = np.argpartition(dists, k - 1)[:k]
        sorted_idx = idx[np.argsort(dists[idx])]

        neighbors = [
            {
                "index": int(i),
                "label": self.labels[i],
                "distance": float(dists[i]),
                "path": self.paths[i],
                "source": self.sources[i],
            }
            for i in sorted_idx
        ]

        votes: Dict[str, int] = {}
        for nb in neighbors:
            votes[nb["label"]] = votes.get(nb["label"], 0) + 1
        best_vote = max(votes.values())
        tied = [label for label, count in votes.items() if count == best_vote]
        pred = tied[0] if len(tied) == 1 else neighbors[0]["label"]
        probs = {label: votes[label] / len(neighbors) for label in votes}
        if not return_context:
            return pred, neighbors, probs, resolved_path

        context = {
            "feature_vector": feat,
            "neighbor_indices": [int(i) for i in sorted_idx],
            "distance_vector": dists,
        }
        return pred, neighbors, probs, resolved_path, context

    def save(self, path: str) -> None:
        self._ensure_ready()
        dump(
            {
                "X": self.X,
                "labels": self.labels,
                "paths": self.paths,
                "sources": self.sources,
                "meta": self.meta,
            },
            path,
        )

    def load(self, path: str) -> "KNNSpectralClassifier":
        payload = load(path)
        self.X = payload["X"]
        self.labels = list(payload["labels"])
        self.paths = list(payload["paths"])
        self.sources = list(payload.get("sources", []))
        self.meta = dict(payload.get("meta", {}))
        if "config" in self.meta:
            try:
                self.cfg = SpectralConfig(**self.meta["config"])
            except Exception:
                pass
        return self
