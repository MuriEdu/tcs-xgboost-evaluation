from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import polars as pl

try:
    import pycatch22
except ImportError as _pycatch22_err:
    raise ImportError(
        "pycatch22 não está instalado. Execute: pip install pycatch22"
    ) from _pycatch22_err
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier

from .base import BaseTimeSeriesClassifier


def _extract_features(subseq: np.ndarray) -> np.ndarray:
    """Extrai 25 features de uma subsequência: mean, std, slope + 22 Catch22."""
    mean = np.mean(subseq)
    std = np.std(subseq)
    x = np.arange(len(subseq), dtype=float)
    slope = np.polyfit(x, subseq, 1)[0]
    catch22_vals = pycatch22.catch22_all(subseq.tolist())["values"]
    features = np.array([mean, std, slope] + catch22_vals, dtype=float)
    return np.nan_to_num(features, nan=0.0)


class _CIFTree:
    """Uma única árvore do ensemble CIF."""

    def __init__(self, min_interval_length: int) -> None:
        self._min_interval_length = min_interval_length
        self._intervals: List[Tuple[int, int]] = []
        self._tree: Optional[DecisionTreeClassifier] = None

    def _sample_intervals(self, n_timepoints: int, k: int, rng: np.random.RandomState) -> List[Tuple[int, int]]:
        intervals = []
        seen = set()
        max_attempts = k * 10
        attempts = 0
        while len(intervals) < k and attempts < max_attempts:
            start = rng.randint(0, max(1, n_timepoints - self._min_interval_length + 1))
            end = rng.randint(start + self._min_interval_length, n_timepoints + 1)
            key = (start, end)
            if key not in seen:
                seen.add(key)
                intervals.append(key)
            attempts += 1
        return intervals

    def _build_feature_matrix(self, X_np: np.ndarray) -> np.ndarray:
        n_samples = X_np.shape[0]
        n_features_per_interval = 25
        matrix = np.zeros((n_samples, len(self._intervals) * n_features_per_interval))
        for col_offset, (start, end) in enumerate(self._intervals):
            feat_start = col_offset * n_features_per_interval
            feat_end = feat_start + n_features_per_interval
            for row in range(n_samples):
                subseq = X_np[row, start:end]
                matrix[row, feat_start:feat_end] = _extract_features(subseq)
        return matrix

    def fit(self, X_np: np.ndarray, y_np: np.ndarray, rng: np.random.RandomState) -> "_CIFTree":
        n_samples, n_timepoints = X_np.shape
        # bootstrap
        boot_idx = rng.randint(0, n_samples, size=n_samples)
        X_boot = X_np[boot_idx]
        y_boot = y_np[boot_idx]

        k = math.ceil(math.sqrt(n_timepoints))
        self._intervals = self._sample_intervals(n_timepoints, k, rng)
        if not self._intervals:
            raise ValueError(
                f"Não foi possível sortear intervalos válidos para uma série com {n_timepoints} pontos "
                f"e min_interval_length={self._min_interval_length}."
            )

        feature_matrix = self._build_feature_matrix(X_boot)
        self._tree = DecisionTreeClassifier(random_state=rng.randint(0, 2**31))
        self._tree.fit(feature_matrix, y_boot)
        return self

    def transform(self, X_np: np.ndarray) -> np.ndarray:
        if self._tree is None:
            raise NotFittedError("_CIFTree não foi treinada. Chame fit() primeiro.")
        feature_matrix = self._build_feature_matrix(X_np)
        return self._tree.predict_proba(feature_matrix)


class CanonicalIntervalForestClassifier(BaseTimeSeriesClassifier):
    """Canonical Interval Forest: ensemble de árvores sobre intervalos aleatórios da série temporal."""

    def __init__(
        self,
        n_estimators: int = 200,
        min_interval_length: int = 3,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.min_interval_length = min_interval_length
        self.random_state = random_state
        self._n_classes_: int = 0
        self._classes_: np.ndarray = np.array([])

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "CanonicalIntervalForestClassifier":
        if y is None:
            raise ValueError("y é obrigatório para o treinamento do CIF.")
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators deve ser >= 1, recebido {self.n_estimators}.")
        if self.min_interval_length < 2:
            raise ValueError(f"min_interval_length deve ser >= 2, recebido {self.min_interval_length}.")
        if X.width < self.min_interval_length:
            raise ValueError(
                f"X tem {X.width} colunas, mas min_interval_length={self.min_interval_length}. "
                "A série temporal é curta demais."
            )
        if len(y) != X.height:
            raise ValueError(f"X tem {X.height} amostras mas y tem {len(y)} entradas.")

        X_np = X.to_numpy().astype(float)
        y_np = y.to_numpy().astype(int)

        rng_master = np.random.RandomState(self.random_state)
        seeds = rng_master.randint(0, 2**31, size=self.n_estimators)
        self._trees: List[_CIFTree] = []
        for i in range(self.n_estimators):
            rng = np.random.RandomState(int(seeds[i]))
            tree = _CIFTree(min_interval_length=self.min_interval_length)
            tree.fit(X_np, y_np, rng)
            self._trees.append(tree)

        self._classes_ = np.unique(y_np)
        self._n_classes_ = len(self._classes_)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if not hasattr(self, "_trees"):
            raise NotFittedError(
                "Este CanonicalIntervalForestClassifier ainda não foi treinado. Chame fit() primeiro."
            )
        X_np = X.to_numpy().astype(float)
        n_classes = self._n_classes_
        aligned = []
        for tree in self._trees:
            proba = tree.transform(X_np)
            if proba.shape[1] != n_classes:
                full = np.zeros((proba.shape[0], n_classes))
                for j, cls in enumerate(tree._tree.classes_):
                    idx = np.searchsorted(self._classes_, cls)
                    full[:, idx] = proba[:, j]
                aligned.append(full)
            else:
                aligned.append(proba)
        combined = np.concatenate(aligned, axis=1)
        col_names = [f"cif_tree{i}_class{c}" for i in range(self.n_estimators) for c in range(n_classes)]
        return pl.DataFrame(combined, schema=col_names)
