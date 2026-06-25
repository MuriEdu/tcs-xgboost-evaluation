from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import polars as pl
from joblib import Parallel, delayed, effective_n_jobs

try:
    import pycatch22
except ImportError as _pycatch22_err:
    raise ImportError(
        "pycatch22 não está instalado. Execute: pip install pycatch22"
    ) from _pycatch22_err
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier

from .base import BaseTimeSeriesClassifier


N_FEATURES_PER_INTERVAL = 25

Interval = Tuple[int, int]


def _interval_feature_block(sub: np.ndarray) -> np.ndarray:
    """25 features (mean, std, slope + 22 Catch22) for every row of ``sub`` ([:, start:end]).

    mean/std/slope vectorized across rows; Catch22 once per row (pycatch22 is
    one-series-at-a-time).
    """
    n_samples, length = sub.shape

    mean = sub.mean(axis=1)
    std = sub.std(axis=1)
    # Least-squares slope, vectorized: sum((x-x̄)·y)/sum((x-x̄)²) == np.polyfit(x,y,1)[0].
    x_centered = np.arange(length, dtype=float)
    x_centered -= x_centered.mean()
    slope = (sub @ x_centered) / float(x_centered @ x_centered)

    block = np.empty((n_samples, N_FEATURES_PER_INTERVAL), dtype=np.float32)
    block[:, 0] = mean
    block[:, 1] = std
    block[:, 2] = slope
    for row in range(n_samples):
        block[row, 3:] = pycatch22.catch22_all(sub[row].tolist())["values"]
    return np.nan_to_num(block, nan=0.0)


def _interval_blocks_for_rows(X_rows: np.ndarray, intervals: List[Interval]) -> np.ndarray:
    """Stack every interval's feature block for one chunk of rows.

    Runs inside a joblib worker — receives only its row slice of X (cheap to
    pickle), computes all intervals for those rows.
    """
    out = np.empty((X_rows.shape[0], len(intervals) * N_FEATURES_PER_INTERVAL), dtype=np.float32)
    for c, (start, end) in enumerate(intervals):
        out[:, c * N_FEATURES_PER_INTERVAL:(c + 1) * N_FEATURES_PER_INTERVAL] = _interval_feature_block(
            X_rows[:, start:end]
        )
    return out


def _sample_intervals(
    n_timepoints: int, k: int, min_interval_length: int, rng: np.random.RandomState
) -> List[Interval]:
    intervals: List[Interval] = []
    seen = set()
    max_attempts = k * 10
    attempts = 0
    while len(intervals) < k and attempts < max_attempts:
        start = rng.randint(0, max(1, n_timepoints - min_interval_length + 1))
        end = rng.randint(start + min_interval_length, n_timepoints + 1)
        key = (int(start), int(end))
        if key not in seen:
            seen.add(key)
            intervals.append(key)
        attempts += 1
    return intervals


class CanonicalIntervalForestClassifier(BaseTimeSeriesClassifier):
    """Canonical Interval Forest: ensemble de árvores sobre intervalos aleatórios da série temporal.

    Two-stage design for cross-validation efficiency:

    * The interval set and per-tree seeds are derived only from ``random_state``
      and ``n_timepoints`` — they do NOT depend on the data rows or labels.
    * Catch22 features per (row, interval) are therefore identical across CV
      folds. The expensive extraction is built ONCE over all rows
      (``prepare_cache``) and only the cheap DecisionTree training is redone per
      fold (``fit_indexed`` / ``transform_indexed``), indexing the shared cache.

    The cache build is parallelized over row-chunks with joblib (``n_jobs``).
    """

    # Lets the pipeline build the Catch22 cache once on all rows, then retrain
    # trees per fold by row index — see validate_pipeline.
    supports_indexed_fit = True

    def __init__(
        self,
        n_estimators: int = 200,
        min_interval_length: int = 3,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.min_interval_length = min_interval_length
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._n_classes_: int = 0
        self._classes_: np.ndarray = np.array([])

    # ------------------------------------------------------------------ #
    # Stage 1 — data-independent interval set + parallel Catch22 cache
    # ------------------------------------------------------------------ #
    def _plan_trees(self, n_timepoints: int) -> None:
        """Draw per-tree (interval, bootstrap, tree) seeds and interval sets.

        Independent RNG streams per role so the interval set does not depend on
        the bootstrap draw size (and thus is identical across folds of any size).
        """
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators deve ser >= 1, recebido {self.n_estimators}.")
        if self.min_interval_length < 2:
            raise ValueError(f"min_interval_length deve ser >= 2, recebido {self.min_interval_length}.")
        if n_timepoints < self.min_interval_length:
            raise ValueError(
                f"X tem {n_timepoints} colunas, mas min_interval_length={self.min_interval_length}. "
                "A série temporal é curta demais."
            )

        rng_master = np.random.RandomState(self.random_state)
        # columns: [interval_seed, bootstrap_seed, tree_seed]
        self._seeds = rng_master.randint(0, 2**31, size=(self.n_estimators, 3))
        k = math.ceil(math.sqrt(n_timepoints))
        self._tree_intervals: List[List[Interval]] = []
        for i in range(self.n_estimators):
            rng = np.random.RandomState(int(self._seeds[i, 0]))
            intervals = _sample_intervals(n_timepoints, k, self.min_interval_length, rng)
            if not intervals:
                raise ValueError(
                    f"Não foi possível sortear intervalos válidos para uma série com {n_timepoints} pontos "
                    f"e min_interval_length={self.min_interval_length}."
                )
            self._tree_intervals.append(intervals)
        self._n_timepoints = n_timepoints

    def _build_cache(self, X_np: np.ndarray, intervals: Iterable[Interval]) -> Dict[Interval, np.ndarray]:
        """Compute each unique interval's feature block over all rows of X_np, in parallel."""
        unique = list(dict.fromkeys(intervals))
        n = X_np.shape[0]
        # Split rows into one chunk per worker so the parallel pool is actually
        # used (effective_n_jobs resolves -1 to the real core count).
        n_chunks = max(1, min(effective_n_jobs(self.n_jobs), n))
        chunks = [c for c in np.array_split(np.arange(n), n_chunks) if len(c) > 0]

        parts = Parallel(n_jobs=self.n_jobs)(
            delayed(_interval_blocks_for_rows)(X_np[c], unique) for c in chunks
        )
        stacked = np.vstack(parts)  # (n, len(unique) * 25)
        return {
            iv: stacked[:, c * N_FEATURES_PER_INTERVAL:(c + 1) * N_FEATURES_PER_INTERVAL]
            for c, iv in enumerate(unique)
        }

    def prepare_cache(self, X: pl.DataFrame) -> "CanonicalIntervalForestClassifier":
        """Plan trees and build the Catch22 cache once over all rows in X."""
        X_np = X.to_numpy().astype(float)
        self._plan_trees(X_np.shape[1])
        all_intervals = [iv for ivs in self._tree_intervals for iv in ivs]
        self._cache = self._build_cache(X_np, all_intervals)
        return self

    # ------------------------------------------------------------------ #
    # Stage 2 — per-fold tree training / inference against the cache
    # ------------------------------------------------------------------ #
    def fit_indexed(self, row_idx: np.ndarray, y: pl.Series) -> "CanonicalIntervalForestClassifier":
        """Train trees on a subset of the cached rows (a CV fold), with bootstrap."""
        if not hasattr(self, "_cache"):
            raise NotFittedError("prepare_cache() deve ser chamado antes de fit_indexed().")
        row_idx = np.asarray(row_idx)
        y_np = y.to_numpy().astype(int)
        if len(y_np) != len(row_idx):
            raise ValueError(f"row_idx tem {len(row_idx)} entradas mas y tem {len(y_np)}.")
        n = len(row_idx)

        # Threads share the (large) cache without pickling it; sklearn's tree
        # builder releases the GIL during fitting, so this scales reasonably.
        self._trees: List[Tuple[List[Interval], DecisionTreeClassifier]] = Parallel(
            n_jobs=self.n_jobs, prefer="threads"
        )(delayed(self._fit_one_tree)(i, row_idx, y_np, n) for i in range(self.n_estimators))

        self._classes_ = np.unique(y_np)
        self._n_classes_ = len(self._classes_)
        return self

    def _fit_one_tree(
        self, i: int, row_idx: np.ndarray, y_np: np.ndarray, n: int
    ) -> Tuple[List[Interval], DecisionTreeClassifier]:
        boot = np.random.RandomState(int(self._seeds[i, 1])).randint(0, n, size=n)
        global_rows = row_idx[boot]
        intervals = self._tree_intervals[i]
        feature_matrix = np.hstack([self._cache[iv][global_rows] for iv in intervals])
        tree = DecisionTreeClassifier(random_state=int(self._seeds[i, 2]))
        tree.fit(feature_matrix, y_np[boot])
        return intervals, tree

    def transform_indexed(self, row_idx: np.ndarray) -> pl.DataFrame:
        """Transform a subset of the cached rows (a CV fold)."""
        if not hasattr(self, "_trees"):
            raise NotFittedError("fit_indexed() deve ser chamado antes de transform_indexed().")
        row_idx = np.asarray(row_idx)
        combined = self._combine_proba(self._cache, row_idx)
        return pl.DataFrame(combined, schema=self._proba_columns())

    # ------------------------------------------------------------------ #
    # Standalone API (single dataset, no shared cache)
    # ------------------------------------------------------------------ #
    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "CanonicalIntervalForestClassifier":
        if y is None:
            raise ValueError("y é obrigatório para o treinamento do CIF.")
        if len(y) != X.height:
            raise ValueError(f"X tem {X.height} amostras mas y tem {len(y)} entradas.")
        self.prepare_cache(X)
        self.fit_indexed(np.arange(X.height), y)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if not hasattr(self, "_trees"):
            raise NotFittedError(
                "Este CanonicalIntervalForestClassifier ainda não foi treinado. Chame fit() primeiro."
            )
        X_np = X.to_numpy().astype(float)
        cache = self._build_cache(X_np, (iv for ivs in self._tree_intervals for iv in ivs))
        combined = self._combine_proba(cache, None)
        return pl.DataFrame(combined, schema=self._proba_columns())

    # ------------------------------------------------------------------ #
    # Shared inference helpers
    # ------------------------------------------------------------------ #
    def _combine_proba(self, cache: Dict[Interval, np.ndarray], row_idx: Optional[np.ndarray]) -> np.ndarray:
        n_classes = self._n_classes_
        aligned = []
        for intervals, tree in self._trees:
            if row_idx is None:
                feature_matrix = np.hstack([cache[iv] for iv in intervals])
            else:
                feature_matrix = np.hstack([cache[iv][row_idx] for iv in intervals])
            proba = tree.predict_proba(feature_matrix)
            if proba.shape[1] != n_classes:
                full = np.zeros((proba.shape[0], n_classes))
                for j, cls in enumerate(tree.classes_):
                    idx = np.searchsorted(self._classes_, cls)
                    full[:, idx] = proba[:, j]
                aligned.append(full)
            else:
                aligned.append(proba)
        return np.concatenate(aligned, axis=1)

    def _proba_columns(self) -> List[str]:
        return [
            f"cif_tree{i}_class{c}"
            for i in range(self.n_estimators)
            for c in range(self._n_classes_)
        ]
