"""Shapelet-based time-series classifier.

This implementation uses Random Shapelets to extract meta-features. 
A shapelet is a time-series subsequence that is representative of a class.
By calculating the minimum distance between a sample and various random shapelets,
we can generate discriminative features for the final classifier.
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np
import polars as pl

from .base import BaseTimeSeriesClassifier


class ShapeletTimeSeriesClassifier(BaseTimeSeriesClassifier):
    """Random Shapelet Transform: extracts distances to random subsequences.

    Produces num_shapelets meta-features per sample.
    """

    # Distances are unsupervised and data-deterministic: the same row always maps
    # to the same shapelet distances, so the pipeline may fit+transform once on all
    # rows and slice per fold instead of recomputing it inside every CV fold.
    supports_global_transform = True

    def __init__(
        self, 
        num_shapelets: int = 100, 
        min_shapelet_length: int = 3, 
        max_shapelet_length: Optional[int] = None,
        random_state: Optional[int] = 42
    ) -> None:
        self.num_shapelets = num_shapelets
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.random_state = random_state
        self._shapelets: List[np.ndarray] = []

    def _generate_random_shapelets(self, data: np.ndarray) -> List[np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        n_samples, seq_len = data.shape
        
        max_len = self.max_shapelet_length or seq_len
        shapelets = []
        
        for _ in range(self.num_shapelets):
            # Select a random sample
            sample_idx = rng.integers(0, n_samples)
            sample = data[sample_idx]
            
            # Select a random length and start position
            sh_len = rng.integers(self.min_shapelet_length, max_len + 1)
            start_pos = rng.integers(0, seq_len - sh_len + 1)
            
            shapelet = sample[start_pos : start_pos + sh_len]
            # Z-normalize shapelet for scale invariance
            std = shapelet.std()
            if std > 0:
                shapelet = (shapelet - shapelet.mean()) / std
            
            shapelets.append(shapelet)
            
        return shapelets

    @staticmethod
    def _min_dist_batch(data: np.ndarray, shapelet: np.ndarray) -> np.ndarray:
        """Min z-normalized squared distance from each row to a shapelet.

        Vectorized over all samples and sliding-window positions at once,
        replacing the former per-window / per-sample Python loops. ``data`` is
        (n_samples, seq_len); returns (n_samples,).
        """
        sh_len = len(shapelet)
        # windows: (n_samples, n_windows, sh_len)
        windows = np.lib.stride_tricks.sliding_window_view(data, sh_len, axis=1)

        mean = windows.mean(axis=2, keepdims=True)
        std = windows.std(axis=2, keepdims=True)
        # Z-normalize each window; leave constant windows (std==0) untouched, matching
        # the original scalar implementation.
        nonzero = std > 0
        safe_std = np.where(nonzero, std, 1.0)
        norm = np.where(nonzero, (windows - mean) / safe_std, windows)

        # mean squared diff over the shapelet length, then min over windows
        dist = ((norm - shapelet) ** 2).mean(axis=2)  # (n_samples, n_windows)
        return dist.min(axis=1)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "ShapeletTimeSeriesClassifier":
        data = X.to_numpy().astype(np.float32)
        self._shapelets = self._generate_random_shapelets(data)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if not self._shapelets:
            raise RuntimeError("ShapeletTimeSeriesClassifier must be fitted before calling transform.")

        data = X.to_numpy().astype(np.float32)
        n_samples = data.shape[0]
        output = np.empty((n_samples, self.num_shapelets), dtype=np.float32)

        for s_idx in range(self.num_shapelets):
            output[:, s_idx] = self._min_dist_batch(data, self._shapelets[s_idx])

        col_names = [f"shapelet_dist_{i}" for i in range(self.num_shapelets)]
        return pl.DataFrame(output, schema=col_names)
