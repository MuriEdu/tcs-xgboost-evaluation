"""Strict generic interface definition for time-series classifier transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import polars as pl


class BaseTimeSeriesClassifier(ABC):
    """Generic time-series classifier interface for transform-based pipelines."""

    # When True, transform is unsupervised and data-deterministic: the same row
    # always maps to the same meta-features regardless of any train/test split.
    # The pipeline may then fit+transform ONCE on all rows and slice per fold,
    # instead of recomputing the transform inside every cross-validation fold.
    supports_global_transform: bool = False

    # When True, the model exposes prepare_cache / fit_indexed / transform_indexed:
    # a fold-independent feature cache is built ONCE over all rows, then trees are
    # retrained per fold by row index (the cheap, supervised part). Used by
    # validate_pipeline to avoid recomputing per-fold features across CV folds.
    supports_indexed_fit: bool = False

    @abstractmethod
    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "BaseTimeSeriesClassifier":
        """Train the time-series classifier on time-series data."""
        pass

    @abstractmethod
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform raw time-series data into meta-features for the final model."""
        pass

    def fit_transform(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> pl.DataFrame:
        """Train and transform in one step for pipeline convenience."""
        return self.fit(X, y).transform(X)
