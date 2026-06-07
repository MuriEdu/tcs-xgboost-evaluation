"""Strict generic interface definition for time-series classifier transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import polars as pl


class BaseTimeSeriesClassifier(ABC):
    """Generic time-series classifier interface for transform-based pipelines."""

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
