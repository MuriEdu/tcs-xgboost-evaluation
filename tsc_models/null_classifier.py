"""Null TSC implementation that passes through temporal data unchanged."""

from __future__ import annotations

from typing import Optional

import polars as pl

from .base import BaseTimeSeriesClassifier


class NullTimeSeriesClassifier(BaseTimeSeriesClassifier):
    """Pass-through time-series classifier. Returns the input data unchanged."""

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "NullTimeSeriesClassifier":
        self.feature_names_ = X.columns
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        return X.clone()
