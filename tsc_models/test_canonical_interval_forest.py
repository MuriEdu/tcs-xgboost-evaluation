import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from tsc_models.canonical_interval_forest import CanonicalIntervalForestClassifier


def _make_dataset(n_samples: int = 40, n_timepoints: int = 18, seed: int = 0):
    rng = np.random.RandomState(seed)
    X_np = rng.randn(n_samples, n_timepoints).astype(float)
    y_np = rng.randint(0, 2, size=n_samples)
    X = pl.DataFrame(X_np, schema=[f"t{i}" for i in range(n_timepoints)])
    y = pl.Series("y", y_np)
    return X, y


def test_fit_transform_shape():
    X, y = _make_dataset()
    n_estimators = 5
    clf = CanonicalIntervalForestClassifier(n_estimators=n_estimators, random_state=42)
    result = clf.fit_transform(X, y)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (len(X), n_estimators * 2)


def test_transform_before_fit_raises():
    X, _ = _make_dataset()
    clf = CanonicalIntervalForestClassifier(n_estimators=3, random_state=0)
    with pytest.raises(NotFittedError):
        clf.transform(X)


def test_reproducibility():
    X, y = _make_dataset()
    clf_a = CanonicalIntervalForestClassifier(n_estimators=5, random_state=7)
    clf_b = CanonicalIntervalForestClassifier(n_estimators=5, random_state=7)
    result_a = clf_a.fit_transform(X, y)
    result_b = clf_b.fit_transform(X, y)
    assert result_a.equals(result_b)


def test_fit_transform_separate_train_test():
    X_train, y_train = _make_dataset(n_samples=60, seed=1)
    X_test, _ = _make_dataset(n_samples=20, seed=2)
    n_estimators = 4
    clf = CanonicalIntervalForestClassifier(n_estimators=n_estimators, random_state=10)
    clf.fit(X_train, y_train)
    result = clf.transform(X_test)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (20, n_estimators * 2)


def test_fit_raises_on_y_none():
    X, _ = _make_dataset()
    clf = CanonicalIntervalForestClassifier(n_estimators=3, random_state=0)
    with pytest.raises(ValueError, match="y é obrigatório"):
        clf.fit(X)


def test_fit_raises_on_mismatched_lengths():
    X, y = _make_dataset(n_samples=40)
    y_short = y[:30]
    clf = CanonicalIntervalForestClassifier(n_estimators=3, random_state=0)
    with pytest.raises(ValueError, match="amostras"):
        clf.fit(X, y_short)


def test_fit_raises_on_series_too_short():
    rng = np.random.RandomState(0)
    X_short = pl.DataFrame(rng.randn(20, 2), schema=["t0", "t1"])
    y = pl.Series("y", rng.randint(0, 2, 20))
    clf = CanonicalIntervalForestClassifier(n_estimators=3, min_interval_length=3, random_state=0)
    with pytest.raises(ValueError):
        clf.fit(X_short, y)
