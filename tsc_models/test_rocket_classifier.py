import numpy as np
import polars as pl
import pytest

from tsc_models.rocket_classifier import RocketTimeSeriesClassifier


def _make_dataset(n_samples: int = 40, n_timepoints: int = 18, seed: int = 0):
    rng = np.random.RandomState(seed)
    X_np = rng.randn(n_samples, n_timepoints).astype(np.float32)
    X = pl.DataFrame(X_np, schema=[f"t{i}" for i in range(n_timepoints)])
    return X


def test_fit_transform_shape():
    X = _make_dataset()
    num_kernels = 50
    clf = RocketTimeSeriesClassifier(num_kernels=num_kernels, random_state=42)
    result = clf.fit_transform(X)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (len(X), num_kernels * 2)


def test_transform_before_fit_raises():
    X = _make_dataset()
    clf = RocketTimeSeriesClassifier(num_kernels=10)
    with pytest.raises(RuntimeError, match="fitted"):
        clf.transform(X)


def test_reproducibility():
    X = _make_dataset()
    clf_a = RocketTimeSeriesClassifier(num_kernels=20, random_state=7)
    clf_b = RocketTimeSeriesClassifier(num_kernels=20, random_state=7)
    result_a = clf_a.fit_transform(X)
    result_b = clf_b.fit_transform(X)
    assert result_a.equals(result_b)


def test_fit_transform_separate_train_test():
    X_train = _make_dataset(n_samples=40, seed=1)
    X_test = _make_dataset(n_samples=15, seed=2)
    num_kernels = 20
    clf = RocketTimeSeriesClassifier(num_kernels=num_kernels, random_state=10)
    clf.fit(X_train)
    result = clf.transform(X_test)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (15, num_kernels * 2)


def test_output_column_names():
    X = _make_dataset()
    num_kernels = 3
    clf = RocketTimeSeriesClassifier(num_kernels=num_kernels, random_state=0)
    result = clf.fit_transform(X)
    expected = [f"rocket_{feat}_{k}" for k in range(num_kernels) for feat in ("max", "ppv")]
    assert result.columns == expected
