import numpy as np
import polars as pl
import pytest

from tsc_models.lstm_classifier import LSTMTimeSeriesClassifier


def _make_dataset(n_samples: int = 40, n_timepoints: int = 12, seed: int = 0):
    rng = np.random.RandomState(seed)
    X_np = rng.randn(n_samples, n_timepoints).astype(np.float32)
    y_np = rng.randint(0, 2, size=n_samples)
    X = pl.DataFrame(X_np, schema=[f"t{i}" for i in range(n_timepoints)])
    y = pl.Series("y", y_np)
    return X, y


def test_fit_transform_shape():
    X, y = _make_dataset()
    hidden_size = 8
    clf = LSTMTimeSeriesClassifier(hidden_size=hidden_size, epochs=2)
    result = clf.fit_transform(X, y)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (len(X), hidden_size)


def test_transform_before_fit_raises():
    X, _ = _make_dataset()
    clf = LSTMTimeSeriesClassifier()
    with pytest.raises(RuntimeError, match="fitted"):
        clf.transform(X)


def test_fit_transform_separate_train_test():
    X_train, y_train = _make_dataset(n_samples=40, seed=1)
    X_test, _ = _make_dataset(n_samples=15, seed=2)
    hidden_size = 8
    clf = LSTMTimeSeriesClassifier(hidden_size=hidden_size, epochs=2)
    clf.fit(X_train, y_train)
    result = clf.transform(X_test)
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (15, hidden_size)


def test_output_column_names():
    X, y = _make_dataset()
    hidden_size = 4
    clf = LSTMTimeSeriesClassifier(hidden_size=hidden_size, epochs=2)
    result = clf.fit_transform(X, y)
    assert result.columns == [f"lstm_h{i}" for i in range(hidden_size)]


def test_fit_without_labels():
    X, _ = _make_dataset()
    clf = LSTMTimeSeriesClassifier(hidden_size=4, epochs=2)
    clf.fit(X)
    result = clf.transform(X)
    assert result.shape == (len(X), 4)
