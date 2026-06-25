"""ROCKET time-series classifier (Dempster et al., 2020).

RandOm Convolutional KErnel Transform: applies a large number of random convolutional
kernels to each time series and extracts two global features per kernel — max pooling
and proportion of positive values (PPV). This yields highly discriminative features
with very low compute cost.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from .base import BaseTimeSeriesClassifier


class RocketTimeSeriesClassifier(BaseTimeSeriesClassifier):
    """ROCKET transform: random kernels -> max + PPV features per kernel.

    Produces 2 * num_kernels meta-features per sample, no training labels required.
    """

    # Kernels depend only on seq_len; transform is unsupervised and deterministic.
    supports_global_transform = True

    def __init__(self, num_kernels: int = 2_000, random_state: Optional[int] = 42) -> None:
        self.num_kernels = num_kernels
        self.random_state = random_state
        self._kernels: Optional[dict] = None

    def _generate_kernels(self, seq_len: int) -> dict:
        rng = np.random.default_rng(self.random_state)

        # Candidate kernel lengths (odd values only, capped at seq_len)
        candidate_lengths = np.array([v for v in [7, 9, 11] if v <= seq_len], dtype=np.int32)
        if candidate_lengths.size == 0:
            candidate_lengths = np.array([min(3, seq_len)], dtype=np.int32)

        lengths = rng.choice(candidate_lengths, size=self.num_kernels)

        weights = []
        biases = rng.uniform(-1.0, 1.0, size=self.num_kernels)
        dilations = []
        paddings = []

        for length in lengths:
            w = rng.standard_normal(length).astype(np.float32)
            w -= w.mean()
            weights.append(w)

            max_dilation_exp = int(np.floor(np.log2((seq_len - 1) / (length - 1)))) if length > 1 else 0
            max_dilation_exp = max(max_dilation_exp, 0)
            dilation = 2 ** rng.integers(0, max_dilation_exp + 1)
            dilations.append(int(dilation))

            use_padding = rng.integers(0, 2)
            paddings.append(int(use_padding))

        return {
            "lengths": lengths,
            "weights": weights,
            "biases": biases.astype(np.float32),
            "dilations": dilations,
            "paddings": paddings,
        }

    @staticmethod
    def _apply_kernel_batch(
        data: np.ndarray,
        weight: np.ndarray,
        bias: float,
        dilation: int,
        use_padding: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply one kernel to all samples at once.

        ``data`` is (n_samples, seq_len). Returns (max_vals, ppv_vals), each
        shaped (n_samples,). Vectorized over samples and output positions via a
        dilated sliding window, replacing the former per-element Python loops.
        """
        n_samples = data.shape[0]
        kernel_len = len(weight)
        # Effective receptive field accounting for dilation
        effective_len = (kernel_len - 1) * dilation + 1

        if use_padding:
            pad_width = effective_len // 2
            data = np.pad(
                data, ((0, 0), (pad_width, pad_width)), mode="constant", constant_values=0.0
            )

        seq_len = data.shape[1]
        output_len = seq_len - effective_len + 1
        if output_len <= 0:
            return (
                np.full(n_samples, bias, dtype=np.float32),
                np.zeros(n_samples, dtype=np.float32),
            )

        # windows: (n_samples, output_len, effective_len); pick every dilation-th
        # tap to recover the kernel_len positions the kernel actually touches.
        windows = np.lib.stride_tricks.sliding_window_view(data, effective_len, axis=1)
        taps = windows[:, :, ::dilation]  # (n_samples, output_len, kernel_len)
        conv = taps @ weight + bias  # (n_samples, output_len)

        max_vals = conv.max(axis=1).astype(np.float32)
        ppv_vals = (conv > 0).mean(axis=1).astype(np.float32)
        return max_vals, ppv_vals

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "RocketTimeSeriesClassifier":
        self._kernels = self._generate_kernels(X.width)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if self._kernels is None:
            raise RuntimeError("RocketTimeSeriesClassifier must be fitted before calling transform.")

        data = X.to_numpy().astype(np.float32)
        n_samples = data.shape[0]
        n_features = self.num_kernels * 2
        output = np.empty((n_samples, n_features), dtype=np.float32)

        kernels = self._kernels
        for k in range(self.num_kernels):
            max_vals, ppv_vals = self._apply_kernel_batch(
                data,
                kernels["weights"][k],
                kernels["biases"][k],
                kernels["dilations"][k],
                kernels["paddings"][k],
            )
            output[:, k * 2] = max_vals
            output[:, k * 2 + 1] = ppv_vals

        col_names = [f"rocket_{feat}_{k}" for k in range(self.num_kernels) for feat in ("max", "ppv")]
        return pl.DataFrame(output, schema=col_names)
