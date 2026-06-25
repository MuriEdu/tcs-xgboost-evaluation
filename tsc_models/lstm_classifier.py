"""LSTM-based time-series classifier for meta-feature generation."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseTimeSeriesClassifier

logger = logging.getLogger(__name__)


class _LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size) — take last layer
        return h_n[-1]


class LSTMTimeSeriesClassifier(BaseTimeSeriesClassifier):
    """LSTM encoder that maps temporal sequences to fixed-size meta-features.

    Each temporal column is treated as one timestep with a single feature dimension.
    The encoder is trained as an autoencoder to learn compact representations.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 30,
        # Larger batch keeps the GPU busy (~4-6x faster/epoch). LR scaled by the
        # sqrt rule (batch 64->512 = 8x, lr 1e-3 * sqrt(8) ≈ 3e-3) so the fewer
        # gradient steps per epoch don't degrade convergence.
        batch_size: int = 512,
        learning_rate: float = 3e-3,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self._encoder: Optional[_LSTMEncoder] = None

    def _to_tensor(self, X: pl.DataFrame) -> torch.Tensor:
        arr = X.to_numpy().astype(np.float32)
        # shape: (n_samples, seq_len, 1) — kept on CPU; moved to device per batch
        # so DataLoader can pin memory and overlap host->GPU copies.
        return torch.from_numpy(arr).unsqueeze(-1)

    def fit(self, X: pl.DataFrame, y: Optional[pl.Series] = None) -> "LSTMTimeSeriesClassifier":
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        seq_len = X.width
        encoder = _LSTMEncoder(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Decoder reconstructs the input sequence from the hidden state
        decoder = nn.Linear(self.hidden_size, seq_len).to(self.device)

        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        x_tensor = self._to_tensor(X)  # CPU
        # Target for reconstruction: original sequence flattened per sample
        target = x_tensor.squeeze(-1)  # (n_samples, seq_len)

        use_cuda = self.device.type == "cuda"
        logger.info("LSTM training on device=%s", self.device)
        dataset = TensorDataset(x_tensor, target)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, pin_memory=use_cuda
        )

        encoder.train()
        decoder.train()
        for _ in range(self.epochs):
            for x_batch, t_batch in loader:
                x_batch = x_batch.to(self.device, non_blocking=use_cuda)
                t_batch = t_batch.to(self.device, non_blocking=use_cuda)
                optimizer.zero_grad()
                h = encoder(x_batch)
                reconstruction = decoder(h)
                loss = loss_fn(reconstruction, t_batch)
                loss.backward()
                optimizer.step()

        self._encoder = encoder
        self._encoder.eval()
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        if self._encoder is None:
            raise RuntimeError("LSTMTimeSeriesClassifier must be fitted before calling transform.")

        x_tensor = self._to_tensor(X).to(self.device)
        with torch.no_grad():
            hidden = self._encoder(x_tensor).cpu().numpy()  # (n_samples, hidden_size)

        col_names = [f"lstm_h{i}" for i in range(hidden.shape[1])]
        return pl.DataFrame(hidden, schema=col_names)
