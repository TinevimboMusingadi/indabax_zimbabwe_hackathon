"""PyTorch MLP with focal loss for tabular binary classification."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class MLPModel(BaseModel):
    """Simple MLP with focal loss, dropout, and early stopping."""

    name = "mlp"

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        focal_gamma: float = 2.0,
        patience: int = 10,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.focal_gamma = focal_gamma
        self.patience = patience
        self._model = None
        self._device = "cpu"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        in_dim = X_train.shape[1]

        self._model = self._build_net(in_dim).to(self._device)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=1e-5
        )

        pos_weight = float(
            (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        )

        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        train_dl = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )

        best_val_loss = float("inf")
        wait = 0

        for epoch in range(self.epochs):
            self._model.train()
            for xb, yb in train_dl:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                logits = self._model(xb).squeeze(-1)
                loss = self._focal_loss(logits, yb, pos_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if X_val is not None and y_val is not None:
                val_loss = self._eval_loss(X_val, y_val, pos_weight)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        logger.info(
                            "MLP early stop at epoch %d", epoch + 1
                        )
                        break

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("Call fit() first.")
        self._model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self._device)
            logits = self._model(t).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def _build_net(self, in_dim: int):  # type: ignore[no-untyped-def]
        import torch.nn as nn

        layers = []
        prev = in_dim
        for dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            prev = dim
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def _focal_loss(self, logits, targets, pos_weight):  # type: ignore[no-untyped-def]
        import torch
        import torch.nn.functional as F

        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(pos_weight, device=logits.device),
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        return (focal_weight * bce).mean()

    def _eval_loss(self, X, y, pos_weight):  # type: ignore[no-untyped-def]
        import torch

        self._model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self._device)
            yt = torch.FloatTensor(y).to(self._device)
            logits = self._model(t).squeeze(-1)  # type: ignore[union-attr]
            loss = self._focal_loss(logits, yt, pos_weight)
        return loss.item()
