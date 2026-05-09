"""Entity Embedding Network for categorical features.

Trains a shallow embedding model, then extracts dense vectors
for downstream use as features in any model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class EmbeddingModel(BaseModel):
    """Train entity embeddings for categoricals, expose extract()."""

    name = "embedder"

    def __init__(
        self,
        cat_dims: list[int] | None = None,
        emb_dims: list[int] | None = None,
        num_cont: int = 0,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.cat_dims = cat_dims or []
        self.emb_dims = emb_dims or []
        self.num_cont = num_cont
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._net = None
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

        if not self.cat_dims:
            logger.warning("No cat_dims provided — nothing to embed.")
            return

        self._net = _EmbeddingNet(
            self.cat_dims, self.emb_dims, self.num_cont
        ).to(self._device)

        n_cat = len(self.cat_dims)
        X_cat = torch.LongTensor(X_train[:, :n_cat].astype(int))
        X_cont = torch.FloatTensor(X_train[:, n_cat:])
        y_t = torch.FloatTensor(y_train)

        ds = TensorDataset(X_cat, X_cont, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            self._net.train()
            for xc, xn, yb in dl:
                xc = xc.to(self._device)
                xn = xn.to(self._device)
                yb = yb.to(self._device)
                out = self._net(xc, xn).squeeze(-1)
                loss = criterion(out, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self._net is None:
            raise RuntimeError("Call fit() first.")
        self._net.eval()
        n_cat = len(self.cat_dims)
        with torch.no_grad():
            xc = torch.LongTensor(X[:, :n_cat].astype(int)).to(self._device)
            xn = torch.FloatTensor(X[:, n_cat:]).to(self._device)
            logits = self._net(xc, xn).squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()

    def extract_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract concatenated embedding vectors for downstream use."""
        import torch

        if self._net is None:
            raise RuntimeError("Call fit() first.")
        self._net.eval()
        n_cat = len(self.cat_dims)
        with torch.no_grad():
            xc = torch.LongTensor(X[:, :n_cat].astype(int)).to(self._device)
            embedded = [
                emb(xc[:, i])
                for i, emb in enumerate(self._net.embeddings)
            ]
            return torch.cat(embedded, dim=1).cpu().numpy()


class _EmbeddingNet:
    """PyTorch entity embedding network (internal)."""

    def __new__(cls, cat_dims, emb_dims, num_cont):  # type: ignore[no-untyped-def]
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, cat_dims, emb_dims, num_cont):  # type: ignore[no-untyped-def]
                super().__init__()
                self.embeddings = nn.ModuleList([
                    nn.Embedding(n_cat, emb_dim)
                    for n_cat, emb_dim in zip(cat_dims, emb_dims)
                ])
                self.emb_drop = nn.Dropout(0.2)
                total_emb = sum(emb_dims)
                self.fc = nn.Sequential(
                    nn.Linear(total_emb + num_cont, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )

            def forward(self, x_cat, x_cont):  # type: ignore[no-untyped-def]
                embedded = [
                    emb(x_cat[:, i])
                    for i, emb in enumerate(self.embeddings)
                ]
                import torch
                x = torch.cat(embedded + [x_cont], dim=1)
                x = self.emb_drop(x)
                return self.fc(x)

        return Net(cat_dims, emb_dims, num_cont)
