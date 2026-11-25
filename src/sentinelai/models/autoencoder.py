"""Stage 3: anomaly / zero-day detection with an autoencoder trained on benign traffic only.

The autoencoder learns to reconstruct benign flows. Its anomaly score is
the reconstruction error; flows it reconstructs badly are flagged.

* ``fit`` refuses training data that contains attack rows: the whole point
  of the stage is that it never sees an attack.
* The alert threshold is the ``threshold_percentile`` of reconstruction
  errors on **benign validation** rows, so the target false-positive rate
  (100 - percentile) is set without looking at attacks either.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..data.preprocess import FlowPreprocessor
from .common import binary_metrics


def _torch():
    import torch
    return torch


class BenignOnlyError(ValueError):
    pass


def _net(n_in: int, hidden: list[int]):
    nn = _torch().nn
    enc, dec, d = [], [], n_in
    for h in hidden:
        enc += [nn.Linear(d, h), nn.ReLU()]
        d = h
    for h in list(reversed(hidden[:-1])) + [n_in]:
        dec += [nn.Linear(d, h)] + ([nn.ReLU()] if h != n_in else [])
        d = h
    return nn.Sequential(*enc, *dec)


class Autoencoder:
    def __init__(self, feature_names=None, hidden=(64, 16), epochs=20, batch_size=4096, lr=1e-3,
                 threshold_percentile=99.0, seed=42, device=None, patience=3):
        self.features, self.hidden = feature_names, list(hidden)
        self.epochs, self.batch_size, self.lr, self.seed = epochs, batch_size, lr, seed
        self.threshold_percentile, self.device, self.patience = threshold_percentile, device, patience
        self.history_: list[dict] = []

    def _dev(self):
        return self.device or ("cuda" if _torch().cuda.is_available() else "cpu")

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, label_col="Label"):
        for name, part in (("train", train), ("val", val)):
            if (part[label_col] != 0).any():
                raise BenignOnlyError(f"{name} contains {(part[label_col] != 0).sum()} attack rows; "
                                      "the autoencoder must be fitted on benign traffic only")
        torch = _torch()
        torch.manual_seed(self.seed)
        t0 = time.time()
        dev = self._dev()
        self.pp_ = FlowPreprocessor(features=self.features).fit(train)
        Xtr = torch.tensor(self.pp_.transform(train).to_numpy("float32"))
        Xva = torch.tensor(self.pp_.transform(val).to_numpy("float32"))
        self.net_ = _net(Xtr.shape[1], self.hidden).to(dev)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = float("inf"), None, 0
        for ep in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(Xtr), generator=g)
            total = 0.0
            for b in range(0, len(Xtr), self.batch_size):
                x = Xtr[perm[b:b + self.batch_size]].to(dev)
                opt.zero_grad()
                loss = ((self.net_(x) - x) ** 2).mean()
                loss.backward()
                opt.step()
                total += float(loss) * len(x)
            val_loss = float(self._errors(Xva).mean())
            self.history_.append({"epoch": ep + 1, "train_mse": total / len(Xtr), "val_mse": val_loss})
            if val_loss < best - 1e-6:
                best, bad = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.threshold_ = float(np.percentile(self._errors(Xva), self.threshold_percentile))
        self.fit_seconds_ = round(time.time() - t0, 1)
        return self

    def _errors(self, X) -> np.ndarray:
        torch = _torch()
        dev = self._dev()
        self.net_.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(X), 65536):
                x = X[b:b + 65536].to(dev)
                out.append(((self.net_(x) - x) ** 2).mean(1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Reconstruction error per flow (higher = more anomalous)."""
        X = _torch().tensor(self.pp_.transform(df).to_numpy("float32"))
        return self._errors(X)

    def feature_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature squared error, for explaining an anomaly (Phase 4)."""
        torch = _torch()
        Xdf = self.pp_.transform(df)
        X = torch.tensor(Xdf.to_numpy("float32")).to(self._dev())
        self.net_.eval()
        with torch.no_grad():
            err = ((self.net_(X) - X) ** 2).cpu().numpy()
        return pd.DataFrame(err, columns=Xdf.columns, index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        s = self.score(df)
        return pd.DataFrame({"anomaly_score": s, "is_anomaly": (s > self.threshold_).astype(int)}, index=df.index)

    def evaluate(self, df: pd.DataFrame, label_col="Label") -> dict:
        p = self.predict(df)
        return binary_metrics(df[label_col], p.is_anomaly, p.anomaly_score)

    def __getstate__(self):
        s = self.__dict__.copy()
        if "net_" in s:
            s["net_state_"] = {k: v.detach().cpu() for k, v in s.pop("net_").state_dict().items()}
        return s

    def __setstate__(self, s):
        state = s.pop("net_state_", None)
        self.__dict__.update(s)
        if state is not None:
            self.net_ = _net(len(self.pp_.feature_names_), self.hidden)
            self.net_.load_state_dict(state)
            self.net_.to(self._dev())
