"""Stage 2: temporal detection over time-ordered flow windows (GRU or LSTM + attention).

For every flow, the model sees that flow plus the previous ``window - 1``
flows **sent by the same source host**, in time order, and predicts the
tactic of the last flow (``Benign`` included). Besides the per-flow
features, each step carries the log time gap since the host's previous
flow, which is what makes periodic beaconing visible.

The attention layer returns one weight per time step. Phase 4 uses these
weights to show which earlier flows drove a decision.

Splitting: ``block_split`` cuts every day into fixed time blocks (default
5 min) and assigns whole blocks at random to train / val / test. Windows
are built per (host, block), so no window ever spans two splits. A single
time cut per day was tried first and rejected: attacks happen in bursts, so
a cut put whole attack classes on one side only (e.g. ToN password traffic
only in test, DDoS only in train).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..data.preprocess import FlowPreprocessor
from ..data.taxonomy import BENIGN
from .common import balanced_weights, binary_metrics, multiclass_metrics


def _torch():
    import torch
    return torch


# ----------------------------------------------------------------- data
def temporal_split(df: pd.DataFrame, val_size=0.15, test_size=0.15, by="slice",
                   time_col="FLOW_START_MILLISECONDS"):
    """Cut every slice by time: earliest rows train, then val, then test."""
    parts = {"train": [], "val": [], "test": []}
    for _, g in df.groupby(by, sort=False):
        cut1, cut2 = g[time_col].quantile([1 - val_size - test_size, 1 - test_size])
        parts["train"].append(g[g[time_col] < cut1])
        parts["val"].append(g[(g[time_col] >= cut1) & (g[time_col] < cut2)])
        parts["test"].append(g[g[time_col] >= cut2])
    return {k: pd.concat(v) for k, v in parts.items()}


def block_split(df: pd.DataFrame, val_size=0.15, test_size=0.15, block_minutes: int = 5, seed: int = 42,
                by="slice", time_col="FLOW_START_MILLISECONDS", host_col="IPV4_SRC_ADDR"):
    """Assign whole time blocks to train/val/test, separately inside every slice (day).

    Per slice, blocks are shuffled and filled into test, then val, then train
    until each holds its share of that slice's **rows**, so every day appears
    in every split in the right proportion. Adds ``seq_key`` =
    slice|block|host, which keeps each window inside one host and one block.
    """
    rng = np.random.default_rng(seed)
    block = (df[time_col] // (block_minutes * 60_000)).astype("int64")
    key = df[by].astype(str) + "|" + block.astype(str)
    sizes = key.value_counts()
    slice_of = pd.Series(sizes.index.str.split("|").str[0], index=sizes.index)
    assign = {}
    for sl in slice_of.unique():
        ks = sizes[slice_of == sl]
        ks = ks.iloc[rng.permutation(len(ks))]
        frac = ks.cumsum() / ks.sum()
        part = np.where(frac <= test_size, "test", np.where(frac <= test_size + val_size, "val", "train"))
        if len(ks) >= 3:                       # every split gets at least one block per slice
            part[0], part[-1] = "test", "train"
            if "val" not in part:
                part[1] = "val"
        assign.update(dict(zip(ks.index, part)))
    d = df.assign(seq_key=key + "|" + df[host_col].astype(str), _part=key.map(assign).to_numpy())
    return {k: d[d._part == k].drop(columns="_part") for k in ("train", "val", "test")}


class WindowIndex:
    """Index of per-host windows over a frame sorted by (host, time).

    Stores only start offsets; windows are assembled on the fly, so memory
    stays O(rows), not O(rows x window).
    """

    def __init__(self, df: pd.DataFrame, window: int, host_col="IPV4_SRC_ADDR",
                 time_col="FLOW_START_MILLISECONDS"):
        d = df.sort_values([host_col, time_col], kind="stable")
        self.order = d.index.to_numpy()
        hosts = d[host_col].to_numpy()
        new_host = np.ones(len(hosts), dtype=bool)
        new_host[1:] = hosts[1:] != hosts[:-1]
        host_start = np.maximum.accumulate(np.where(new_host, np.arange(len(d)), 0))
        self.start = np.maximum(np.arange(len(d)) - window + 1, host_start)
        t = d[time_col].to_numpy(dtype="float64")
        gap = np.zeros(len(t))
        gap[1:] = np.diff(t) / 1000.0
        gap[new_host] = 0.0
        self.log_gap = np.log1p(np.clip(gap, 0, None)).astype("float32")
        self.window = window

    def __len__(self):
        return len(self.order)


class SequenceBatches:
    """Vectorised window batches: (x[B, W, d+1], mask[B, W], y[B]) for flows in positions order.

    Windows are right-aligned (the last step is the flow being labelled) and
    zero-padded on the left where a host has fewer than ``window`` earlier flows.
    """

    def __init__(self, X: np.ndarray, idx: WindowIndex, y: np.ndarray | None, batch_size: int,
                 shuffle: bool, seed: int = 0):
        self.X = np.concatenate([X, idx.log_gap[:, None]], axis=1).astype("float32")
        self.idx, self.y, self.bs, self.shuffle = idx, y, batch_size, shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (len(self.idx) + self.bs - 1) // self.bs

    def window(self, pos: np.ndarray):
        W = self.idx.window
        rows = pos[:, None] - (W - 1) + np.arange(W)[None, :]
        mask = rows >= self.idx.start[pos][:, None]
        x = self.X[np.clip(rows, 0, None)] * mask[..., None]
        return x.astype("float32"), mask

    def __iter__(self):
        torch = _torch()
        n = len(self.idx)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for b in range(0, n, self.bs):
            pos = order[b:b + self.bs]
            x, mask = self.window(pos)
            y = torch.from_numpy(self.y[pos].astype("int64")) if self.y is not None else None
            yield torch.from_numpy(x), torch.from_numpy(mask), y


# ---------------------------------------------------------------- model
def build_net(n_in: int, n_out: int, hidden: int = 64, cell: str = "gru"):
    torch = _torch()
    nn = torch.nn

    class AttnRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU())
            rnn = nn.GRU if cell == "gru" else nn.LSTM
            self.rnn = rnn(hidden, hidden, batch_first=True)
            self.att = nn.Linear(hidden, 1)
            self.out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

        def forward(self, x, mask):
            h, _ = self.rnn(self.inp(x))                          # [B, W, H]
            scores = self.att(torch.tanh(h)).squeeze(-1)          # [B, W]
            scores = scores.masked_fill(~mask, float("-inf"))
            alpha = torch.softmax(scores, dim=1)                  # attention weights
            ctx = (alpha.unsqueeze(-1) * h).sum(1)
            last = h[:, -1]                                       # state at the labelled flow
            return self.out(torch.cat([ctx, last], 1)), alpha

    if cell not in ("gru", "lstm"):
        raise ValueError(f"cell must be gru or lstm, got {cell!r}")
    return AttnRNN()


class SequenceModel:
    def __init__(self, feature_names=None, window=32, hidden=64, cell="gru", epochs=8,
                 batch_size=1024, lr=1e-3, seed=42, device=None):
        self.features, self.window, self.hidden, self.cell = feature_names, window, hidden, cell
        self.epochs, self.batch_size, self.lr, self.seed = epochs, batch_size, lr, seed
        self.device = device
        self.history_: list[dict] = []

    def _dev(self):
        torch = _torch()
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _loader(self, df, shuffle, with_y=True):
        # seq_key (from block_split) keeps windows inside one host *and* one time block
        idx = WindowIndex(df, self.window, host_col="seq_key" if "seq_key" in df else "IPV4_SRC_ADDR")
        ordered = df.loc[idx.order]
        X = self.pp_.transform(ordered).to_numpy(dtype="float32")
        y = self.classes_.get_indexer(ordered.tactic) if with_y else None
        if with_y and (y < 0).any():
            raise ValueError(f"unknown tactic labels {sorted(set(ordered.tactic[y < 0]))}")
        return SequenceBatches(X, idx, y, self.batch_size, shuffle, self.seed), ordered

    def fit(self, train: pd.DataFrame, val: pd.DataFrame):
        torch = _torch()
        torch.manual_seed(self.seed)
        t0 = time.time()
        self.pp_ = FlowPreprocessor(features=self.features).fit(train)
        self.classes_ = pd.Index(sorted(train.tactic.unique()))
        dev = self._dev()
        self.net_ = build_net(len(self.pp_.feature_names_) + 1, len(self.classes_), self.hidden, self.cell).to(dev)
        w = balanced_weights(train.tactic)
        cw = pd.Series(w, index=train.tactic.values).groupby(level=0).first().reindex(self.classes_).to_numpy()
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        tr_loader, _ = self._loader(train, shuffle=True)
        best, best_state = -1.0, None
        for ep in range(self.epochs):
            self.net_.train()
            total, n = 0.0, 0
            for x, m, y in tr_loader:
                x, m, y = x.to(dev), m.to(dev), y.to(dev)
                opt.zero_grad()
                logits, _ = self.net_(x, m)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                total += loss.item() * len(y)
                n += len(y)
            val_pred = self.predict(val)
            score = multiclass_metrics(val.loc[val_pred.index, "tactic"], val_pred.tactic_pred)["macro_f1"]
            self.history_.append({"epoch": ep + 1, "train_loss": total / max(n, 1), "val_macro_f1": score})
            if score > best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in self.net_.state_dict().items()}
        self.net_.load_state_dict(best_state)
        self.fit_seconds_ = round(time.time() - t0, 1)
        return self

    def predict(self, df: pd.DataFrame, return_attention: bool = False):
        torch = _torch()
        dev = self._dev()
        self.net_.eval()
        loader, ordered = self._loader(df, shuffle=False, with_y=False)
        probs, atts = [], []
        with torch.no_grad():
            for x, m, _ in loader:
                logits, alpha = self.net_(x.to(dev), m.to(dev))
                probs.append(torch.softmax(logits, 1).cpu().numpy())
                if return_attention:
                    atts.append(alpha.cpu().numpy())
        P = np.concatenate(probs) if probs else np.zeros((0, len(self.classes_)))
        pred = self.classes_[P.argmax(1)] if len(P) else []
        benign_i = self.classes_.get_loc(BENIGN) if BENIGN in self.classes_ else None
        p_attack = 1 - P[:, benign_i] if benign_i is not None else np.ones(len(P))
        out = pd.DataFrame({"tactic_pred": pred, "tactic_conf": P.max(1) if len(P) else [],
                            "p_attack": p_attack}, index=ordered.index)
        out["is_attack"] = (out.tactic_pred != BENIGN).astype(int)
        if return_attention:
            return out, np.concatenate(atts) if atts else np.zeros((0, self.window))
        return out

    def evaluate(self, df: pd.DataFrame) -> dict:
        p = self.predict(df)
        d = df.loc[p.index]
        return {"binary": binary_metrics(d.Label, p.is_attack, p.p_attack),
                "tactic": multiclass_metrics(d.tactic, p.tactic_pred)}

    # torch modules pickle poorly across devices; store weights as CPU tensors
    def __getstate__(self):
        s = self.__dict__.copy()
        if "net_" in s:
            s["net_state_"] = {k: v.detach().cpu() for k, v in s.pop("net_").state_dict().items()}
        return s

    def __setstate__(self, s):
        state = s.pop("net_state_", None)
        self.__dict__.update(s)
        if state is not None:
            self.net_ = build_net(len(self.pp_.feature_names_) + 1, len(self.classes_), self.hidden, self.cell)
            self.net_.load_state_dict(state)
            self.net_.to(self._dev())
