"""Checks of how far the explanations can be trusted.

* **deletion**: replace an alert's top-k features (by SHAP, towards attack) with
  typical benign values and measure the drop in attack probability; compare
  with replacing k random features. Faithful explanations give a larger drop.
* **stability**: near-identical flows should get similar explanations
  (Spearman correlation of their per-feature SHAP vectors), compared with
  random pairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import group_columns


def benign_reference(X_benign: pd.DataFrame) -> pd.Series:
    """Typical benign value of every model column (median; for 0/1 columns this is the majority value)."""
    return X_benign.median()


def deletion_test(predict_p, X: pd.DataFrame, agg: pd.DataFrame, ref: pd.Series, ks=(1, 3, 5),
                  n_random: int = 5, seed: int = 42) -> dict:
    """``predict_p(X) -> p_attack``; ``agg`` = per-original-feature SHAP for the same rows (same order)."""
    rng = np.random.default_rng(seed)
    groups = group_columns(X.columns)
    feats = list(agg.columns)
    base = predict_p(X)
    out = {"rows": len(X), "base_mean_p_attack": float(base.mean())}
    X0, R = X.to_numpy("float64"), ref.to_numpy("float64")
    G = np.zeros((len(feats), X0.shape[1]), bool)                 # feature -> its model columns
    for j, f in enumerate(feats):
        G[j, groups[f]] = True
    vals = agg.to_numpy()
    as_df = lambda a: pd.DataFrame(a, columns=X.columns, index=X.index)
    for k in ks:
        top = np.argsort(-vals, axis=1)[:, :k]                    # strongest push towards attack
        M = G[top].any(1)                                         # n x columns to replace
        drop_top = base - predict_p(as_df(np.where(M, R, X0)))
        drops = []
        for _ in range(n_random):
            rand = np.argsort(rng.random((len(X0), len(feats))), axis=1)[:, :k]
            Mr = G[rand].any(1)
            drops.append(base - predict_p(as_df(np.where(Mr, R, X0))))
        drop_rand = np.mean(drops, axis=0)
        out[f"k={k}"] = {"mean_drop_top": float(drop_top.mean()), "mean_drop_random": float(drop_rand.mean()),
                         "share_top_beats_random": float((drop_top > drop_rand).mean()),
                         "share_flipped_to_benign_top": float(((base >= 0.5) & (base - drop_top < 0.5)).mean()),
                         "share_flipped_to_benign_random": float(((base >= 0.5) & (base - drop_rand < 0.5)).mean())}
    return out


def stability(X: pd.DataFrame, agg: pd.DataFrame, seed: int = 42) -> dict:
    """Spearman correlation of SHAP vectors: each row with its nearest neighbour vs a random row."""
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors
    if len(X) < 3:
        return {"rows": len(X)}
    nn = NearestNeighbors(n_neighbors=2).fit(X.to_numpy("float32"))
    dist, idx = nn.kneighbors(X.to_numpy("float32"))
    rng = np.random.default_rng(seed)
    A = agg.to_numpy()
    rand = rng.integers(0, len(X), len(X))

    def rho(i, j):
        r = spearmanr(A[i], A[j]).statistic
        return np.nan if np.isnan(r) else r

    near = np.array([rho(i, idx[i, 1]) for i in range(len(X))])
    rnd = np.array([rho(i, rand[i]) for i in range(len(X)) if rand[i] != i])
    return {"rows": len(X), "median_neighbour_distance": float(np.median(dist[:, 1])),
            "median_rho_nearest_neighbour": float(np.nanmedian(near)), "median_rho_random_pair": float(np.nanmedian(rnd))}


def global_importance(agg: pd.DataFrame, by: pd.Series | None = None, top: int = 15) -> dict:
    """Mean |SHAP| per original feature, overall and per group (e.g. true tactic or source)."""
    imp = agg.abs()
    out = {"overall": imp.mean().sort_values(ascending=False).head(top).round(6).to_dict()}
    if by is not None:
        out["by_group"] = {str(g): imp[by.to_numpy() == g].mean().sort_values(ascending=False).head(top).round(6).to_dict()
                           for g in pd.unique(by)}
    return out
