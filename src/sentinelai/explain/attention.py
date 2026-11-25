"""Stage 2: attention weights per window, and a test of whether they point at what mattered.

Attention weights are not guaranteed to be explanations. ``occlusion`` removes
one earlier flow at a time from a window (zeroes it and masks it out, exactly
as left padding is handled) and records how much the attack probability moves.
``attention_vs_occlusion`` then measures how often the most-attended earlier
flow is also the one whose removal moves the prediction most.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _torch():
    import torch
    return torch


def windows_for(model, df: pd.DataFrame, positions: np.ndarray | None = None):
    """Window tensors for flows of ``df`` (in the model's host/time order).

    Returns (x [n, W, d], mask [n, W], ordered frame, positions used).
    """
    loader, ordered = model._loader(df, shuffle=False, with_y=False)
    pos = np.arange(len(ordered)) if positions is None else np.asarray(positions)
    x, mask = loader.window(pos)
    return x, mask, ordered, pos, loader


def _forward(model, x: np.ndarray, mask: np.ndarray, batch: int = 4096):
    torch = _torch()
    dev = model._dev()
    model.net_.eval()
    P, A = [], []
    with torch.no_grad():
        for b in range(0, len(x), batch):
            logits, alpha = model.net_(torch.from_numpy(x[b:b + batch]).to(dev), torch.from_numpy(mask[b:b + batch]).to(dev))
            P.append(torch.softmax(logits, 1).cpu().numpy())
            A.append(alpha.cpu().numpy())
    return np.concatenate(P), np.concatenate(A)


def p_attack(model, P: np.ndarray) -> np.ndarray:
    return 1 - P[:, model.classes_.get_loc("Benign")]


def occlusion(model, x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each window: attention weights, and |change in p_attack| when each earlier step is removed.

    The last step (the flow being labelled) is never removed. Returns
    (p_attack [n], attention [n, W], delta [n, W] with NaN where nothing was removed).
    """
    P, A = _forward(model, x, mask)
    base = p_attack(model, P)
    n, W = mask.shape
    delta = np.full((n, W), np.nan)
    for j in range(W - 1):
        rows = np.where(mask[:, j])[0]
        if not len(rows):
            continue
        xj, mj = x[rows].copy(), mask[rows].copy()
        xj[:, j] = 0
        mj[:, j] = False
        Pj, _ = _forward(model, xj, mj)
        delta[rows, j] = np.abs(p_attack(model, Pj) - base[rows])
    return base, A, delta


def attention_vs_occlusion(att: np.ndarray, delta: np.ndarray, mask: np.ndarray, seed: int = 0) -> dict:
    """How well attention ranks the earlier flows that matter, against a random pick.

    Only windows with at least 2 earlier flows count.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    W = mask.shape[1]
    hits, rand_hits, rhos, k_list, n = 0, 0, [], [], 0
    for i in range(len(att)):
        prev = np.where(mask[i, :W - 1])[0]
        if len(prev) < 2 or np.all(np.isnan(delta[i, prev])):
            continue
        n += 1
        a, d = att[i, prev], delta[i, prev]
        top_true = prev[np.nanargmax(d)]
        hits += prev[np.argmax(a)] == top_true
        rand_hits += rng.choice(prev) == top_true
        k_list.append(len(prev))
        if np.nanstd(d) > 0 and np.std(a) > 0:
            rhos.append(spearmanr(a, d).statistic)
    return {"windows": n,
            "top_attended_is_most_influential": hits / n if n else None,
            "random_pick_is_most_influential": rand_hits / n if n else None,
            "expected_random_rate": float(np.mean([1 / k for k in k_list])) if k_list else None,
            "median_spearman_attention_vs_occlusion": float(np.median(rhos)) if rhos else None,
            "windows_with_rank_correlation": len(rhos)}
