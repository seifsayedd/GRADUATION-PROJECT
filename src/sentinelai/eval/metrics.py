"""Evaluation metrics: confusion matrices, per-class / per-source rates,
block-bootstrap confidence intervals, validation-only thresholds, calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


# ------------------------------------------------------------------ thresholds
def threshold_at_fpr(benign_val_scores, target_fpr: float = 0.01) -> float:
    """Alert threshold that flags ``target_fpr`` of **benign validation** scores.

    Only benign validation scores are accepted, so the threshold cannot be
    tuned on test data or on attacks. A flow is flagged when ``score > threshold``.
    """
    s = np.asarray(benign_val_scores, dtype="float64")
    if not len(s):
        raise ValueError("no benign validation scores to set a threshold on")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0, 1)")
    return float(np.quantile(s, 1 - target_fpr, method="higher"))


# ------------------------------------------------------------------- counting
def counts(y_true, flag) -> dict:
    y, f = np.asarray(y_true).astype(int), np.asarray(flag).astype(int)
    tp = int(((y == 1) & (f == 1)).sum())
    fp = int(((y == 0) & (f == 1)).sum())
    fn = int(((y == 1) & (f == 0)).sum())
    tn = int(((y == 0) & (f == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def rates(c: dict) -> dict:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    return {"n": tp + fp + fn + tn, "positives": tp + fn, "precision": prec, "recall": rec,
            "f1": f1, "fpr": fpr, **c}


def binary(y_true, flag) -> dict:
    return rates(counts(y_true, flag))


def confusion(y_true, y_pred, labels: list[str] | None = None) -> dict:
    """Confusion matrix with counts and row-normalised rates (rows = true class)."""
    yt, yp = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    labels = labels or sorted(set(yt) | set(yp))
    m = confusion_matrix(yt, yp, labels=labels)
    rows = m.sum(1, keepdims=True)
    norm = np.divide(m, rows, out=np.zeros_like(m, dtype="float64"), where=rows > 0)
    return {"labels": labels, "counts": m.tolist(), "row_normalised": norm.round(4).tolist()}


def per_class_detection(df: pd.DataFrame, flag, class_col: str = "Attack") -> dict:
    """Share of flows flagged, per class (for attacks = detection rate, for Benign = FPR)."""
    f = pd.Series(np.asarray(flag).astype(int), index=df.index)
    out = {}
    for cls, g in df.groupby(class_col):
        out[str(cls)] = {"n": int(len(g)), "flagged": float(f.loc[g.index].mean()),
                         "source": ", ".join(sorted(g.source.unique())) if "source" in g else None}
    return out


def per_source(df: pd.DataFrame, flag, label_col: str = "Label") -> dict:
    f = pd.Series(np.asarray(flag).astype(int), index=df.index)
    out = {}
    for src, g in df.groupby("source"):
        b = g[label_col] == 0
        out[str(src)] = {"benign_n": int(b.sum()),
                         "benign_fpr": float(f.loc[g.index[b]].mean()) if b.any() else None,
                         "attack_n": int((~b).sum()),
                         "attack_detected": float(f.loc[g.index[~b]].mean()) if (~b).any() else None}
    return out


# ------------------------------------------------------------------ bootstrap
def block_ids(df: pd.DataFrame, block_minutes: int = 5, by: str = "source",
              time_col: str = "FLOW_START_MILLISECONDS") -> np.ndarray:
    """Group id per (``by``, time block). Neighbouring flows are near-copies,
    so whole blocks, not rows, are the units that get resampled."""
    blk = (df[time_col].to_numpy("int64") // (block_minutes * 60_000)).astype(str)
    key = df[by].astype(str).to_numpy() + "|" + blk
    return pd.factorize(key)[0]


_STATS = {
    "recall": lambda tp, fp, fn, tn: tp / np.maximum(tp + fn, 1),
    "fpr": lambda tp, fp, fn, tn: fp / np.maximum(fp + tn, 1),
    "precision": lambda tp, fp, fn, tn: tp / np.maximum(tp + fp, 1),
    "f1": lambda tp, fp, fn, tn: 2 * tp / np.maximum(2 * tp + fp + fn, 1),
}


def block_bootstrap_ci(y_true, flag, groups, stats=("recall", "fpr", "precision", "f1"),
                       n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Percentile bootstrap CI that resamples whole groups (blocks) with replacement.

    Counts are aggregated per group first, so each resample is one matrix
    product. Pass ``groups=np.arange(n)`` for an ordinary row bootstrap.
    """
    y, f = np.asarray(y_true).astype(int), np.asarray(flag).astype(int)
    g = pd.factorize(np.asarray(groups))[0]
    G = g.max() + 1 if len(g) else 0
    C = np.zeros((G, 4))
    for j, mask in enumerate([(y == 1) & (f == 1), (y == 0) & (f == 1), (y == 1) & (f == 0), (y == 0) & (f == 0)]):
        C[:, j] = np.bincount(g[mask], minlength=G)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(G, np.full(G, 1 / G), size=n_boot)       # B x G resample counts
    B = W @ C                                                     # B x 4
    point = C.sum(0)
    out = {"n_groups": int(G), "n_boot": n_boot}
    for s in stats:
        fn_ = _STATS[s]
        vals = fn_(*B.T)
        out[s] = {"point": float(fn_(*point)),
                  "lo": float(np.quantile(vals, alpha / 2)), "hi": float(np.quantile(vals, 1 - alpha / 2))}
    return out


# ------------------------------------------------------------ curves/calibration
def curves(y_true, score, max_points: int = 300) -> dict:
    """Down-sampled ROC and precision-recall curve points (for figures)."""
    y, s = np.asarray(y_true).astype(int), np.asarray(score, dtype="float64")
    if y.min() == y.max():
        return {}
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)

    def thin(a, b):
        i = np.unique(np.linspace(0, len(a) - 1, min(max_points, len(a))).astype(int))
        return a[i].round(5).tolist(), b[i].round(5).tolist()

    rf, rt = thin(fpr, tpr)
    pr, pp = thin(rec, prec)
    return {"roc": {"fpr": rf, "tpr": rt}, "pr": {"recall": pr, "precision": pp}}


def calibration(y_true, p, bins: int = 10) -> dict:
    """Reliability table, Brier score and expected calibration error for probabilities."""
    y, p = np.asarray(y_true).astype(int), np.clip(np.asarray(p, dtype="float64"), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = float(p[m].mean()), float(y[m].mean())
            ece += m.mean() * abs(conf - acc)
            table.append({"bin": [round(edges[b], 2), round(edges[b + 1], 2)], "n": int(m.sum()),
                          "mean_predicted": conf, "observed_attack_rate": acc})
    return {"brier": float(((p - y) ** 2).mean()), "ece": float(ece), "bins": table}
