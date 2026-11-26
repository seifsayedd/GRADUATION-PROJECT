"""Shared helpers for the Phase 2 models: feature sets, quick metrics, the model registry."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from ..data.mapping import available_features

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"            # git-ignored
SIZE_LIMIT_MB = 50                     # project rule: never commit binaries above this


def feature_set(name: str) -> list[str] | None:
    """``full`` = every usable BigFlow feature (None = no restriction); ``lab`` = lab-fillable only."""
    if name == "full":
        return None
    if name == "lab":
        return available_features()
    raise ValueError(f"unknown feature set {name!r}")


def balanced_weights(y) -> np.ndarray:
    """Sample weights that give every class the same total weight."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([w[v] for v in y], dtype="float64")


def binary_metrics(y_true, y_pred, score=None) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {"n": int(len(y_true)), "positives": int(y_true.sum()),
         "precision": float(precision_score(y_true, y_pred, zero_division=0)),
         "recall": float(recall_score(y_true, y_pred, zero_division=0)),
         "f1": float(f1_score(y_true, y_pred, zero_division=0)),
         "fpr": float(fp / (fp + tn)) if fp + tn else None,
         "fnr": float(fn / (fn + tp)) if fn + tp else None,
         "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    if score is not None and 0 < y_true.sum() < len(y_true):
        m["roc_auc"] = float(roc_auc_score(y_true, score))
        m["pr_auc"] = float(average_precision_score(y_true, score))
    return m


def multiclass_metrics(y_true, y_pred) -> dict:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    yt, yp = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    return {"n": int(len(yt)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
            "per_class": classification_report(yt, yp, labels=labels, output_dict=True, zero_division=0),
            "labels": labels,
            "confusion": confusion_matrix(yt, yp, labels=labels).tolist()}


def save_model(obj, stage: str, name: str, meta: dict, root: Path | None = None) -> dict:
    """Save ``obj`` to models/<stage>/<name>/ with a meta.json; report size against the 50 MB rule."""
    d = Path(root or MODELS_DIR) / stage / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "model.joblib"
    joblib.dump(obj, path, compress=3)
    size_mb = path.stat().st_size / 1e6
    meta = {**meta, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "file": str(path.relative_to(d.parents[1])),
            "size_mb": round(size_mb, 2), "commit_allowed": size_mb <= SIZE_LIMIT_MB}
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=_json), encoding="utf-8")
    return meta


def load_model(stage: str, name: str, root: Path | None = None):
    d = Path(root or MODELS_DIR) / stage / name
    return joblib.load(d / "model.joblib"), json.loads((d / "meta.json").read_text(encoding="utf-8"))


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json), encoding="utf-8")
