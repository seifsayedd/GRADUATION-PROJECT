"""Stage 1: per-flow detection with Random Forest or XGBoost.

Two layers, as in docs/ai-team-data-requirements.md:

1. **binary**: benign vs attack, trained on every training row
2. **tactic**: ATT&CK tactic, trained on attack rows only and applied to
   flows the binary layer flags

Class imbalance is handled with balanced sample weights, not duplicated
rows. Preprocessing (``FlowPreprocessor``) is fitted on the training split
only and travels inside the model object, so lab data and test data go
through exactly the same transform.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from ..data.preprocess import FlowPreprocessor
from ..data.taxonomy import BENIGN
from .common import balanced_weights, binary_metrics, multiclass_metrics


def _xgb():
    import xgboost  # imported lazily so the data layer works without it
    return xgboost


def gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class FlowClassifier:
    """Hierarchical (binary -> tactic) flow classifier."""

    def __init__(self, algo: str = "xgboost", feature_names: list[str] | None = None,
                 params: dict | None = None, seed: int = 42, threshold: float = 0.5):
        if algo not in ("random_forest", "xgboost"):
            raise ValueError(f"unknown algo {algo!r}")
        self.algo, self.features, self.params = algo, feature_names, dict(params or {})
        self.seed, self.threshold = seed, threshold
        self.history_: list[dict] = []

    # ------------------------------------------------------------------ fitting
    def fit(self, train: pd.DataFrame, val: pd.DataFrame, label_col="Label", tactic_col="tactic"):
        t0 = time.time()
        self.pp_ = FlowPreprocessor(features=self.features).fit(train)
        Xtr, Xva = self.pp_.transform(train), self.pp_.transform(val)
        ytr, yva = train[label_col].to_numpy(int), val[label_col].to_numpy(int)
        self.binary_ = self._fit_one(Xtr, ytr, Xva, yva, n_classes=2, task="binary")

        atr, ava = ytr == 1, yva == 1
        self.tactic_enc_ = LabelEncoder().fit(train.loc[atr, tactic_col])
        if set(val.loc[ava, tactic_col]) - set(self.tactic_enc_.classes_):
            ava = ava & val[tactic_col].isin(self.tactic_enc_.classes_).to_numpy()
        ttr = self.tactic_enc_.transform(train.loc[atr, tactic_col])
        tva = self.tactic_enc_.transform(val.loc[ava, tactic_col])
        self.tactic_ = self._fit_one(Xtr[atr], ttr, Xva[ava], tva,
                                     n_classes=len(self.tactic_enc_.classes_), task="tactic")
        self.fit_seconds_ = round(time.time() - t0, 1)
        self.feature_names_ = self.pp_.feature_names_
        return self

    def _new_model(self, n_classes: int, extra: dict):
        p = {**self.params, **extra}
        if self.algo == "random_forest":
            return RandomForestClassifier(
                n_estimators=p.get("n_estimators", 150), max_depth=p.get("max_depth", 24),
                min_samples_leaf=p.get("min_samples_leaf", 3), max_samples=p.get("max_samples", 0.5),
                n_jobs=-1, random_state=self.seed)
        xgb = _xgb()
        kw = dict(n_estimators=p.get("n_estimators", 600), max_depth=p.get("max_depth", 8),
                  learning_rate=p.get("learning_rate", 0.1), subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", device="cuda" if gpu_available() else "cpu",
                  early_stopping_rounds=p.get("early_stopping_rounds", 30), random_state=self.seed,
                  eval_metric="logloss" if n_classes == 2 else "mlogloss")
        if n_classes > 2:
            kw["objective"] = "multi:softprob"
        return xgb.XGBClassifier(**kw)

    def _fit_one(self, Xtr, ytr, Xva, yva, n_classes: int, task: str):
        w = balanced_weights(ytr)
        if self.algo == "random_forest":
            m = self._new_model(n_classes, {})
            m.fit(Xtr, ytr, sample_weight=w)
            score = self._score(m, Xva, yva, n_classes)
            self.history_.append({"task": task, "params": {}, "val_score": score})
            return m
        # XGBoost: small grid, early stopping and model choice on the validation split
        grid = self.params.get("grid") or {}
        keys = sorted(grid)
        combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))] or [{}]
        best, best_score = None, -1.0
        for combo in combos:
            m = self._new_model(n_classes, combo)
            m.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xva, yva)], sample_weight_eval_set=[balanced_weights(yva)],
                  verbose=False)
            score = self._score(m, Xva, yva, n_classes)
            self.history_.append({"task": task, "params": combo, "val_score": score,
                                  "best_iteration": int(getattr(m, "best_iteration", -1) or -1)})
            if score > best_score:
                best, best_score = m, score
        best.set_params(device="cpu")   # predict on CPU: portable, and no host/GPU copy per call
        return best

    @staticmethod
    def _score(m, X, y, n_classes):
        """Model-selection score on validation: F1 (binary) or macro-F1 (tactic)."""
        pred = m.predict(X)
        return float(f1_score(y, pred, average="binary" if n_classes == 2 else "macro", zero_division=0))

    # --------------------------------------------------------------- prediction
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.pp_.transform(df)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.transform(df)
        p_attack = self.binary_.predict_proba(X)[:, 1]
        is_attack = p_attack >= self.threshold
        tactic = np.full(len(df), BENIGN, dtype=object)
        tactic_conf = np.zeros(len(df))
        if is_attack.any():
            proba = self.tactic_.predict_proba(X[is_attack])
            idx = proba.argmax(1)
            tactic[is_attack] = self.tactic_enc_.inverse_transform(idx)
            tactic_conf[is_attack] = proba.max(1)
        return pd.DataFrame({"p_attack": p_attack, "is_attack": is_attack.astype(int),
                             "tactic_pred": tactic, "tactic_conf": tactic_conf}, index=df.index)

    def evaluate(self, df: pd.DataFrame, label_col="Label", tactic_col="tactic") -> dict:
        """Quick metrics used during training; the full evaluation is Phase 3."""
        p = self.predict(df)
        out = {"binary": binary_metrics(df[label_col], p.is_attack, p.p_attack),
               "tactic_end_to_end": multiclass_metrics(df[tactic_col], p.tactic_pred)}
        atk = df[label_col].to_numpy(int) == 1
        if atk.any():
            X = self.transform(df[atk])
            known = df.loc[atk, tactic_col].isin(self.tactic_enc_.classes_).to_numpy()
            if known.any():
                pred = self.tactic_enc_.inverse_transform(self.tactic_.predict(X[known]))
                out["tactic_given_attack"] = multiclass_metrics(df.loc[atk, tactic_col][known], pred)
        return out
