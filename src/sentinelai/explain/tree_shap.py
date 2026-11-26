"""SHAP values for the Stage 1 tree models.

* XGBoost: exact TreeSHAP built into XGBoost (``pred_contribs``). Values are in
  log-odds (binary) or per-class margin (tactic) space.
* Random Forest: ``shap.TreeExplainer``. Values are in probability space.

Every call checks additivity against the model's real output: the base value
plus the contributions must give back the score the model actually produced.
A mismatch raises ``AdditivityError`` instead of returning a wrong explanation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class AdditivityError(AssertionError):
    pass


@dataclass
class ShapResult:
    values: np.ndarray        # n x columns
    base: np.ndarray          # n
    output: np.ndarray        # n, the model output being explained (margin or probability)
    space: str                # "log-odds", "margin" or "probability"
    columns: list
    max_additivity_error: float


def check_additivity(values, base, output, tol: float) -> float:
    err = float(np.max(np.abs(np.asarray(values).sum(1) + np.asarray(base) - np.asarray(output)))) if len(output) else 0.0
    if err > tol:
        raise AdditivityError(f"contributions do not add up to the model output (max error {err:.2e} > {tol:.0e})")
    return err


def _iter_range(m):
    bi = getattr(m, "best_iteration", None)
    return (0, int(bi) + 1) if bi is not None and bi >= 0 else (0, 0)


def _xgb(m, X: pd.DataFrame, cls: np.ndarray | None, tol: float) -> ShapResult:
    import xgboost as xgb
    booster = m.get_booster()
    dm = xgb.DMatrix(X)
    rng = _iter_range(m)
    contrib = booster.predict(dm, pred_contribs=True, iteration_range=rng)
    margin = booster.predict(dm, output_margin=True, iteration_range=rng)
    proba = m.predict_proba(X)
    if contrib.ndim == 2:                                   # binary: n x (F+1), log-odds
        vals, base, out, space = contrib[:, :-1], contrib[:, -1], margin, "log-odds"
        # the margin must be the same score the classifier reports
        if not np.allclose(1 / (1 + np.exp(-margin)), proba[:, 1], atol=1e-4):
            raise AdditivityError("XGBoost margin does not match predict_proba (iteration range?)")
    else:                                                   # multiclass: n x K x (F+1)
        r = np.arange(len(X))
        vals, base, out, space = contrib[r, cls, :-1], contrib[r, cls, -1], margin[r, cls], "margin"
        sm = np.exp(margin - margin.max(1, keepdims=True))
        if not np.allclose(sm / sm.sum(1, keepdims=True), proba, atol=1e-4):
            raise AdditivityError("XGBoost margins do not match predict_proba (iteration range?)")
    err = check_additivity(vals, base, out, tol)
    return ShapResult(vals, base, out, space, list(X.columns), err)


def _rf(m, X: pd.DataFrame, cls: np.ndarray | None, tol: float) -> ShapResult:
    import shap
    ex = shap.TreeExplainer(m)
    sv = np.asarray(ex.shap_values(X.to_numpy("float64"), check_additivity=False))   # n x F x K
    if sv.ndim == 3 and sv.shape[0] != len(X):                                     # older layout K x n x F
        sv = np.transpose(sv, (1, 2, 0))
    ev = np.atleast_1d(ex.expected_value)
    k = np.ones(len(X), int) if cls is None else cls
    r = np.arange(len(X))
    vals, base = sv[r, :, k], ev[k]
    out = m.predict_proba(X)[r, k]
    err = check_additivity(vals, base, out, tol)
    return ShapResult(vals, base, out, "probability", list(X.columns), err)


def explain_trees(model, X: pd.DataFrame, cls: np.ndarray | None = None, tol: float | None = None) -> ShapResult:
    """SHAP for one fitted tree classifier. ``cls`` = class index per row (multiclass), None = binary attack class."""
    import xgboost as xgb
    if isinstance(model, xgb.XGBClassifier):
        return _xgb(model, X, cls, tol or 1e-3)
    return _rf(model, X, cls, tol or 1e-6)


def explain_flow_classifier(fc, df: pd.DataFrame, tactic: bool = True) -> dict:
    """Binary SHAP for every row of ``df``; tactic SHAP (for the predicted tactic) too if ``tactic``."""
    X = fc.transform(df)
    res = {"binary": explain_trees(fc.binary_, X)}
    if tactic:
        proba = fc.tactic_.predict_proba(X)
        cls = proba.argmax(1)
        res["tactic"] = explain_trees(fc.tactic_, X, cls)
        res["tactic_pred"] = fc.tactic_enc_.inverse_transform(cls)
        res["tactic_conf"] = proba.max(1)
    res["p_attack"] = fc.binary_.predict_proba(X)[:, 1]
    res["X"] = X
    return res
