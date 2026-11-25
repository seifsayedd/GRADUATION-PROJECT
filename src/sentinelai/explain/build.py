"""Build explanation objects for Stage 1, 2 and 3 alerts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .attention import occlusion, windows_for
from .features import aggregate, value_text
from .schema import contributions, finalize, flow_info
from .tree_shap import explain_flow_classifier


def _truth(row) -> dict | None:
    if "Label" not in row:
        return None
    g = {"label": float(row["Label"])}
    for k, c in (("attack", "Attack"), ("tactic", "tactic")):
        if c in row and isinstance(row[c], str):
            g[k] = row[c]
    return g


def _base(row, aid, model_info):
    e = {"alert_id": aid, "flow": flow_info(row), "model": model_info}
    t = _truth(row)
    if t:
        e["ground_truth"] = t
    return e


def stage1(fc, df: pd.DataFrame, model_info: dict, caveats: list[str], threshold: float = 0.5,
           operating_point: str = "default", top_k: int = 8, id_prefix: str = "s1") -> tuple[list[dict], dict]:
    """Explanation per row of ``df`` plus the aggregated SHAP frames (for global analysis and checks)."""
    r = explain_flow_classifier(fc, df)
    cols = list(r["X"].columns)
    aggB = aggregate(r["binary"].values, cols)
    aggT = aggregate(r["tactic"].values, cols)
    out = []
    for i, (idx, row) in enumerate(df.iterrows()):
        p = float(r["p_attack"][i])
        flagged = p >= threshold
        e = _base(row, f"{id_prefix}-{idx}", model_info)
        e["decision"] = {"score": round(p, 6), "score_kind": "probability of attack", "threshold": float(threshold),
                         "operating_point": operating_point, "flagged": bool(flagged)}
        b = r["binary"]
        e["contributions"] = contributions(aggB.iloc[i], row, b.base[i], b.output[i], b.space, b.max_additivity_error, top_k)
        if flagged:
            tac = str(r["tactic_pred"][i])
            e["tactic"] = {"predicted": tac, "confidence": round(float(r["tactic_conf"][i]), 6)}
            if tac == "Reconnaissance/Discovery":
                e["tactic"]["taxonomy_note"] = "Reconnaissance and Discovery are one merged class (see docs/attack-taxonomy.md)."
            t = r["tactic"]
            e["tactic_contributions"] = contributions(aggT.iloc[i], row, t.base[i], t.output[i], t.space,
                                                      t.max_additivity_error, top_k, kind="tactic")
        e["caveats"] = list(caveats)
        out.append(finalize(e))
    frames = {"binary": aggB.set_axis(df.index), "tactic": aggT.set_axis(df.index), "X": r["X"],
              "p_attack": pd.Series(r["p_attack"], index=df.index), "tactic_pred": pd.Series(r["tactic_pred"], index=df.index),
              "additivity": {"binary": r["binary"].max_additivity_error, "tactic": r["tactic"].max_additivity_error,
                             "space": {"binary": r["binary"].space, "tactic": r["tactic"].space}}}
    return out, frames


def stage3(ae, df: pd.DataFrame, model_info: dict, caveats: list[str], top_k: int = 8,
           id_prefix: str = "s3") -> tuple[list[dict], pd.DataFrame]:
    """Reconstruction-error explanation: per-column squared error / n_columns sums exactly to the score."""
    fe = ae.feature_errors(df)
    contrib = fe.to_numpy() / fe.shape[1]
    agg = aggregate(contrib, list(fe.columns)).set_axis(df.index)
    score = ae.score(df)
    err = float(np.max(np.abs(contrib.sum(1) - score))) if len(df) else 0.0
    out = []
    for i, (idx, row) in enumerate(df.iterrows()):
        e = _base(row, f"{id_prefix}-{idx}", model_info)
        e["decision"] = {"score": round(float(score[i]), 6), "score_kind": "reconstruction error",
                         "threshold": round(float(ae.threshold_), 6), "operating_point": "1% of benign validation flagged",
                         "flagged": bool(score[i] > ae.threshold_)}
        e["contributions"] = contributions(agg.iloc[i], row, 0.0, score[i], "squared error", err, top_k, kind="error")
        e["caveats"] = list(caveats)
        out.append(finalize(e))
    return out, agg


def _flow_brief(r, gap_s: float) -> str:
    proto = value_text("PROTOCOL", r["PROTOCOL"]).split(" ")[-1].strip("()")
    port = f":{int(r['L4_DST_PORT'])}" if "L4_DST_PORT" in r else ""
    return f"{proto} to {r['IPV4_DST_ADDR']}{port}, {gap_s:,.1f} s before the labelled flow"


def stage2(model, df: pd.DataFrame, positions: np.ndarray, model_info: dict, caveats: list[str], threshold: float,
           operating_point: str, top_k: int = 5, id_prefix: str = "s2") -> tuple[list[dict], dict]:
    """Attention + remove-one-flow explanation for the flows at ``positions`` (model order)."""
    x, mask, ordered, pos, _ = windows_for(model, df, positions)
    p, att, delta = occlusion(model, x, mask)
    W = mask.shape[1]
    t = ordered.FLOW_START_MILLISECONDS.to_numpy("float64")
    out = []
    for n, ps in enumerate(pos):
        row = ordered.iloc[ps]
        e = _base(row, f"{id_prefix}-{ordered.index[ps]}", model_info)
        e["decision"] = {"score": round(float(p[n]), 6), "score_kind": "1 - P(Benign)", "threshold": float(threshold),
                         "operating_point": operating_point, "flagged": bool(p[n] > threshold)}
        steps = [j for j in np.argsort(-att[n]) if mask[n, j]][:top_k]
        items = []
        for j in steps:
            k = ps - (W - 1 - j)
            a = {"offset": float(j - (W - 1)), "weight": round(float(att[n, j]), 6)}
            if not np.isnan(delta[n, j]):
                a["occlusion_delta"] = round(float(delta[n, j]), 6)
            a["flow"] = "the labelled flow itself" if j == W - 1 else _flow_brief(ordered.iloc[k], (t[ps] - t[k]) / 1000)
            items.append(a)
        e["attention"] = items
        e["caveats"] = list(caveats)
        out.append(finalize(e))
    return out, {"p": p, "att": att, "delta": delta, "mask": mask}
