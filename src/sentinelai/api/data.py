"""Read-only views over the repository's real outputs. Every view names the file it came from."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

# Lab explanation objects are named <prefix><row>, where row is the lab conversation's index.
LAB_MODELS = {"xgboost_temporal_lab": "lab-xgboost_temporal_lab-",
              "random_forest_temporal_lab": "lab-random_forest_temporal_lab-",
              "autoencoder_temporal_lab": "lab-ae-"}


class MissingOutput(FileNotFoundError):
    """A report file an endpoint needs has not been generated yet."""


@lru_cache(maxsize=64)
def _load(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json(root: Path, rel: str) -> dict:
    p = root / rel
    if not p.exists():
        raise MissingOutput(f"{rel} is missing; run the phase that produces it")
    return _load(str(p), p.stat().st_mtime)


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def metric(op: dict) -> dict:
    """Test metrics of one operating point (Phase 3 layout), with the F1 bootstrap CI when present."""
    t = op["test"] if "test" in op else op
    ci = op.get("test_ci", op.get("default_ci", {})).get("f1", {}) if "test" in op else {}
    return {"f1": _r(t["f1"]), "f1_lo": _r(ci.get("lo")), "f1_hi": _r(ci.get("hi")),
            "precision": _r(t["precision"]), "recall": _r(t["recall"]), "fpr": _r(t["fpr"]), "n": t["n"]}


def loso_f1_range(loso: dict, feature_set: str) -> list[float] | None:
    f1 = [r["val_fpr_0.01"]["f1"] for r in loso["runs"] if r["feature_set"] == feature_set]
    return [_r(min(f1)), _r(max(f1))] if f1 else None


def summary(root: Path) -> dict:
    """Headline numbers, each with its source file and the JSON path it was read from."""
    models = load_json(root, "reports/phase3/models.json")["models"]
    xgb = metric(models["stage1_xgboost_temporal_full"]["operating_points"]["val_fpr_0.01"])
    loso = load_json(root, "reports/phase3/loso.json")["loso"]
    lab = load_json(root, "reports/phase3/lab.json")["lab"]
    lab2 = load_json(root, "reports/phase2/lab_scores_summary.json")
    hold = load_json(root, "reports/phase4/holdout.json")["holdout"]

    cards = [
        {"id": "seen", "label": "Stage 1 XGBoost F1 on later flows of the training networks",
         "value": xgb["f1"], "range": [xgb["f1_lo"], xgb["f1_hi"]],
         "note": f"FPR {xgb['fpr']} with a threshold set for 1% validation FPR; range is a 95% block-bootstrap CI",
         "source": "reports/phase3/models.json",
         "path": "models.stage1_xgboost_temporal_full.operating_points.val_fpr_0.01"},
        {"id": "unseen_full", "label": "F1 on a network left out of training (all features)",
         "value": None, "range": loso_f1_range(loso, "full"), "note": "leave-one-source-out over 4 sources",
         "source": "reports/phase3/loso.json", "path": "loso.runs[feature_set=full].val_fpr_0.01.f1"},
        {"id": "unseen_lab", "label": "F1 on a network left out of training (lab features)",
         "value": None, "range": loso_f1_range(loso, "lab"), "note": "leave-one-source-out over 4 sources",
         "source": "reports/phase3/loso.json", "path": "loso.runs[feature_set=lab].val_fpr_0.01.f1"},
        {"id": "lab", "label": "Lab capture v1 conversations", "value": lab["rows"], "range": None,
         "note": f"{lab['attacks']} of them an attack; no model flagged it",
         "source": "reports/phase3/lab.json", "path": "lab.rows, lab.attacks"},
    ]
    lab_flags = [{"model": k, "flagged": v["fp"] + v["tp"], "false_alarms": v["fp"], "attack_caught": v["tp"] > 0}
                 for k, v in lab2["models"].items()]
    holdout = [{"model": k, "class": c, "detected": _r(v["detected"])}
               for k, cls in hold.items() for c, v in cls.items() if isinstance(v, dict) and "detected" in v]
    return {"cards": cards,
            "lab_flags": {"source": "reports/phase2/lab_scores_summary.json", "rows": lab_flags},
            "holdout": {"source": "reports/phase4/holdout.json",
                        "note": "attack classes removed from training (Phase 4)", "rows": holdout}}


def evaluation(root: Path) -> dict:
    models = load_json(root, "reports/phase3/models.json")["models"]
    loso = load_json(root, "reports/phase3/loso.json")["loso"]
    rows = [{"model": name, **metric(m["operating_points"]["val_fpr_0.01"]), "threshold": _r(m["threshold_val"], 4)}
            for name, m in models.items()]
    lrows = [{"feature_set": r["feature_set"], "test_source": r["test_source"], **metric(r["val_fpr_0.01"]),
              "roc_auc": _r(r["roc_auc"])} for r in loso["runs"]]
    return {"operating_point": "threshold set for 1% FPR on benign validation flows",
            "models": {"source": "reports/phase3/models.json", "rows": rows},
            "loso": {"source": "reports/phase3/loso.json", "rows": lrows}}


def alert_row(e: dict) -> dict:
    d, f = e["decision"], e["flow"]
    return {"alert_id": e["alert_id"], "model": e["model"]["name"], "stage": e["model"]["stage"],
            "dataset": f.get("dataset"), "src": f.get("src"), "dst": f.get("dst"), "protocol": f.get("protocol"),
            "dst_port": f.get("dst_port"), "start_ms": f.get("start_ms"), "score": _r(d["score"], 4),
            "threshold": _r(d["threshold"], 4), "score_kind": d["score_kind"], "flagged": bool(d["flagged"]),
            "true_tactic": e.get("ground_truth", {}).get("tactic"),
            "predicted_tactic": (e.get("tactic") or {}).get("predicted")}


def lab_index(alert_id: str) -> tuple[str, int] | None:
    for model, prefix in LAB_MODELS.items():
        rest = alert_id[len(prefix):]
        if alert_id.startswith(prefix) and rest.isdigit():
            return model, int(rest)
    return None


def timeline(root: Path, explanations: dict[str, dict]) -> dict:
    """The lab capture's conversations in time order, with each model's verdict."""
    by_row: dict[int, dict] = defaultdict(dict)
    for aid, e in explanations.items():
        li = lab_index(aid)
        if li:
            by_row[li[1]][li[0]] = e
    gru: dict[int, dict] = {}
    p = root / "reports/phase2/lab_scores.csv"
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                gru[i] = {"session": r["session_id"], "gru_p_attack": _r(r["s2_p_attack"], 4)}
    rows = []
    for i, ms in by_row.items():
        e0 = next(iter(ms.values()))
        f = e0["flow"]
        rows.append({"row": i, "start_ms": f.get("start_ms"), "src": f.get("src"), "dst": f.get("dst"),
                     "protocol": f.get("protocol"), "dst_port": f.get("dst_port"),
                     "truth": e0.get("ground_truth", {}).get("attack"),
                     "true_tactic": e0.get("ground_truth", {}).get("tactic"),
                     "verdicts": {m: {"alert_id": e["alert_id"], "flagged": bool(e["decision"]["flagged"]),
                                      "score": _r(e["decision"]["score"], 4)} for m, e in ms.items()},
                     **gru.get(i, {})})
    rows.sort(key=lambda r: (r["start_ms"] is None, r["start_ms"] or 0, r["row"]))
    return {"source": "reports/phase4/lab/*.json (explanation objects); GRU scores from reports/phase2/lab_scores.csv",
            "note": "Replay of lab capture v1, not live traffic. The GRU has no lab explanation objects; "
                    "Phase 2 lab scoring flagged none of the 54 conversations with it.",
            "models": list(LAB_MODELS), "rows": rows}


_TACTIC_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$")


def tactics(root: Path) -> list[str]:
    """Tactic order from the first table in docs/attack-taxonomy.md."""
    out, inside = [], False
    text = (root / "docs/attack-taxonomy.md").read_text(encoding="utf-8")
    for line in (ln.strip() for ln in text.splitlines() if ln.strip()):   # the file has some \r\r\n endings
        if line.startswith("| Tactic |"):
            inside = True
            continue
        if inside:
            if not line.startswith("|"):
                break
            m = _TACTIC_ROW.match(line)
            if m and not m.group(1).startswith("-"):
                out.append(m.group(1))
    return out


def attack_matrix(root: Path, explanations: dict[str, dict]) -> dict:
    """Tactic x model counts over the stored explanation objects (a sample, not a full test set)."""
    order = tactics(root)
    cells: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for e in explanations.values():
        c = cells[(e.get("ground_truth", {}).get("tactic") or "Unknown", e["model"]["name"])]
        c["objects"] += 1
        c["flagged"] += bool(e["decision"]["flagged"])
    models = sorted({m for _, m in cells})
    seen = {t for t, _ in cells}
    rows = order + sorted(seen - set(order))
    return {"source": "explanation objects in reports/phase4/lab and reports/phase4/examples; "
                      "tactic order from docs/attack-taxonomy.md",
            "note": "Counts of stored explanation objects by true tactic. These are sampled examples, so the "
                    "counts show what the sample covers, not detection rates.",
            "tactics": rows, "models": models,
            "cells": [{"tactic": t, "model": m, "objects": cells[(t, m)]["objects"],
                       "flagged": cells[(t, m)]["flagged"]} for t in rows for m in models if (t, m) in cells]}
