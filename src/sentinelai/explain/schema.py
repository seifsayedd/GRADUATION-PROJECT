"""The per-alert explanation object: JSON schema, builder, validator and template summary.

The object is what Phase 5 (GenAI reports) and Phase 6 (dashboard) consume.
``docs/schema/explanation.schema.json`` is written from ``SCHEMA`` below.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .features import value_text

SCHEMA_VERSION = "1.0"

_num = {"type": "number"}
_str = {"type": "string"}
_contrib_item = {"type": "object", "required": ["feature", "value", "contribution", "direction"],
                 "properties": {"feature": _str, "value": _str, "contribution": _num,
                                "direction": {"type": "string", "enum": ["towards attack", "towards benign", "towards this tactic",
                                                                         "against this tactic", "abnormal"]}}}
_contribs = {"type": "object", "required": ["space", "base_value", "output", "items", "rest", "additivity_error"],
             "properties": {"space": {"type": "string", "enum": ["log-odds", "margin", "probability", "squared error"]},
                            "base_value": _num, "output": _num, "items": {"type": "array", "items": _contrib_item},
                            "rest": _num, "additivity_error": _num}}
SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SentinelAI alert explanation", "type": "object",
    # Stage 1 and 3 explanations carry `contributions`; Stage 2 carries `attention` (checked in finalize).
    "required": ["schema_version", "alert_id", "flow", "model", "decision", "caveats", "summary"],
    "properties": {
        "schema_version": {"type": "string", "enum": [SCHEMA_VERSION]},
        "alert_id": _str,
        "flow": {"type": "object", "required": ["src", "dst", "protocol", "start_ms"],
                 "properties": {"src": _str, "src_port": _num, "dst": _str, "dst_port": _num,
                                "protocol": _str, "start_ms": _num, "dataset": _str}},
        "ground_truth": {"type": "object", "properties": {"label": _num, "attack": _str, "tactic": _str}},
        "model": {"type": "object", "required": ["stage", "name", "feature_set", "git_commit"],
                  "properties": {"stage": _num, "name": _str, "feature_set": _str, "git_commit": _str}},
        "decision": {"type": "object", "required": ["score", "score_kind", "threshold", "operating_point", "flagged"],
                     "properties": {"score": _num, "score_kind": _str, "threshold": _num, "operating_point": _str,
                                    "flagged": {"type": "boolean"}}},
        "tactic": {"type": "object", "required": ["predicted", "confidence"],
                   "properties": {"predicted": _str, "confidence": _num, "taxonomy_note": _str}},
        "contributions": _contribs,
        "tactic_contributions": _contribs,
        "attention": {"type": "array", "items": {"type": "object", "required": ["offset", "weight"],
                                                 "properties": {"offset": _num, "weight": _num, "occlusion_delta": _num,
                                                                "flow": _str}}},
        "caveats": {"type": "array", "items": _str},
        "summary": _str,
    },
}

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}


class SchemaError(ValueError):
    pass


def validate(obj, schema: dict = SCHEMA, path: str = "$") -> None:
    """Small JSON-schema check (type, required, properties, items, enum). Raises SchemaError."""
    t = schema.get("type")
    if t == "number":
        if isinstance(obj, bool) or not isinstance(obj, (int, float)) or (isinstance(obj, float) and not np.isfinite(obj)):
            raise SchemaError(f"{path}: expected a finite number, got {obj!r}")
    elif t and not isinstance(obj, _TYPES[t]):
        raise SchemaError(f"{path}: expected {t}, got {type(obj).__name__}")
    if "enum" in schema and obj not in schema["enum"]:
        raise SchemaError(f"{path}: {obj!r} not in {schema['enum']}")
    if t == "object":
        for k in schema.get("required", []):
            if k not in obj:
                raise SchemaError(f"{path}: missing {k!r}")
        for k, sub in schema.get("properties", {}).items():
            if k in obj:
                validate(obj[k], sub, f"{path}.{k}")
        extra = set(obj) - set(schema.get("properties", {}))
        if schema.get("properties") and extra:
            raise SchemaError(f"{path}: unexpected keys {sorted(extra)}")
    if t == "array":
        for i, v in enumerate(obj):
            validate(v, schema.get("items", {}), f"{path}[{i}]")


def write_schema(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SCHEMA, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------ builder
def contributions(agg_row, raw_row, base: float, output: float, space: str, add_err: float, top_k: int,
                  kind: str = "binary") -> dict:
    """Top-k original features by |contribution|; everything else is summed into ``rest``."""
    s = agg_row.sort_values(key=np.abs, ascending=False)
    top = s.iloc[:top_k]
    pos, neg = {"binary": ("towards attack", "towards benign"), "tactic": ("towards this tactic", "against this tactic"),
                "error": ("abnormal", "abnormal")}[kind]
    items = [{"feature": f, "value": value_text(f, raw_row.get(f)), "contribution": round(float(v), 6),
              "direction": pos if v >= 0 else neg} for f, v in top.items()]
    return {"space": space, "base_value": round(float(base), 6), "output": round(float(output), 6), "items": items,
            "rest": round(float(s.iloc[top_k:].sum()), 6), "additivity_error": float(add_err)}


def flow_info(row) -> dict:
    proto = value_text("PROTOCOL", row["PROTOCOL"]) if "PROTOCOL" in row else "unknown"
    f = {"src": str(row.get("IPV4_SRC_ADDR", "unknown")), "dst": str(row.get("IPV4_DST_ADDR", "unknown")),
         "protocol": proto, "start_ms": float(row.get("FLOW_START_MILLISECONDS", 0))}
    for k, c in (("src_port", "L4_SRC_PORT"), ("dst_port", "L4_DST_PORT")):
        if c in row and row[c] == row[c]:
            f[k] = float(row[c])
    if "source" in row and isinstance(row["source"], str):
        f["dataset"] = row["source"]
    return f


def _fmt(v: float) -> str:
    return f"{v:+.3f}"


def summary(e: dict) -> str:
    """Fixed-template paragraph. Every number in it is copied from the object."""
    d, m, f = e["decision"], e["model"], e["flow"]
    port = lambda k: f":{int(f[k])}" if f.get(k) else ""
    stage = {1: "Stage 1", 2: "Stage 2", 3: "Stage 3"}[int(m["stage"])]
    verb = "flagged" if d["flagged"] else "did not flag"
    proto = f["protocol"].split("(")[-1].rstrip(")") if "(" in f["protocol"] else f["protocol"]
    s = [f"{stage} ({m['name']}) {verb} the {proto} flow {f['src']}{port('src_port')} -> {f['dst']}{port('dst_port')} "
         f"(score {d['score']:.3f}, threshold {d['threshold']:.3f}, {d['score_kind']})."]
    if "tactic" in e:
        s.append(f"Predicted tactic: {e['tactic']['predicted']} (confidence {e['tactic']['confidence']:.3f}).")
    if "contributions" in e:
        items = e["contributions"]["items"]
        up = [i for i in items if i["direction"] in ("towards attack", "abnormal")][:3]
        down = [i for i in items if i["direction"] == "towards benign"][:2]
        unit = e["contributions"]["space"]
        if up:
            s.append(("Most abnormal features" if up[0]["direction"] == "abnormal" else "Strongest evidence for attack")
                     + f" ({unit}): " + "; ".join(f"{i['feature']} = {i['value']} ({_fmt(i['contribution'])})" for i in up) + ".")
        if down:
            s.append(f"Strongest evidence against ({unit}): " + "; ".join(f"{i['feature']} = {i['value']} ({_fmt(i['contribution'])})" for i in down) + ".")
    earlier = [a for a in e.get("attention", []) if a["offset"] < 0]
    if earlier:
        a = earlier[0]
        s.append(f"Most attended earlier flow: {int(-a['offset'])} flows back (weight {a['weight']:.3f}"
                 + (f", removing it moves the attack probability by {a['occlusion_delta']:.3f})." if "occlusion_delta" in a else ")."))
    if e["caveats"]:
        s.append("Caveat: " + e["caveats"][0])
    return " ".join(s)


def numbers_in(text: str) -> list[str]:
    return re.findall(r"[-+]?\d[\d,]*\.?\d*", text)


def finalize(e: dict) -> dict:
    e = {"schema_version": SCHEMA_VERSION, **e}
    if "contributions" not in e and "attention" not in e:
        raise SchemaError("an explanation needs feature contributions or attention weights")
    e["summary"] = summary(e)
    validate(e)
    return e
