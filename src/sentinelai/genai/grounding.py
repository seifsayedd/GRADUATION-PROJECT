"""Groundedness checks for generated text.

A report passes only if every number and every IPv4 address it states can be
found in the sources it was given, and every citation names a source that was
actually provided. Numbers may be rounded (``0.0669`` supports ``0.067``) and
a fraction supports the same value as a percentage (``0.069`` supports ``6.9%``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?%?(?![\w])", re.I)


def ips(text: str) -> set[str]:
    return set(IP.findall(text))


def numbers(text: str) -> list[str]:
    t = IP.sub(" ", text)
    return [m.group(0) for m in NUM.finditer(t)]


def _value(tok: str) -> tuple[Decimal, int, bool] | None:
    """(value, decimals, is_percent) for a numeric token."""
    pct = tok.endswith("%")
    s = tok.rstrip("%").replace(",", "").lstrip("+")
    try:
        v = Decimal(s)
    except InvalidOperation:
        return None
    dec = len(s.split(".")[1]) if "." in s and "e" not in s.lower() else 0
    return v, dec, pct


def _source_values(texts: list[str]) -> list[Decimal]:
    vals = []
    for t in texts:
        for tok in numbers(t):
            p = _value(tok)
            if p:
                v, _, pct = p
                vals.append(abs(v / 100 if pct else v))
    return vals


def supported(tok: str, src_vals: list[Decimal]) -> bool:
    p = _value(tok)
    if p is None:
        return True
    v, dec, pct = p
    v = abs(v)
    half = Decimal(1).scaleb(-dec) / 2
    for s in src_vals:
        for cand in ((s * 100,) if pct else (s, s * 100)):
            if abs(cand - v) <= half + Decimal("1e-12"):
                return True
    return False


@dataclass
class GroundingResult:
    ok: bool
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_ips: list[str] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)


def walk_strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() if k != "citations" for s in walk_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in walk_strings(v)]
    return []


def walk_citations(obj) -> list[str]:
    if isinstance(obj, dict):
        return list(obj.get("citations", [])) + [c for k, v in obj.items() if k != "citations" for c in walk_citations(v)]
    if isinstance(obj, list):
        return [c for v in obj for c in walk_citations(v)]
    return []


def check(output: dict, sources: dict[str, str]) -> GroundingResult:
    """``sources``: source id -> full source text that was shown to the model."""
    texts = walk_strings(output)
    src_texts = list(sources.values())
    src_vals = _source_values(src_texts)
    src_ips = set().union(*(ips(t) for t in src_texts)) if src_texts else set()
    bad_num = sorted({n for t in texts for n in numbers(t) if not supported(n, src_vals)})
    bad_ip = sorted({i for t in texts for i in ips(t) if i not in src_ips})
    bad_cite = sorted({c for c in walk_citations(output) if c not in sources})
    return GroundingResult(not (bad_num or bad_ip or bad_cite), bad_num, bad_ip, bad_cite)
