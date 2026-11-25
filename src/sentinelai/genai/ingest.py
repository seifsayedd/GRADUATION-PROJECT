"""Build the knowledge base: split project documents into chunks and load explanation objects."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .store import Chunk

DOCS = ["docs/phase2-findings.md", "docs/phase3-findings.md", "docs/phase4-findings.md", "docs/attack-taxonomy.md",
        "docs/lab-architecture.md", "docs/feature-mapping.md", "data/README.md"]
REPORTS = ["reports/phase2/summary.md", "reports/phase3/report.md", "reports/phase4/report.md"]
EXPLANATIONS = ["reports/phase4/lab/*.json", "reports/phase4/examples/*.json"]

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_markdown(text: str, source: str, kind: str = "doc", max_chars: int = 1200) -> list[Chunk]:
    """Split on headings, then on blank lines so no chunk exceeds ``max_chars``. Deterministic ids."""
    path: list[str] = []
    blocks: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            blocks.append((" > ".join(path) or "(top)", body))
        buf.clear()

    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()

    out: list[Chunk] = []
    for section, body in blocks:
        part = ""
        for para in re.split(r"\n\s*\n", body):
            if part and len(part) + len(para) > max_chars:
                out.append(Chunk(f"{source}#{len(out) + 1}", source, section, kind, part.strip()))
                part = ""
            part += para + "\n\n"
        if part.strip():
            out.append(Chunk(f"{source}#{len(out) + 1}", source, section, kind, part.strip()))
    return out


def explanation_chunk(obj: dict, source: str) -> Chunk:
    """One searchable chunk per explanation object: its template summary plus the key fields."""
    items = obj.get("contributions", {}).get("items", [])
    feats = "; ".join(f"{i['feature']} {i['value']} {i['direction']}" for i in items[:6])
    text = f"{obj['summary']} Top features: {feats}." if feats else obj["summary"]
    return Chunk(f"alert:{obj['alert_id']}", source, obj["model"]["name"], "explanation", text,
                 {"alert_id": obj["alert_id"], "stage": obj["model"]["stage"], "flagged": obj["decision"]["flagged"]})


def load_explanations(root: Path, patterns=EXPLANATIONS) -> list[tuple[dict, str]]:
    out, seen = [], set()
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            for obj in json.loads(p.read_text(encoding="utf-8")):
                if obj["alert_id"] not in seen:
                    seen.add(obj["alert_id"])
                    out.append((obj, p.relative_to(root).as_posix()))
    return out


def build(root: Path, store, embedder, docs=DOCS, reports=REPORTS, log=print) -> dict:
    chunks: list[Chunk] = []
    for kind, files in (("doc", docs), ("report", reports)):
        for f in files:
            p = root / f
            if p.exists():
                chunks += chunk_markdown(p.read_text(encoding="utf-8"), f, kind)
            else:
                log(f"skipping missing {f}")
    exps = load_explanations(root)
    for obj, src in exps:
        store.add_explanation(obj)
        chunks.append(explanation_chunk(obj, src))
    log(f"embedding {len(chunks):,} chunks with {embedder.name} ...")
    store.add(chunks, embedder.embed([f"{c.section}\n{c.text}" for c in chunks]))
    return {"chunks": len(chunks), "explanations": len(exps),
            "by_kind": {k: sum(c.kind == k for c in chunks) for k in ("doc", "report", "explanation")}}
