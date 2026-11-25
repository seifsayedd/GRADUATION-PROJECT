"""Storage behind one interface: document chunks with embeddings, alert explanations, reports.

* ``MemoryStore``: numpy cosine search in memory; can be saved to and loaded from disk.
* ``PgVectorStore``: PostgreSQL + pgvector, schema in ``db/001_init.sql``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class Chunk:
    id: str
    source: str            # repository path the text came from
    section: str           # heading path inside that file
    kind: str              # doc | report | explanation
    text: str
    meta: dict = field(default_factory=dict)


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, k: int, kinds: tuple[str, ...] | None = None) -> list[tuple[Chunk, float]]: ...
    def add_explanation(self, obj: dict) -> None: ...
    def get_explanation(self, alert_id: str) -> dict | None: ...
    def list_alerts(self) -> list[str]: ...
    def save_report(self, alert_id: str | None, report: dict, citations: list[str], provider: str) -> str: ...


class MemoryStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, dim), dtype="float32")
        self.explanations: dict[str, dict] = {}
        self.reports: list[dict] = []

    def add(self, chunks, vectors):
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.shape != (len(chunks), self.dim):
            raise ValueError(f"expected vectors of shape ({len(chunks)}, {self.dim}), got {vectors.shape}")
        known = {c.id for c in self.chunks}
        dup = [c.id for c in chunks if c.id in known]
        if dup:
            raise ValueError(f"duplicate chunk ids: {dup[:3]}")
        self.chunks += list(chunks)
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, vector, k, kinds=None):
        if not self.chunks:
            return []
        v = np.asarray(vector, dtype="float32").reshape(-1)
        sims = self.vectors @ (v / (np.linalg.norm(v) or 1))
        order = np.argsort(-sims)
        out = [(self.chunks[i], float(sims[i])) for i in order if kinds is None or self.chunks[i].kind in kinds]
        return out[:k]

    def add_explanation(self, obj):
        self.explanations[obj["alert_id"]] = obj

    def get_explanation(self, alert_id):
        return self.explanations.get(alert_id)

    def list_alerts(self):
        return sorted(self.explanations)

    def save_report(self, alert_id, report, citations, provider):
        rid = f"report-{len(self.reports) + 1}"
        self.reports.append({"id": rid, "alert_id": alert_id, "report": report, "citations": citations, "provider": provider})
        return rid

    # -- persistence (git-ignored data/interim/phase5/)
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks]), encoding="utf-8")
        (d / "explanations.json").write_text(json.dumps(self.explanations), encoding="utf-8")

    @staticmethod
    def load(d: Path) -> "MemoryStore":
        v = np.load(d / "vectors.npy")
        s = MemoryStore(v.shape[1])
        s.chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text(encoding="utf-8"))]
        s.vectors = v
        s.explanations = json.loads((d / "explanations.json").read_text(encoding="utf-8"))
        return s


class PgVectorStore:
    """PostgreSQL + pgvector. Needs ``DATABASE_URL`` and the schema from ``db/001_init.sql``."""

    def __init__(self, database_url: str, dim: int):
        import psycopg
        from pgvector.psycopg import register_vector
        self.dim = dim
        self.conn = psycopg.connect(database_url, autocommit=True)
        register_vector(self.conn)

    def init_schema(self, sql_path: Path) -> None:
        sql = sql_path.read_text(encoding="utf-8")
        if f"vector({self.dim})" not in sql:
            raise ValueError(f"{sql_path} does not declare vector({self.dim})")
        self.conn.execute(sql)

    def add(self, chunks, vectors):
        with self.conn.cursor() as cur:
            for c, v in zip(chunks, np.asarray(vectors, dtype="float32")):
                cur.execute("INSERT INTO chunks (id, source, section, kind, text, meta, embedding) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                            (c.id, c.source, c.section, c.kind, c.text, json.dumps(c.meta), v))

    def search(self, vector, k, kinds=None):
        q = "SELECT id, source, section, kind, text, meta, 1 - (embedding <=> %s) AS sim FROM chunks"
        args: list = [np.asarray(vector, dtype="float32")]
        if kinds:
            q += " WHERE kind = ANY(%s)"
            args.append(list(kinds))
        q += " ORDER BY embedding <=> %s LIMIT %s"
        args += [np.asarray(vector, dtype="float32"), k]
        rows = self.conn.execute(q, args).fetchall()
        return [(Chunk(r[0], r[1], r[2], r[3], r[4], r[5] or {}), float(r[6])) for r in rows]

    def add_explanation(self, obj):
        self.conn.execute("INSERT INTO explanations (alert_id, stage, model, flagged, body) VALUES (%s, %s, %s, %s, %s) "
                          "ON CONFLICT (alert_id) DO UPDATE SET body = EXCLUDED.body",
                          (obj["alert_id"], int(obj["model"]["stage"]), obj["model"]["name"], obj["decision"]["flagged"],
                           json.dumps(obj)))

    def get_explanation(self, alert_id):
        r = self.conn.execute("SELECT body FROM explanations WHERE alert_id = %s", (alert_id,)).fetchone()
        return r[0] if r else None

    def list_alerts(self):
        return [r[0] for r in self.conn.execute("SELECT alert_id FROM explanations ORDER BY alert_id").fetchall()]

    def save_report(self, alert_id, report, citations, provider):
        r = self.conn.execute("INSERT INTO reports (alert_id, body, citations, provider) VALUES (%s, %s, %s, %s) RETURNING id",
                              (alert_id, json.dumps(report), citations, provider)).fetchone()
        return str(r[0])
