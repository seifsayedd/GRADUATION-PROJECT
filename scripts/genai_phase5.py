#!/usr/bin/env python3
"""Phase 5: grounded incident reports and analyst Q&A.

Settings come only from environment variables (see .env.example). Without
DATABASE_URL the knowledge base lives in memory and is cached in
data/interim/phase5/ (git-ignored). The default provider is "template"
(no LLM, offline); set SENTINEL_LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY
to use Claude.

Usage:
    python scripts/genai_phase5.py ingest
    python scripts/genai_phase5.py alerts                 # list alert ids
    python scripts/genai_phase5.py report --alert <id>
    python scripts/genai_phase5.py ask "Why does XGBoost flag benign lab traffic?"
    python scripts/genai_phase5.py init-db                # PostgreSQL only (needs DATABASE_URL)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinelai.genai import ingest, rag  # noqa: E402
from sentinelai.genai.config import Settings  # noqa: E402
from sentinelai.genai.embeddings import get_embedder  # noqa: E402
from sentinelai.genai.providers import get_provider  # noqa: E402
from sentinelai.genai.store import MemoryStore, PgVectorStore  # noqa: E402

CACHE = ROOT / "data" / "interim" / "phase5"
SCHEMA = ROOT / "db" / "001_init.sql"


def open_store(s: Settings, fresh: bool = False):
    if s.database_url:
        return PgVectorStore(s.database_url, s.embed_dim)
    if fresh or not (CACHE / "vectors.npy").exists():
        if not fresh:
            sys.exit("No knowledge base yet: run `python scripts/genai_phase5.py ingest` first.")
        return MemoryStore(s.embed_dim)
    return MemoryStore.load(CACHE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    sub.add_parser("ingest")
    sub.add_parser("alerts")
    r = sub.add_parser("report")
    r.add_argument("--alert", required=True)
    q = sub.add_parser("ask")
    q.add_argument("question")
    a = ap.parse_args()
    s = Settings.from_env()

    if a.cmd == "init-db":
        if not s.database_url:
            sys.exit("DATABASE_URL is not set; the in-memory store needs no schema.")
        PgVectorStore(s.database_url, s.embed_dim).init_schema(SCHEMA)
        print("schema applied")
    elif a.cmd == "ingest":
        st = open_store(s, fresh=True)
        print(json.dumps(ingest.build(ROOT, st, get_embedder(s)), indent=1))
        if isinstance(st, MemoryStore):
            st.save(CACHE)
            print(f"saved to {CACHE}")
    elif a.cmd == "alerts":
        st = open_store(s)
        for aid in st.list_alerts():
            e = st.get_explanation(aid)
            print(f"{aid:45s} {'FLAGGED' if e['decision']['flagged'] else 'not flagged':12s} {e['model']['name']}")
    else:
        st, emb, prov = open_store(s), get_embedder(s), get_provider(s)
        try:
            out = (rag.report_for_alert(st, emb, prov, a.alert, s.top_k) if a.cmd == "report"
                   else rag.ask(st, emb, prov, a.question, s.top_k))
        except rag.UngroundedOutputError as e:
            print(f"REJECTED (not stored): {e}", file=sys.stderr)
            return 2
        if out.get("fallback_used"):
            print("note: Claude declined on safety grounds and a fallback model produced this output", file=sys.stderr)
        print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
