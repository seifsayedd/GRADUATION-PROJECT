#!/usr/bin/env python3
"""Phase 6: run the SentinelAI API (and the built dashboard, if present) on 127.0.0.1.

Settings come only from environment variables (see .env.example). The default
provider is the offline template, so no API key is needed. There is no login:
keep the default host (127.0.0.1) so only this machine can reach it.

Usage:
    python scripts/serve_phase6.py     # API on http://127.0.0.1:8000/api, dashboard from dashboard/dist
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402

from sentinelai.api.app import create_app  # noqa: E402
from sentinelai.api.settings import ApiSettings  # noqa: E402


def main() -> int:
    a = ApiSettings.from_env()
    dist = ROOT / "dashboard" / "dist"
    app = create_app(ROOT, api=a, static_dir=dist)
    where = f"http://{a.host}:{a.port}"
    print(f"SentinelAI API: {where}/api/health   docs: {where}/docs   "
          + (f"dashboard: {where}/" if dist.exists() else "dashboard not built (cd dashboard && npm run build)"))
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
