"""Command-line entry point for the MFQ daemon."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn

from mfqd.api import create_app
from mfqd.backend import OpenAIChatBackend
from mfqd.service import MfqdService
from mfqd.storage import SessionStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    configured_web_root = os.environ.get("MFQD_WEB_ROOT")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("MFQD_DB", "mfqd.sqlite3")),
        help="SQLite conversation database (default: MFQD_DB or ./mfqd.sqlite3)",
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("MFQD_BACKEND_URL", "http://127.0.0.1:8080"),
        help="MFQ OpenAI-compatible server URL",
    )
    parser.add_argument(
        "--backend-api-key-env",
        default="MFQD_BACKEND_API_KEY",
        help="environment variable containing the backend bearer token",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--web-root",
        type=Path,
        default=Path(configured_web_root) if configured_web_root else None,
        help="optional built Web UI directory (default: MFQD_WEB_ROOT)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api_key = os.environ.get(args.backend_api_key_env, "")
    backend_host = urlsplit(args.backend_url).hostname
    service = MfqdService(
        SessionStore(args.db),
        OpenAIChatBackend(
            args.backend_url,
            api_key=api_key,
            local_tensor_files=backend_host in {"127.0.0.1", "localhost", "::1"},
        ),
    )
    uvicorn.run(
        create_app(service, web_root=args.web_root),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
