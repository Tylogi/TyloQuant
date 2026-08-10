"""Generate and verify the checked-in MFQd OpenAPI contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mfqd.api import create_contract_app
from mfqd.models import PROTOCOL_VERSION

REALTIME_EVENTS = [
    "input_audio.delta",
    "input_audio.commit",
    "response.text.delta",
    "response.reasoning.delta",
    "response.tool_call.delta",
    "response.audio.delta",
    "response.interrupted",
    "response.completed",
    "session.state",
    "runtime.metrics",
    "error",
]


def build_openapi_schema() -> dict[str, Any]:
    schema: dict[str, Any] = create_contract_app().openapi()
    schema["x-mfqd-protocol-version"] = PROTOCOL_VERSION
    schema["x-mfqd-websocket"] = {
        "path": "/api/v1/realtime",
        "frameSchema": "#/components/schemas/RealtimeFrame",
        "events": REALTIME_EVENTS,
    }
    from mfqd.models import RealtimeFrame

    realtime_schema = RealtimeFrame.model_json_schema(ref_template="#/components/schemas/{model}")
    nested_schemas = realtime_schema.pop("$defs", {})
    schema["components"]["schemas"].update(nested_schemas)
    schema["components"]["schemas"]["RealtimeFrame"] = realtime_schema
    return schema


def render_openapi() -> str:
    return json.dumps(build_openapi_schema(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rendered = render_openapi()
    if args.check:
        if not args.path.is_file() or args.path.read_text(encoding="utf-8") != rendered:
            print(f"OpenAPI contract is stale: {args.path}")
            return 1
        return 0
    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
