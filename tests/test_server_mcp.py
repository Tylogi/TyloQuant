from __future__ import annotations

import asyncio
import stat
import textwrap
from pathlib import Path

import pytest

from mfq.server.mcp import McpClient
from mfq.server.models import (
    CreateMcpServerRequest,
    McpToolCallRequest,
    McpTransport,
)
from mfq.server.service import ServerService, ServiceError
from mfq.server.storage import SessionStore
from tests.test_server_service import FakeBackend


def _fake_mcp(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import sys

            for line in sys.stdin:
                request = json.loads(line)
                if 'id' not in request:
                    continue
                if request['method'] == 'initialize':
                    result = {
                        'protocolVersion': '2025-06-18',
                        'capabilities': {'tools': {}},
                        'serverInfo': {'name': 'fake', 'version': '1'},
                    }
                elif request['method'] == 'tools/list':
                    result = {'tools': [{
                        'name': 'echo',
                        'description': 'Echo text',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {'text': {'type': 'string'}},
                            'required': ['text'],
                        },
                    }]}
                elif request['method'] == 'tools/call':
                    text = request['params']['arguments']['text']
                    result = {
                        'content': [{'type': 'text', 'text': text}],
                        'structuredContent': {'echo': text},
                    }
                print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}), flush=True)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_stdio_mcp_lists_and_calls_tools(tmp_path: Path) -> None:
    async def scenario() -> None:
        executable = tmp_path / "fake-mcp"
        _fake_mcp(executable)
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        server = store.create_mcp_server(
            CreateMcpServerRequest(
                name="local",
                transport=McpTransport.STDIO,
                enabled=True,
                command=str(executable),
            )
        )
        tools = await McpClient(server).list_tools()
        assert tools[0].qualified_name == "local.echo"
        result = await McpClient(server).call_tool("echo", {"text": "MFQ"})
        assert result.structured_content == {"echo": "MFQ"}

    asyncio.run(scenario())


def test_service_requires_confirmation_and_audits_tool_calls(tmp_path: Path) -> None:
    async def scenario() -> None:
        executable = tmp_path / "fake-mcp"
        _fake_mcp(executable)
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        service = ServerService(store, FakeBackend())
        await service.create_mcp_server(
            CreateMcpServerRequest(
                name="local",
                transport=McpTransport.STDIO,
                enabled=True,
                command=str(executable),
            )
        )
        listed = await service.list_mcp_tools()
        assert listed.data[0].qualified_name == "local.echo"
        with pytest.raises(ServiceError) as rejected:
            await service.call_mcp_tool(
                McpToolCallRequest(name="local.echo", arguments={"text": "MFQ"})
            )
        assert rejected.value.detail.code == "tool_confirmation_required"
        result = await service.call_mcp_tool(
            McpToolCallRequest(name="local.echo", arguments={"text": "MFQ"}, confirm=True)
        )
        assert result.content[0]["text"] == "MFQ"
        logs = store.list_runtime_logs()
        assert logs[-1].message == "MCP tool call completed"
        assert "MFQ" not in logs[-1].model_dump_json()

    asyncio.run(scenario())


def test_mcp_configuration_stores_env_names_not_secret_values(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "mfq.server.sqlite3")
    server = store.create_mcp_server(
        CreateMcpServerRequest(
            name="remote",
            transport=McpTransport.STREAMABLE_HTTP,
            url="https://example.invalid/mcp",
            header_env={"Authorization": "MCP_AUTHORIZATION"},
        )
    )
    assert server.header_env == {"Authorization": "MCP_AUTHORIZATION"}
    assert "secret" not in (tmp_path / "mfq.server.sqlite3").read_bytes().decode(
        "latin-1", errors="ignore"
    )
