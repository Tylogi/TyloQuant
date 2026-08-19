"""Minimal MCP client for managed stdio and Streamable HTTP tool servers."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from typing import Any

import httpx

from mfq.server.models import (
    McpServerResource,
    McpToolCallResult,
    McpToolResource,
    McpTransport,
)

PROTOCOL_VERSION = "2025-06-18"
MAX_MESSAGE_BYTES = 4 * 1024 * 1024


class McpError(RuntimeError):
    pass


def qualified_tool_name(server: str, tool: str) -> str:
    return f"{server}.{tool}"


class McpClient:
    def __init__(self, server: McpServerResource) -> None:
        self.server = server

    async def list_tools(self) -> list[McpToolResource]:
        result = await self._session_request("tools/list", {})
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise McpError("MCP tools/list returned no tools array")
        parsed: list[McpToolResource] = []
        for item in tools:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise McpError("MCP tools/list returned an invalid tool")
            parsed.append(
                McpToolResource(
                    server_id=self.server.id,
                    server=self.server.name,
                    name=item["name"],
                    qualified_name=qualified_tool_name(self.server.name, item["name"]),
                    description=item.get("description"),
                    input_schema=item.get("inputSchema") or {"type": "object"},
                )
            )
        return parsed

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolCallResult:
        result = await self._session_request("tools/call", {"name": name, "arguments": arguments})
        content = result.get("content") or []
        if not isinstance(content, list):
            raise McpError("MCP tools/call returned invalid content")
        structured = result.get("structuredContent")
        return McpToolCallResult(
            server=self.server.name,
            name=name,
            content=[item for item in content if isinstance(item, dict)],
            structured_content=structured if isinstance(structured, dict) else None,
            is_error=result.get("isError") is True,
        )

    async def _session_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.server.transport == McpTransport.STDIO:
            return await self._stdio_session(method, params)
        return await self._http_session(method, params)

    async def _stdio_session(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.server.command:
            raise McpError("MCP stdio command is missing")
        process = await asyncio.create_subprocess_exec(
            self.server.command,
            *self.server.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
            limit=MAX_MESSAGE_BYTES,
        )
        try:
            await self._stdio_exchange(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "mfq-server", "version": "1.0"},
                    },
                },
            )
            await self._stdio_send(
                process,
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            response = await self._stdio_exchange(
                process,
                {"jsonrpc": "2.0", "id": 2, "method": method, "params": params},
            )
            return self._result(response)
        finally:
            if process.returncode is None:
                process.terminate()
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=2.0)
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def _stdio_send(
        self, process: asyncio.subprocess.Process, message: dict[str, Any]
    ) -> None:
        assert process.stdin is not None
        process.stdin.write(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n")
        await process.stdin.drain()

    async def _stdio_exchange(
        self, process: asyncio.subprocess.Process, message: dict[str, Any]
    ) -> dict[str, Any]:
        await self._stdio_send(process, message)
        assert process.stdout is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.server.timeout_seconds
        while loop.time() < deadline:
            try:
                line = await asyncio.wait_for(
                    process.stdout.readline(), timeout=max(0.01, deadline - loop.time())
                )
            except (asyncio.TimeoutError, ValueError) as error:
                raise McpError("MCP stdio request timed out or exceeded the size limit") from error
            if not line:
                raise McpError("MCP stdio server closed unexpectedly")
            if len(line) > MAX_MESSAGE_BYTES:
                raise McpError("MCP stdio response exceeded the size limit")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as error:
                raise McpError("MCP stdio server returned invalid JSON") from error
            if not isinstance(response, dict):
                raise McpError("MCP stdio response must be an object")
            if response.get("id") == message.get("id"):
                return response
        raise McpError("MCP stdio request timed out")

    async def _http_session(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.server.url:
            raise McpError("MCP HTTP URL is missing")
        headers = {
            header: os.environ[variable]
            for header, variable in self.server.header_env.items()
            if variable in os.environ
        }
        timeout = httpx.Timeout(self.server.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            initialized, session_id = await self._http_exchange(
                client,
                headers,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "mfq-server", "version": "1.0"},
                    },
                },
            )
            self._result(initialized)
            session_headers = dict(headers)
            if session_id:
                session_headers["MCP-Session-Id"] = session_id
            await self._http_notification(
                client,
                session_headers,
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            response, _ = await self._http_exchange(
                client,
                session_headers,
                {"jsonrpc": "2.0", "id": 2, "method": method, "params": params},
            )
            return self._result(response)

    async def _http_exchange(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        message: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        response = await client.post(
            self.server.url,
            headers={
                **headers,
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "MCP-Protocol-Version": PROTOCOL_VERSION,
            },
            json=message,
        )
        if response.status_code >= 400:
            raise McpError(f"MCP HTTP request failed with status {response.status_code}")
        if len(response.content) > MAX_MESSAGE_BYTES:
            raise McpError("MCP HTTP response exceeded the size limit")
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            payload = self._last_sse_json(response.text)
        else:
            try:
                payload = response.json()
            except ValueError as error:
                raise McpError("MCP HTTP server returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise McpError("MCP HTTP response must be an object")
        return payload, response.headers.get("mcp-session-id")

    async def _http_notification(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        message: dict[str, Any],
    ) -> None:
        response = await client.post(
            self.server.url,
            headers={
                **headers,
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "MCP-Protocol-Version": PROTOCOL_VERSION,
            },
            json=message,
        )
        if response.status_code >= 400:
            raise McpError(f"MCP initialized notification failed with {response.status_code}")

    @staticmethod
    def _last_sse_json(body: str) -> dict[str, Any]:
        values: list[str] = []
        for block in body.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(
                line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")
            )
            if data:
                values.append(data)
        if not values:
            raise McpError("MCP SSE response contained no data")
        try:
            payload = json.loads(values[-1])
        except json.JSONDecodeError as error:
            raise McpError("MCP SSE response returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise McpError("MCP SSE response must be an object")
        return payload

    @staticmethod
    def _result(response: dict[str, Any]) -> dict[str, Any]:
        error = response.get("error")
        if isinstance(error, dict):
            raise McpError(str(error.get("message") or "MCP request failed"))
        result = response.get("result")
        if not isinstance(result, dict):
            raise McpError("MCP response contains no result object")
        return result
