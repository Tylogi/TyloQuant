from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.backend import BackendDelta
from mfq.server.models import ResponsePerformance, SamplingParams, TokenUsage
from mfq.server.openai_compat import (
    collect_chat_completion,
    parse_chat_request,
    stream_chat_completion,
)


class _Backend:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def stream(self, **request: Any) -> AsyncIterator[BackendDelta]:
        self.requests.append(request)

        async def generate() -> AsyncIterator[BackendDelta]:
            yield BackendDelta(reasoning_delta="plan")
            yield BackendDelta(content_delta="answer")
            yield BackendDelta(finish_reason="stop")
            yield BackendDelta(
                performance=ResponsePerformance(
                    prefill_tokens=3,
                    ttft_ms=2.0,
                    prefill_ms=1.0,
                    prefill_tps=3000.0,
                    decode_ms=1.0,
                    decode_tps=2000.0,
                    generation_ms=3.0,
                    generation_tps=666.0,
                    sampling=SamplingParams(),
                )
            )
            yield BackendDelta(
                usage=TokenUsage(
                    prompt_tokens=3,
                    completion_tokens=2,
                    total_tokens=5,
                )
            )

        return generate()


class _Service:
    def __init__(self, backend: _Backend) -> None:
        self.backend = backend

    async def start(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def runtime_models(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "DeepSeek-V4-Flash"}]}


def _request(**updates: Any):
    body: dict[str, Any] = {
        "model": "DeepSeek-V4-Flash",
        "messages": [{"role": "user", "content": "hello"}],
    }
    body.update(updates)
    return parse_chat_request(body)


def test_openai_request_defaults_keep_thinking_vision_and_mtp_enabled() -> None:
    request = _request()

    assert request.sampling.enable_thinking is True
    assert request.sampling.enable_vision is True
    assert request.sampling.enable_mtp is True
    assert request.sampling.mtp_max_draft_tokens == 5


def test_openai_tools_accept_standard_strict_field_and_null_choice() -> None:
    request = _request(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                    "strict": False,
                },
            }
        ],
        tool_choice=None,
    )

    assert request.tools[0].function.strict is False
    assert request.tool_choice == "auto"


def test_nonstream_response_keeps_reasoning_out_of_visible_content() -> None:
    async def run() -> None:
        result = await collect_chat_completion(_Backend(), _request())
        message = result["choices"][0]["message"]
        assert message["reasoning_content"] == "plan"
        assert message["content"] == "answer"
        assert result["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }

    asyncio.run(run())


def test_stream_uses_one_id_and_separate_reasoning_and_content_deltas() -> None:
    async def run() -> None:
        values = [
            value
            async for value in stream_chat_completion(
                _Backend(),
                _request(stream=True, stream_options={"include_usage": True}),
            )
        ]
        assert values[-1] == "data: [DONE]\n\n"
        payloads = [
            json.loads(value.removeprefix("data: "))
            for value in values[:-1]
            if value.startswith("data: ")
        ]
        ids = {payload["id"] for payload in payloads}
        assert len(ids) == 1
        assert payloads[0]["choices"][0]["delta"] == {
            "role": "assistant",
            "content": "",
        }
        visible = [
            payload["choices"][0]["delta"]
            for payload in payloads
            if payload["choices"] and payload["choices"][0]["delta"]
        ]
        assert {"reasoning_content": "plan"} in visible
        assert {"content": "answer"} in visible
        assert all("<think>" not in json.dumps(payload) for payload in payloads)
        usage_payloads = [payload for payload in payloads if payload["choices"] == []]
        assert len(usage_payloads) == 1
        assert usage_payloads[0]["usage"]["total_tokens"] == 5
        assert usage_payloads[0]["mfq_metrics"]["prefill_tokens"] == 3

    asyncio.run(run())


def test_stream_keepalive_detects_disconnect_and_closes_backend() -> None:
    class SlowBackend:
        closed = False

        def stream(self, **_: Any) -> AsyncIterator[BackendDelta]:
            async def generate() -> AsyncIterator[BackendDelta]:
                try:
                    await asyncio.sleep(60)
                    yield BackendDelta(content_delta="too late")
                finally:
                    self.closed = True

            return generate()

    async def run() -> None:
        backend = SlowBackend()
        disconnect_checks = 0

        async def disconnected() -> bool:
            nonlocal disconnect_checks
            disconnect_checks += 1
            return disconnect_checks >= 2

        stream = stream_chat_completion(
            backend,  # type: ignore[arg-type]
            _request(stream=True),
            disconnected=disconnected,
            keepalive_seconds=0.001,
        )
        assert "\"role\":\"assistant\"" in await anext(stream)
        assert await anext(stream) == ": keep-alive\n\n"
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert backend.closed

    asyncio.run(run())


def test_public_openai_routes_have_readable_models_and_standard_messages() -> None:
    async def run() -> None:
        backend = _Backend()
        app = create_app(_Service(backend))  # type: ignore[arg-type]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            models = await client.get("/v1/models")
            assert models.status_code == 200
            assert models.json() == {
                "object": "list",
                "data": [
                    {
                        "id": "DeepSeek-V4-Flash",
                        "object": "model",
                        "created": 0,
                        "owned_by": "mfq",
                    }
                ],
            }
            completion = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "DeepSeek-V4-Flash",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert completion.status_code == 200
            message = completion.json()["choices"][0]["message"]
            assert message == {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "plan",
            }

    asyncio.run(run())


def test_openai_routes_share_api_key_protection() -> None:
    async def run() -> None:
        app = create_app(_Service(_Backend()), api_key="unit-key")  # type: ignore[arg-type]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            denied = await client.get("/v1/models")
            assert denied.status_code == 401
            allowed = await client.get(
                "/v1/models",
                headers={"Authorization": "Bearer unit-key"},
            )
            assert allowed.status_code == 200

    asyncio.run(run())
