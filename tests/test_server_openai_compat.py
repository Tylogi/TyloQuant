from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.runtime.backend import BackendDelta, BackendError
from mfq.server.protocol.models import ResponsePerformance, SamplingParams, TokenUsage
from mfq.server.api.openai_compat import (
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

    def resolve_model_alias(self, model: str) -> str:
        return model

    async def start(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def runtime_models(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "DeepSeek-V4-Flash"}]}

    async def advertised_models(self) -> dict[str, Any]:
        return await self.runtime_models()


def _request(**updates: Any):
    body: dict[str, Any] = {
        "model": "DeepSeek-V4-Flash",
        "messages": [{"role": "user", "content": "hello"}],
    }
    body.update(updates)
    return parse_chat_request(body)


@pytest.mark.parametrize("endpoint,roles", [
    ("/v1/chat/completions", ("system", "developer", "user", "assistant", "tool")),
    ("/v1/messages", ("user", "assistant")),
])
def test_message_role_validation(endpoint, roles):
    async def run():
        backend = _Backend()
        app = create_app(_Service(backend))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for role in roles:
                body = {"model": "model", "max_tokens": 8, "messages": [{"role": role, "content": "hello"}]}
                if endpoint == "/v1/messages":
                    body["system"] = "Top-level system prompt"
                response = await client.post(endpoint, json=body)
                assert response.status_code == 200, response.text
                assert backend.requests[-1]["messages"][-1]["role"] == role
            count = len(backend.requests)
            invalid_roles = ("wizard", "", None, 3, [], {}) + tuple(
                role for role in ("system", "tool") if role not in roles)
            for role in invalid_roles:
                response = await client.post(endpoint, json={"model": "model", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hello"}, {"role": role, "content": "bad"}]})
                assert response.status_code == 400
                error = response.json()["error"]
                assert error["type"] == "invalid_request_error"
                assert "messages[1].role" in error["message"]
                assert all(role in error["message"] for role in roles)
                if endpoint == "/v1/chat/completions":
                    assert error["param"] == "messages[1].role"
            assert len(backend.requests) == count
    asyncio.run(run())


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize("content_type,status", [
    ("application/json", 200), ("application/json; charset=utf-8", 200),
    ("Application/JSON", 200), ("application/vnd.client+json", 200),
    ("text/plain", 415), ("text/json", 415), ("application/octet-stream", 415),
    ("application/x-www-form-urlencoded", 415), (None, 200),
])
def test_json_content_type_validation(endpoint, content_type, status):
    async def run():
        backend = _Backend()
        app = create_app(_Service(backend))
        headers = {"Content-Type": content_type} if content_type else {}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(endpoint, headers=headers, content=json.dumps({
                "model": "model", "max_tokens": 8, "messages": [{"role": "user", "content": "hello"}]}))
        assert response.status_code == status, response.text
        assert len(backend.requests) == (1 if status == 200 else 0)
        if status != 200:
            assert response.json()["error"]["type"] == "invalid_request_error"
            assert "Content-Type" in response.json()["error"]["message"]
    asyncio.run(run())


@pytest.mark.parametrize("endpoint,temperature_max", [("/v1/chat/completions", 2), ("/v1/messages", 1)])
def test_sampling_ranges_and_json_types(endpoint, temperature_max):
    async def run():
        backend = _Backend()
        app = create_app(_Service(backend))
        cases = [(name, value, 200) for name, values in (
            ("temperature", (0, temperature_max)), ("top_p", (0, 1)), ("top_k", (0, 1)),
            ("max_tokens", (1,)), ("seed", (0, 42)),
        ) for value in values]
        cases += [(name, value, 400) for name, values in (
            ("temperature", (-0.1, temperature_max + 0.1, 3, True, "1")),
            ("top_p", (-0.1, 1.1, True, "0.5")), ("top_k", (-1, 1.5, True, "1")),
            ("max_tokens", (0, -1, 1.5, True, "8")), ("seed", (True, 1.5, "42")),
        ) for value in values]
        if endpoint == "/v1/chat/completions":
            for name in ("presence_penalty", "frequency_penalty"):
                cases += [(name, value, 200) for value in (-2, 0, 2)]
                cases += [(name, value, 400) for value in (-2.1, 2.1, True, "0")]
            cases += [("max_completion_tokens", value, 400) for value in (0, True, "8")]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for name, value, status in cases:
                count = len(backend.requests)
                response = await client.post(endpoint, json={"model": "model", "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hello"}], name: value})
                assert response.status_code == status, (name, value, response.text)
                assert len(backend.requests) == count + (1 if status == 200 else 0)
                if status == 200:
                    assert getattr(backend.requests[-1]["sampling"], name) == value
                else:
                    assert response.json()["error"]["type"] == "invalid_request_error"
                    if endpoint == "/v1/chat/completions":
                        assert response.json()["error"]["param"] == name
    asyncio.run(run())


@pytest.mark.parametrize("with_web_root", [False, True])
def test_legacy_completions_returns_actionable_openai_error(tmp_path, with_web_root):
    async def run():
        backend = _Backend()
        if with_web_root:
            (tmp_path / "index.html").write_text("<html>Studio</html>", encoding="utf-8")
        app = create_app(_Service(backend), web_root=tmp_path if with_web_root else None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/completions", json={"model": "model", "prompt": "hello"})
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "unsupported_endpoint"
        assert "/v1/chat/completions" in error["message"]
        assert not backend.requests
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_missing_model_returns_not_found_for_both_transports(tmp_path, stream) -> None:
    from mfq.server.runtime.runtime_pool import RuntimePool
    from mfq.server.state.catalog import ModelCatalog

    async def run() -> None:
        pool = RuntimePool(ModelCatalog([]), tmp_path / "runtime")
        try:
            app = create_app(_Service(pool))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/v1/chat/completions", json={"model": "missing-model",
                    "messages": [{"role": "user", "content": "hello"}], "stream": stream})
            assert response.status_code == 404
            assert response.json()["error"]["type"] == "model_not_loaded"
            assert not pool._instances
        finally:
            await pool.aclose()

    asyncio.run(run())


def test_openai_request_defaults_keep_thinking_vision_and_mtp_enabled() -> None:
    request = _request()

    assert request.sampling.enable_thinking is True
    assert request.sampling.enable_vision is True
    assert request.sampling.enable_mtp is True
    assert request.sampling.mtp_max_draft_tokens == 5
    assert "enable_mtp" not in request.sampling.model_fields_set


def test_openai_request_tracks_explicit_sampling_overrides() -> None:
    request = _request(
        enable_mtp=False,
        temperature=0.25,
        seed=None,
        chat_template_kwargs={"enable_thinking": False},
    )

    assert request.sampling.enable_mtp is False
    assert request.sampling.enable_thinking is False
    assert request.sampling.model_fields_set == {
        "enable_mtp",
        "enable_thinking",
        "seed",
        "temperature",
    }


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


def test_scheduling_extensions_are_forwarded_to_the_backend() -> None:
    async def run() -> None:
        backend = _Backend()
        await collect_chat_completion(backend, _request(priority=5, timeout_seconds=2.5))
        await collect_chat_completion(backend, _request())
        scheduled, default = backend.requests
        assert scheduled["priority"] == 5
        assert scheduled["timeout_seconds"] == 2.5
        assert default["priority"] == 0
        assert default["timeout_seconds"] is None

    asyncio.run(run())


def test_scheduling_extensions_accept_the_engine_bounds() -> None:
    highest = _request(priority=2**31 - 1, timeout_seconds=86400)
    lowest = _request(priority=-(2**31))
    assert highest.priority == 2**31 - 1
    assert highest.timeout_seconds == 86400.0
    assert lowest.priority == -(2**31)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("priority", "high"),
        ("priority", 1.5),
        ("priority", True),
        ("priority", 2**31),
        ("priority", -(2**31) - 1),
        ("timeout_seconds", "1"),
        ("timeout_seconds", 0),
        ("timeout_seconds", -1),
        ("timeout_seconds", 86400.5),
    ],
)
def test_out_of_range_scheduling_extensions_return_422(field: str, value: Any) -> None:
    async def run() -> None:
        app = create_app(_Service(_Backend()))
        body = {
            "model": "DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "hello"}],
            field: value,
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/chat/completions", json=body)
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["param"] == field

    asyncio.run(run())


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


def test_streaming_route_rejects_failed_preflight_before_committing_200() -> None:
    class UnavailableBackend(_Backend):
        async def preflight(self, **_: Any) -> None:
            raise BackendError(
                "model_not_ready",
                "model worker exited",
                retryable=True,
                status_code=503,
            )

    async def run() -> None:
        backend = UnavailableBackend()
        app = create_app(_Service(backend))  # type: ignore[arg-type]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "DeepSeek-V4-Flash",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"]["type"] == "model_not_ready"
        assert backend.requests == []

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
