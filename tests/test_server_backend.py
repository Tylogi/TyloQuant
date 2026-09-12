from __future__ import annotations

import asyncio
import json
from uuid import UUID

import httpx
import pytest

from mfq.server.backend import BackendError, BackendProtocolError, OpenAIChatBackend
from mfq.server.models import (
    JsonSchemaResponseFormat,
    NamedToolChoice,
    SamplingParams,
    ToolDefinition,
)


def test_backend_stream_parses_cpp_sse_and_preserves_request_fields() -> None:
    captured: dict[str, object] = {}
    backend_key = "unit-test-key"

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        captured["payload"] = json.loads(request.content)
        events = [
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "think",
                            "content": "answer",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {"name": "lookup", "arguments": '{"q":'},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '"mfq"}'}}]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
                "mfq_metrics": {
                    "prefill_tokens": 4,
                    "ttft_ms": 12.0,
                    "prefill_ms": 10.0,
                    "prefill_tps": 400.0,
                    "multimodal_ms": 3.0,
                    "model_prefill_ms": 7.0,
                    "decode_ms": 2.0,
                    "decode_tps": 1500.0,
                    "generation_ms": 14.0,
                    "generation_tps": 214.0,
                    "mtp_selected_depth": 5.0,
                    "mtp_depth_5_cycles": 3.0,
                    "mtp_position_5_acceptance_rate": 0.5,
                    "mtp_depth_5_cycle_ms": 2.1,
                    "future_runtime_metric": 123.0,
                    "sampling": {
                        "max_tokens": 12,
                        "temperature": 0.25,
                        "top_k": 20,
                        "top_p": 0.8,
                        "presence_penalty": 0.0,
                        "frequency_penalty": 0.0,
                        "repetition_penalty": 1.0,
                        "seed": 7,
                    },
                },
            },
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        body += "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAIChatBackend("http://backend", api_key=backend_key, client=client)
        deltas = [
            delta
            async for delta in backend.stream(
                model="model-a",
                messages=[{"role": "user", "content": "hello"}],
                sampling=SamplingParams(
                    max_tokens=12,
                    temperature=0.25,
                    seed=7,
                    enable_thinking=True,
                    reasoning_effort="high",
                ),
                session_id=UUID("11111111-1111-4111-8111-111111111111"),
                tools=[
                    ToolDefinition.model_validate(
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "description": "Look up a value",
                                "strict": False,
                                "parameters": {
                                    "type": "object",
                                    "properties": {"q": {"type": "string"}},
                                    "required": ["q"],
                                },
                            },
                        }
                    )
                ],
                tool_choice=NamedToolChoice.model_validate(
                    {"type": "function", "function": {"name": "lookup"}}
                ),
                response_format=JsonSchemaResponseFormat.model_validate(
                    {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer",
                            "schema": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            },
                        },
                    }
                ),
            )
        ]
        await client.aclose()
        assert deltas[1].reasoning_delta == "think"
        assert deltas[1].content_delta == "answer"
        assert deltas[1].tool_calls[0].name == "lookup"
        assert deltas[2].tool_calls[0].arguments_delta == '"mfq"}'
        assert deltas[2].finish_reason == "tool_calls"
        assert deltas[3].usage is not None and deltas[3].usage.total_tokens == 7
        assert deltas[3].performance is not None
        assert deltas[3].performance.multimodal_ms == 3.0
        assert deltas[3].performance.model_prefill_ms == 7.0
        assert deltas[3].performance.mtp_selected_depth == 5
        assert deltas[3].performance.mtp_depth_5_cycles == 3
        assert deltas[3].performance.mtp_position_5_acceptance_rate == 0.5
        assert deltas[3].performance.mtp_depth_5_cycle_ms == 2.1
        assert deltas[3].performance.model_dump()["future_runtime_metric"] == 123.0

    asyncio.run(run())
    assert captured["authorization"] == f"Bearer {backend_key}"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "model-a"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["max_tokens"] == 12
    assert payload["seed"] == 7
    assert payload["enable_vision"] is True
    assert payload["enable_mtp"] is True
    assert payload["mtp_max_draft_tokens"] == 3
    assert payload["mfq_session_id"] == "11111111-1111-4111-8111-111111111111"
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "high",
    }
    assert payload["tools"][0]["function"]["name"] == "lookup"
    assert payload["tools"][0]["function"]["strict"] is False
    assert payload["tool_choice"]["function"]["name"] == "lookup"
    assert payload["response_format"]["json_schema"]["schema"]["type"] == "object"


def test_backend_splits_deepseek_v4_raw_reasoning_across_sse_chunks() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        events = [
            {"id": "native-id", "choices": [{"delta": {"content": "plan</thi"}}]},
            {"id": "native-id", "choices": [{"delta": {"content": "nk>final"}}]},
            {
                "id": "native-id",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        body += "data: [DONE]\n\n"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAIChatBackend("http://backend", client=client)
        backend._model_type = "deepseek_v4_vision"
        deltas = [
            delta
            async for delta in backend.stream(
                model="DeepSeek-V4-Flash-Vision-Exp",
                messages=[{"role": "user", "content": "hello"}],
                sampling=SamplingParams(enable_thinking=True),
            )
        ]
        await client.aclose()
        assert "".join(delta.reasoning_delta for delta in deltas) == "plan"
        assert "".join(delta.content_delta for delta in deltas) == "final"
        assert deltas[-1].finish_reason == "stop"
        assert {delta.backend_request_id for delta in deltas} == {"native-id"}

    asyncio.run(run())
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["reasoning_format"] == "none"


def test_backend_converts_deepseek_v4_dsml_content_to_openai_tool_calls() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        events = [
            {
                "id": "native-tool-id",
                "choices": [
                    {
                        "delta": {
                            "content": "I'll use it.\n\n<｜DSML｜tool_ca",
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "native-tool-id",
                "choices": [
                    {
                        "delta": {
                            "content": (
                                "lls>\n<｜DSML｜invoke name=\"write\">\n"
                                "<｜DSML｜parameter name=\"content\" string=\"true\">"
                                "hello</｜DSML｜parameter>\n"
                                "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
                            ),
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "native-tool-id",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        body += "data: [DONE]\n\n"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAIChatBackend("http://backend", client=client)
        backend._model_type = "deepseek_v4_vision"
        tool = ToolDefinition.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "Write content",
                    "parameters": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"],
                    },
                },
            }
        )
        deltas = [
            delta
            async for delta in backend.stream(
                model="DeepSeek-V4-Flash-Vision-Exp",
                messages=[{"role": "user", "content": "write hello"}],
                sampling=SamplingParams(enable_thinking=False),
                tools=[tool],
                tool_choice="required",
            )
        ]
        await client.aclose()

        content = "".join(delta.content_delta for delta in deltas)
        tool_deltas = [tool for delta in deltas for tool in delta.tool_calls]
        assert content == "I'll use it.\n\n"
        assert "DSML" not in content
        assert len(tool_deltas) == 1
        assert tool_deltas[0].index == 0
        assert tool_deltas[0].call_id is not None
        assert tool_deltas[0].call_id.startswith("call_")
        assert tool_deltas[0].name == "write"
        assert json.loads(tool_deltas[0].arguments_delta) == {"content": "hello"}
        assert deltas[-1].finish_reason == "tool_calls"

    asyncio.run(run())
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["reasoning_format"] == "none"
    assert payload["tool_choice"] == "required"


def test_backend_explicit_cancel_retries_the_native_activation_boundary() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        assert request.method == "POST"
        assert request.url.path == (
            "/api/runtime/sessions/11111111-1111-4111-8111-111111111111/cancel"
        )
        attempts += 1
        return httpx.Response(200, json={"status": "ok", "cancelled": attempts >= 2})

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAIChatBackend("http://backend", client=client)
        assert await backend.cancel_response(
            UUID("11111111-1111-4111-8111-111111111111")
        )
        await client.aclose()

    asyncio.run(run())
    assert attempts == 2


def test_backend_proxies_runtime_console_resources() -> None:
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, body))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "model-a"}]})
        if request.url.path == "/realtime/capabilities":
            return httpx.Response(200, json={"available": True})
        return httpx.Response(200, json={"model": "model-a", "max_context": 8192})

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAIChatBackend("http://backend", client=client)
        assert (await backend.runtime_status())["max_context"] == 8192
        assert (await backend.runtime_models())["data"][0]["id"] == "model-a"
        assert (await backend.realtime_capabilities())["available"] is True
        assert (await backend.reload_runtime(16384))["model"] == "model-a"
        await client.aclose()

    asyncio.run(run())
    assert requests == [
        ("GET", "/api/status", None),
        ("GET", "/v1/models", None),
        ("GET", "/realtime/capabilities", None),
        ("POST", "/api/reload", {"context_size": 16384}),
    ]


def test_backend_forwards_runtime_session_lifecycle() -> None:
    requests: list[tuple[str, str, dict[str, object] | None, str | None]] = []
    backend_key = "unit-test-key"

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        requests.append(
            (
                request.method,
                request.url.path,
                body,
                request.headers.get("authorization"),
            )
        )
        return httpx.Response(200, json={"status": "ok"})

    async def run() -> None:
        source = UUID("11111111-1111-4111-8111-111111111111")
        target = UUID("22222222-2222-4222-8222-222222222222")
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAIChatBackend("http://backend", api_key=backend_key, client=client)
        assert await backend.fork_session(source, target)
        assert await backend.close_session(target)
        await client.aclose()

    asyncio.run(run())
    assert requests == [
        (
            "POST",
            "/api/runtime/sessions/fork",
            {
                "source_session_id": "11111111-1111-4111-8111-111111111111",
                "target_session_id": "22222222-2222-4222-8222-222222222222",
            },
            f"Bearer {backend_key}",
        ),
        (
            "DELETE",
            "/api/runtime/sessions/22222222-2222-4222-8222-222222222222",
            None,
            f"Bearer {backend_key}",
        ),
    ]


def test_backend_reads_registered_model_capabilities_from_health() -> None:
    captured_authorization: list[str | None] = []
    test_credential = "-".join(("test", "credential"))

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_authorization.append(request.headers.get("authorization"))
        return httpx.Response(
            200,
            json={
                "model": "MiniCPM-o-4_5-S4-S",
                "model_type": "minicpmo",
                "duplex_available": True,
            },
        )

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAIChatBackend(
            "http://backend",
            api_key=test_credential,
            client=client,
        )
        capabilities = await backend.capabilities()
        await client.aclose()
        assert capabilities.model == "MiniCPM-o-4_5-S4-S"
        assert capabilities.model_capabilities.architecture_family == "minicpmo"
        assert capabilities.model_capabilities.features.video_input is True
        assert capabilities.model_capabilities.features.full_duplex is True
        assert capabilities.duplex_available is True

    asyncio.run(run())
    assert captured_authorization == [f"Bearer {test_credential}"]


def test_backend_session_lifecycle_is_optional_for_older_runtimes() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async def run() -> None:
        session_id = UUID("11111111-1111-4111-8111-111111111111")
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAIChatBackend("http://backend", client=client)
        assert not await backend.close_session(session_id)
        await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (
            httpx.Response(200, headers={"content-type": "text/event-stream"}, text=""),
            "backend_protocol_error",
        ),
        (
            httpx.Response(200, headers={"content-type": "application/json"}, json={}),
            "backend_protocol_error",
        ),
        (
            httpx.Response(
                503,
                json={"error": {"code": "overloaded", "message": "try later"}},
            ),
            "overloaded",
        ),
    ],
)
def test_backend_stream_rejects_incomplete_or_failed_responses(
    response: httpx.Response,
    code: str,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return response

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = OpenAIChatBackend("http://backend", client=client)
        with pytest.raises(BackendError) as caught:
            _ = [
                delta
                async for delta in backend.stream(
                    model="model-a",
                    messages=[],
                    sampling=SamplingParams(),
                )
            ]
        await client.aclose()
        assert caught.value.code == code
        if response.status_code == 503:
            assert caught.value.retryable
        else:
            assert isinstance(caught.value, BackendProtocolError)

    asyncio.run(run())
