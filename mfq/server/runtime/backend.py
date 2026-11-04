"""Python API adapter for MFQ native runtime transports."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from mfq.server.protocol.input_protocols import render_preformatted_prompt
from mfq.server.protocol.models import (
    ModelCapabilities,
    ResponseFormat,
    ResponsePerformance,
    RuntimeCapabilitiesResource,
    SamplingParams,
    TokenUsage,
    ToolChoice,
    ToolDefinition,
)
from mfq.server.protocol.output_protocols import (
    ParsedToolCall,
    output_protocol_for_architecture,
)
from mfq.server.runtime.capabilities import capabilities_for_architecture
from mfq.server.runtime.client import (
    BackendError,
    BackendProtocolError,
    RuntimeClient,
)
from mfq.server.vision import (
    MiniCPMO45VisionProcessor,
    VisionProcessingError,
    multimodal_processor_for_architecture,
)


@dataclass(frozen=True)
class BackendToolCallDelta:
    index: int
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


@dataclass(frozen=True)
class BackendDelta:
    content_delta: str = ""
    reasoning_delta: str = ""
    tool_calls: tuple[BackendToolCallDelta, ...] = ()
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    performance: ResponsePerformance | None = None
    backend_request_id: str | None = None


class ChatBackend(Protocol):
    async def preflight(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
    ) -> None: ...

    def stream(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
        priority: int = 0,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[BackendDelta]: ...

    async def fork_session(self, source_session_id: UUID, target_session_id: UUID) -> bool: ...

    async def close_session(self, session_id: UUID) -> bool: ...

    async def cancel_response(self, session_id: UUID) -> bool: ...

    async def capabilities(self) -> RuntimeCapabilitiesResource: ...

    async def runtime_status(self) -> dict[str, Any]: ...

    async def runtime_models(self) -> dict[str, Any]: ...

    async def realtime_capabilities(self) -> dict[str, Any]: ...

    async def reload_runtime(self, context_size: int) -> dict[str, Any]: ...

    async def clear_runtime_cache(self) -> dict[str, Any]: ...

    async def trim_runtime_cache(self, target_bytes: int = 0) -> dict[str, Any]: ...

    def realtime_connect(self, *, mode: str = "audio") -> Any: ...

    async def aclose(self) -> None: ...


async def preflight_backend_request(
    backend: ChatBackend,
    *,
    model: str,
    messages: Sequence[dict[str, Any]],
    sampling: SamplingParams,
    session_id: UUID | None = None,
    tools: Sequence[ToolDefinition] = (),
    tool_choice: ToolChoice = "auto",
    response_format: ResponseFormat | None = None,
) -> None:
    """Run an optional cheap readiness gate before an HTTP stream commits."""

    preflight = getattr(backend, "preflight", None)
    if not callable(preflight):
        return
    await preflight(
        model=model,
        messages=messages,
        sampling=sampling,
        session_id=session_id,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
    )


@asynccontextmanager
async def closing_backend_stream(
    stream: AsyncIterator[BackendDelta],
) -> AsyncIterator[AsyncIterator[BackendDelta]]:
    """Close a nested backend iterator when its forwarding stream stops early."""

    try:
        yield stream
    finally:
        close = getattr(stream, "aclose", None)
        if callable(close):
            await close()


class OpenAIChatBackend:
    """Translate Python-owned chat semantics to native runtime operations."""

    def __init__(
        self,
        runtime: RuntimeClient,
        *,
        avfoundation_video_library: str | Path | None = None,
        local_tensor_files: bool = False,
        model_type: str | None = None,
    ) -> None:
        self._runtime = runtime
        # Managed local workers already have a canonical architecture from the
        # model catalog.  Seed it here so the first text-only request selects
        # the correct prompt/output protocol without depending on a prior
        # /health request from the UI.
        self._model_type = model_type
        self._vision_processor = MiniCPMO45VisionProcessor(
            avfoundation_library=avfoundation_video_library
        )
        self._avfoundation_video_library = avfoundation_video_library
        self._local_tensor_files = local_tensor_files and os.name in {"posix", "nt"}
        self._runtime_metric_overrides: dict[str, dict[str, float]] = {}

    async def count_tokens(self, prompt: str) -> dict[str, Any]:
        result = await self._runtime.count_tokens({"prompt": prompt})
        if not isinstance(result, dict) or result.get("tokenization") != "raw-no-special-tokens":
            raise BackendProtocolError("invalid native token count response")
        for key in ("prompt_tokens", "context_size"):
            if type(result.get(key)) is not int or result[key] < 1:
                raise BackendProtocolError(f"invalid native {key}")
        return result

    async def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        import math
        result = await self._runtime.score(payload)
        scores = result.get("scores")
        if result.get("log_base") != "e" or not isinstance(result.get("prompt_tokens"), int) or result["prompt_tokens"] < 1 or not isinstance(scores, list) or len(scores) != len(payload["continuations"]):
            raise BackendProtocolError("invalid native probability response")
        for score in scores:
            if not isinstance(score, dict):
                raise BackendProtocolError("invalid native probability row")
            tokens, values, total = score.get("token_ids"), score.get("token_logprobs"), score.get("log_likelihood")
            if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens) or not isinstance(values, list) or len(tokens) != len(values):
                raise BackendProtocolError("invalid native probability token alignment")
            if any(type(value) not in {int, float} or not math.isfinite(value) or value > 1e-5 for value in values) or type(total) not in {int, float} or not math.isfinite(total) or not math.isclose(total, math.fsum(values), rel_tol=1e-7, abs_tol=1e-5):
                raise BackendProtocolError("invalid native probability values")
            if payload.get("mode") == "next_token" and len(tokens) != 1:
                raise BackendProtocolError("next_token requires exactly one scored token")
        return result

    async def probability_available(self) -> bool:
        return (await self._runtime.health()).get("probability_available") is True

    async def benchmark(self, payload: dict[str, Any]) -> dict[str, Any]:
        metrics = None
        usage = None
        finish_reason = None
        async with self._runtime.generate(payload) as events:
            async for event in events:
                if event is None:
                    continue
                if event.get("mfq_metrics") is not None:
                    metrics = event["mfq_metrics"]
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        if not isinstance(metrics, dict) or not isinstance(usage, dict):
            raise BackendProtocolError("benchmark requires native metrics and token usage")
        return {"metrics": metrics, "usage": usage, "finish_reason": finish_reason}

    async def preflight(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
    ) -> None:
        del session_id, tools, tool_choice, response_format
        if self._contains_visual_media(messages) and not sampling.enable_vision:
            raise BackendError(
                "vision_disabled",
                "vision is disabled for this request; enable Vision to send images or video",
                status_code=400,
            )
        capabilities = await self.capabilities()
        if model == capabilities.model:
            return
        models = await self.runtime_models()
        data = models.get("data", []) if isinstance(models, dict) else []
        available = {
            str(item.get("id"))
            for item in data
            if isinstance(item, dict) and item.get("id")
        }
        if model not in available:
            raise BackendError(
                "model_not_loaded",
                f"model is not loaded: {model}",
                status_code=404,
            )

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
        priority: int = 0,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[BackendDelta]:
        output_protocol = output_protocol_for_architecture(
            self._model_type or model
        )
        reasoning_parser = (
            output_protocol.create_reasoning_parser()
            if sampling.enable_thinking
            else None
        )
        tool_call_parser = (
            output_protocol.create_tool_call_parser(
                {
                    tool.function.name: tool.function.parameters
                    for tool in tools
                }
            )
            if tools and tool_choice != "none"
            else None
        )
        pending_protocol_calls: list[ParsedToolCall] = []
        native_tool_calls_seen = False
        next_protocol_call_index = 0
        last_backend_request_id: str | None = None
        backend_messages = list(messages)
        multimodal: dict[str, Any] | None = None
        cleanup_paths: tuple[Path, ...] = ()
        processor_ms = 0.0
        if self._contains_visual_media(backend_messages) and not sampling.enable_vision:
            raise BackendError(
                "vision_disabled",
                "vision is disabled for this request; enable Vision to send images or video",
            )
        if self._contains_media(backend_messages):
            if self._model_type is None:
                await self.capabilities()
            processor = (
                self._vision_processor
                if self._model_type == "minicpmo"
                else multimodal_processor_for_architecture(
                    self._model_type or "",
                    avfoundation_library=self._avfoundation_video_library,
                )
            )
            if processor is not None:
                processor_started = time.perf_counter()
                try:
                    processed = await asyncio.to_thread(
                        processor.prepare_openai_messages,
                        backend_messages,
                        use_binary_file=self._local_tensor_files,
                    )
                except VisionProcessingError as error:
                    raise BackendError("media_processing_error", str(error)) from error
                processor_ms = (time.perf_counter() - processor_started) * 1000.0
                if processed is not None:
                    backend_messages = processed.messages
                    multimodal = processed.tensors
                    cleanup_paths = processed.cleanup_paths
            else:
                raise BackendError(
                    "media_processing_unsupported",
                    f"the loaded {self._model_type or 'unknown'} model has no registered media processor",
                )
        payload = {
            "model": model,
            "messages": backend_messages,
            "max_tokens": sampling.max_tokens,
            "temperature": sampling.temperature,
            "top_k": sampling.top_k,
            "top_p": sampling.top_p,
            "presence_penalty": sampling.presence_penalty,
            "frequency_penalty": sampling.frequency_penalty,
            "repetition_penalty": sampling.repetition_penalty,
            "enable_vision": sampling.enable_vision,
            "enable_mtp": sampling.enable_mtp,
            "mtp_max_draft_tokens": sampling.mtp_max_draft_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Protocol parsing belongs at this API boundary rather than in an
            # architecture runtime, so every backend presents one OpenAI-
            # compatible reasoning and tool-call contract.
            "reasoning_format": output_protocol.reasoning_format,
            "chat_template_kwargs": {
                "enable_thinking": sampling.enable_thinking,
            },
        }
        if sampling.reasoning_effort:
            payload["chat_template_kwargs"]["reasoning_effort"] = sampling.reasoning_effort
        if sampling.seed is not None:
            payload["seed"] = sampling.seed
        if tools:
            payload["tools"] = [tool.model_dump(mode="json", by_alias=True) for tool in tools]
            payload["tool_choice"] = (
                tool_choice if isinstance(tool_choice, str) else tool_choice.model_dump(mode="json")
            )
        if response_format is not None and response_format.type != "text":
            payload["response_format"] = response_format.model_dump(mode="json", by_alias=True)
        if session_id is not None:
            payload["mfq_session_id"] = str(session_id)
        if priority != 0:
            payload["priority"] = priority
        if timeout_seconds is not None:
            # The deadline is relative so clock drift between the server and a
            # worker cannot move it; the server already subtracted its queue
            # wait from the client timeout.
            payload["mfq_timeout_seconds"] = timeout_seconds
        if multimodal is not None:
            payload["mfq_multimodal"] = multimodal
        preformatted_prompt = render_preformatted_prompt(
            self._model_type or model,
            backend_messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            enable_thinking=sampling.enable_thinking,
            reasoning_effort=sampling.reasoning_effort,
            parallel_tool_calls=True,
        )
        if preformatted_prompt is not None:
            # Processor-owned prompt protocols are selected by the registry;
            # messages and tools remain present for native output constraints.
            payload["mfq_preformatted_prompt"] = preformatted_prompt
        try:
            async with self._runtime.generate(payload) as events:
                saw_done = False
                async for data in events:
                    if data is None:
                        trailing_reasoning = ""
                        trailing_content = ""
                        if reasoning_parser is not None:
                            trailing_reasoning, trailing_content = reasoning_parser.finish()
                            reasoning_parser = None
                        if tool_call_parser is not None:
                            try:
                                visible, parsed = tool_call_parser.feed(trailing_content)
                                final_visible, final_parsed = tool_call_parser.finish()
                            except ValueError as error:
                                raise BackendProtocolError(
                                    "invalid "
                                    f"{output_protocol.tool_call_protocol_name} "
                                    f"tool-call output: {error}"
                                ) from error
                            trailing_content = visible + final_visible
                            pending_protocol_calls.extend(parsed)
                            pending_protocol_calls.extend(final_parsed)
                            tool_call_parser = None
                        parsed_tool_deltas: tuple[BackendToolCallDelta, ...] = ()
                        if pending_protocol_calls and not native_tool_calls_seen:
                            parsed_tool_deltas = self._parsed_tool_call_deltas(
                                pending_protocol_calls,
                                start_index=next_protocol_call_index,
                            )
                            next_protocol_call_index += len(parsed_tool_deltas)
                        pending_protocol_calls.clear()
                        if trailing_reasoning or trailing_content or parsed_tool_deltas:
                            yield BackendDelta(
                                content_delta=trailing_content,
                                reasoning_delta=trailing_reasoning,
                                tool_calls=parsed_tool_deltas,
                                finish_reason=(
                                    "tool_calls" if parsed_tool_deltas else None
                                ),
                                backend_request_id=last_backend_request_id,
                            )
                        saw_done = True
                        break
                    delta = self._parse_event(data)
                    if delta.backend_request_id is not None:
                        last_backend_request_id = delta.backend_request_id
                    if reasoning_parser is not None:
                        if delta.reasoning_delta:
                            # A backend with a working structured parser is
                            # authoritative; do not parse its content twice.
                            reasoning_parser = None
                        else:
                            reasoning, content = reasoning_parser.feed(
                                delta.content_delta
                            )
                            if delta.finish_reason is not None:
                                trailing_reasoning, trailing_content = (
                                    reasoning_parser.finish()
                                )
                                reasoning += trailing_reasoning
                                content += trailing_content
                                reasoning_parser = None
                            delta = replace(
                                delta,
                                content_delta=content,
                                reasoning_delta=reasoning,
                            )
                    if delta.tool_calls:
                        native_tool_calls_seen = True
                    if tool_call_parser is not None:
                        try:
                            content, parsed = tool_call_parser.feed(delta.content_delta)
                            pending_protocol_calls.extend(parsed)
                            if delta.finish_reason is not None:
                                trailing, parsed = tool_call_parser.finish()
                                content += trailing
                                pending_protocol_calls.extend(parsed)
                                tool_call_parser = None
                        except ValueError as error:
                            raise BackendProtocolError(
                                "invalid "
                                f"{output_protocol.tool_call_protocol_name} "
                                f"tool-call output: {error}"
                            ) from error

                        parsed_tool_deltas = ()
                        finish_reason = delta.finish_reason
                        if finish_reason is not None and pending_protocol_calls:
                            if not native_tool_calls_seen:
                                parsed_tool_deltas = self._parsed_tool_call_deltas(
                                    pending_protocol_calls,
                                    start_index=next_protocol_call_index,
                                )
                                next_protocol_call_index += len(parsed_tool_deltas)
                                finish_reason = "tool_calls"
                            pending_protocol_calls.clear()
                        delta = replace(
                            delta,
                            content_delta=content,
                            tool_calls=delta.tool_calls + parsed_tool_deltas,
                            finish_reason=finish_reason,
                        )
                    if delta.performance is not None and processor_ms > 0.0:
                        performance = self._with_processor_timing(
                            delta.performance,
                            processor_ms,
                        )
                        delta = replace(delta, performance=performance)
                        if delta.backend_request_id:
                            self._remember_runtime_metric_override(
                                delta.backend_request_id,
                                performance,
                            )
                    yield delta
                if not saw_done:
                    raise BackendProtocolError("backend stream ended without [DONE]")
        finally:
            for path in cleanup_paths:
                path.unlink(missing_ok=True)

    async def aclose(self) -> None:
        await self._runtime.aclose()

    @staticmethod
    def _parsed_tool_call_deltas(
        calls: Sequence[ParsedToolCall],
        *,
        start_index: int,
    ) -> tuple[BackendToolCallDelta, ...]:
        return tuple(
            BackendToolCallDelta(
                index=start_index + offset,
                call_id=f"call_{uuid4().hex}",
                name=call.name,
                arguments_delta=call.arguments,
            )
            for offset, call in enumerate(calls)
        )

    async def fork_session(self, source_session_id: UUID, target_session_id: UUID) -> bool:
        return await self._runtime.fork_session(
            str(source_session_id),
            str(target_session_id),
        )

    async def close_session(self, session_id: UUID) -> bool:
        return await self._runtime.close_session(str(session_id))

    async def cancel_response(self, session_id: UUID) -> bool:
        # A stop can race the native request becoming visible after Python-side
        # media preprocessing. Briefly retry so cancellation remains reliable
        # at that boundary without delaying an already-active decode.
        for attempt in range(20):
            try:
                if await self._runtime.cancel_response(str(session_id)):
                    return True
            except BackendError as error:
                if error.retryable:
                    return False
                raise
            if attempt < 19:
                await asyncio.sleep(0.025)
        return False

    async def capabilities(self) -> RuntimeCapabilitiesResource:
        payload = await self._runtime.health()
        model = str(payload.get("model") or "mfq-model")
        model_type = str(payload.get("model_type") or "unknown")
        self._model_type = model_type
        raw_capabilities = payload.get("model_capabilities")
        try:
            capabilities = (
                ModelCapabilities.model_validate(raw_capabilities)
                if isinstance(raw_capabilities, dict)
                else capabilities_for_architecture(model_type)
            )
        except ValueError as error:
            raise BackendProtocolError(
                f"backend returned invalid model capabilities: {error}"
            ) from error
        return RuntimeCapabilitiesResource(
            model=model,
            model_type=model_type,
            model_capabilities=capabilities,
            vision_available=bool(
                payload.get("vision_available", False)
                or payload.get("video_available", False)
            ),
            mtp_available=bool(payload.get("mtp_available", False)),
            duplex_available=payload.get("duplex_available") is True,
            batch_capacity=max(1, int(payload.get("batch_capacity") or 1)),
        )

    @staticmethod
    def _contains_media(messages: Sequence[dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            if any(
                isinstance(item, dict)
                and item.get("type") in {"image_url", "video_url", "input_audio"}
                for item in content
            ):
                return True
        return False

    @staticmethod
    def _contains_visual_media(messages: Sequence[dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            if any(
                isinstance(item, dict)
                and item.get("type") in {"image_url", "video_url"}
                for item in content
            ):
                return True
        return False

    async def runtime_status(self) -> dict[str, Any]:
        try:
            status = await self._runtime.status()
        except BackendError as error:
            if error.code not in {"backend_http_404", "not_found"}:
                raise
            return {**(await self._runtime.health()), "limited": True}
        last_request = status.get("last_request")
        if isinstance(last_request, dict):
            request_id = last_request.get("id")
            override = (
                self._runtime_metric_overrides.get(request_id)
                if isinstance(request_id, str)
                else None
            )
            if override is not None:
                status = dict(status)
                status["last_request"] = {**last_request, **override}
        return status

    async def runtime_health(self) -> dict[str, Any]:
        """Expose engine health independently of older status metric payloads."""
        return await self._runtime.health()

    @staticmethod
    def _with_processor_timing(
        performance: ResponsePerformance,
        processor_ms: float,
    ) -> ResponsePerformance:
        # The established product-level multimodal Prefill boundary ends at
        # first-token availability.  Keep native model timings intact and add
        # the Python media preparation that happened before the native request.
        complete_prefill_ms = processor_ms + performance.ttft_ms
        complete_prefill_tps = (
            1000.0 * performance.prefill_tokens / complete_prefill_ms
            if performance.prefill_tokens > 0 and complete_prefill_ms > 0.0
            else 0.0
        )
        return performance.model_copy(
            update={
                "processor_ms": processor_ms,
                "complete_prefill_ms": complete_prefill_ms,
                "complete_prefill_tps": complete_prefill_tps,
                "complete_generation_ms": processor_ms + performance.generation_ms,
            }
        )

    def _remember_runtime_metric_override(
        self,
        request_id: str,
        performance: ResponsePerformance,
    ) -> None:
        self._runtime_metric_overrides[request_id] = {
            "processor_ms": performance.processor_ms,
            "complete_prefill_ms": performance.complete_prefill_ms,
            "complete_prefill_tps": performance.complete_prefill_tps,
            "complete_generation_ms": performance.complete_generation_ms,
        }
        while len(self._runtime_metric_overrides) > 128:
            self._runtime_metric_overrides.pop(next(iter(self._runtime_metric_overrides)))

    async def runtime_models(self) -> dict[str, Any]:
        return await self._runtime.models()

    async def realtime_capabilities(self) -> dict[str, Any]:
        return await self._runtime.realtime_capabilities()

    async def reload_runtime(self, context_size: int) -> dict[str, Any]:
        return await self._runtime.reload(context_size)

    async def clear_runtime_cache(self) -> dict[str, Any]:
        return await self._runtime.clear_cache()

    async def trim_runtime_cache(self, target_bytes: int = 0) -> dict[str, Any]:
        if target_bytes < 0:
            raise ValueError("target_bytes must be non-negative")
        return await self._runtime.trim_cache(target_bytes)

    async def set_prefix_cache_budget(self, target_bytes: int, disk_target_bytes: int | None = None,
                                    ram_target_bytes: int | None = None) -> dict[str, Any]:
        if any(value is not None and value < 0 for value in (target_bytes, disk_target_bytes, ram_target_bytes)):
            raise ValueError("cache budgets must be non-negative")
        if ram_target_bytes is not None:
            return await self._runtime.set_cache_budget(target_bytes, disk_target_bytes, ram_target_bytes)
        if disk_target_bytes is not None:
            return await self._runtime.set_cache_budget(target_bytes, disk_target_bytes)
        return await self._runtime.set_cache_budget(target_bytes)

    async def refresh_prefix_cache_index(self) -> dict[str, Any]:
        return await self._runtime.refresh_cache_index()

    async def set_resident_memory_budget(self, target_bytes: int, weight_target_bytes: int | None = None) -> dict[str, Any]:
        if target_bytes < 0 or weight_target_bytes is not None and weight_target_bytes < 0:
            raise ValueError("resident budget must be non-negative")
        if weight_target_bytes is not None:
            return await self._runtime.set_resident_memory_budget(target_bytes, weight_target_bytes)
        return await self._runtime.set_resident_memory_budget(target_bytes)

    def realtime_connect(self, *, mode: str = "audio") -> Any:
        return self._runtime.realtime_connect(mode=mode)

    @staticmethod
    def _parse_event(event: dict[str, Any]) -> BackendDelta:
        raw_request_id = event.get("id")
        backend_request_id = raw_request_id if isinstance(raw_request_id, str) else None
        if "error" in event:
            detail = event["error"]
            if isinstance(detail, dict):
                raw_status = detail.get("status_code")
                raise BackendError(
                    str(detail.get("code") or detail.get("type") or "backend_error"),
                    str(detail.get("message") or "backend request failed"),
                    retryable=detail.get("retryable") is True,
                    status_code=(
                        raw_status
                        if isinstance(raw_status, int) and not isinstance(raw_status, bool)
                        else None
                    ),
                )
            raise BackendError("backend_error", str(detail))

        usage = None
        raw_usage = event.get("usage")
        if raw_usage is not None:
            try:
                usage = TokenUsage.model_validate(raw_usage)
            except ValueError as error:
                raise BackendProtocolError(f"invalid backend usage object: {error}") from error
        performance = None
        raw_performance = event.get("mfq_metrics")
        if raw_performance is not None:
            try:
                performance = ResponsePerformance.model_validate(raw_performance)
            except ValueError as error:
                raise BackendProtocolError(
                    f"invalid backend performance object: {error}"
                ) from error

        choices = event.get("choices")
        if choices == []:
            return BackendDelta(
                usage=usage,
                performance=performance,
                backend_request_id=backend_request_id,
            )
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise BackendProtocolError("backend SSE event must contain one choice or usage")
        choice = choices[0]
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise BackendProtocolError("backend choice delta must be an object")
        tool_calls = OpenAIChatBackend._parse_tool_call_deltas(delta.get("tool_calls"))
        content = delta.get("content") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if not isinstance(content, str) or not isinstance(reasoning, str):
            raise BackendProtocolError("backend text deltas must be strings")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise BackendProtocolError("backend finish_reason must be a string or null")
        return BackendDelta(
            content_delta=content,
            reasoning_delta=reasoning,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            performance=performance,
            backend_request_id=backend_request_id,
        )

    @staticmethod
    def _parse_tool_call_deltas(value: Any) -> tuple[BackendToolCallDelta, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise BackendProtocolError("backend tool_calls delta must be an array")
        parsed: list[BackendToolCallDelta] = []
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("index"), int):
                raise BackendProtocolError("backend tool call delta requires an integer index")
            function = item.get("function") or {}
            if not isinstance(function, dict):
                raise BackendProtocolError("backend tool call function delta must be an object")
            call_id = item.get("id")
            name = function.get("name")
            arguments = function.get("arguments") or ""
            if call_id is not None and not isinstance(call_id, str):
                raise BackendProtocolError("backend tool call id must be a string")
            if name is not None and not isinstance(name, str):
                raise BackendProtocolError("backend tool call name must be a string")
            if not isinstance(arguments, str):
                raise BackendProtocolError("backend tool call arguments delta must be a string")
            parsed.append(
                BackendToolCallDelta(
                    index=item["index"],
                    call_id=call_id,
                    name=name,
                    arguments_delta=arguments,
                )
            )
        return tuple(parsed)
