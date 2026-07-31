"""Synchronous, transport-independent ownership of one TPQ chat engine."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .chat_adapters import (
    AssistantOutput,
    ChatAdapter,
    ChatMessage,
    ChatOptions,
    PromptPlan,
    StreamDelta,
)


class ChatQueueFull(RuntimeError):
    """Raised when all configured waiting slots are occupied."""


@dataclass(frozen=True)
class GenerationMetrics:
    """Content-free measurements for one completed generation."""

    request_id: str
    prompt_tokens: int
    processed_tokens: int
    completion_tokens: int
    queue_delay_ms: float
    kv_mode: str
    kv_reason: str
    prefill_ms: float | None
    ttft_ms: float | None
    generation_ms: float
    finish_reason: str
    tokens_per_second: float
    output_token_sha256: str
    periodic_tail_detected: bool
    cancelled: bool

    @property
    def token_rate(self) -> float:
        """Concise alias used by transports and diagnostics."""
        return self.tokens_per_second

    @property
    def generation_duration_ms(self) -> float:
        return self.generation_ms


@dataclass(frozen=True)
class HotConversation:
    """The adapter ledger corresponding to the engine's canonical history."""

    model: str
    adapter_name: str
    ledger: object


@dataclass(frozen=True)
class GenerationReady:
    """Content-free identity available immediately before generation."""

    request_id: str
    created: int
    model: str


@dataclass
class ChatResult:
    request_id: str
    created: int
    model: str
    output: AssistantOutput
    output_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    metrics: GenerationMetrics


@dataclass(frozen=True)
class _Admission:
    acquired: bool
    queued: bool
    admitted_at: float


def _token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _periodic_tail_detected(
    token_ids: list[int],
    *,
    repeats: int = 4,
    min_period: int = 1,
    max_period: int = 64,
) -> bool:
    """Detect an exact repeated token block without retaining its contents."""
    maximum = min(max_period, len(token_ids) // repeats)
    for period in range(min_period, maximum + 1):
        pattern = token_ids[-period:]
        if all(
            token_ids[
                len(token_ids) - (repeat + 1) * period : len(token_ids)
                - repeat * period
            ]
            == pattern
            for repeat in range(1, repeats)
        ):
            return True
    return False


def _metric_value(stats: object | None, name: str, default: Any) -> Any:
    if stats is None:
        return default
    if isinstance(stats, dict):
        return stats.get(name, default)
    return getattr(stats, name, default)


class _StopTextFilter:
    """Hide stop strings without delaying text beyond an unstable suffix."""

    def __init__(self, stops: tuple[str, ...]) -> None:
        self._stops = tuple(stop for stop in stops if stop)
        self._buffer = ""
        self._offset = 0
        self.stopped = False
        self.stop_at: int | None = None

    def feed(self, text: str) -> str:
        if self.stopped or not text:
            return ""
        self._buffer += text
        matches = [
            self._buffer.find(stop)
            for stop in self._stops
            if self._buffer.find(stop) >= 0
        ]
        if matches:
            stop_at = min(matches)
            visible = self._buffer[:stop_at]
            self.stop_at = self._offset + stop_at
            self._buffer = ""
            self.stopped = True
            return visible
        retain = max(
            (
                length
                for stop in self._stops
                for length in range(
                    1,
                    min(len(stop), len(self._buffer) + 1),
                )
                if self._buffer.endswith(stop[:length])
            ),
            default=0,
        )
        split = len(self._buffer) - retain
        visible, self._buffer = self._buffer[:split], self._buffer[split:]
        self._offset += len(visible)
        return visible

    def finish(self) -> str:
        if self.stopped:
            return ""
        visible, self._buffer = self._buffer, ""
        self._offset += len(visible)
        return visible


class ChatService:
    """Serialize requests through one engine and retain one exact-token ledger."""

    def __init__(
        self,
        engine: object,
        *,
        adapter: ChatAdapter,
        served_model_name: str,
        default_reasoning: bool | None = None,
        spec: int = 0,
        max_queue: int = 16,
        metrics_jsonl: str | Path | None = None,
    ) -> None:
        if isinstance(max_queue, bool) or not isinstance(max_queue, int):
            raise TypeError("max_queue must be an integer")
        if max_queue < 0:
            raise ValueError("max_queue must be non-negative")
        if isinstance(spec, bool) or not isinstance(spec, int):
            raise TypeError("spec must be an integer")
        if spec < 0:
            raise ValueError("spec must be non-negative")
        self.engine = engine
        self.adapter = adapter
        self.served_model_name = served_model_name
        self.default_reasoning = default_reasoning
        self.spec = spec
        self.max_queue = max_queue
        self.metrics_jsonl = Path(metrics_jsonl) if metrics_jsonl is not None else None

        self._engine_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._waiting_count = 0
        self._hot_conversation: HotConversation | None = None

    @property
    def busy(self) -> bool:
        return self._engine_lock.locked()

    @property
    def waiting_count(self) -> int:
        with self._state_lock:
            return self._waiting_count

    @property
    def hot_conversation(self) -> HotConversation | None:
        with self._state_lock:
            return self._hot_conversation

    def _admit(self) -> _Admission:
        admitted_at = time.perf_counter()
        with self._state_lock:
            if self._engine_lock.acquire(blocking=False):
                return _Admission(
                    acquired=True,
                    queued=False,
                    admitted_at=admitted_at,
                )
            if self._waiting_count >= self.max_queue:
                raise ChatQueueFull(f"chat queue is full ({self.max_queue} waiting)")
            self._waiting_count += 1
            return _Admission(
                acquired=False,
                queued=True,
                admitted_at=admitted_at,
            )

    def _wait_for_owner(
        self,
        admission: _Admission,
        cancel_event: threading.Event,
    ) -> bool:
        if admission.acquired:
            return True
        cancelled_after_acquire = False
        try:
            while not cancel_event.is_set():
                if self._engine_lock.acquire(timeout=0.05):
                    if cancel_event.is_set():
                        cancelled_after_acquire = True
                        return False
                    return True
            return False
        finally:
            with self._state_lock:
                self._waiting_count -= 1
                if self._waiting_count < 0:
                    self._waiting_count = 0
                    raise RuntimeError("chat waiting count underflow")
                if cancelled_after_acquire:
                    self._engine_lock.release()

    def _release_owner(self) -> None:
        # Admission and release share the state lock, so a request cannot
        # observe a transiently free engine while the waiting count changes.
        with self._state_lock:
            self._engine_lock.release()

    def _generate(
        self,
        plan: PromptPlan,
        options: ChatOptions,
        callback: object,
        should_stop: object,
    ) -> list[int]:
        common = {
            "max_new": options.max_new,
            "callback": callback,
            "should_stop": should_stop,
            "kv_baseline_len": plan.kv_baseline_len,
        }
        if self.spec > 0:
            return self.engine.generate_speculative(
                plan.input_ids,
                k=self.spec,
                **common,
            )
        return self.engine.generate(
            plan.input_ids,
            temp=options.temperature,
            top_p=options.top_p,
            rep_penalty=options.repetition_penalty,
            no_repeat_ngram=options.no_repeat_ngram_size,
            **common,
        )

    @staticmethod
    def _finish_reason(
        *,
        output: AssistantOutput,
        output_count: int,
        prompt_count: int,
        options: ChatOptions,
        max_context: int | None,
        stopped: bool,
    ) -> str:
        if stopped:
            return "stop"
        if options.max_new is not None and output_count >= options.max_new:
            return "length"
        if max_context is not None and prompt_count + output_count >= max_context:
            return "length"
        if output.tool_calls:
            return "tool_calls"
        return "stop"

    def _write_metrics(self, metrics: GenerationMetrics) -> None:
        if self.metrics_jsonl is None:
            return
        line = json.dumps(
            asdict(metrics),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._metrics_lock:
            with self.metrics_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def complete(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
        *,
        request_id: str | None = None,
        cancel_event: threading.Event | None = None,
        on_ready: Callable[[GenerationReady], None] | None = None,
        on_stream_delta: Callable[[StreamDelta], None] | None = None,
    ) -> ChatResult:
        """Generate one complete response while exclusively owning the engine."""
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")
        if not isinstance(options, ChatOptions):
            raise TypeError("options must be ChatOptions")
        request_id = request_id or f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        cancellation = cancel_event or threading.Event()
        admission = self._admit()
        owns_engine = self._wait_for_owner(admission, cancellation)
        if not owns_engine:
            now = time.perf_counter()
            empty_output = AssistantOutput(
                reasoning_content=None,
                content="",
                tool_calls=[],
            )
            metrics = GenerationMetrics(
                request_id=request_id,
                prompt_tokens=0,
                processed_tokens=0,
                completion_tokens=0,
                queue_delay_ms=(now - admission.admitted_at) * 1000,
                kv_mode="not-started",
                kv_reason="cancelled-in-queue",
                prefill_ms=None,
                ttft_ms=None,
                generation_ms=0.0,
                finish_reason="stop",
                tokens_per_second=0.0,
                output_token_sha256=_token_ids_sha256([]),
                periodic_tail_detected=False,
                cancelled=True,
            )
            self._write_metrics(metrics)
            return ChatResult(
                request_id=request_id,
                created=created,
                model=self.served_model_name,
                output=empty_output,
                output_ids=[],
                finish_reason="stop",
                prompt_tokens=0,
                completion_tokens=0,
                metrics=metrics,
            )

        queue_acquired_at = time.perf_counter()
        generation_started = queue_acquired_at
        try:
            hot = self.hot_conversation
            hot_ledger = (
                hot.ledger
                if hot is not None
                and hot.model == self.served_model_name
                and hot.adapter_name
                == getattr(self.adapter, "name", type(self.adapter).__name__)
                else None
            )
            plan = self.adapter.prepare(
                self.engine,
                messages,
                options,
                hot_ledger,
            )
            if on_ready is not None:
                on_ready(
                    GenerationReady(
                        request_id=request_id,
                        created=created,
                        model=self.served_model_name,
                    )
                )
            decode_stream = self.engine.new_decode_stream(skip_special_tokens=False)
            parser = self.adapter.new_stream_parser(self.engine, options)
            stop_filter = _StopTextFilter(options.stop)
            callback_ids: list[int] = []
            decoded_length = 0
            text_ranges: list[tuple[int, int]] = []
            first_token_at: float | None = None

            def on_token(token_id: int, _ignored_piece: str) -> None:
                nonlocal decoded_length, first_token_at
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                callback_ids.append(token_id)
                chunk = decode_stream.step(self.engine.tok, token_id) or ""
                start = decoded_length
                decoded_length += len(chunk)
                text_ranges.append((start, decoded_length))
                visible = stop_filter.feed(chunk)
                if visible:
                    for delta in parser.feed(visible):
                        if on_stream_delta is not None:
                            on_stream_delta(delta)

            def should_stop() -> bool:
                return cancellation.is_set() or stop_filter.stopped

            generation_started = time.perf_counter()
            generated_ids = list(
                self._generate(
                    plan,
                    options,
                    on_token,
                    should_stop,
                )
            )
            generation_finished = time.perf_counter()
            kv_stats = getattr(self.engine, "last_kv_stats", None)

            if callback_ids != generated_ids:
                raise RuntimeError(
                    "engine callback token IDs must exactly match generated output"
                )

            if not stop_filter.stopped:
                visible_tail = stop_filter.finish()
                if visible_tail:
                    for delta in parser.feed(visible_tail):
                        if on_stream_delta is not None:
                            on_stream_delta(delta)
            parsed, final_deltas = parser.finish()
            if on_stream_delta is not None:
                for delta in final_deltas:
                    on_stream_delta(delta)

            visible_ids = generated_ids
            hidden_stop = stop_filter.stopped
            if stop_filter.stop_at is not None:
                stop_at = stop_filter.stop_at
                visible_count = next(
                    (
                        index
                        for index, (_start, end) in enumerate(text_ranges)
                        if end > stop_at
                    ),
                    len(generated_ids),
                )
                visible_ids = generated_ids[:visible_count]

            max_context = getattr(
                getattr(self.engine, "model", None),
                "max_ctx",
                None,
            )
            finish_reason = self._finish_reason(
                output=parsed,
                output_count=len(visible_ids),
                prompt_count=len(plan.input_ids),
                options=options,
                max_context=max_context,
                stopped=(cancellation.is_set() or hidden_stop),
            )

            context_exhausted_without_generation = (
                max_context is not None
                and len(plan.input_ids) >= max_context
                and not generated_ids
            )
            if (
                cancellation.is_set()
                or hidden_stop
                or context_exhausted_without_generation
            ):
                # The live KV contains output not belonging to a successfully
                # committed canonical response. Reset it before another
                # request can inherit that state.
                self.engine.reset()
            else:
                ledger = self.adapter.commit(
                    self.engine,
                    plan,
                    visible_ids,
                    parsed,
                )
                committed_hot = HotConversation(
                    model=self.served_model_name,
                    adapter_name=getattr(
                        self.adapter,
                        "name",
                        type(self.adapter).__name__,
                    ),
                    ledger=ledger,
                )
                with self._state_lock:
                    self._hot_conversation = committed_hot

            generation_seconds = max(
                0.0,
                generation_finished - generation_started,
            )
            completion_count = len(visible_ids)
            metrics = GenerationMetrics(
                request_id=request_id,
                prompt_tokens=len(plan.input_ids),
                processed_tokens=_metric_value(
                    kv_stats,
                    "processed_tokens",
                    len(plan.input_ids),
                ),
                completion_tokens=completion_count,
                queue_delay_ms=(queue_acquired_at - admission.admitted_at) * 1000,
                kv_mode=str(_metric_value(kv_stats, "mode", "unknown")),
                kv_reason=str(_metric_value(kv_stats, "reason", "unknown")),
                prefill_ms=_metric_value(kv_stats, "prefill_ms", None),
                ttft_ms=(
                    None
                    if first_token_at is None
                    else (first_token_at - generation_started) * 1000
                ),
                generation_ms=generation_seconds * 1000,
                finish_reason=finish_reason,
                tokens_per_second=(
                    completion_count / generation_seconds
                    if generation_seconds > 0
                    else 0.0
                ),
                output_token_sha256=_token_ids_sha256(visible_ids),
                periodic_tail_detected=_periodic_tail_detected(visible_ids),
                cancelled=cancellation.is_set(),
            )
            self._write_metrics(metrics)
            return ChatResult(
                request_id=request_id,
                created=created,
                model=self.served_model_name,
                output=parsed,
                output_ids=list(visible_ids),
                finish_reason=finish_reason,
                prompt_tokens=len(plan.input_ids),
                completion_tokens=completion_count,
                metrics=metrics,
            )
        except BaseException:
            # A failed engine call can leave KV partially advanced. Reset it,
            # but retain the previous canonical ledger so an exact extension
            # can be safely rebuilt from token IDs on a later request.
            self.engine.reset()
            raise
        finally:
            self._release_owner()


__all__ = [
    "ChatQueueFull",
    "ChatResult",
    "ChatService",
    "GenerationReady",
    "GenerationMetrics",
    "HotConversation",
]
