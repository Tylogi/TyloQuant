"""Kimi K3 XTML chat adapter for text-only inference.

The structural format follows Moonshot's ``encoding_k3.py``.  Control tokens
are encoded as specials while all message text and attribute values are
encoded as ordinary BPE, preventing user text from injecting protocol tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import (
    AssistantOutput,
    ChatMessage,
    ChatOptions,
    PromptPlan,
    StreamDelta,
    StreamParser,
    UnsupportedChatCapability,
)


OPEN = "<|open|>"
CLOSE = "<|close|>"
SEP = "<|sep|>"
END_OF_MSG = "<|end_of_msg|>"


def _escape_attr(value: str) -> str:
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


def _open_tag(tag: str, **attrs: str) -> list[tuple[str, bool]]:
    segments = [(OPEN, True), (tag, False)]
    for key, value in attrs.items():
        segments.extend([
            (f" {key}", False),
            ('="', False),
            (_escape_attr(value), False),
            ('"', False),
        ])
    segments.append((SEP, True))
    return segments


def _close_tag(tag: str) -> list[tuple[str, bool]]:
    return [(CLOSE, True), (tag, False), (SEP, True)]


def _message(
    role: str,
    content: str,
) -> list[tuple[str, bool]]:
    return [
        *_open_tag("message", role=role),
        (content, False),
        *_close_tag("message"),
        (END_OF_MSG, True),
    ]


def _thinking_effort_message(effort: str) -> list[tuple[str, bool]]:
    body = (
        "`thinking_effort` guides on how much to think in your "
        "thinking channel (not including the response channel), "
        "supported values include `low`, `medium`, `high`, and `max`.\n"
        f"Now the system is invoked with `thinking_effort={effort}`."
    )
    return [
        *_open_tag(
            "message",
            role="system",
            type="thinking-effort",
        ),
        (body, False),
        *_close_tag("message"),
        (END_OF_MSG, True),
    ]


def _assistant_message(
    message: ChatMessage,
    *,
    thinking: bool,
) -> list[tuple[str, bool]]:
    segments = _open_tag("message", role="assistant")
    if thinking:
        segments.extend(_open_tag("think"))
        if message.reasoning_content:
            segments.append((message.reasoning_content, False))
        segments.extend(_close_tag("think"))
    segments.extend(_open_tag("response"))
    if message.content:
        segments.append((message.content, False))
    segments.extend(_close_tag("response"))
    segments.extend(_close_tag("message"))
    segments.append((END_OF_MSG, True))
    return segments


def _reject_unsupported(
    messages: tuple[ChatMessage, ...],
    options: ChatOptions,
) -> None:
    if options.tools or options.tool_choice not in (None, "none"):
        raise UnsupportedChatCapability("kimi_k3", "tools")
    if options.response_format is not None:
        raise UnsupportedChatCapability("kimi_k3", "response_format")
    if any(
        message.role == "tool"
        or message.tool_calls
        or message.tool_call_id is not None
        for message in messages
    ):
        raise UnsupportedChatCapability("kimi_k3", "tools")


def _render(
    messages: tuple[ChatMessage, ...],
    options: ChatOptions,
) -> list[tuple[str, bool]]:
    thinking = options.thinking_mode == "thinking"
    segments: list[tuple[str, bool]] = []
    if thinking:
        effort = options.reasoning_effort or "max"
        if effort not in {"low", "medium", "high", "max"}:
            raise UnsupportedChatCapability(
                "kimi_k3",
                f"reasoning_effort={effort}",
            )
        segments.extend(_thinking_effort_message(effort))
    for message in messages:
        if message.role in {"system", "developer"}:
            segments.extend(_message("system", message.content))
        elif message.role == "user":
            segments.extend(_message("user", message.content))
        elif message.role == "assistant":
            segments.extend(_assistant_message(
                message,
                thinking=thinking,
            ))
        else:
            raise UnsupportedChatCapability(
                "kimi_k3",
                f"{message.role} messages",
            )
    segments.extend(_open_tag("message", role="assistant"))
    segments.extend(_open_tag("think" if thinking else "response"))
    return segments


def _encode_segments(
    engine: object,
    segments: list[tuple[str, bool]],
) -> list[int]:
    ids: list[int] = []
    tokenizer = getattr(engine, "tok", None)
    for text, allow_special in segments:
        if not text:
            continue
        if tokenizer is not None:
            encoded = tokenizer.encode(
                text,
                allow_special_tokens=allow_special,
            )
            ids.extend(encoded.ids)
        else:
            ids.extend(engine.encode(text))
    return ids


def _decode_raw(engine: object, output_ids: list[int]) -> str:
    tokenizer = getattr(engine, "tok", None)
    if tokenizer is not None:
        return tokenizer.decode(
            output_ids,
            skip_special_tokens=False,
        )
    return engine.decode(output_ids)


_THINK_TO_RESPONSE = (
    f"{CLOSE}think{SEP}{OPEN}response{SEP}"
)
_RESPONSE_END = f"{CLOSE}response{SEP}"


def _parse_text(text: str, *, thinking: bool) -> AssistantOutput:
    reasoning = None
    content = text
    if thinking:
        if _THINK_TO_RESPONSE not in text:
            return AssistantOutput(
                reasoning_content=text or None,
                content="",
                tool_calls=[],
            )
        reasoning, content = text.split(_THINK_TO_RESPONSE, 1)
    if _RESPONSE_END in content:
        content = content.split(_RESPONSE_END, 1)[0]
    return AssistantOutput(
        reasoning_content=reasoning or None,
        content=content,
        tool_calls=[],
    )


class _KimiStreamParser:
    def __init__(self, *, thinking: bool):
        self._thinking = thinking
        self._phase = "reasoning" if thinking else "content"
        self._buffer = ""
        self._reasoning = ""
        self._content = ""
        self._finished = False

    def _drain(
        self,
        marker: str,
        kind: str,
    ) -> tuple[StreamDelta, ...]:
        found = self._buffer.find(marker)
        if found >= 0:
            text = self._buffer[:found]
            self._buffer = self._buffer[found + len(marker):]
            if kind == "reasoning":
                self._reasoning += text
                self._phase = "content"
            else:
                self._content += text
                self._phase = "done"
            return (StreamDelta(kind=kind, text=text),) if text else ()
        safe = max(0, len(self._buffer) - len(marker) + 1)
        if safe == 0:
            return ()
        text = self._buffer[:safe]
        self._buffer = self._buffer[safe:]
        if kind == "reasoning":
            self._reasoning += text
        else:
            self._content += text
        return (StreamDelta(kind=kind, text=text),) if text else ()

    def feed(self, text: str) -> tuple[StreamDelta, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished Kimi stream parser")
        self._buffer += text
        deltas: list[StreamDelta] = []
        while True:
            previous = self._phase
            if self._phase == "reasoning":
                deltas.extend(self._drain(
                    _THINK_TO_RESPONSE,
                    "reasoning",
                ))
            elif self._phase == "content":
                deltas.extend(self._drain(
                    _RESPONSE_END,
                    "content",
                ))
            else:
                self._buffer = ""
                break
            if self._phase == previous:
                break
        return tuple(deltas)

    def finish(self) -> tuple[AssistantOutput, tuple[StreamDelta, ...]]:
        if self._finished:
            raise RuntimeError("Kimi stream parser is already finished")
        self._finished = True
        deltas: tuple[StreamDelta, ...] = ()
        if self._phase == "reasoning" and self._buffer:
            self._reasoning += self._buffer
            deltas = (
                StreamDelta(kind="reasoning", text=self._buffer),
            )
        elif self._phase == "content" and self._buffer:
            self._content += self._buffer
            deltas = (
                StreamDelta(kind="content", text=self._buffer),
            )
        self._buffer = ""
        return (
            AssistantOutput(
                reasoning_content=self._reasoning or None,
                content=self._content,
                tool_calls=[],
            ),
            deltas,
        )


@dataclass
class KimiTokenLedger:
    committed_messages: tuple[ChatMessage, ...] = ()
    completed_ids: list[int] | None = None
    thinking_mode: str | None = None

    def clear(self) -> None:
        self.committed_messages = ()
        self.completed_ids = None
        self.thinking_mode = None


class KimiK3ChatAdapter:
    name = "kimi_k3"

    def prepare(
        self,
        engine: object,
        messages: list[ChatMessage],
        options: ChatOptions,
        hot_ledger: object | None,
    ) -> PromptPlan:
        normalized = tuple(messages)
        _reject_unsupported(normalized, options)
        if (
            isinstance(hot_ledger, KimiTokenLedger)
            and hot_ledger.thinking_mode == options.thinking_mode
            and hot_ledger.completed_ids is not None
            and getattr(engine, "_cache_ids", None)
            == hot_ledger.completed_ids
            and len(normalized)
            == len(hot_ledger.committed_messages) + 1
            and normalized[:-1] == hot_ledger.committed_messages
            and normalized[-1].role == "user"
        ):
            suffix = [
                (END_OF_MSG, True),
                *_message("user", normalized[-1].content),
                *_open_tag("message", role="assistant"),
                *_open_tag(
                    "think"
                    if options.thinking_mode == "thinking"
                    else "response"
                ),
            ]
            input_ids = [
                *hot_ledger.completed_ids,
                *_encode_segments(engine, suffix),
            ]
        else:
            input_ids = _encode_segments(
                engine,
                _render(normalized, options),
            )
        return PromptPlan(
            input_ids=input_ids,
            kv_baseline_len=len(input_ids),
            normalized_messages=normalized,
            canonical_prefix_ids=list(input_ids),
            adapter_state={
                "thinking_mode": options.thinking_mode,
            },
        )

    def parse_complete(
        self,
        engine: object,
        output_ids: list[int],
        options: ChatOptions,
    ) -> AssistantOutput:
        return _parse_text(
            _decode_raw(engine, list(output_ids)),
            thinking=options.thinking_mode == "thinking",
        )

    def new_stream_parser(
        self,
        engine: object,
        options: ChatOptions,
    ) -> StreamParser:
        del engine
        return _KimiStreamParser(
            thinking=options.thinking_mode == "thinking",
        )

    def commit(
        self,
        engine: object,
        plan: PromptPlan,
        output_ids: list[int],
        parsed: AssistantOutput,
    ) -> KimiTokenLedger:
        completed_ids = [*plan.input_ids, *output_ids]
        if getattr(engine, "_cache_ids", None) != completed_ids:
            completed_ids = None
        return KimiTokenLedger(
            committed_messages=plan.normalized_messages + (
                ChatMessage(
                    role="assistant",
                    content=parsed.content,
                    reasoning_content=parsed.reasoning_content,
                ),
            ),
            completed_ids=completed_ids,
            thinking_mode=plan.adapter_state["thinking_mode"],
        )


__all__ = [
    "KimiK3ChatAdapter",
    "KimiTokenLedger",
]
