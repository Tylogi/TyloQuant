"""Presentation-layer parsing for tagged reasoning model output.

The native runtimes normally return ``reasoning_content`` separately.  Some
chat templates, notably DeepSeek V4 variants whose generation prompt already
ends in ``<think>``, instead stream the reasoning body through ``content`` and
only emit the closing marker.  This module keeps that backend quirk out of the
public API and handles markers split across arbitrary transport chunks.
"""

from __future__ import annotations


class TaggedReasoningParser:
    """Split a stream into reasoning and visible answer deltas.

    ``start_in_reasoning`` is used when the opening tag belongs to the rendered
    prompt rather than to generated text.  Incomplete tag prefixes are retained
    until the next chunk, so protocol markers never leak into either channel.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, *, start_in_reasoning: bool = False) -> None:
        self._in_reasoning = start_in_reasoning
        self._buffer = ""

    @staticmethod
    def _protected_suffix(value: str, markers: tuple[str, ...]) -> int:
        return max(
            (
                width
                for marker in markers
                for width in range(1, min(len(value), len(marker) - 1) + 1)
                if value.endswith(marker[:width])
            ),
            default=0,
        )

    def feed(self, text: str) -> tuple[str, str]:
        """Consume a generated text fragment and return reasoning/content deltas."""

        if not text:
            return "", ""
        self._buffer += text
        reasoning: list[str] = []
        content: list[str] = []

        while self._buffer:
            matches = [
                (position, marker)
                for marker in (self.OPEN, self.CLOSE)
                if (position := self._buffer.find(marker)) >= 0
            ]
            if matches:
                position, marker = min(matches, key=lambda match: match[0])
                visible = self._buffer[:position]
                self._buffer = self._buffer[position + len(marker) :]
                if self._in_reasoning:
                    reasoning.append(visible)
                else:
                    content.append(visible)
                self._in_reasoning = marker == self.OPEN
                if marker == self.CLOSE:
                    self._buffer = self._buffer.lstrip("\r\n")
                continue

            protected = self._protected_suffix(
                self._buffer,
                (self.OPEN, self.CLOSE),
            )
            visible = self._buffer[:-protected] if protected else self._buffer
            self._buffer = self._buffer[-protected:] if protected else ""
            if self._in_reasoning:
                reasoning.append(visible)
            else:
                content.append(visible)
            break

        return "".join(reasoning), "".join(content)

    def finish(self) -> tuple[str, str]:
        """Flush buffered text without leaking a truncated protocol marker.

        A model can occasionally end without ``</think>``.  That means it did
        not produce a visible answer, so the partial text remains reasoning;
        it must never be duplicated into ``content``.
        """

        partial, self._buffer = self._buffer, ""
        if partial and any(marker.startswith(partial) for marker in (self.OPEN, self.CLOSE)):
            return "", ""
        if self._in_reasoning:
            return partial, ""
        return "", partial


def split_tagged_reasoning(
    text: str,
    *,
    start_in_reasoning: bool = False,
) -> tuple[str, str]:
    """Split a complete response with the same rules as the streaming parser."""

    parser = TaggedReasoningParser(start_in_reasoning=start_in_reasoning)
    reasoning, content = parser.feed(text)
    trailing_reasoning, trailing_content = parser.finish()
    return reasoning + trailing_reasoning, content + trailing_content
