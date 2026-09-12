"""Incremental parsing for DeepSeek's native DSML tool-call protocol.

DeepSeek V4 models emit tool invocations as tagged text.  Native backends may
occasionally stream that text through ``content`` instead of translating it to
OpenAI ``tool_calls``.  This parser owns that presentation boundary and keeps
partial protocol markers out of user-visible content.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_DSML_PREFIX = "<｜DSML｜"
_BLOCK_NAMES = ("tool_calls", "function_calls")
_BLOCK_STARTS = tuple(f"{_DSML_PREFIX}{name}>" for name in _BLOCK_NAMES)


class DSMLParseError(ValueError):
    """Raised when generated DSML is incomplete or malformed."""


@dataclass(frozen=True)
class DSMLToolCall:
    name: str
    arguments: str


class DSMLStreamParser:
    """Extract complete DSML calls from arbitrarily fragmented text."""

    def __init__(self, tool_schemas: Mapping[str, Mapping[str, Any]]) -> None:
        self._tool_schemas = dict(tool_schemas)
        self._buffer = ""
        self._block_name: str | None = None

    @staticmethod
    def _protected_prefix_width(value: str) -> int:
        """Return a suffix width that could grow into the DSML prefix."""

        return max(
            (
                width
                for width in range(1, min(len(value), len(_DSML_PREFIX) - 1) + 1)
                if value.endswith(_DSML_PREFIX[:width])
            ),
            default=0,
        )

    @staticmethod
    def _skip_whitespace(value: str, position: int) -> int:
        while position < len(value) and value[position].isspace():
            position += 1
        return position

    @staticmethod
    def _read_name_attribute(
        value: str,
        position: int,
        *,
        prefix: str,
    ) -> tuple[str, int]:
        if not value.startswith(prefix, position):
            raise DSMLParseError(f"expected {prefix!r} in DSML tool block")
        position += len(prefix)
        end = value.find('"', position)
        if end < 0 or end == position:
            raise DSMLParseError("DSML name attribute is missing or empty")
        name = value[position:end]
        if not value.startswith('">', end):
            raise DSMLParseError("malformed DSML name attribute")
        return name, end + 2

    @staticmethod
    def _parameter_type_allows_string(schema: Mapping[str, Any], name: str) -> bool:
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            return False
        definition = properties.get(name)
        if not isinstance(definition, Mapping):
            return False
        value_type = definition.get("type")
        return value_type == "string" or (
            isinstance(value_type, list) and "string" in value_type
        )

    @classmethod
    def _decode_value(
        cls,
        value: str,
        *,
        is_string: bool,
        schema: Mapping[str, Any],
        name: str,
    ) -> Any:
        if is_string:
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            # Some DSV4 checkpoints occasionally mark a schema-declared
            # string as string="false".  The schema gives us an unambiguous,
            # safe recovery path for that otherwise-valid invocation.
            if cls._parameter_type_allows_string(schema, name):
                return value
            raise DSMLParseError(
                f"DSML parameter {name!r} is not valid JSON"
            ) from error

    def _parse_invoke(self, value: str, position: int) -> tuple[DSMLToolCall, int]:
        name, position = self._read_name_attribute(
            value,
            position,
            prefix=f'{_DSML_PREFIX}invoke name="',
        )
        schema = self._tool_schemas.get(name)
        if schema is None:
            raise DSMLParseError(f"DSML invokes unavailable tool {name!r}")

        arguments: dict[str, Any] = {}
        close_invoke = f"</{_DSML_PREFIX[1:]}invoke>"
        parameter_prefix = f'{_DSML_PREFIX}parameter name="'
        close_parameter = f"</{_DSML_PREFIX[1:]}parameter>"

        while True:
            position = self._skip_whitespace(value, position)
            if value.startswith(close_invoke, position):
                position += len(close_invoke)
                break
            if not value.startswith(parameter_prefix, position):
                raise DSMLParseError("expected DSML parameter or invoke closing tag")
            position += len(parameter_prefix)
            name_end = value.find('"', position)
            if name_end < 0 or name_end == position:
                raise DSMLParseError("DSML parameter name is missing or empty")
            parameter_name = value[position:name_end]
            position = name_end + 1
            if parameter_name in arguments:
                raise DSMLParseError(
                    f"duplicate DSML parameter {parameter_name!r}"
                )
            string_prefix = ' string="'
            if not value.startswith(string_prefix, position):
                raise DSMLParseError("DSML parameter is missing its string attribute")
            position += len(string_prefix)
            flag_end = value.find('"', position)
            if flag_end < 0:
                raise DSMLParseError("unterminated DSML string attribute")
            string_flag = value[position:flag_end]
            if string_flag not in {"true", "false"}:
                raise DSMLParseError(
                    f"invalid DSML string attribute {string_flag!r}"
                )
            if not value.startswith('">', flag_end):
                raise DSMLParseError("malformed DSML parameter opening tag")
            position = flag_end + 2
            value_end = value.find(close_parameter, position)
            if value_end < 0:
                raise DSMLParseError(
                    f"DSML parameter {parameter_name!r} has no closing tag"
                )
            parameter_value = value[position:value_end]
            position = value_end + len(close_parameter)
            arguments[parameter_name] = self._decode_value(
                parameter_value,
                is_string=string_flag == "true",
                schema=schema,
                name=parameter_name,
            )

        return (
            DSMLToolCall(
                name=name,
                arguments=json.dumps(
                    arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
            position,
        )

    def _parse_block(self, block_name: str, body: str) -> tuple[DSMLToolCall, ...]:
        calls: list[DSMLToolCall] = []
        position = 0
        while True:
            position = self._skip_whitespace(body, position)
            if position == len(body):
                break
            call, position = self._parse_invoke(body, position)
            calls.append(call)
        if not calls:
            raise DSMLParseError(f"empty DSML {block_name} block")
        return tuple(calls)

    def feed(self, text: str) -> tuple[str, tuple[DSMLToolCall, ...]]:
        """Consume one text delta and return visible text plus complete calls."""

        self._buffer += text
        visible: list[str] = []
        calls: list[DSMLToolCall] = []

        while self._buffer:
            if self._block_name is not None:
                close = f"</{_DSML_PREFIX[1:]}{self._block_name}>"
                close_at = self._buffer.find(close)
                if close_at < 0:
                    break
                body = self._buffer[:close_at]
                self._buffer = self._buffer[close_at + len(close) :]
                block_name, self._block_name = self._block_name, None
                calls.extend(self._parse_block(block_name, body))
                continue

            marker_at = self._buffer.find(_DSML_PREFIX)
            if marker_at >= 0:
                fragment = self._buffer[marker_at:]
                complete = next(
                    (marker for marker in _BLOCK_STARTS if fragment.startswith(marker)),
                    None,
                )
                if complete is not None:
                    visible.append(self._buffer[:marker_at])
                    self._buffer = fragment[len(complete) :]
                    self._block_name = complete[len(_DSML_PREFIX) : -1]
                    continue
                if any(marker.startswith(fragment) for marker in _BLOCK_STARTS):
                    visible.append(self._buffer[:marker_at])
                    self._buffer = fragment
                    break
                raise DSMLParseError("unexpected or malformed DSML protocol marker")

            protected = self._protected_prefix_width(self._buffer)
            if protected:
                visible.append(self._buffer[:-protected])
                self._buffer = self._buffer[-protected:]
            else:
                visible.append(self._buffer)
                self._buffer = ""
            break

        return "".join(visible), tuple(calls)

    def finish(self) -> tuple[str, tuple[DSMLToolCall, ...]]:
        """Flush visible text, rejecting truncated DSML protocol output."""

        if self._block_name is not None:
            raise DSMLParseError(f"incomplete DSML {self._block_name} block")
        trailing, self._buffer = self._buffer, ""
        distinctive = "<｜DSML"
        if trailing.startswith(distinctive) or (
            len(trailing) >= len("<｜DS") and distinctive.startswith(trailing)
        ):
            raise DSMLParseError("incomplete DSML protocol marker")
        return trailing, ()
