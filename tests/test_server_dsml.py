from __future__ import annotations

import json

import pytest

from mfq.server.dsml import DSMLParseError, DSMLStreamParser

TOOLS = {
    "write": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "count": {"type": "integer"},
            "options": {"type": "object"},
        },
    },
    "lookup": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    },
}


def test_dsml_parser_withholds_fragmented_markers_and_decodes_calls() -> None:
    parser = DSMLStreamParser(TOOLS)
    chunks = (
        "I'll write it.\n\n<｜DS",
        "ML｜tool_calls>\n<｜DSML｜inv",
        'oke name="write">\n<｜DSML｜parameter name="content" string="true">hel',
        "lo</｜DSML｜parameter>\n<｜DSML｜parameter name=\"count\" ",
        'string="false">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n',
        "</｜DSML｜tool_calls>",
    )

    visible: list[str] = []
    calls = []
    for chunk in chunks:
        content, parsed = parser.feed(chunk)
        visible.append(content)
        calls.extend(parsed)
        assert "DSML" not in content
    trailing, parsed = parser.finish()
    visible.append(trailing)
    calls.extend(parsed)

    assert "".join(visible) == "I'll write it.\n\n"
    assert len(calls) == 1
    assert calls[0].name == "write"
    assert json.loads(calls[0].arguments) == {"content": "hello", "count": 2}


def test_dsml_parser_supports_parallel_function_calls_and_json_values() -> None:
    parser = DSMLStreamParser(TOOLS)
    content, calls = parser.feed(
        "<｜DSML｜function_calls>\n"
        '<｜DSML｜invoke name="lookup">\n'
        '<｜DSML｜parameter name="query" string="true">mfq</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="write">\n'
        '<｜DSML｜parameter name="options" string="false">'
        '{"append":true}</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜function_calls>"
    )

    assert content == ""
    assert [call.name for call in calls] == ["lookup", "write"]
    assert json.loads(calls[0].arguments) == {"query": "mfq"}
    assert json.loads(calls[1].arguments) == {"options": {"append": True}}
    assert parser.finish() == ("", ())


def test_dsml_parser_recovers_false_flag_for_schema_declared_string() -> None:
    parser = DSMLStreamParser(TOOLS)
    _content, calls = parser.feed(
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="write">\n'
        '<｜DSML｜parameter name="content" string="false">hello'
        "</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )

    assert json.loads(calls[0].arguments) == {"content": "hello"}


@pytest.mark.parametrize(
    ("generated", "message"),
    [
        (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="missing">\n</｜DSML｜invoke>\n'
            "</｜DSML｜tool_calls>",
            "unavailable tool",
        ),
        ("<｜DSML｜tool_calls>\n", "incomplete DSML tool_calls block"),
        ("answer<｜DSML｜tool_ca", "incomplete DSML protocol marker"),
    ],
)
def test_dsml_parser_rejects_unknown_or_incomplete_protocol(
    generated: str,
    message: str,
) -> None:
    parser = DSMLStreamParser(TOOLS)
    if "missing" in generated:
        with pytest.raises(DSMLParseError, match=message):
            parser.feed(generated)
        return

    parser.feed(generated)
    with pytest.raises(DSMLParseError, match=message):
        parser.finish()
