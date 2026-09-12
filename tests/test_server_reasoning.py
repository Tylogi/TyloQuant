from mfq.server.reasoning import TaggedReasoningParser, split_tagged_reasoning


def test_reasoning_parser_hides_tags_split_across_transport_chunks() -> None:
    parser = TaggedReasoningParser(start_in_reasoning=True)

    assert parser.feed("work</thi") == ("work", "")
    assert parser.feed("nk>\nanswer<th") == ("", "answer")
    assert parser.feed("ink>more</think>") == ("more", "")
    assert parser.finish() == ("", "")


def test_reasoning_parser_swallows_redundant_protocol_markers() -> None:
    parser = TaggedReasoningParser(start_in_reasoning=True)

    assert parser.feed("<thi") == ("", "")
    assert parser.feed("nk>plan</think>final</think>") == ("plan", "final")
    assert parser.finish() == ("", "")


def test_reasoning_parser_never_duplicates_unfinished_thinking_into_content() -> None:
    parser = TaggedReasoningParser(start_in_reasoning=True)

    assert parser.feed("unfinished reasoning") == ("unfinished reasoning", "")
    assert parser.finish() == ("", "")


def test_reasoning_parser_drops_truncated_protocol_marker() -> None:
    parser = TaggedReasoningParser(start_in_reasoning=True)

    assert parser.feed("work</thi") == ("work", "")
    assert parser.finish() == ("", "")


def test_complete_split_preserves_tag_free_visible_content() -> None:
    assert split_tagged_reasoning("plain answer") == ("", "plain answer")
