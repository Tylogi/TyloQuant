import numpy as np

from mfq.runtime.minicpmo45_realtime import (
    AppleToken2Wav,
    DEFAULT_DUPLEX_SYSTEM_PROMPT,
    SAMPLE_RATE_OUT,
    _session_system_prompt,
)


def fake_renderer():
    renderer = AppleToken2Wav.__new__(AppleToken2Wav)
    renderer.token_buffer = [4218, 4218, 4218]
    calls = []

    def stream(tokens, *, last_chunk):
        calls.append((list(tokens), last_chunk))
        return np.ones(1_200, dtype=np.float32)

    renderer._stream = stream
    renderer.reset_turn = lambda: setattr(
        renderer, "token_buffer", [4218, 4218, 4218]
    )
    return renderer, calls


def test_first_tts_chunk_force_flushes_with_official_lookahead():
    renderer, calls = fake_renderer()

    waveform = renderer.push(
        [1, 2, 3, 4, 5],
        end_of_turn=False,
        force_flush=True,
    )

    assert calls == [([4218, 4218, 4218, 1, 2, 3, 4, 5], False)]
    assert renderer.token_buffer == [3, 4, 5]
    assert waveform.shape == (SAMPLE_RATE_OUT,)
    assert np.count_nonzero(waveform[:-1_200]) == 0
    assert np.all(waveform[-1_200:] == 1)


def test_regular_tts_chunk_waits_for_25_codes_plus_lookahead():
    renderer, calls = fake_renderer()

    assert renderer.push(
        list(range(24)),
        end_of_turn=False,
    ).size == 0
    waveform = renderer.push([24], end_of_turn=False)

    assert len(calls) == 1
    assert len(calls[0][0]) == 28
    assert calls[0][1] is False
    assert renderer.token_buffer == [22, 23, 24]
    assert waveform.shape == (SAMPLE_RATE_OUT,)


def test_end_of_turn_flushes_and_resets_token2wav_state():
    renderer, calls = fake_renderer()

    waveform = renderer.push([7, 8], end_of_turn=True)

    assert calls == [([4218, 4218, 4218, 7, 8], True)]
    assert renderer.token_buffer == [4218, 4218, 4218]
    assert waveform.shape == (1_200,)


def test_duplex_system_prompt_uses_profile_then_explicit_api_value():
    profile = {"system_prompt": "architecture prompt"}
    assert _session_system_prompt({}, profile) == "architecture prompt"
    assert (
        _session_system_prompt({"system_prompt": "request prompt"}, profile)
        == "request prompt"
    )
    assert _session_system_prompt({}, {}) == DEFAULT_DUPLEX_SYSTEM_PROMPT
