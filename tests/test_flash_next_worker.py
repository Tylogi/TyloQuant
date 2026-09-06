from __future__ import annotations

import asyncio
import base64
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import mlx.core as mx
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

import mfq.runtime.flash_next_worker as worker_module
from mfq.formats import io
from mfq.formats.assets import (
    HF_CHAT_TEMPLATE_ASSET,
    HF_GENERATION_CONFIG_ASSET,
    HF_TOKENIZER_CONFIG_ASSET,
    HF_TOKENIZER_JSON_ASSET,
    MODEL_CONFIG_ASSET,
    MODEL_GRAPH_ASSET,
)
from mfq.formats.header import FileHeader
from mfq.runtime.flash_next_worker import (
    FlashNextTextWorker,
    FlashNextWorkerError,
    _GenerationSummary,
    _MultimodalInput,
    _parse_multimodal_images,
    _PreparedRequest,
    _ReasoningParser,
    _sampling_log_probs,
    create_app,
    load_flash_next_tokenizer,
)
from mfq.server.backend import OpenAIChatBackend
from mfq.server.models import SamplingParams
from mfq.server.native import (
    NativeRuntime,
    python_mlx_runtime_command,
    resolve_runtime_route,
)


def _tiny_tokenizer() -> bytes:
    backend = Tokenizer(
        models.WordLevel(
            {
                "<unk>": 0,
                "user": 1,
                "assistant": 2,
                ":": 3,
                "hello": 4,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return backend.to_str().encode()


def test_embedded_flash_next_tokenizer_renders_without_remote_code(tmp_path: Path) -> None:
    model = tmp_path / "tiny.mfq"
    template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}assistant:"
    )
    io.save(
        model,
        FileHeader(version=2, model_arch="qwen4_exp-test"),
        {
            "weight": np.ones((1,), dtype=np.float16),
            HF_TOKENIZER_JSON_ASSET: _tiny_tokenizer(),
            HF_TOKENIZER_CONFIG_ASSET: json.dumps(
                {"unk_token": "<unk>", "chat_template": template}
            ).encode(),
            HF_CHAT_TEMPLATE_ASSET: template.encode(),
            HF_GENERATION_CONFIG_ASSET: b'{"max_new_tokens":17}',
        },
    )

    with io.open_mmap(model) as store:
        tokenizer, generation = load_flash_next_tokenizer(store, model)

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered == "user: hello\nassistant:"
    assert generation["max_new_tokens"] == 17


def test_reasoning_parser_hides_fragmented_protocol_tags() -> None:
    parser = _ReasoningParser(starts_in_reasoning=True)

    assert parser.feed("work</thi") == (("reasoning", "work"),)
    assert parser.feed("nk>\nanswer<th") == (("content", "answer"),)
    assert parser.feed("ink>more</think>") == (("reasoning", "more"),)
    assert parser.finish() == ()
    assert parser.reasoning_text == "workmore"
    assert parser.content == "answer"


class _CacheModel:
    max_context = 64

    def __init__(self) -> None:
        self.config = SimpleNamespace(eos_token_ids=(63,))
        self.position = 0
        self.prefills: list[tuple[int, ...]] = []
        self.extensions: list[tuple[int, ...]] = []
        self.reset_count = 0

    @staticmethod
    def _logits(ids: np.ndarray) -> mx.array:
        values = np.asarray(ids, dtype=np.float32)[..., None]
        return mx.array(np.repeat(values, 64, axis=-1))

    def prefill(self, input_ids: np.ndarray) -> mx.array:
        values = tuple(int(item) for item in input_ids.reshape(-1))
        self.prefills.append(values)
        self.position = len(values)
        return self._logits(input_ids)

    def forward(self, input_ids: np.ndarray, *, use_cache: bool) -> mx.array:
        assert use_cache
        values = tuple(int(item) for item in input_ids.reshape(-1))
        self.extensions.append(values)
        self.position += len(values)
        return self._logits(input_ids)

    def reset_cache(self, _batch: int = 1) -> None:
        self.position = 0
        self.reset_count += 1

    def close(self) -> None:
        pass


def _prepared(session_id: str, *tokens: int) -> _PreparedRequest:
    return _PreparedRequest(
        request_id=f"request-{session_id}",
        session_id=session_id,
        input_ids=tuple(tokens),
        prompt_ends_in_thinking=False,
        max_tokens=1,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        seed=None,
        sampling_payload={},
    )


def _prepare_worker() -> FlashNextTextWorker:
    class TokenizerStub:
        eos_token_id = 63

        @staticmethod
        def apply_chat_template(*_args, **_kwargs) -> str:
            return "user: hello\nassistant:"

        @staticmethod
        def encode(_rendered: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            return [1, 2, 3]

    model = SimpleNamespace(
        max_context=64,
        config=SimpleNamespace(eos_token_ids=(63,)),
    )
    return FlashNextTextWorker(
        model,
        TokenizerStub(),
        model_name="Flash",
        model_type="glm5_next",
        generation_config={"max_new_tokens": 17},
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_tokens", 0, "max_tokens must be positive"),
        ("max_tokens", 1.5, "max_tokens must be an integer"),
        ("max_tokens", None, "max_tokens must be an integer"),
        ("temperature", "warm", "temperature must be a number"),
        ("top_k", 1025, "top_k must be in [0,1024]"),
        ("top_k", True, "top_k must be an integer"),
        ("top_p", None, "top_p must be a number"),
        ("presence_penalty", float("nan"), "presence_penalty must be finite"),
        ("frequency_penalty", 2.1, "frequency_penalty must be finite"),
        ("repetition_penalty", 0, "repetition_penalty must be finite"),
        ("seed", -1, "seed must be non-negative"),
        ("seed", 1.25, "seed must be an integer"),
        ("enable_vision", 1, "enable_vision must be a boolean"),
        ("enable_mtp", "yes", "enable_mtp must be a boolean"),
    ],
)
def test_flash_next_prepare_rejects_invalid_sampling_as_request_errors(
    field: str,
    value: object,
    message: str,
) -> None:
    worker = _prepare_worker()
    request = {
        "model": "Flash",
        "messages": [{"role": "user", "content": "hello"}],
        field: value,
    }

    with pytest.raises(FlashNextWorkerError, match=message.replace("[", r"\[").replace("]", r"\]")):
        worker.prepare(request)


def test_flash_next_prepare_preserves_default_and_explicit_sampling_values() -> None:
    worker = _prepare_worker()
    request = {
        "model": "Flash",
        "messages": [{"role": "user", "content": "hello"}],
    }
    default = worker.prepare(request)
    explicit = worker.prepare(
        {
            **request,
            "max_tokens": 9,
            "temperature": 0.25,
            "top_k": 1024,
            "top_p": 0.75,
            "presence_penalty": -0.25,
            "frequency_penalty": 0.5,
            "repetition_penalty": 1.1,
            "seed": 7,
            "enable_vision": False,
            "enable_mtp": False,
        }
    )

    assert default.max_tokens == 17
    assert (
        explicit.max_tokens,
        explicit.temperature,
        explicit.top_k,
        explicit.top_p,
        explicit.seed,
    ) == (9, 0.25, 1024, 0.75, 7)
    assert explicit.presence_penalty == -0.25
    assert explicit.frequency_penalty == 0.5
    assert explicit.repetition_penalty == 1.1
    assert default.enable_vision and default.enable_mtp
    assert not explicit.enable_vision and not explicit.enable_mtp


def test_flash_next_prepare_rejects_media_when_vision_is_manually_disabled() -> None:
    worker = _prepare_worker()
    with pytest.raises(FlashNextWorkerError, match="vision is disabled"):
        worker.prepare(
            {
                "model": "Flash",
                "messages": [{"role": "user", "content": "hello"}],
                "enable_vision": False,
                "mfq_multimodal": {},
            }
        )


def test_worker_family_registry_assigns_every_optional_component_once() -> None:
    families = {item.family: item for item in worker_module._WORKER_FAMILY_REGISTRY}
    assert set(families) == {"qwen3_5", "qwen4_exp", "glm5_next"}
    for registration in families.values():
        assert callable(registration.parse_config)
        assert callable(registration.load)
        assert callable(registration.model_type.from_mfq)
        assert callable(registration.vision_type.load_if_present)
        assert callable(registration.mtp_type.load_if_present)
        assert registration.mtp_supported


def test_flash_next_http_rejects_bad_json_and_sampling_with_400() -> None:
    async def run() -> None:
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(_prepare_worker())),
            base_url="http://worker",
        )
        try:
            invalid_json = await client.post(
                "/v1/chat/completions",
                content=b"{",
                headers={"content-type": "application/json"},
            )
            invalid_top_k = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "Flash",
                    "messages": [{"role": "user", "content": "hello"}],
                    "top_k": 1025,
                },
            )
        finally:
            await client.aclose()

        assert invalid_json.status_code == 400
        assert invalid_json.json()["error"]["message"] == "request body must be valid JSON"
        assert invalid_top_k.status_code == 400
        assert invalid_top_k.json()["error"]["message"] == "top_k must be in [0,1024]"

    asyncio.run(run())


def test_flash_next_hot_prefix_reuses_only_exact_live_session_prefixes() -> None:
    model = _CacheModel()
    worker = FlashNextTextWorker(
        model,
        SimpleNamespace(),
        model_name="Flash",
        model_type="qwen4_exp",
    )

    first = worker._prefill_prompt(_prepared("source", 1, 2, 3))
    assert (first.reused_tokens, first.computed_tokens) == (0, 3)
    assert model.prefills == [(1, 2, 3)]

    extended = worker._prefill_prompt(_prepared("source", 1, 2, 3, 4, 5))
    assert (extended.reused_tokens, extended.computed_tokens) == (3, 2)
    assert model.extensions == [(4, 5)]

    assert worker.fork_session("source", "fork") == 1
    forked = worker._prefill_prompt(_prepared("fork", 1, 2, 3, 4, 5, 6))
    assert (forked.reused_tokens, forked.computed_tokens) == (5, 1)
    assert model.extensions[-1] == (6,)

    edited = worker._prefill_prompt(_prepared("fork", 1, 2, 9))
    assert (edited.reused_tokens, edited.computed_tokens) == (0, 3)
    assert model.prefills[-1] == (1, 2, 9)

    other = worker._prefill_prompt(_prepared("other", 1, 2, 9, 10))
    assert (other.reused_tokens, other.computed_tokens) == (0, 4)
    assert model.prefills[-1] == (1, 2, 9, 10)
    assert worker.status()["prefix_cache_hits"] == 2
    assert worker.status()["prefix_cache_hit_tokens"] == 8


def test_flash_next_hot_prefix_fork_close_and_clear_lifecycle() -> None:
    model = _CacheModel()
    worker = FlashNextTextWorker(
        model,
        SimpleNamespace(),
        model_name="Flash",
        model_type="glm5_next",
    )
    worker._prefill_prompt(_prepared("source", 7, 8))
    assert worker.fork_session("missing", "empty") == 0
    assert worker.fork_session("source", "fork") == 1
    released, cancelled = worker.close_session("source")
    assert (released, cancelled) == (1, False)
    assert worker.status()["prefix_cache_sessions"] == 1
    assert worker.clear_cache()
    assert model.reset_count == 1
    status = worker.status()
    assert status["prefix_cache_sessions"] == 0
    assert status["prefix_cache_snapshots"] == 0
    assert status["prefix_cache_tokens"] == 0


def test_flash_next_sampling_log_probs_match_top_k_then_nucleus_contract() -> None:
    logits = mx.array([[4.0, 3.0, 2.0, 1.0]], dtype=mx.float32)
    actual = _sampling_log_probs(
        logits,
        temperature=1.0,
        top_k=3,
        top_p=0.8,
    )
    mx.eval(actual)
    values = np.asarray(actual)[0]
    expected = np.asarray([4.0, 3.0], dtype=np.float32)
    expected -= np.log(np.exp(expected).sum())
    np.testing.assert_allclose(values[:2], expected, rtol=1e-6, atol=1e-6)
    assert np.isneginf(values[2:]).all()


def test_flash_next_ordinary_decode_applies_prompt_and_generated_penalties() -> None:
    vocab = 16

    class Target:
        max_context = 64

        def __init__(self) -> None:
            self.config = SimpleNamespace(eos_token_ids=(15,))
            self.position = 0
            self.decoded: list[int] = []

        @staticmethod
        def _scores(primary: int, secondary: int) -> mx.array:
            values = np.full((1, 1, vocab), -100.0, dtype=np.float32)
            values[0, 0, primary] = 10.0
            values[0, 0, secondary] = 9.0
            return mx.array(values)

        def prefill(self, input_ids: np.ndarray) -> mx.array:
            self.position = int(input_ids.shape[1])
            return self._scores(1, 2)

        def decode(self, input_ids: np.ndarray) -> mx.array:
            token = int(input_ids.reshape(-1)[0])
            self.decoded.append(token)
            self.position += 1
            return self._scores(2, 3)

        def close(self) -> None:
            pass

    backend = Tokenizer(
        models.WordLevel(
            {"<unk>": 0, "a": 1, "b": 2, "c": 3},
            unk_token="<unk>",
        )
    )
    worker = FlashNextTextWorker(
        Target(),
        SimpleNamespace(backend_tokenizer=backend, eos_token_id=15),
        model_name="Flash",
        model_type="glm5_next",
    )
    prepared = replace(
        _prepared("penalties", 1, 1),
        session_id=None,
        max_tokens=2,
        presence_penalty=2.0,
        frequency_penalty=1.0,
        sampling_payload={
            "presence_penalty": 2.0,
            "frequency_penalty": 1.0,
            "repetition_penalty": 1.0,
        },
    )

    summary = worker.generate(prepared, lambda _delta: None)

    assert summary.completion_tokens == 2
    assert summary.content == "b c"
    assert worker.model.decoded == [2]


def test_flash_next_depth_one_mtp_accepts_and_rolls_back_exactly() -> None:
    vocab = 16

    def logits_for(tokens: np.ndarray) -> mx.array:
        transitions = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 9: 10}
        result = np.full((*tokens.shape, vocab), -100.0, dtype=np.float32)
        for row, token in enumerate(tokens.reshape(-1)):
            result.reshape(-1, vocab)[row, transitions.get(int(token), 0)] = 100.0
        return mx.array(result)

    class Target:
        max_context = 64

        def __init__(self) -> None:
            self.config = SimpleNamespace(eos_token_ids=(15,))
            self.position = 7
            self.calls: list[tuple[tuple[int, ...], int]] = []
            self.checkpoint: int | None = None
            self.commits = 0
            self.rollbacks = 0

        def forward_with_hidden(self, ids, *, use_cache, n_confirmed=0):
            assert use_cache
            values = np.asarray(ids, dtype=np.int32)
            tokens = tuple(int(item) for item in values.reshape(-1))
            self.calls.append((tokens, int(n_confirmed)))
            start = self.position
            self.position += len(tokens)
            if n_confirmed:
                self.checkpoint = start + int(n_confirmed)
            hidden = mx.array(values[..., None].astype(np.float32))
            return logits_for(values), hidden

        def commit_speculative_cache(self):
            self.commits += 1
            self.checkpoint = None

        def rollback_speculative_cache(self):
            assert self.checkpoint is not None
            self.rollbacks += 1
            self.position = self.checkpoint
            self.checkpoint = None

    class Mtp:
        def reset_cache(self, _batch=1):
            pass

        def forward(self, ids, _hidden, *, use_cache):
            assert use_cache
            token = int(np.asarray(ids).reshape(-1)[0])
            draft = {2: 3, 4: 9, 5: 6}[token]
            return mx.array([[[float(draft)]]], dtype=mx.float32)

        @staticmethod
        def compute_logits(hidden):
            draft = int(np.asarray(hidden).reshape(-1)[0])
            result = np.full((1, 1, vocab), -100.0, dtype=np.float32)
            result[0, 0, draft] = 100.0
            return mx.array(result)

    target = Target()
    worker = FlashNextTextWorker(
        target,
        SimpleNamespace(),
        model_name="Flash",
        model_type="glm5_next",
        mtp=Mtp(),
    )
    prepared = replace(
        _prepared("mtp", 11),
        max_tokens=6,
        presence_penalty=0.25,
    )
    initial = np.full((1, 1, vocab), -100.0, dtype=np.float32)
    initial[0, 0, 1] = 100.0
    emitted: list[int] = []
    penalty_histories: list[np.ndarray] = []
    original_penalized_logits = worker._penalized_logits

    def trace_penalties(prepared_request, logits, counts):
        assert counts is not None
        mx.eval(counts)
        penalty_histories.append(np.asarray(counts).copy())
        return original_penalized_logits(prepared_request, logits, counts)

    worker._penalized_logits = trace_penalties

    generated, finish, first_at, stats = worker._generate_with_mtp(
        prepared,
        mx.array(initial),
        threading.Event(),
        {15},
        emitted.append,
    )

    assert generated == emitted == [1, 2, 3, 4, 5, 6]
    assert finish == "length"
    assert first_at is not None
    assert target.calls == [((1,), 0), ((2, 3), 1), ((4, 9), 1), ((5, 6), 1)]
    assert (target.commits, target.rollbacks, target.position) == (2, 1, 13)
    assert (stats.cycles, stats.drafted_tokens, stats.accepted_tokens) == (3, 3, 2)
    # The third draft sees the corrected token 5 but never the rejected draft 9.
    assert penalty_histories[7][5] == 1
    assert penalty_histories[7][9] == 0


def test_flash_next_multimodal_tensor_contract_rejects_wrong_family_and_nan() -> None:
    vision = SimpleNamespace(
        in_channels=3,
        temporal_patch_size=2,
        patch_size=2,
        spatial_merge_size=2,
    )
    config = SimpleNamespace(
        family="glm5_next",
        vision=vision,
        image_token_id=9,
    )
    pixels = np.arange(4 * 24, dtype="<f4").reshape(4, 24)
    grid = np.asarray([[1, 2, 2]], dtype="<i4")

    def tensor(value: np.ndarray, dtype: str) -> dict[str, object]:
        return {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_base64": base64.b64encode(value.tobytes()).decode("ascii"),
        }

    payload = {
        "version": 3,
        "processor": "grid_vision.v1",
        "pixel_values": tensor(pixels, "float32"),
        "image_grid_thw": tensor(grid, "int32"),
    }
    parsed = _parse_multimodal_images(payload, config)
    np.testing.assert_array_equal(parsed.pixel_values, pixels)
    np.testing.assert_array_equal(parsed.image_grid_thw, grid)

    wrong_family = dict(payload, processor="foreign_grid.v1")
    with pytest.raises(RuntimeError, match="does not match"):
        _parse_multimodal_images(wrong_family, config)
    invalid_pixels = pixels.copy()
    invalid_pixels[0, 0] = np.nan
    invalid = dict(payload, pixel_values=tensor(invalid_pixels, "float32"))
    with pytest.raises(RuntimeError, match="non-finite"):
        _parse_multimodal_images(invalid, config)


def test_flash_next_multimodal_tensor_contract_preserves_mixed_media_order() -> None:
    vision = SimpleNamespace(
        in_channels=3,
        temporal_patch_size=2,
        patch_size=2,
        spatial_merge_size=2,
    )
    config = SimpleNamespace(
        family="glm5_next",
        vision=vision,
        image_token_id=9,
        video_start_token_id=7,
        video_end_token_id=8,
    )
    pixels = np.arange(12 * 24, dtype="<f4").reshape(12, 24)
    image_grid = np.asarray([[1, 2, 2]], dtype="<i4")
    video_grid = np.asarray([[2, 2, 2]], dtype="<i4")
    combined_grid = np.asarray([[2, 2, 2], [1, 2, 2]], dtype="<i4")
    media_types = np.asarray([2, 1], dtype="<i4")

    def tensor(value: np.ndarray, dtype: str) -> dict[str, object]:
        return {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_base64": base64.b64encode(value.tobytes()).decode("ascii"),
        }

    payload = {
        "version": 3,
        "processor": "grid_vision.v1",
        "pixel_values": tensor(pixels, "float32"),
        "image_grid_thw": tensor(image_grid, "int32"),
        "video_grid_thw": tensor(video_grid, "int32"),
        "vision_grid_thw": tensor(combined_grid, "int32"),
        "vision_types": tensor(media_types, "int32"),
    }
    parsed = _parse_multimodal_images(payload, config)
    np.testing.assert_array_equal(parsed.pixel_values, pixels)
    np.testing.assert_array_equal(parsed.image_grid_thw, image_grid)
    np.testing.assert_array_equal(parsed.video_grid_thw, video_grid)
    np.testing.assert_array_equal(parsed.vision_grid_thw, combined_grid)
    np.testing.assert_array_equal(parsed.vision_types, media_types)

    wrong_order = dict(
        payload,
        vision_grid_thw=tensor(combined_grid[::-1].copy(), "int32"),
    )
    with pytest.raises(RuntimeError, match="grids disagree"):
        _parse_multimodal_images(wrong_order, config)

    wrong_pixels = dict(
        payload,
        pixel_values=tensor(pixels[:-1].copy(), "float32"),
    )
    with pytest.raises(RuntimeError, match="pixel count disagrees"):
        _parse_multimodal_images(wrong_pixels, config)


class _MultimodalCacheModel(_CacheModel):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(eos_token_ids=(63,), image_token_id=9)
        self.forwarded_embeddings: np.ndarray | None = None
        self.forwarded_chunk_sizes: list[int] = []

    @staticmethod
    def embedding(input_ids: mx.array) -> mx.array:
        values = np.asarray(input_ids, dtype=np.float32)[..., None]
        return mx.array(np.repeat(values, 4, axis=-1))

    def forward_embeddings(self, embeddings: mx.array, *, use_cache: bool) -> mx.array:
        assert use_cache
        chunk = np.asarray(embeddings)
        self.forwarded_chunk_sizes.append(int(chunk.shape[1]))
        self.forwarded_embeddings = (
            chunk
            if self.forwarded_embeddings is None
            else np.concatenate((self.forwarded_embeddings, chunk), axis=1)
        )
        self.position += int(embeddings.shape[1])
        token_values = chunk[..., :1]
        return mx.array(np.repeat(token_values, 64, axis=-1))


class _VisionStub:
    def __call__(self, pixel_values, grid_thw):
        assert np.asarray(pixel_values).shape == (4, 24)
        np.testing.assert_array_equal(grid_thw, [[1, 2, 2]])
        merged = mx.array([[100.0, 101.0, 102.0, 103.0]])
        return merged, merged


def test_flash_next_multimodal_prefill_never_reuses_text_only_prefix_cache() -> None:
    model = _MultimodalCacheModel()
    worker = FlashNextTextWorker(
        model,
        SimpleNamespace(),
        model_name="Flash",
        model_type="glm5_next",
        vision=_VisionStub(),
    )
    worker._prefill_prompt(_prepared("source", 1, 2))
    multimodal = _MultimodalInput(
        pixel_values=np.zeros((4, 24), dtype=np.float32),
        image_grid_thw=np.asarray([[1, 2, 2]], dtype=np.int32),
    )
    result = worker._prefill_prompt(replace(_prepared("source", 1, 9, 2), multimodal=multimodal))

    assert (result.reused_tokens, result.computed_tokens) == (0, 3)
    assert result.multimodal_ms >= 0.0
    assert model.reset_count == 1
    assert model.forwarded_embeddings is not None
    np.testing.assert_array_equal(
        model.forwarded_embeddings[0, 1],
        [100.0, 101.0, 102.0, 103.0],
    )
    status = worker.status()
    assert status["prefix_cache_sessions"] == 0
    assert status["prefix_cache_snapshots"] == 0
    assert status["prefix_cache_hits"] == 0


def test_flash_next_multimodal_prefill_is_chunked_without_reordering_embeddings() -> None:
    model = _MultimodalCacheModel()
    worker = FlashNextTextWorker(
        model,
        SimpleNamespace(),
        model_name="Flash",
        model_type="glm5_next",
        vision=_VisionStub(),
        prefill_chunk_size=2,
    )
    multimodal = _MultimodalInput(
        pixel_values=np.zeros((4, 24), dtype=np.float32),
        image_grid_thw=np.asarray([[1, 2, 2]], dtype=np.int32),
    )

    worker._prefill_prompt(
        replace(
            _prepared("source", 1, 9, 2, 3, 4),
            multimodal=multimodal,
        )
    )

    assert model.forwarded_chunk_sizes == [2, 2, 1]
    assert model.position == 5
    assert model.forwarded_embeddings is not None
    np.testing.assert_array_equal(
        model.forwarded_embeddings[0, 1],
        [100.0, 101.0, 102.0, 103.0],
    )


def test_flash_next_worker_closes_text_model_when_vision_initialization_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeStore:
        records = {worker_module.MODEL_CONFIG_ASSET: object()}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __getitem__(self, name: str) -> bytes:
            assert name == worker_module.MODEL_CONFIG_ASSET
            return b"{}"

    class FakeConfig:
        family = "qwen4_exp"
        max_position_embeddings = 4096
        vision = object()

    class FakeModel:
        def __init__(self) -> None:
            self.model = SimpleNamespace(tensors={"model.visual.patch_embed.proj.weight": object()})
            self.closed = False

        def close(self) -> None:
            self.closed = True

    loaded = FakeModel()

    class FakeModelType:
        @classmethod
        def from_mfq(cls, *_args, **_kwargs):
            return loaded

    class FailingVision:
        @classmethod
        def load_if_present(cls, *_args, **_kwargs):
            raise RuntimeError("broken vision fixture")

    class EmptyMtp:
        @classmethod
        def load_if_present(cls, *_args, **_kwargs):
            return None

    registration = worker_module._WorkerFamilyRegistration(
        family="qwen4_exp",
        aliases=("qwen4_exp",),
        config_parser=lambda _payload: FakeConfig(),
        config_type=FakeConfig,
        model_type=FakeModelType,
        vision_type=FailingVision,
        mtp_type=EmptyMtp,
    )

    monkeypatch.setattr(worker_module, "open_mmap", lambda _path: FakeStore())
    monkeypatch.setattr(
        worker_module,
        "load_flash_next_tokenizer",
        lambda *_args: (SimpleNamespace(), {}),
    )
    monkeypatch.setattr(worker_module, "_worker_family_for_payload", lambda _payload: registration)
    monkeypatch.setattr(worker_module, "_worker_family_for_config", lambda _config: registration)

    with pytest.raises(RuntimeError, match="broken vision fixture"):
        FlashNextTextWorker.from_mfq(
            tmp_path / "fixture.mfq",
            model_name="Flash",
            max_context=1024,
        )

    assert loaded.closed


def test_python_mlx_runtime_command_reenters_mfq_cli(tmp_path: Path) -> None:
    command = python_mlx_runtime_command(
        ("python", "-m", "mfq.cli"),
        model="model.mfq",
        model_name="Flash",
        host="127.0.0.1",
        port=1234,
        context_size=8192,
    )
    assert command == [
        "python",
        "-m",
        "mfq.cli",
        "_flash-next-worker",
        "--mfq",
        "model.mfq",
        "--host",
        "127.0.0.1",
        "--port",
        "1234",
        "--model-name",
        "Flash",
        "--ctx-size",
        "8192",
        "--prefill-chunk-size",
        "2048",
    ]
    model = tmp_path / "glm5.mfq"
    io.save(
        model,
        FileHeader(version=2, model_arch="legacy-name-is-ignored"),
        {
            MODEL_GRAPH_ASSET: json.dumps(
                {
                    "schema_version": 1,
                    "architecture": "glm5_next",
                    "graph": {"kind": "causal_lm", "backbone": "glm5_next"},
                    "components": [
                        {
                            "kind": "text",
                            "tensor_root": "model",
                            "implementation": "glm5_next",
                        }
                    ],
                    "canonical_naming": {
                        "namespace": "mfq.tensor",
                        "version": 1,
                        "component_roots": ["model"],
                    },
                }
            ).encode(),
        },
    )
    runtime = NativeRuntime(
        executable=Path("/runtime/mfq-decode-metal"),
        model=model,
        model_name="Flash",
        backend="metal",
        context_size=4096,
        architecture="glm5_next-hf-mfq-nint-recipe",
        controller_command=("mfq-cli",),
    )
    assert runtime.command(9001)[0:2] == ["mfq-cli", "_flash-next-worker"]
    assert "--server" not in runtime.command(9001)


def test_runtime_route_uses_graph_backbone_and_native_qwen_components(
    tmp_path: Path,
) -> None:
    model = tmp_path / "qwen35-vl.mfq"
    io.save(
        model,
        FileHeader(version=2, model_arch="qwen3_5-hf-full-mfq"),
        {
            MODEL_CONFIG_ASSET: json.dumps(
                {
                    "model_type": "qwen3_5",
                    "language_model_only": False,
                    "vision_config": {"model_type": "qwen3_5"},
                }
            ).encode(),
            "model.visual.patch_embed.proj.weight": np.ones((1,), dtype=np.float16),
        },
    )

    route = resolve_runtime_route("qwen3_5-hf-full-mfq", model)
    assert route.architecture_family == "qwen3_5"
    assert route.backbone == "qwen3_5"
    assert route.vision_available
    assert not route.python_mlx_worker
    assert not route.requires_mfq
    runtime = NativeRuntime(
        executable=Path("/runtime/mfq-decode-metal"),
        model=model,
        model_name="Qwen3.5-VL",
        backend="metal",
        context_size=4096,
        architecture="qwen3_5-hf-full-mfq",
        controller_command=("mfq-cli",),
    )
    assert runtime.command(9002)[0] == "/runtime/mfq-decode-metal"
    assert "--server" in runtime.command(9002)

    text_only = tmp_path / "qwen35-text.mfq"
    io.save(
        text_only,
        FileHeader(version=2, model_arch="qwen3_5-hf-full-mfq"),
        {
            MODEL_CONFIG_ASSET: b'{"model_type":"qwen3_5","language_model_only":true}',
            "weight": np.ones((1,), dtype=np.float16),
        },
    )
    route = resolve_runtime_route("qwen3_5-hf-full-mfq", text_only)
    assert not route.vision_available
    assert not route.python_mlx_worker
    assert not route.requires_mfq

    flash_next = resolve_runtime_route("glm5_next-hf-mfq-nint-recipe", text_only)
    assert flash_next.architecture_family == "qwen3_5"
    assert not flash_next.python_mlx_worker
    assert not flash_next.requires_mfq


class _FakeWorker:
    model_name = "Flash"
    model_type = "qwen4_exp"

    def __init__(self) -> None:
        self.model = SimpleNamespace(max_context=4096, reset_cache=lambda: None)
        self._generation_lock = threading.Lock()
        self.cancelled: list[str] = []

    def status(self) -> dict[str, object]:
        return {
            "status": "ok",
            "ready": True,
            "model": self.model_name,
            "model_type": self.model_type,
            "max_context": 4096,
            "context_capacity": 4096,
        }

    def prepare(self, request: dict[str, object]) -> _PreparedRequest:
        assert request["model"] == self.model_name
        return _PreparedRequest(
            request_id="chatcmpl-test",
            session_id=str(request.get("mfq_session_id") or "session"),
            input_ids=(1, 2, 3),
            prompt_ends_in_thinking=True,
            max_tokens=8,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            seed=None,
            sampling_payload=SamplingParams(
                max_tokens=8,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
            ).model_dump(mode="json"),
        )

    def generate(self, prepared, emit):
        emit({"reasoning": "checked"})
        emit({"content": "Hello"})
        sampling = SamplingParams(
            max_tokens=8,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
        ).model_dump(mode="json")
        metrics = {
            "prefill_tokens": 3,
            "ttft_ms": 2.0,
            "prefill_ms": 1.5,
            "prefill_tps": 2000.0,
            "multimodal_ms": 0.0,
            "model_prefill_ms": 1.5,
            "processor_ms": 0.0,
            "complete_prefill_ms": 2.0,
            "complete_prefill_tps": 1500.0,
            "decode_ms": 1.0,
            "decode_tps": 1000.0,
            "generation_ms": 3.0,
            "complete_generation_ms": 3.0,
            "generation_tps": 333.3,
            "mtp_available": True,
            "mtp_used": True,
            "mtp_cycles": 1,
            "mtp_drafted_tokens": 1,
            "mtp_accepted_tokens": 1,
            "mtp_acceptance_rate": 1.0,
            "mtp_target_ms": 0.4,
            "mtp_head_ms": 0.2,
            "mtp_rollback_ms": 0.0,
            "sampling": sampling,
        }
        return _GenerationSummary(
            request_id=prepared.request_id,
            prompt_tokens=3,
            completion_tokens=1,
            finish_reason="stop",
            content="Hello",
            reasoning="checked",
            metrics=metrics,
        )

    def cancel(self, session_id: str) -> bool:
        self.cancelled.append(session_id)
        return True


def test_flash_next_worker_protocol_matches_common_backend() -> None:
    async def run() -> None:
        worker = _FakeWorker()
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(worker)),
            base_url="http://worker",
        )
        backend = OpenAIChatBackend("http://worker", client=client)
        try:
            capabilities = await backend.capabilities()
            assert capabilities.model == "Flash"
            assert capabilities.model_type == "qwen4_exp"
            assert capabilities.model_capabilities.features.text
            deltas = [
                delta
                async for delta in backend.stream(
                    model="Flash",
                    messages=[{"role": "user", "content": "Hello"}],
                    sampling=SamplingParams(
                        max_tokens=8,
                        temperature=0.0,
                        top_k=0,
                        top_p=1.0,
                    ),
                )
            ]
            assert "".join(delta.reasoning_delta for delta in deltas) == "checked"
            assert "".join(delta.content_delta for delta in deltas) == "Hello"
            assert next(delta.usage for delta in deltas if delta.usage).total_tokens == 4
            assert (
                next(delta.performance for delta in deltas if delta.performance).decode_tps == 1000
            )
            performance = next(delta.performance for delta in deltas if delta.performance)
            assert performance.mtp_used
            assert performance.mtp_accepted_tokens == 1
            assert await backend.cancel_response("session")
        finally:
            await client.aclose()

    asyncio.run(run())
