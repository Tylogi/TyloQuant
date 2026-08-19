from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from safetensors.torch import save_file

from mfq.calibration.allocator import GroupCandidate, allocate, allocate_lp_rounded
from mfq.calibration.artifact import (
    CalibrationScheme,
    ExpertSelection,
    ExpertTensorSelection,
    TensorSelection,
    load_scheme,
    save_scheme,
)
from mfq.calibration.collector import HiddenTrace, validate_layerwise
from mfq.calibration.dataset import (
    CalibrationCorpus,
    TraceSource,
    build_corpus_from_records,
    build_trace_corpus_from_jsonl,
    eaddario_sources,
)
from mfq.calibration.evaluator import (
    LayerStrategy,
    TensorCandidateEvaluation,
    _load_packed_candidate,
    _materialize_scheme_candidate_cache,
    _read_candidate_cache,
    _unpack_packed_q,
    build_compensated_layer_strategies,
    build_global_layer_strategies,
    build_layer_strategies,
    evaluate_tensor_candidate,
    nint_storage_bits,
)
from mfq.calibration.inint import build_inint_selector, load_inint_selector
from mfq.calibration.qwen35 import Qwen35LinearTarget, qwen35_head_weight_name
from mfq.calibration.refinement import (
    StrategyTrace,
    _select_pareto_knee,
    refine_layerwise,
    refine_layerwise_global,
)
from mfq.calibration.statistics import (
    Qwen35StatisticsCollector,
    TensorStatistics,
    _chunked_head_gradient,
    _statistics_layout,
    load_statistics,
    save_statistics,
)
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize as quantize_nint_cpu
from mfq.runtime.torch_linear import TorchNintLinear
from mfq.formats.io import unpack_nint_moe
from mfq.quantize.expert_nint import dequantize_expertwise
from mfq.tools.quantize_hf_to_mfq import (
    _nint_moe_blob_nbytes,
    _plan,
    _write_nint_moe_axis0_blob,
)


def test_qwen35_head_weight_name_supports_tied_embeddings() -> None:
    explicit = type("Index", (), {"weight_map": {"lm_head.weight": "model.safetensors"}})()
    tied = type(
        "Index",
        (),
        {"weight_map": {"model.language_model.embed_tokens.weight": "model.safetensors"}},
    )()
    assert qwen35_head_weight_name(explicit) == "lm_head.weight"
    assert (
        qwen35_head_weight_name(tied)
        == "model.language_model.embed_tokens.weight"
    )


def test_statistics_layout_keeps_complete_short_masked_validation_trace(
    tmp_path: Path,
) -> None:
    corpus = CalibrationCorpus(
        root=tmp_path,
        manifest={"domains": ["test"]},
        tokens=np.arange(12, dtype=np.int32),
        offsets=np.asarray([0, 8, 12], dtype=np.int64),
        split_ids=np.asarray([0, 1], dtype=np.uint8),
        domain_ids=np.asarray([0, 0], dtype=np.uint16),
        loss_mask=np.ones(12, dtype=np.bool_),
    )
    layout, counts = _statistics_layout(
        corpus,
        window_length=8,
        token_limits={"train": 8, "validation": 4},
        seed=17,
        seed_offsets={"train": 0, "validation": 1},
    )

    assert counts == {"train": 8, "validation": 4}
    assert [(item.split, item.end - item.start) for item in layout] == [
        ("train", 8),
        ("validation", 4),
    ]


class _Tokenizer:
    eos_token_id = 2
    vocab_size = 512
    name_or_path = "fake-tokenizer"
    chat_template = ""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3 + ord(value) % 251 for value in text]


class _TraceTokenizer(_Tokenizer):
    chat_template = "test-trace-template"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_kwargs,
    ) -> list[int]:
        assert tokenize
        ids = [17]
        for index, message in enumerate(messages):
            role = str(message["role"])
            if role == "assistant" and index == len(messages) - 1:
                ids.append(19)
                reasoning = str(message.get("reasoning_content", ""))
                content = reasoning + str(message["content"])
            else:
                ids.append(20 + len(role))
                content = "context"
            ids.extend(30 + index % 211 for index, _word in enumerate(content.split()))
        if add_generation_prompt:
            ids.append(19)
        else:
            ids.append(self.eos_token_id)
        return ids



def _trace_row(index: int, mode: str, *, model: str = "qwen3.5-9b") -> dict:
    return {
        "model": model,
        "mode": mode,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Solve trace task {index}."},
        ],
        "reasoning": f"Reasoning for task {index}." if mode == "thinking" else None,
        "output": f"Answer for task {index} in {mode} mode.",
        "finish_reason": "stop",
        "usage": {"total_tokens": 20 + index % 7},
        "created": index,
        "source_dataset": "synthetic",
    }


def _write_trace_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _target(rows: int = 4, columns: int = 8) -> Qwen35LinearTarget:
    return Qwen35LinearTarget(
        name="model.language_model.layers.0.mlp.gate_proj.weight",
        source_name="model.language_model.layers.0.mlp.gate_proj.weight",
        module_name="model.layers.0.mlp.gate_proj",
        rows=rows,
        columns=columns,
        row_start=0,
        row_end=rows,
        group="layer.0.ffn_gate_up",
        role="ffn_gate",
        gguf_name="blk.0.ffn_gate.weight",
    )


def test_chunked_head_gradient_matches_full_softmax() -> None:
    generator = torch.Generator().manual_seed(31)
    hidden = torch.randn((2, 4, 5), generator=generator)
    head_weight = torch.randn((11, 5), generator=generator)
    input_ids = torch.randint(0, 11, (2, 4), generator=generator)

    reference_hidden = hidden.clone().requires_grad_(True)
    logits = functional.linear(reference_hidden[:, :-1], head_weight)
    losses = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    )
    reference_generator = torch.Generator().manual_seed(97)
    signs = torch.empty_like(losses)
    signs.bernoulli_(0.5, generator=reference_generator).mul_(2).sub_(1)
    reference_probe = (losses * signs).sum() / losses.numel() ** 0.5
    reference_probe.backward()

    for row_chunk in (4, 11, 0):
        chunk_generator = torch.Generator().manual_seed(97)
        gradient, probe = _chunked_head_gradient(
            hidden,
            input_ids,
            torch.nn.Identity(),
            head_weight,
            chunk_generator,
            row_chunk=row_chunk,
        )
        torch.testing.assert_close(gradient, reference_hidden.grad, rtol=2e-6, atol=2e-6)
        assert abs(probe - float(reference_probe.item())) < 2e-6


def test_statistics_collector_installs_on_streamed_layer() -> None:
    target = Qwen35LinearTarget(
        name="model.language_model.layers.3.proj.weight",
        source_name="model.language_model.layers.3.proj.weight",
        module_name="model.layers.3.proj",
        rows=4,
        columns=5,
        row_start=0,
        row_end=4,
        group="layer.3.proj",
        role="test",
        gguf_name="blk.3.test.weight",
    )
    layer = torch.nn.Module()
    layer.proj = torch.nn.Linear(5, 4, bias=False)
    collector = Qwen35StatisticsCollector((target,), torch.device("cpu"))
    collector.mode = "input"
    collector.install_layer(layer, 3)
    value = torch.randn((2, 3, 5), generator=torch.Generator().manual_seed(7))
    layer.proj(value)
    collector.close()
    assert collector.input_counts["train"][target.module_name] == 6
    torch.testing.assert_close(
        collector.input_sums["train"][target.module_name],
        value.double().square().sum(dim=(0, 1)),
    )


def _statistics(rows: int = 4, columns: int = 8) -> TensorStatistics:
    target = _target(rows, columns)
    return TensorStatistics(
        target=target,
        train_input_second_moment=np.linspace(0.5, 1.5, columns, dtype=np.float32),
        validation_input_second_moment=np.linspace(0.6, 1.4, columns, dtype=np.float32),
        train_row_fisher=np.linspace(0.5, 1.0, rows, dtype=np.float32),
        validation_row_fisher=np.linspace(0.6, 1.1, rows, dtype=np.float32),
        train_input_count=64,
        validation_input_count=32,
        train_fisher_probes=4,
        validation_fisher_probes=2,
    )


def test_calibration_corpus_is_exact_and_reproducible(tmp_path: Path) -> None:
    records = {
        "language": [f"language sample {index}" for index in range(20)],
        "code": [f"code sample {index}" for index in range(20)],
    }
    kwargs = dict(
        train_tokens=40,
        validation_tokens=20,
        sequence_length=8,
        domain_weights={"language": 0.6, "code": 0.4},
        seed=17,
        render_mode="plain",
    )
    first = build_corpus_from_records(records, _Tokenizer(), tmp_path / "first", **kwargs)
    second = build_corpus_from_records(records, _Tokenizer(), tmp_path / "second", **kwargs)

    assert first.token_count("train") == 40
    assert first.token_count("validation") == 20
    np.testing.assert_array_equal(first.tokens, second.tokens)
    np.testing.assert_array_equal(first.offsets, second.offsets)
    np.testing.assert_array_equal(first.split_ids, second.split_ids)
    assert (
        sum(
            batch.input_ids.size
            for batch in first.iter_batches("validation", window_length=8, batch_size=2)
        )
        == 20
    )
    first.close()
    second.close()
    assert first.tokens._mmap.closed
    assert second.tokens._mmap.closed


def test_calibration_corpus_can_right_pad_nearby_lengths(tmp_path: Path) -> None:
    corpus = CalibrationCorpus(
        root=tmp_path,
        manifest={"domains": ["language"]},
        tokens=np.arange(20, dtype=np.int32),
        offsets=np.asarray([0, 3, 8, 20], dtype=np.int64),
        split_ids=np.asarray([0, 0, 0], dtype=np.int8),
        domain_ids=np.asarray([0, 0, 0], dtype=np.int16),
    )
    batches = tuple(
        corpus.iter_batches(
            "train",
            window_length=16,
            batch_size=2,
            pad_to_multiple=4,
        )
    )
    assert batches
    assert all(batch.input_ids.shape[0] <= 2 for batch in batches)
    assert all(batch.input_ids.shape[1] % 4 == 0 for batch in batches)
    assert sum(int(batch.attention_mask.sum()) for batch in batches) == 20
    for batch in batches:
        assert batch.padding_mask is not None
        assert np.all(batch.input_ids[~batch.padding_mask] == 0)
        assert batch.loss_mask is None


def test_eaddario_source_presets_keep_domain_weights_and_scale() -> None:
    medium = eaddario_sources("medium")
    assert [item.filename for item in medium] == [
        "text_all_medium.parquet",
        "code_medium.parquet",
        "math_medium.parquet",
        "tools_medium.parquet",
    ]
    assert sum(item.weight for item in medium) == 1.0
    with np.testing.assert_raises_regex(ValueError, "unknown eaddario source size"):
        eaddario_sources("giant")


def test_trace_corpus_preserves_records_and_is_reproducible(tmp_path: Path) -> None:
    nonthinking = tmp_path / "nonthinking.jsonl"
    thinking = tmp_path / "thinking.jsonl"
    nonthinking_rows = [_trace_row(index, "nonthinking") for index in range(80)]
    length_limited = _trace_row(1000, "nonthinking")
    length_limited["finish_reason"] = "length"
    nonthinking_rows.append(length_limited)
    thinking_rows = [_trace_row(index, "thinking") for index in range(80)]
    empty_output = _trace_row(1000, "thinking")
    empty_output["output"] = ""
    thinking_rows.append(empty_output)
    _write_trace_rows(nonthinking, nonthinking_rows)
    _write_trace_rows(thinking, thinking_rows)
    sources = (
        (nonthinking, TraceSource(nonthinking.name, "nonthinking")),
        (thinking, TraceSource(thinking.name, "thinking")),
    )
    tokenizer = _TraceTokenizer()
    kwargs = dict(
        expected_generator_model="qwen3.5-9b",
        train_tokens=512,
        validation_tokens=128,
        sequence_length=64,
        seed=41,
        source_metadata={"repo_id": "test/traces", "resolved_revision": "abc123"},
    )
    first = build_trace_corpus_from_jsonl(sources, tokenizer, tmp_path / "first", **kwargs)
    second = build_trace_corpus_from_jsonl(sources, tokenizer, tmp_path / "second", **kwargs)

    assert first.token_count("train") == 512
    assert first.token_count("validation") == 128
    assert first.manifest["corpus_kind"] == "model_trace"
    assert first.manifest["trace"]["boundary_preserved"] is True
    assert first.manifest["trace"]["partial_records"] == {"train": 1, "validation": 1}
    assert set(first.manifest["domains"]) == {"thinking", "nonthinking"}
    assert first.manifest["sources"]["audit"]["records"] == 160
    assert first.manifest["sources"]["audit"]["excluded_records"] == 2
    assert first.manifest["sources"]["audit"]["exclusion_reasons"] == {
        "empty_output": 1,
        "finish_reason_length": 1,
    }
    assert (
        first.manifest["prompt_sets"]["train"]["sha256"]
        != first.manifest["prompt_sets"]["validation"]["sha256"]
    )
    np.testing.assert_array_equal(first.tokens, second.tokens)
    np.testing.assert_array_equal(first.offsets, second.offsets)
    np.testing.assert_array_equal(first.split_ids, second.split_ids)
    assert first.manifest["token_sha256"] == second.manifest["token_sha256"]

    full_sequences = []
    for index in range(80):
        for mode in ("nonthinking", "thinking"):
            row = _trace_row(index, mode)
            messages = row["messages"] + [
                {
                    "role": "assistant",
                    "content": row["output"],
                    "reasoning_content": row["reasoning"] or "",
                }
            ]
            full_sequences.append(
                np.asarray(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                    ),
                    dtype=np.int64,
                )
            )
    for chunk_index in range(first.chunks):
        chunk = first.chunk_tokens(chunk_index)
        assert any(np.array_equal(chunk, value[: chunk.size]) for value in full_sequences)
    first.close()
    second.close()


def test_trace_corpus_rejects_wrong_generator_model(tmp_path: Path) -> None:
    source = tmp_path / "wrong.jsonl"
    _write_trace_rows(source, [_trace_row(0, "nonthinking", model="other-model")])
    with np.testing.assert_raises_regex(ValueError, "trace generator model"):
        build_trace_corpus_from_jsonl(
            ((source, TraceSource(source.name, "nonthinking")),),
            _TraceTokenizer(),
            tmp_path / "corpus",
            expected_generator_model="qwen3.5-9b",
            train_tokens=8,
            validation_tokens=4,
            sequence_length=64,
        )


def test_execution_packed_candidates_reuse_exact_quantized_values(tmp_path: Path) -> None:
    generator = np.random.default_rng(29)
    weight = generator.normal(size=(4, 8)).astype(np.float32)
    for bits in (4, 5, 6, 8):
        spec = NintSpec(bits, 4, 4)
        encoded = quantize_nint_cpu(weight, spec, axis=0)
        arrays = TorchNintLinear.deploy_arrays(encoded)
        np.testing.assert_array_equal(
            _unpack_packed_q(arrays["q_packed"], bits, spec.groupsize),
            encoded.q,
        )

    statistics = _statistics(rows=4, columns=8)
    spec = NintSpec(4, 4, 4)
    encoded = quantize_nint_cpu(weight, spec, axis=0)
    arrays = TorchNintLinear.deploy_arrays(encoded)
    metadata = np.frombuffer(
        json.dumps(
            {
                "name": statistics.target.name,
                "bits": spec.bits,
                "groupsize": spec.groupsize,
                "sub_bits": spec.sub_bits,
                "shape": [4, 8],
                "axis": 0,
                "neuron_len": 8,
            },
            separators=(",", ":"),
        ).encode("utf-8"),
        dtype=np.uint8,
    )
    path = tmp_path / "candidate.nint-exec.npz"
    with path.open("wb") as stream:
        np.savez(stream, metadata=metadata, **arrays)
    loaded = _load_packed_candidate(path, statistics, "NINT4", spec)
    np.testing.assert_array_equal(loaded.q, encoded.q)
    np.testing.assert_array_equal(loaded.sub_scale, encoded.sub_scale)
    np.testing.assert_array_equal(loaded.sub_min, encoded.sub_min)
    direct = evaluate_tensor_candidate(
        torch.from_numpy(weight), statistics, "NINT4", spec, backend="cpu"
    )
    reused = evaluate_tensor_candidate(
        torch.from_numpy(weight),
        statistics,
        "NINT4",
        spec,
        backend="cpu",
        encoded=loaded,
    )
    np.testing.assert_array_equal(reused.train_row_loss, direct.train_row_loss)
    np.testing.assert_array_equal(reused.validation_row_loss, direct.validation_row_loss)


def test_statistics_roundtrip_and_cpu_candidate_evaluation(tmp_path: Path) -> None:
    statistics = _statistics()
    path = tmp_path / "statistics.npz"
    save_statistics(path, {statistics.target.name: statistics}, {"seed": 17})
    loaded = load_statistics(path)
    np.testing.assert_allclose(
        loaded.require(statistics.target.name).train_input_second_moment,
        statistics.train_input_second_moment,
    )

    generator = torch.Generator().manual_seed(11)
    weight = torch.randn((4, 8), generator=generator)
    evaluation = evaluate_tensor_candidate(
        weight,
        loaded.require(statistics.target.name),
        "TEST",
        NintSpec(4, 4, 4),
        backend="cpu",
        device="cpu",
        row_chunk=2,
    )
    assert evaluation.train_row_loss.shape == (4,)
    assert evaluation.validation_row_loss.shape == (4,)
    assert np.isfinite(evaluation.train_row_loss).all()
    assert evaluation.train_loss >= 0
    assert evaluation.storage_bits == nint_storage_bits(4, 8, NintSpec(4, 4, 4))


def test_global_allocator_obeys_multiple_choice_budget() -> None:
    spec4 = NintSpec(4, 24, 6)
    spec8 = NintSpec(8, 48, 7)
    candidates = [
        GroupCandidate("a", "low", {"a": spec4}, 10, 10.0, 11.0),
        GroupCandidate("a", "high", {"a": spec8}, 20, 1.0, 1.2),
        GroupCandidate("b", "low", {"b": spec4}, 10, 7.0, 8.0),
        GroupCandidate("b", "high", {"b": spec8}, 20, 2.0, 2.2),
    ]
    result = allocate(candidates, 30)
    assert result.actual_storage_bits == 30
    assert result.selected["a"].profile == "high"
    assert result.selected["b"].profile == "low"


def test_global_allocator_limits_changed_groups() -> None:
    spec4 = NintSpec(4, 24, 6)
    spec8 = NintSpec(8, 48, 7)
    candidates = [
        GroupCandidate("a", "low", {"a": spec4}, 10, 10.0, 10.0),
        GroupCandidate("a", "high", {"a": spec8}, 20, 1.0, 1.0),
        GroupCandidate("b", "low", {"b": spec4}, 10, 8.0, 8.0),
        GroupCandidate("b", "high", {"b": spec8}, 20, 2.0, 2.0),
    ]
    result = allocate(
        candidates,
        40,
        baseline_profiles={"a": "low", "b": "low"},
        max_changed_groups=1,
    )
    assert result.selected["a"].profile == "high"
    assert result.selected["b"].profile == "low"


def test_lp_rounded_allocator_returns_feasible_integer_assignment() -> None:
    spec4 = NintSpec(4, 24, 6)
    spec8 = NintSpec(8, 48, 7)
    candidates = [
        GroupCandidate("a", "low", {"a": spec4}, 10, 8.0, 8.0),
        GroupCandidate("a", "high", {"a": spec8}, 20, 0.0, 0.0),
        GroupCandidate("b", "low", {"b": spec4}, 10, 7.0, 7.0),
        GroupCandidate("b", "high", {"b": spec8}, 20, 0.0, 0.0),
    ]

    result = allocate_lp_rounded(candidates, 35)

    assert set(result.selected) == {"a", "b"}
    assert result.actual_storage_bits <= 35
    assert result.actual_storage_bits == sum(
        item.storage_bits for item in result.selected.values()
    )
    assert result.solver == "scipy.optimize.linprog/highs+integer-rounding"



def _evaluation(
    name: str,
    profile: str,
    spec: NintSpec,
    train_rows: list[float],
    validation_rows: list[float],
    columns: int = 24,
) -> TensorCandidateEvaluation:
    rows = len(train_rows)
    return TensorCandidateEvaluation(
        name=name,
        group="group.0",
        profile=profile,
        spec=spec,
        rows=rows,
        columns=columns,
        storage_bits=nint_storage_bits(rows, columns, spec),
        train_loss=float(sum(train_rows)),
        validation_loss=float(sum(validation_rows)),
        train_nmse_percent=1.0,
        validation_nmse_percent=1.0,
        train_row_loss=np.asarray(train_rows, dtype=np.float32),
        validation_row_loss=np.asarray(validation_rows, dtype=np.float32),
    )


def test_scheme_candidate_cache_records_and_checks_identity(tmp_path: Path) -> None:
    value = _evaluation(
        "tensor",
        "NINT4",
        NintSpec(4, 24, 6),
        [1.0, 2.0],
        [1.5, 2.5],
    )
    path = tmp_path / "stable.npz"
    _materialize_scheme_candidate_cache(path, value, "identity-a")

    loaded = _read_candidate_cache(path)
    assert loaded is not None
    assert loaded[0]["identity"] == "identity-a"
    _materialize_scheme_candidate_cache(path, value, "identity-a")
    with np.testing.assert_raises_regex(ValueError, "another model"):
        _materialize_scheme_candidate_cache(path, value, "identity-b")


def test_inint_selects_rows_by_function_loss_under_budget(tmp_path: Path) -> None:
    name = "tensor"
    evaluations = {
        name: {
            "NINT4": _evaluation(
                name,
                "NINT4",
                NintSpec(4, 24, 6),
                [10.0, 9.0, 2.0, 1.0, 0.9, 0.8, 0.7, 0.6],
                [8, 7, 2, 1, 0.9, 0.8, 0.7, 0.6],
            ),
            "NINT5": _evaluation(
                name,
                "NINT5",
                NintSpec(5, 28, 7),
                [6.0, 6.0, 1.5, 0.8, 0.8, 0.7, 0.6, 0.5],
                [5, 5, 1.5, 0.8, 0.8, 0.7, 0.6, 0.5],
            ),
            "NINT8": _evaluation(
                name,
                "NINT8",
                NintSpec(8, 48, 7),
                [1.0, 8.0, 1.8, 0.9, 0.85, 0.75, 0.65, 0.55],
                [1, 6, 1.8, 0.9, 0.85, 0.75, 0.65, 0.55],
            ),
        }
    }
    path = tmp_path / "selector.npz"
    selector = build_inint_selector(
        evaluations,
        path,
        target_profile="NINT5",
        exact_row_limit=100,
    )
    loaded = load_inint_selector(path)
    np.testing.assert_array_equal(selector.selectors[name], loaded.selectors[name])
    assert loaded.selectors[name][0]
    assert loaded.metadata["actual_storage_bits"] <= loaded.metadata["target_storage_bits"]
    assert loaded.metadata["train_loss_reduction"] > 0


class _ToyBackend:
    num_layers = 2
    hidden_size = 2
    teacher_dtype = torch.float32
    quantized_dtype = torch.float32
    device = torch.device("cpu")

    def initial_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        value = input_ids.float()
        return torch.stack((value, torch.ones_like(value)), dim=-1)

    def release_initial_state(self) -> None:
        return None

    @contextmanager
    def layer(self, layer_index: int, *, quantized: bool) -> Iterator[tuple[int, bool]]:
        yield layer_index, quantized

    def forward_layer(
        self,
        layer: tuple[int, bool],
        layer_index: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert layer[0] == layer_index
        output = hidden_states * float(layer_index + 2)
        return output + 0.25 if layer[1] else output


def test_layerwise_replay_accumulates_quantized_path_error(tmp_path: Path) -> None:
    corpus = build_corpus_from_records(
        {"language": [f"sample {index}" for index in range(10)]},
        _Tokenizer(),
        tmp_path / "corpus",
        train_tokens=12,
        validation_tokens=8,
        sequence_length=4,
        seed=5,
        render_mode="plain",
    )
    report = tmp_path / "report.json"
    traces = validate_layerwise(
        _ToyBackend(),
        corpus,
        report,
        work_dir=tmp_path / "hidden",
        window_length=4,
        batch_size=2,
        max_tokens=8,
    )
    assert len(traces) == 2
    assert traces[1].squared_error > traces[0].squared_error
    assert json.loads(report.read_text(encoding="utf-8"))["token_count"] == 8
    assert not (tmp_path / "hidden" / "teacher-hidden.bin").exists()


def test_hf_conversion_plan_uses_calibrated_nint_spec(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    name = "model.language_model.layers.0.mlp.gate_proj.weight"
    save_file({name: torch.zeros((4, 8), dtype=torch.bfloat16)}, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps({"text_config": {}}), encoding="utf-8")
    spec = NintSpec(6, 24, 7)
    scheme_path = tmp_path / "scheme.json"
    save_scheme(
        scheme_path,
        CalibrationScheme(
            path=None,
            target_profile="NINT6",
            target_storage_bits=nint_storage_bits(4, 8, spec),
            selections={
                name: TensorSelection(
                    name=name,
                    group="layer.0.ffn_gate_up",
                    spec=spec,
                    rows=4,
                    columns=8,
                    storage_bits=nint_storage_bits(4, 8, spec),
                    train_loss=1.0,
                    validation_loss=1.1,
                )
            },
            metadata={},
            candidate_table={},
        ),
    )
    scheme = load_scheme(scheme_path)
    plan = _plan(root, True, None, "F16", scheme)
    assert len(plan) == 1
    assert plan[0].target_dtype == "NINT6"
    assert plan[0].target_spec == spec


def test_expertwise_scheme_roundtrip_plan_and_stream_writer(tmp_path: Path) -> None:
    root = tmp_path / "moe-model"
    root.mkdir()
    name = "model.language_model.layers.0.mlp.experts.down_proj.weight"
    shape = (4, 3, 48)
    weight = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape) / 1000
    save_file({name: weight.to(torch.bfloat16)}, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps({"text_config": {}}), encoding="utf-8")

    specs = (
        NintSpec(4, 24, 6),
        NintSpec(6, 24, 7),
        NintSpec(4, 24, 6),
        NintSpec(6, 24, 7),
    )
    experts = tuple(
        ExpertSelection(
            expert_id=expert,
            spec=profile,
            storage_bits=nint_storage_bits(shape[1], shape[2], profile),
            train_loss=float(expert + 1),
            validation_loss=float(expert + 1.5),
        )
        for expert, profile in enumerate(specs)
    )
    selection = ExpertTensorSelection(
        name=name,
        group="layer.0.expert_down",
        n_experts=shape[0],
        rows_per_expert=shape[1],
        columns=shape[2],
        selections=experts,
    )
    scheme_path = tmp_path / "expert-scheme.json"
    save_scheme(
        scheme_path,
        CalibrationScheme(
            path=None,
            target_profile="EXPERT_WISE",
            target_storage_bits=selection.storage_bits,
            selections={},
            metadata={},
            candidate_table={},
            expert_selections={name: selection},
        ),
    )
    scheme = load_scheme(scheme_path)
    assert scheme.require_expert(name).specs == specs
    assert scheme.storage_bits == selection.storage_bits
    assert scheme.weight_count == int(np.prod(shape))

    plan = _plan(root, True, None, "F16", scheme)
    assert len(plan) == 1
    assert plan[0].target_dtype == "NINTM"
    assert plan[0].expert_shape == shape
    assert plan[0].expert_specs == specs

    blob_path = tmp_path / "expert.blob"
    nbytes = _write_nint_moe_axis0_blob(
        weight,
        shape,
        shape,
        specs,
        blob_path,
        row_chunk=4,
        quant_backend="cpu",
        device="cpu",
    )
    tensor = unpack_nint_moe(blob_path.read_bytes())
    assert nbytes == _nint_moe_blob_nbytes(shape, specs)
    assert tensor.expert_profiles == tuple(profile.profile_label for profile in specs)
    assert dequantize_expertwise(tensor).shape == shape


def test_hf_conversion_plan_splits_linear_qk_and_v_specs(tmp_path: Path) -> None:
    root = tmp_path / "linear-model"
    root.mkdir()
    source = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    save_file({source: torch.zeros((7, 8), dtype=torch.bfloat16)}, root / "model.safetensors")
    (root / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "linear_num_key_heads": 1,
                    "linear_key_head_dim": 2,
                    "linear_num_value_heads": 1,
                    "linear_value_head_dim": 3,
                }
            }
        ),
        encoding="utf-8",
    )
    qk_name = "model.language_model.layers.0.linear_attn.in_proj_qk.weight"
    v_name = "model.language_model.layers.0.linear_attn.in_proj_v.weight"
    qk_spec = NintSpec(4, 24, 6)
    v_spec = NintSpec(6, 24, 7)
    selections = {
        qk_name: TensorSelection(
            qk_name,
            "layer.0.linear_qk",
            qk_spec,
            4,
            8,
            nint_storage_bits(4, 8, qk_spec),
            1.0,
            1.0,
        ),
        v_name: TensorSelection(
            v_name,
            "layer.0.linear_v",
            v_spec,
            3,
            8,
            nint_storage_bits(3, 8, v_spec),
            1.0,
            1.0,
        ),
    }
    scheme = CalibrationScheme(
        path=None,
        target_profile="NINT5",
        target_storage_bits=sum(item.storage_bits for item in selections.values()),
        selections=selections,
        metadata={},
        candidate_table={},
    )
    plan = _plan(root, True, None, "F16", scheme)
    assert [(item.name, item.row_start, item.row_end, item.target_spec) for item in plan] == [
        (qk_name, 0, 4, qk_spec),
        (v_name, 4, 7, v_spec),
    ]


class _RefinementBackend(_ToyBackend):
    num_layers = 1

    def __init__(self) -> None:
        super().__init__()
        self.strategy_loads: list[int] = []

    def prepare_layer_strategies(
        self, layer_index: int, strategies: list[dict[str, NintSpec]]
    ) -> None:
        assert layer_index == 0
        assert strategies

    @contextmanager
    def layer_for_strategy(self, layer_index: int, specs: dict[str, NintSpec]) -> Iterator[int]:
        assert layer_index == 0
        bits = next(iter(specs.values())).bits
        self.strategy_loads.append(bits)
        yield bits

    def clear_quantized_cache(self, layer_index: int | None = None) -> None:
        del layer_index

    def forward_layer(self, layer, layer_index, hidden_states):
        if isinstance(layer, int):
            error = {4: 0.5, 5: 0.05, 8: 0.01}[layer]
            return hidden_states * 2.0 + error
        return super().forward_layer(layer, layer_index, hidden_states)


def test_layerwise_refinement_uses_real_hidden_error(tmp_path: Path) -> None:
    name = "model.language_model.layers.0.mlp.gate_proj.weight"
    specs = {
        "NINT4": NintSpec(4, 24, 6),
        "NINT5": NintSpec(5, 28, 7),
        "NINT8": NintSpec(8, 48, 7),
    }
    evaluations = {
        name: {
            profile: _evaluation(
                name,
                profile,
                spec,
                [0.1 if profile == "NINT4" else 1.0] * 8,
                [0.2 if profile == "NINT4" else 1.1] * 8,
            )
            for profile, spec in specs.items()
        }
    }
    base_value = evaluations[name]["NINT5"]
    base = CalibrationScheme(
        path=None,
        target_profile="NINT5",
        target_storage_bits=base_value.storage_bits,
        selections={
            name: TensorSelection(
                name=name,
                group=base_value.group,
                spec=base_value.spec,
                rows=base_value.rows,
                columns=base_value.columns,
                storage_bits=base_value.storage_bits,
                train_loss=base_value.train_loss,
                validation_loss=base_value.validation_loss,
            )
        },
        metadata={},
        candidate_table={},
    )
    strategies = build_layer_strategies(evaluations, base, top_k=2)
    assert any(item.specs[name].bits == 4 for item in strategies[0])
    assert any(item.specs[name].bits == 5 for item in strategies[0])

    corpus = build_corpus_from_records(
        {"language": [f"refine sample {index}" for index in range(10)]},
        _Tokenizer(),
        tmp_path / "refine-corpus",
        train_tokens=8,
        validation_tokens=4,
        sequence_length=4,
        seed=7,
        render_mode="plain",
    )
    backend = _RefinementBackend()
    refined, records = refine_layerwise(
        backend,
        corpus,
        base,
        evaluations,
        strategies,
        tmp_path / "refined.json",
        tmp_path / "refinement-report.json",
        work_dir=tmp_path / "refinement-hidden",
        window_length=4,
        max_tokens=8,
    )
    assert len(backend.strategy_loads) == len(strategies[0]) + 1
    assert backend.strategy_loads[-1] == 5
    assert records[0].selected.specs[name].bits == 5
    assert refined.selections[name].spec.bits == 5


def test_global_layer_strategy_frontier_crosses_original_layer_budget() -> None:
    name = "model.language_model.layers.0.mlp.gate_proj.weight"
    profiles = {
        "NINT4": (NintSpec(4, 24, 6), 9.0),
        "NINT5": (NintSpec(5, 28, 7), 4.0),
        "NINT6": (NintSpec(6, 24, 7), 3.0),
        "NINT8": (NintSpec(8, 48, 7), 0.5),
    }
    evaluations = {
        name: {
            profile: _evaluation(
                name,
                profile,
                spec,
                [loss / 8.0] * 8,
                [loss / 8.0] * 8,
            )
            for profile, (spec, loss) in profiles.items()
        }
    }
    base_value = evaluations[name]["NINT5"]
    base = CalibrationScheme(
        path=None,
        target_profile="NINT5",
        target_storage_bits=base_value.storage_bits,
        selections={
            name: TensorSelection(
                name,
                base_value.group,
                base_value.spec,
                base_value.rows,
                base_value.columns,
                base_value.storage_bits,
                base_value.train_loss,
                base_value.validation_loss,
            )
        },
        metadata={},
        candidate_table={},
    )
    strategies = build_global_layer_strategies(evaluations, base)[0]
    assert min(item.storage_bits for item in strategies) < base.storage_bits
    assert max(item.storage_bits for item in strategies) > base.storage_bits
    assert any(item.specs[name] == base_value.spec for item in strategies)


class _GlobalRefinementBackend(_ToyBackend):
    num_layers = 2

    def __init__(self) -> None:
        super().__init__()
        self.strategy_loads: list[tuple[int, int]] = []

    def prepare_layer_strategies(
        self,
        layer_index: int,
        strategies: list[dict[str, NintSpec]],
    ) -> None:
        assert strategies
        assert all(len(item) == 1 for item in strategies)
        assert 0 <= layer_index < self.num_layers

    @contextmanager
    def layer_for_strategy(
        self,
        layer_index: int,
        specs: dict[str, NintSpec],
    ) -> Iterator[tuple[str, int, int]]:
        bits = next(iter(specs.values())).bits
        self.strategy_loads.append((layer_index, bits))
        yield "strategy", layer_index, bits

    def clear_quantized_cache(self, layer_index: int | None = None) -> None:
        del layer_index

    def forward_layer(self, layer, layer_index, hidden_states):
        if len(layer) == 3:
            assert layer[:2] == ("strategy", layer_index)
            errors = {
                0: {5: 0.5, 6: 0.01},
                1: {4: 0.02, 5: 0.01},
            }
            return hidden_states * float(layer_index + 2) + errors[layer_index][layer[2]]
        return super().forward_layer(layer, layer_index, hidden_states)


class _RejectingRefinementBackend(_GlobalRefinementBackend):
    def forward_layer(self, layer, layer_index, hidden_states):
        if len(layer) == 3:
            bits = layer[2]
            if layer_index == 0:
                return hidden_states if bits == 5 else hidden_states + 1.0
            return hidden_states * 10.0 + (12.0 if bits == 5 else 0.0)
        if layer_index == 0:
            return hidden_states
        return hidden_states * 10.0


class _CompensatedRefinementBackend(_GlobalRefinementBackend):
    def forward_layer(self, layer, layer_index, hidden_states):
        if len(layer) == 3:
            errors = {
                0: {4: 0.01, 5: 0.5},
                1: {5: 0.5, 6: 0.01},
            }
            return hidden_states * float(layer_index + 2) + errors[layer_index][layer[2]]
        return super().forward_layer(layer, layer_index, hidden_states)


def test_global_refinement_transfers_storage_and_replays_path(tmp_path: Path) -> None:
    specs = {
        "NINT4": NintSpec(4, 24, 6),
        "NINT5": NintSpec(5, 28, 7),
        "NINT6": NintSpec(6, 24, 7),
    }
    names = {
        layer: f"model.language_model.layers.{layer}.mlp.gate_proj.weight" for layer in range(2)
    }
    layer_profiles = {0: ("NINT5", "NINT6"), 1: ("NINT4", "NINT5")}
    evaluations: dict[str, dict[str, TensorCandidateEvaluation]] = {}
    strategies: dict[int, list[LayerStrategy]] = {}
    selections: dict[str, TensorSelection] = {}
    for layer, name in names.items():
        evaluations[name] = {
            profile: _evaluation(
                name,
                profile,
                specs[profile],
                [1.0] * 8,
                [1.0] * 8,
            )
            for profile in layer_profiles[layer]
        }
        strategies[layer] = []
        for profile in layer_profiles[layer]:
            value = evaluations[name][profile]
            strategies[layer].append(
                LayerStrategy(
                    layer=layer,
                    name=profile,
                    specs={name: value.spec},
                    profiles={f"layer.{layer}": profile},
                    storage_bits=value.storage_bits,
                    train_loss=value.train_loss,
                    validation_loss=value.validation_loss,
                )
            )
        base_value = evaluations[name]["NINT5"]
        selections[name] = TensorSelection(
            name,
            base_value.group,
            base_value.spec,
            base_value.rows,
            base_value.columns,
            base_value.storage_bits,
            base_value.train_loss,
            base_value.validation_loss,
        )

    base = CalibrationScheme(
        path=None,
        target_profile="NINT5",
        target_storage_bits=sum(item.storage_bits for item in selections.values()),
        selections=selections,
        metadata={},
        candidate_table={},
    )
    corpus = build_corpus_from_records(
        {"language": [f"global refine sample {index}" for index in range(10)]},
        _Tokenizer(),
        tmp_path / "global-refine-corpus",
        train_tokens=8,
        validation_tokens=4,
        sequence_length=4,
        seed=11,
        render_mode="plain",
    )
    backend = _GlobalRefinementBackend()
    refined, rounds = refine_layerwise_global(
        backend,
        corpus,
        base,
        evaluations,
        strategies,
        tmp_path / "global-refined.json",
        tmp_path / "global-refinement-report.json",
        work_dir=tmp_path / "global-refinement-hidden",
        window_length=4,
        max_tokens=8,
        max_iterations=3,
    )

    assert refined.storage_bits <= base.target_storage_bits
    assert refined.selections[names[0]].spec.bits == 6
    assert refined.selections[names[1]].spec.bits == 4
    assert len(rounds) == 2
    assert rounds[0].changed_layers == (0, 1)
    assert rounds[0].attempts[0].accepted
    assert (
        rounds[0].attempts[0].replayed_normalized_squared_error
        < rounds[0].attempts[0].current_normalized_squared_error
    )
    assert rounds[1].converged
    report = json.loads((tmp_path / "global-refinement-report.json").read_text(encoding="utf-8"))
    assert report["converged"] is True
    assert len(report["final_path"]) == 2
    assert not (
        tmp_path / "global-refinement-hidden" / "global-refine-quantized-hidden.bin"
    ).exists()


def test_compensated_greedy_carries_saved_bits_forward(tmp_path: Path) -> None:
    specs = {
        "NINT4": NintSpec(4, 24, 6),
        "NINT5": NintSpec(5, 28, 7),
        "NINT6": NintSpec(6, 24, 7),
    }
    profiles = {0: ("NINT4", "NINT5"), 1: ("NINT5", "NINT6")}
    surrogate_losses = {
        0: {"NINT4": 2.0, "NINT5": 1.0},
        1: {"NINT5": 2.0, "NINT6": 1.0},
    }
    names = {
        layer: f"model.language_model.layers.{layer}.mlp.gate_proj.weight" for layer in range(2)
    }
    evaluations: dict[str, dict[str, TensorCandidateEvaluation]] = {}
    selections: dict[str, TensorSelection] = {}
    for layer, name in names.items():
        evaluations[name] = {
            profile: _evaluation(
                name,
                profile,
                specs[profile],
                [surrogate_losses[layer][profile] / 8.0] * 8,
                [surrogate_losses[layer][profile] / 8.0] * 8,
            )
            for profile in profiles[layer]
        }
        base_value = evaluations[name]["NINT5"]
        selections[name] = TensorSelection(
            name,
            base_value.group,
            base_value.spec,
            base_value.rows,
            base_value.columns,
            base_value.storage_bits,
            base_value.train_loss,
            base_value.validation_loss,
        )

    base = CalibrationScheme(
        path=None,
        target_profile="NINT5",
        target_storage_bits=sum(item.storage_bits for item in selections.values()),
        selections=selections,
        metadata={},
        candidate_table={},
    )
    strategies = build_compensated_layer_strategies(evaluations, base)
    corpus = build_corpus_from_records(
        {"language": [f"compensated sample {index}" for index in range(10)]},
        _Tokenizer(),
        tmp_path / "compensated-corpus",
        train_tokens=8,
        validation_tokens=4,
        sequence_length=4,
        seed=17,
        render_mode="plain",
    )
    refined, records = refine_layerwise(
        _CompensatedRefinementBackend(),
        corpus,
        base,
        evaluations,
        strategies,
        tmp_path / "compensated-scheme.json",
        tmp_path / "compensated-report.json",
        work_dir=tmp_path / "compensated-hidden",
        window_length=4,
        max_tokens=8,
        cumulative_budget=True,
    )

    assert refined.selections[names[0]].spec.bits == 4
    assert refined.selections[names[1]].spec.bits == 5
    assert refined.storage_bits <= base.target_storage_bits
    assert records[0].target_storage_bits == 1_488
    assert records[0].remaining_credit_bits == 368
    assert records[1].target_storage_bits == 1_488
    assert records[1].available_storage_bits == 1_856
    assert records[1].remaining_credit_bits == 368
    report = json.loads((tmp_path / "compensated-report.json").read_text(encoding="utf-8"))
    assert report["allocation"] == "cumulative_budget_greedy"
    assert report["remaining_credit_bits"] == 368


def test_compensated_greedy_uses_target_profile_layer_budget(tmp_path: Path) -> None:
    specs = {
        "NINT4": NintSpec(4, 24, 6),
        "NINT5": NintSpec(5, 28, 7),
        "NINT6": NintSpec(6, 24, 7),
    }
    profiles = {0: ("NINT5", "NINT6"), 1: ("NINT4", "NINT5")}
    base_profiles = {0: "NINT6", 1: "NINT4"}
    surrogate_losses = {
        0: {"NINT5": 2.0, "NINT6": 1.0},
        1: {"NINT4": 2.0, "NINT5": 1.0},
    }
    names = {
        layer: f"model.language_model.layers.{layer}.mlp.gate_proj.weight" for layer in range(2)
    }
    evaluations: dict[str, dict[str, TensorCandidateEvaluation]] = {}
    selections: dict[str, TensorSelection] = {}
    for layer, name in names.items():
        evaluations[name] = {
            profile: _evaluation(
                name,
                profile,
                specs[profile],
                [surrogate_losses[layer][profile] / 8.0] * 8,
                [surrogate_losses[layer][profile] / 8.0] * 8,
            )
            for profile in profiles[layer]
        }
        base_value = evaluations[name][base_profiles[layer]]
        selections[name] = TensorSelection(
            name,
            base_value.group,
            base_value.spec,
            base_value.rows,
            base_value.columns,
            base_value.storage_bits,
            base_value.train_loss,
            base_value.validation_loss,
        )

    target_storage_bits = sum(evaluations[name]["NINT5"].storage_bits for name in names.values())
    base = CalibrationScheme(
        path=None,
        target_profile="NINT5",
        target_storage_bits=target_storage_bits,
        selections=selections,
        metadata={},
        candidate_table={},
    )
    strategies = build_compensated_layer_strategies(evaluations, base)
    corpus = build_corpus_from_records(
        {"language": [f"target-profile budget sample {index}" for index in range(10)]},
        _Tokenizer(),
        tmp_path / "target-profile-budget-corpus",
        train_tokens=8,
        validation_tokens=4,
        sequence_length=4,
        seed=19,
        render_mode="plain",
    )
    refined, records = refine_layerwise(
        _GlobalRefinementBackend(),
        corpus,
        base,
        evaluations,
        strategies,
        tmp_path / "target-profile-budget-scheme.json",
        tmp_path / "target-profile-budget-report.json",
        work_dir=tmp_path / "target-profile-budget-hidden",
        window_length=4,
        max_tokens=8,
        cumulative_budget=True,
    )

    assert base.storage_bits < base.target_storage_bits
    assert records[0].base_storage_bits == 1_520
    assert records[0].target_storage_bits == 1_488
    assert records[0].available_storage_bits == 1_488
    assert records[0].selected.specs[names[0]].bits == 5
    assert records[0].remaining_credit_bits == 0
    assert records[1].base_storage_bits == 1_120
    assert records[1].target_storage_bits == 1_488
    assert records[1].available_storage_bits == 1_488
    assert records[1].selected.specs[names[1]].bits == 4
    assert records[1].remaining_credit_bits == 368
    assert refined.storage_bits == target_storage_bits - 368


def test_real_pareto_knee_rejects_dominated_points_and_obeys_budget() -> None:
    spec = NintSpec(4, 24, 6)

    def candidate(name: str, storage_bits: int, squared_error: float) -> StrategyTrace:
        strategy = LayerStrategy(
            layer=0,
            name=name,
            specs={name: spec},
            profiles={name: "NINT4"},
            storage_bits=storage_bits,
            train_loss=squared_error,
            validation_loss=squared_error,
        )
        return StrategyTrace(
            strategy,
            HiddenTrace(
                layer=0,
                reference_energy=100.0,
                quantized_energy=100.0,
                squared_error=squared_error,
                dot_product=100.0,
                value_count=1,
            ),
        )

    candidates = [
        candidate("cheap", 100, 100.0),
        candidate("knee", 200, 40.0),
        candidate("dominated", 250, 50.0),
        candidate("middle", 300, 20.0),
        candidate("high", 400, 15.0),
        candidate("maximum", 500, 14.0),
    ]
    full = _select_pareto_knee(candidates, 500)
    limited = _select_pareto_knee(candidates, 150)

    assert [item.strategy.name for item in full.frontier] == [
        "cheap",
        "knee",
        "middle",
        "high",
        "maximum",
    ]
    assert full.unconstrained.strategy.name == "knee"
    assert full.selected.strategy.name == "knee"
    assert full.selected_score > 0.44
    assert limited.unconstrained.strategy.name == "knee"
    assert limited.selected.strategy.name == "cheap"


def test_global_refinement_rejects_replayed_regression(tmp_path: Path) -> None:
    specs = {
        "NINT4": NintSpec(4, 24, 6),
        "NINT5": NintSpec(5, 28, 7),
        "NINT6": NintSpec(6, 24, 7),
    }
    names = {
        layer: f"model.language_model.layers.{layer}.mlp.gate_proj.weight" for layer in range(2)
    }
    profiles = {0: ("NINT4", "NINT5"), 1: ("NINT5", "NINT6")}
    evaluations: dict[str, dict[str, TensorCandidateEvaluation]] = {}
    strategies: dict[int, list[LayerStrategy]] = {}
    selections: dict[str, TensorSelection] = {}
    for layer, name in names.items():
        evaluations[name] = {
            profile: _evaluation(
                name,
                profile,
                specs[profile],
                [1.0] * 8,
                [1.0] * 8,
            )
            for profile in profiles[layer]
        }
        strategies[layer] = [
            LayerStrategy(
                layer,
                profile,
                {name: evaluations[name][profile].spec},
                {f"layer.{layer}": profile},
                evaluations[name][profile].storage_bits,
                evaluations[name][profile].train_loss,
                evaluations[name][profile].validation_loss,
            )
            for profile in profiles[layer]
        ]
        base_value = evaluations[name]["NINT5"]
        selections[name] = TensorSelection(
            name,
            base_value.group,
            base_value.spec,
            base_value.rows,
            base_value.columns,
            base_value.storage_bits,
            base_value.train_loss,
            base_value.validation_loss,
        )

    base = CalibrationScheme(
        path=None,
        target_profile="NINT5",
        target_storage_bits=sum(item.storage_bits for item in selections.values()),
        selections=selections,
        metadata={},
        candidate_table={},
    )
    corpus = build_corpus_from_records(
        {"language": [f"reject proposal sample {index}" for index in range(10)]},
        _Tokenizer(),
        tmp_path / "reject-corpus",
        train_tokens=8,
        validation_tokens=4,
        sequence_length=4,
        seed=13,
        render_mode="plain",
    )
    refined, rounds = refine_layerwise_global(
        _RejectingRefinementBackend(),
        corpus,
        base,
        evaluations,
        strategies,
        tmp_path / "rejected-scheme.json",
        tmp_path / "rejected-report.json",
        work_dir=tmp_path / "reject-hidden",
        window_length=4,
        max_tokens=8,
        max_iterations=2,
    )

    assert all(refined.selections[name].spec.bits == 5 for name in names.values())
    assert len(rounds) == 1
    assert rounds[0].converged
    assert not rounds[0].attempts[0].accepted
    assert (
        rounds[0].attempts[0].replayed_normalized_squared_error
        > rounds[0].attempts[0].current_normalized_squared_error
    )
