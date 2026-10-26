"""Header-only coverage of uniform importance and registered planning policies."""

from __future__ import annotations

import itertools
import json
import math
import struct

import numpy as np
import pytest

from mfq.architectures.hf_config import text_section
from mfq.architectures.tensor_schema import _SCHEMAS, map_source_tensor_name
from mfq.calibration.alphaq_model import model_scope
from mfq.calibration.alphaq_workflow import planned, recipe_types, routing
from mfq.calibration.artifact import CalibrationScheme
from mfq.calibration.ud_recipes import UD_MODELS, automatic_recipe
from mfq.calibration.uniform import uniform_imatrix
from mfq.cli import _build_parser, main
from mfq.quantize.imatrix import load_importance_matrix
from mfq.tools import quantize_hf_to_mfq as q


FAMILIES = (
    "qwen3_5", "qwen4_exp", "glm5_next", "deepseek_v41", "deepseek_v4",
    "minicpmo", "glm_dsa", "gemma4", "nemotron_h_omni",
)
EMPTY = CalibrationScheme(None, "uniform-test", 0, {}, {}, {})


def _checkpoint(root, config, shapes):
    """Write a valid safetensors header with an unused zero-filled payload."""
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    header, offset = {}, 0
    for name, shape in shapes.items():
        size = math.prod(shape) * 2
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with (root / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.truncate(stream.tell() + offset)
    return root


def _qwen_case(family="qwen4_exp", *, packed=True):
    text = dict(model_type=family + "_text", hidden_size=8, num_hidden_layers=1,
                num_experts=2, num_experts_per_tok=1, moe_intermediate_size=4,
                mtp_num_hidden_layers=1, linear_num_key_heads=1, linear_key_head_dim=2,
                linear_num_value_heads=1, linear_value_head_dim=4)
    config = {"model_type": family, "text_config": text, "vision_config": {"depth": 1}}
    shapes = {"model.visual.blocks.0.attn.qkv.weight": (12, 8)}
    for prefix in ("model.language_model.layers.0.", "mtp.layers.0."):
        for suffix in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                       "self_attn.o_proj.weight"):
            shapes[prefix + suffix] = (8, 8)
        shapes[prefix + "mlp.gate.weight"] = (2, 8)
        shapes[prefix + "linear_attn.in_proj_qkv.weight"] = (8, 8)
        if family == "qwen4_exp":
            shapes[prefix + "self_attn.indexer.index_qk_proj.weight"] = (6, 8)
        if packed:
            shapes[prefix + "mlp.experts.gate_up_proj"] = (2, 8, 8)
            shapes[prefix + "mlp.experts.down_proj"] = (2, 8, 4)
        else:
            for expert, projection in itertools.product(range(2), ("gate", "up", "down")):
                shapes[prefix + f"mlp.experts.{expert}.{projection}_proj.weight"] = (
                    (8, 4) if projection == "down" else (4, 8)
                )
    return config, shapes


def _case(family):
    if family in {"qwen3_5", "qwen4_exp"}:
        return _qwen_case(family)
    if family == "nemotron_h_omni":
        from tests.test_quantize.test_nemotron_h_import import synthetic_config, synthetic_tensors

        return synthetic_config(), {name: tuple(value.shape) for name, value in synthetic_tensors().items()}
    if family in {"deepseek_v4", "deepseek_v41"}:
        text = dict(model_type=family + "_text", num_hidden_layers=1, num_nextn_predict_layers=1)
        if family == "deepseek_v41":
            from tests.test_deepseek_v41_architecture import _config

            config = _config()
        else:
            config = {"model_type": family, "text_config": text}
        shapes = {"layers.0.attn.wq_a.weight": (6, 8),
                  "mtp.0.main_proj.weight": (8, 16),
                  "vision.blocks.0.attn.wqkv.weight": (12, 8)}
        if family == "deepseek_v41":
            shapes.update({f"mtp.{stage}.main_proj.weight": (8, 16) for stage in (1, 2)})
        for expert, projection in itertools.product(range(2), ("w1", "w2", "w3")):
            shapes[f"layers.0.ffn.experts.{expert}.{projection}.weight"] = (8, 4) if projection == "w2" else (4, 8)
        return config, shapes
    if family in {"glm5_next", "glm_dsa"}:
        text = dict(model_type="glm5_next_text" if family == "glm5_next" else "glm_moe_dsa",
                    num_hidden_layers=1, hidden_size=8, num_attention_heads=2,
                    kv_lora_rank=6, qk_nope_head_dim=4, v_head_dim=2,
                    n_routed_experts=2, num_experts_per_tok=1, moe_intermediate_size=4,
                    num_nextn_predict_layers=1, first_k_dense_replace=0, moe_layer_freq=1)
        prefix = "model.language_model.layers." if family == "glm5_next" else "model.layers."
        shapes = {prefix + "0.self_attn.kv_b_proj.weight": (12, 6)}
        if family == "glm5_next":
            shapes[prefix + "1.self_attn.kv_b_proj.weight"] = (12, 6)
        else:
            shapes["predictor.block.0.attention.query.weight"] = (8, 8)
        shapes["vision.block.0.attention.qkv.weight"] = (12, 8)
        for expert, projection in itertools.product(range(2), ("gate", "up", "down")):
            shapes[prefix + f"0.mlp.experts.{expert}.{projection}_proj.weight"] = (
                (8, 4) if projection == "down" else (4, 8)
            )
        return {"model_type": family if family == "glm5_next" else "glm_moe_dsa", "text_config": text}, shapes
    if family == "minicpmo":
        return {"model_type": "minicpmo", "version": "4.5", "num_hidden_layers": 1}, {
            "llm.model.layers.0.self_attn.q_proj.weight": (8, 8),
            "vpm.encoder.layers.0.self_attn.q_proj.weight": (8, 8),
            "apm.layers.0.self_attn.q_proj.weight": (8, 8),
            "tts.model.layers.0.self_attn.q_proj.weight": (8, 8),
        }
    return {"model_type": "gemma4", "text_config": {
        "model_type": "gemma4_text", "num_hidden_layers": 1, "num_nextn_predict_layers": 1,
    }}, {
        "model.language_model.layers.0.self_attn.q_proj.weight": (8, 8),
        "model.language_model.layers.0.self_attn.k_proj.weight": (8, 8),
        "vision.block.0.attention.qkv.weight": (12, 8),
        "predictor.block.0.attention.query.weight": (8, 8),
    }


def _recipe(config, shapes):
    values = {}
    for name, shape in shapes.items():
        mapping = map_source_tensor_name(name, config)
        assert mapping is not None, name
        if mapping.recipe_name and mapping.canonical_name.startswith("model."):
            values[mapping.recipe_name] = "Q4_K" if len(shape) > 1 else "F32"
    text = text_section(config)
    layers = int(text.get("num_hidden_layers", len(text.get("layers_block_type", ()))))
    for layer in range(layers):
        for projection in ("gate", "up", "down"):
            values[f"blk.{layer}.ffn_{projection}_exps.weight"] = "Q4_K"
    # The packed Qwen4 indexer has separate Q/K entries in a GGUF recipe.
    for name, value in tuple(values.items()):
        if "indexer" in name and "q_proj" in name:
            values[name.replace("q_proj", "k_proj")] = value
    return values


def _assert_bindings(imatrix, plans):
    assert len({item.name for item in plans}) == len(plans)
    for item in plans:
        if len(item.shape) not in (2, 3) or item.source_dtype in q._HF_INTEGER_DTYPES:
            continue
        match = imatrix.find(q._hf_imatrix_names(item))
        assert match is not None, item.name
        _, entry = match
        shape, storage = q._hf_imatrix_shapes(item)
        assert entry.width == shape[-1], item.name
        assert entry.matrices == (shape[0] if len(shape) == 3 else 1), item.name
        resolved = imatrix.input_for_rows(q._hf_imatrix_names(item), shape, storage, slice(0, 1))
        np.testing.assert_array_equal(resolved[1], np.ones_like(resolved[1]))
    bindings = q._bind_hf_imatrix(imatrix, plans)
    assert set(bindings) == {item.name for item in plans if q._hf_plan_supports_imatrix(item)}


def test_architecture_and_ud_registries_are_covered():
    assert {schema.architecture for schema in _SCHEMAS.values()} == set(FAMILIES)
    assert {_SCHEMAS[name].architecture for name in UD_MODELS} <= set(FAMILIES)


@pytest.mark.parametrize("family", FAMILIES)
def test_uniform_covers_registered_recipe_mtp_vision_plans(tmp_path, family):
    config, shapes = _case(family)
    root = _checkpoint(tmp_path / "hf", config, shapes)
    imatrix = load_importance_matrix(uniform_imatrix(root, tmp_path / "uniform.npz").path)
    assert imatrix.metadata["objective"] == "uniform_calibration_free"
    assert imatrix.metadata["tokens"] == 0
    assert not imatrix.datasets and not imatrix.chunk_count and not imatrix.chunk_size
    for entry in imatrix.entries.values():
        assert not entry.counts.any()
        np.testing.assert_array_equal(entry.values, np.ones_like(entry.values))
    recipe = _recipe(config, shapes)
    for types, vision, mtp in itertools.product((None, recipe), (False, True), (False, True)):
        options = dict(quantize_vision=vision, quantize_mtp=mtp,
                       source_config=config, quantize_ple=True)
        if types is not None and family in {"glm_dsa", "glm5_next", "deepseek_v41", "deepseek_v4"}:
            with pytest.raises(ValueError, match="recipe mapping|no GGUF recipe"):
                q._plan(root, False, types, "F32", EMPTY, **options)
            continue
        plans = q._normalize_hf_expert_storage(q._plan(root, False, types, "F32", EMPTY, **options))
        complete = q._plan(root, False, None, "F32", EMPTY, **options)
        assert {p.name: p.shape for p in plans} == {p.name: p.shape for p in complete}
        assert any(item.name.startswith("vision.") for item in plans)
        if family != "minicpmo":
            assert any(item.name.startswith("predictor.") for item in plans)
        _assert_bindings(imatrix, plans)
    for vision, mtp in itertools.product(("native", "NINT8", "exclude"), ("quantize", "exclude")):
        plans = planned(root, None, EMPTY, {}, "native", vision=vision, mtp=mtp)
        _assert_bindings(imatrix, plans)
        assert any(p.name.startswith("vision.") for p in plans) == (vision != "exclude")
        if mtp == "exclude":
            assert not any(p.name.startswith("predictor.") for p in plans)


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("tier", tuple(UD_MODELS["qwen4_exp_text"]["caps"]))
def test_qwen4_ud_recipe_mtp_indexer_without_recipe_name(tmp_path, packed, tier):
    config, shapes = _qwen_case(packed=packed)
    # Released recipes put sparse attention/indexers on layer 3; layer 0 is linear.
    config["text_config"]["num_hidden_layers"] = 4
    shapes = {name.replace("layers.0.self_attn.", "layers.3.self_attn.")
              if name.startswith("model.") else name: shape for name, shape in shapes.items()}
    root = _checkpoint(tmp_path / "hf", config, shapes)
    registered = {"model_type": "qwen4_exp_text", **UD_MODELS["qwen4_exp_text"]["config"]}
    types = recipe_types(automatic_recipe(registered, tier=tier))
    plans = planned(root, types, EMPTY, {}, "native", vision="native", mtp="quantize")
    predictor = next(p for p in plans if p.name == "predictor.block.0.attention.indexer.query_key.weight")
    assert predictor.gguf_name is None
    assert predictor.target_dtype == "NINT4"
    imatrix = uniform_imatrix(root, tmp_path / "uniform.npz")
    _assert_bindings(imatrix, plans)
    scope = model_scope(plans, imatrix)
    assert len(scope.experts) == 3


def test_uniform_cli_avoids_calibration_and_weight_reads(tmp_path, monkeypatch, capsys):
    config, shapes = _qwen_case()
    root = _checkpoint(tmp_path / "hf", config, shapes)

    def forbidden(*_args, **_kwargs):
        pytest.fail("uniform imatrix must not load weights, a corpus, tokenizer or calibration backend")

    import mfq.calibration.chat as chat
    import mfq.calibration.imatrix as calibration

    monkeypatch.setattr(chat, "prepare_calibration_corpus", forbidden)
    monkeypatch.setattr(calibration, "_backend", forbidden)
    monkeypatch.setattr(q, "_raw_source_for_plan", forbidden)
    original = q.safe_open

    class HeadersOnly:
        def __init__(self, *args, **kwargs):
            self.handle = original(*args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_slice(self, name):
            view = self.handle.get_slice(name)
            # No slicing/indexing interface is exposed to the planner.
            from types import SimpleNamespace

            return SimpleNamespace(get_shape=view.get_shape, get_dtype=view.get_dtype)

    monkeypatch.setattr(q, "safe_open", HeadersOnly)
    output = tmp_path / "uniform.npz"
    assert main(["calibrate", "imatrix", "--model", str(root), "--uniform", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["tokens"] == 0
    assert output.is_file()
    with pytest.raises(FileExistsError):
        uniform_imatrix(root, output)


def test_uniform_requires_no_corpus_and_rejects_combined_sources():
    parser = _build_parser()
    args = ["calibrate", "imatrix", "--model", "model", "--output", "matrix.npz"]
    with pytest.raises(SystemExit):
        parser.parse_args(args)
    with pytest.raises(SystemExit):
        parser.parse_args([*args, "--uniform", "--corpus", "corpus"])


@pytest.mark.parametrize("family", ("qwen4_exp", "glm5_next", "deepseek_v41", "unknown_family"))
def test_unsupported_calibration_names_architecture_before_loading(tmp_path, family, capsys):
    from mfq.calibration.imatrix import _backend, collect_imatrix

    config = {"model_type": family, "text_config": {"model_type": family + "_text"}}
    root = _checkpoint(tmp_path / "hf", config, {"unused.weight": (2, 2)})
    for action in (
        lambda: _backend(root, None, "sdpa"),
        lambda: collect_imatrix(root, None, tmp_path / "unused.npz", device="cpu"),
    ):
        with pytest.raises(ValueError, match=family + "_text.*--uniform") as error:
            action()
        assert error.value.__cause__ is None
    assert main(["calibrate", "imatrix", "--model", str(root), "--corpus", "missing",
                 "--output", str(tmp_path / "calibrated.npz")]) == 2
    stderr = capsys.readouterr().err
    assert family + "_text" in stderr and "--uniform" in stderr
    assert "Traceback" not in stderr


def test_uniform_router_counts_are_never_treated_as_observations(tmp_path):
    config, shapes = _qwen_case()
    root = _checkpoint(tmp_path / "hf", config, shapes)
    imatrix = uniform_imatrix(root, tmp_path / "uniform.npz")
    scope = model_scope(planned(root, None, EMPTY, {}, "native"), imatrix)
    with pytest.raises(ValueError, match="--alphaq-router none"):
        routing(scope, imatrix, config, "imatrix", "", root)
    multipliers, exposure = routing(scope, imatrix, config, "none", "", root)
    assert multipliers is None and set(exposure.values()) == {0.5}


def test_uniform_unknown_architecture(tmp_path):
    root = _checkpoint(tmp_path / "hf", {"model_type": "future_architecture"}, {
        "projection.weight": (6, 8), "projection.bias": (6,),
    })
    imatrix = uniform_imatrix(root, tmp_path / "uniform.npz")
    assert imatrix.metadata["model_type"] == "future_architecture"
    assert set(imatrix.entries) == {"projection.weight"}


@pytest.mark.parametrize("tier", tuple(UD_MODELS["nemotron_h"]["caps"]))
def test_nemotron_registered_tiers_with_mtp_and_vision(tmp_path, tier):
    from tests.test_nemotron_ud_recipes import released_text_config

    config, shapes = _case("nemotron_h_omni")
    config["llm_config"]["layers_block_type"] = released_text_config()["layers_block_type"][:8]
    shapes = {name.replace("backbone.layers.2.", "backbone.layers.7."): shape for name, shape in shapes.items()}
    root = _checkpoint(tmp_path / "hf", config, shapes)
    types = recipe_types(automatic_recipe({"model_type": "nemotron_h", **UD_MODELS["nemotron_h"]["config"]},
                                         tier=tier))
    imatrix = uniform_imatrix(root, tmp_path / "uniform.npz")
    for vision, mtp in itertools.product(("native", "NINT8", "exclude"), ("quantize", "exclude")):
        plans = planned(root, types, EMPTY, {}, "native", vision=vision, mtp=mtp)
        _assert_bindings(imatrix, plans)
        assert len(model_scope(plans, imatrix).experts) == 4


def test_qwen4_recipe_still_rejects_missing_text_and_mismatched_indexer(tmp_path):
    config, shapes = _qwen_case()
    root = _checkpoint(tmp_path / "hf", config, shapes)
    recipe = _recipe(config, shapes)
    recipe.pop("blk.0.indexer.q_proj.weight")
    with pytest.raises(ValueError, match="tensor is absent from GGUF recipe"):
        q._plan(root, False, recipe, "F32", quantize_mtp=True)
    recipe["blk.0.indexer.q_proj.weight"] = "Q8_0"
    with pytest.raises(ValueError, match="indexer Q/K requires matching recipe types"):
        q._plan(root, False, recipe, "F32", quantize_mtp=True)


def test_uniform_alphaq_quantization_on_cpu(tmp_path):
    from mfq.formats.io import open_mmap
    from tests.test_alphaq_workflow import fixture

    root, _calibrated, _count = fixture(tmp_path)
    imatrix = uniform_imatrix(root, tmp_path / "uniform.npz")
    output = tmp_path / "quantized.mfq"
    assert main([
        "quantize", str(root), str(output), "--target-bpw", "12",
        "--imatrix", str(imatrix.path), "--alphaq-router", "none",
        "--alphaq-profiles", "NINT4,NINT8", "--backend", "cpu", "--device", "cpu",
    ]) == 0
    with open_mmap(output) as store:
        assert store.header.extra["alphaq"]["router"] == "none"
        assert all(store.records[f"model.block.0.mlp.experts.{projection}.weight"].dtype == "MFE"
                   for projection in ("gate", "up", "down"))
