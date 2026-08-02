"""Public operator numerical self-tests exposed by ``tpq check``."""

from __future__ import annotations

import os
from typing import Any


def _pack(torch, indices, bits: int):
    values = indices.reshape(-1).to(torch.int64)
    if bits == 8:
        return values.to(torch.uint8)
    if bits == 16:
        return values.to(torch.uint16).view(torch.uint8)
    if bits in (9, 11, 13, 15):
        word = 0
        filled = 0
        output = []
        for value in values.tolist():
            word |= int(value) << filled
            filled += bits
            while filled >= 8:
                output.append(word & 0xFF)
                word >>= 8
                filled -= 8
        if filled:
            output.append(word & 0xFF)
        return torch.tensor(output, dtype=torch.uint8)
    group_size, byte_count = {
        10: (4, 5),
        12: (2, 3),
        14: (4, 7),
    }[bits]
    groups = values.reshape(-1, group_size)
    words = torch.zeros(groups.shape[0], dtype=torch.int64)
    for offset in range(group_size):
        words |= groups[:, offset] << (bits * offset)
    output = torch.empty(
        groups.shape[0], byte_count, dtype=torch.uint8
    )
    for offset in range(byte_count):
        output[:, offset] = (
            (words >> (8 * offset)) & 0xFF
        ).to(torch.uint8)
    return output.reshape(-1)


def _run_projection_layout(
    torch,
    *,
    name: str,
    definitions: tuple[tuple[int, int, int], ...],
    activation: str = "swiglu",
    limit: float = 10.0,
) -> dict[str, Any]:
    from .api import packed_moe_topk

    device = torch.device("cuda:0")
    hidden = 64
    intermediate = 32
    expert_count = 3
    dtype_tags = {
        8: 0, 16: 1, 12: 2, 14: 3, 10: 4, 9: 5,
        11: 6, 13: 7, 15: 8,
    }
    value = torch.randn(
        1, hidden, dtype=torch.bfloat16, device=device
    )
    route_ids = torch.arange(
        expert_count, dtype=torch.long, device=device
    )
    route_weights = torch.rand(expert_count, device=device)
    route_weights = (route_weights / route_weights.sum()).float()
    metadata = torch.zeros(15, expert_count, dtype=torch.long)
    retained = []
    reference_rows = []
    for expert in range(expert_count):
        dense = []
        for projection, (bits, dim, codebook_size) in enumerate(
            definitions
        ):
            rows = intermediate if projection < 2 else hidden
            columns = hidden if projection < 2 else intermediate
            indices = torch.randint(
                codebook_size,
                (rows, columns // dim),
                dtype=torch.int64,
            )
            codebook = torch.randn(
                codebook_size,
                dim,
                dtype=torch.bfloat16,
                device=device,
            )
            packed = _pack(torch, indices, bits).to(device)
            retained.append((packed, codebook))
            base = 5 * projection
            metadata[:, expert][base : base + 5] = torch.tensor(
                [
                    packed.data_ptr(),
                    codebook.data_ptr(),
                    columns // dim,
                    dim,
                    dtype_tags[bits],
                ],
                dtype=torch.long,
            )
            dense.append(
                codebook[indices.to(device)].reshape(rows, columns)
            )
        gate = (value.float() @ dense[0].float().t()).to(torch.bfloat16)
        up = (value.float() @ dense[1].float().t()).to(torch.bfloat16)
        if activation == "situ":
            gate_f = gate.float()
            up_f = up.float()
            activated = (
                4.0
                * torch.tanh(gate_f / 4.0)
                * torch.sigmoid(gate_f)
                * (25.0 * torch.tanh(up_f / 25.0))
            ).to(torch.bfloat16)
        else:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            activated = (
                torch.nn.functional.silu(gate.float()) * up.float()
            ).to(torch.bfloat16)
        reference_rows.append(
            (activated.float() @ dense[2].float().t())
            .to(torch.bfloat16)
            .squeeze(0)
            .float()
        )
    metadata = metadata.to(device)
    hidden_workspace = torch.empty(
        expert_count,
        2 * intermediate,
        dtype=torch.bfloat16,
        device=device,
    )
    output_workspace = torch.empty(
        expert_count,
        hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    result = torch.empty(hidden, dtype=torch.float32, device=device)
    actual = packed_moe_topk(
        value,
        route_ids,
        route_weights,
        metadata,
        activation=activation,
        activation_beta=4.0,
        activation_linear_beta=25.0 if activation == "situ" else 0.0,
        hidden_workspace=hidden_workspace,
        output_workspace=output_workspace,
        result=result,
        grouped_prefix=-1,
        packed_formats=tuple(f"p{item[0]}" for item in definitions),
        code_dims=tuple(item[1] for item in definitions),
        codebook_sizes=tuple(item[2] for item in definitions),
        limit=limit if activation != "situ" else 0.0,
    )
    torch.cuda.synchronize()
    expected = (
        torch.stack(reference_rows) * route_weights[:, None]
    ).sum(0)
    difference = (actual - expected).abs()
    return {
        "layout": name,
        "activation": activation,
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "finite": bool(torch.isfinite(actual).all()),
        "allclose": bool(
            torch.allclose(actual, expected, rtol=0.03, atol=0.2)
        ),
    }


def _run_hc_workspace(torch) -> dict[str, Any]:
    """Verify caller-owned HC decode buffers preserve exact results."""
    from .api import (
        hyper_connection_post,
        hyper_connection_post_moe,
        hyper_connection_pre_norm,
    )

    device = torch.device("cuda:0")
    width = 64
    value = torch.randn(
        1, 1, 4, width, dtype=torch.bfloat16, device=device
    )
    fn = torch.randn(
        24, 4 * width, dtype=torch.bfloat16, device=device
    )
    scale = torch.randn(3, dtype=torch.float32, device=device)
    base = torch.randn(24, dtype=torch.float32, device=device)
    norm = torch.randn(width, dtype=torch.bfloat16, device=device)
    expected = hyper_connection_pre_norm(
        value, fn, scale, base, norm, 20, 1e-6
    )
    buffers = (
        torch.empty(1, width, dtype=torch.bfloat16, device=device),
        torch.empty(1, 4, dtype=torch.bfloat16, device=device),
        torch.empty(1, 16, dtype=torch.bfloat16, device=device),
    )
    actual = hyper_connection_pre_norm(
        value,
        fn,
        scale,
        base,
        norm,
        20,
        1e-6,
        output_buffers=buffers,
    )
    routed = torch.randn(1, width, dtype=torch.float32, device=device)
    shared = torch.randn(
        1, width, dtype=torch.bfloat16, device=device
    )
    expected_post = hyper_connection_post(
        (routed.to(torch.bfloat16) + shared).view(1, 1, width),
        value,
        actual[1],
        actual[2],
    )
    post_output = torch.empty_like(value)
    actual_standard_post = hyper_connection_post(
        (routed.to(torch.bfloat16) + shared).view(1, 1, width),
        value,
        actual[1],
        actual[2],
        output=post_output,
    )
    moe_post_output = torch.empty_like(value)
    actual_post = hyper_connection_post_moe(
        routed,
        shared,
        value,
        actual[1],
        actual[2],
        output=moe_post_output,
    )
    torch.cuda.synchronize()
    differences = torch.cat(
        [
            (got - want).abs().float().reshape(-1)
            for got, want in zip(actual, expected)
        ] + [
            (actual_standard_post - expected_post).abs().float().reshape(-1),
            (actual_post - expected_post).abs().float().reshape(-1),
        ]
    )
    aliases = all(
        got.data_ptr() == workspace.data_ptr()
        for got, workspace in zip(actual, buffers)
    ) and (
        actual_standard_post.data_ptr() == post_output.data_ptr()
        and actual_post.data_ptr() == moe_post_output.data_ptr()
    )
    return {
        "layout": "hyper_connection_workspace",
        "activation": "hc_pre_norm",
        "max_abs": float(differences.max()),
        "mean_abs": float(differences.mean()),
        "finite": bool(
            all(torch.isfinite(item).all() for item in actual)
            and torch.isfinite(actual_standard_post).all()
            and torch.isfinite(actual_post).all()
        ),
        "allclose": bool(float(differences.max()) == 0.0 and aliases),
    }


def _run_packed_route_slots(torch) -> dict[str, Any]:
    """Verify the fixed-address device route directory is graph-safe."""
    from .api import packed_route_slots

    device = torch.device("cuda:0")
    route_ids = torch.tensor([4, 1, 3], dtype=torch.long, device=device)
    directory = (
        torch.arange(1, 1 + 5 * 4, dtype=torch.long, device=device)
        .reshape(5, 4)
        .contiguous()
    )
    # Expert 1 represents a non-resident slot.  Row zero is the raw payload
    # pointer and therefore the authoritative residency marker.
    directory[1].zero_()
    output = torch.empty(4, 3, dtype=torch.long, device=device)
    hit_mask = torch.empty(3, dtype=torch.bool, device=device)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize(device)
    with torch.cuda.graph(graph):
        ok = packed_route_slots(
            route_ids,
            directory,
            output=output,
            hit_mask=hit_mask,
        )
    graph.replay()
    torch.cuda.synchronize(device)
    expected = directory.index_select(0, route_ids).t().contiguous()
    expected_hits = expected[0] != 0
    exact = bool(
        ok
        and torch.equal(output, expected)
        and torch.equal(hit_mask, expected_hits)
    )
    return {
        "layout": "packed_route_slots",
        "activation": "none",
        "max_abs": float((output - expected).abs().max()),
        "mean_abs": float((output - expected).abs().float().mean()),
        "finite": True,
        "allclose": exact,
    }


def _run_paged_mla_plan(torch, *, heads: int) -> dict[str, Any]:
    """Compare the dynamic CUDA planner with FlashInfer's official plan."""
    from .api import attention_step

    layout_name = f"paged_mla_dynamic_plan_h{heads}"
    runner = attention_step(
        "paged_latent_create",
        "cuda",
        device=torch.device("cuda:0"),
        max_ctx=1024,
        heads=heads,
        ckv_dim=512,
        kpe_dim=64,
        dtype=torch.bfloat16,
        qk_head_dim=192,
    )
    if runner is None:
        return {
            "layout": layout_name,
            "activation": "none",
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "finite": True,
            "allclose": True,
            "skipped": "FlashInfer MLA unavailable",
        }
    initialized = bool(
        attention_step(
            "paged_latent_prepare",
            "cuda",
            runner=runner,
            length=1,
        )
    )
    repeated = bool(
        attention_step(
            "paged_latent_prepare",
            "cuda",
            runner=runner,
            length=1,
        )
    )
    dynamic = bool(
        attention_step(
            "paged_latent_prepare",
            "cuda",
            runner=runner,
            length=1024,
        )
    )
    reference = attention_step(
        "paged_latent_create",
        "cuda",
        device=torch.device("cuda:0"),
        max_ctx=1024,
        heads=heads,
        ckv_dim=512,
        kpe_dim=64,
        dtype=torch.bfloat16,
        qk_head_dim=192,
    )
    reference_ready = bool(
        reference is not None
        and attention_step(
            "paged_latent_prepare",
            "cuda",
            runner=reference,
            length=1024,
        )
    )

    plan_info = [int(value) for value in runner._wrapper._plan_info]
    reference_info = (
        []
        if reference is None
        else [
            int(value)
            for value in reference._wrapper._plan_info
        ]
    )

    def schedule_slice(wrapper, offset: int, count: int):
        return wrapper._int_workspace_buffer.narrow(
            0,
            int(offset),
            int(count) * 4,
        ).view(torch.int32)

    schedule_exact = False
    schedule_checks: dict[str, bool] = {}
    mismatch_samples: dict[str, dict[str, list[int]]] = {}
    if reference_ready and plan_info == reference_info:
        clusters = int(plan_info[1])
        sms = int(plan_info[0] * clusters)
        reference_work = schedule_slice(
            reference._wrapper,
            plan_info[15],
            clusters + 1,
        )
        total_works = int(reference_work[-1])
        fields = {
            "q_indptr": (2, total_works),
            "kv_indptr": (3, total_works),
            "partial_indptr": (4, total_works),
            "merge_start": (5, sms),
            "merge_end": (6, sms),
            "merge_partial_start": (7, sms),
            "merge_partial_end": (8, sms),
            "merge_stride": (9, sms),
            "q_len": (10, total_works),
            "kv_len": (11, total_works),
            "q_start": (12, total_works),
            "kv_start": (13, total_works),
            "kv_end": (14, total_works),
            "work_indptr": (15, clusters + 1),
        }
        for name, (field, count) in fields.items():
            actual = schedule_slice(
                runner._wrapper,
                plan_info[field],
                count,
            )
            expected = schedule_slice(
                reference._wrapper,
                reference_info[field],
                count,
            )
            schedule_checks[name] = bool(torch.equal(actual, expected))
            if not schedule_checks[name]:
                mismatch_samples[name] = {
                    "actual": [int(value) for value in actual[:8]],
                    "expected": [int(value) for value in expected[:8]],
                }
        metadata = {
            "kv_indptr_input": (
                runner._kv_indptr_gpu,
                reference._kv_indptr_gpu,
            ),
            "kv_len_input": (
                runner._kv_len_gpu,
                reference._kv_len_gpu,
            ),
            "kv_indices_input": (
                runner._kv_indices_gpu[:2],
                reference._kv_indices_gpu[:2],
            ),
        }
        for name, (actual, expected) in metadata.items():
            schedule_checks[name] = bool(torch.equal(actual, expected))
            if not schedule_checks[name]:
                mismatch_samples[name] = {
                    "actual": [int(value) for value in actual[:8]],
                    "expected": [int(value) for value in expected[:8]],
                }
        schedule_exact = bool(all(schedule_checks.values()))
    error = None
    fast_requested = os.environ.get("TPQ_FLASHINFER_GPU_PLAN", "1") != "0"
    fast_path = bool(runner.gpu_plan_hits >= 2)
    passed = bool(
        initialized
        and repeated
        and dynamic
        and reference_ready
        and schedule_exact
        and (not fast_requested or fast_path)
    )
    if not passed:
        from ..flashinfer_mla import last_error

        current = last_error()
        error = None if current is None else f"{type(current).__name__}: {current}"
    return {
        "layout": layout_name,
        "activation": "none",
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "finite": True,
        "allclose": passed,
        "error": error,
        "plan_info": plan_info,
        "schedule_exact": schedule_exact,
        "schedule_checks": schedule_checks,
        "mismatch_samples": mismatch_samples,
        "gpu_plan_hits": int(runner.gpu_plan_hits),
        "gpu_plan_rejections": int(runner.gpu_plan_rejections),
        "cpu_plan_calls": int(runner.cpu_plan_calls),
        "gpu_fast_path": fast_path,
        "int_workspace_bytes": int(
            runner._wrapper._int_workspace_buffer.numel()
        ),
    }


def projection_cuda_selftest() -> dict[str, Any]:
    """Compile/select the public CUDA op and check DSV4 H/C/S layouts."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "--cuda-ops requires exactly one visible CUDA GPU"
        )
    torch.manual_seed(20260731)
    layouts = [
        _run_projection_layout(
            torch,
            name="kimi_front44",
            definitions=(
                (16, 16, 65536),
                (12, 8, 4096),
                (12, 8, 4096),
            ),
            activation="situ",
            limit=0.0,
        ),
        _run_projection_layout(
            torch,
            name="kimi_tail48",
            definitions=(
                (10, 8, 1024),
                (10, 8, 1024),
                (8, 4, 256),
            ),
            activation="situ",
            limit=0.0,
        ),
        _run_projection_layout(
            torch,
            name="dsv4_h",
            definitions=(
                (16, 8, 65536),
                (14, 8, 16384),
                (12, 4, 4096),
            ),
        ),
        _run_projection_layout(
            torch,
            name="dsv4_c",
            definitions=(
                (12, 8, 4096),
                (10, 8, 1024),
                (10, 8, 1024),
            ),
        ),
        _run_projection_layout(
            torch,
            name="dsv4_s",
            definitions=(
                (14, 8, 16384),
                (10, 8, 1024),
                (16, 8, 65536),
            ),
        ),
        _run_projection_layout(
            torch,
            name="dsv4_search_i01",
            definitions=(
                (11, 8, 2048),
                (13, 8, 8192),
                (15, 8, 32768),
            ),
        ),
        _run_hc_workspace(torch),
        _run_packed_route_slots(torch),
        _run_paged_mla_plan(torch, heads=64),
        _run_paged_mla_plan(torch, heads=96),
    ]
    return {
        "device": torch.cuda.get_device_name(0),
        "layouts": layouts,
        "all_passed": all(
            item["finite"] and item["allclose"]
            for item in layouts
        ),
    }


__all__ = ["projection_cuda_selftest"]
