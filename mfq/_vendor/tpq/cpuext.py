"""Optional x86 CPU kernels.

The module is deliberately lazy: CUDA inference never compiles or loads the
CPU extension.  CPU inference attempts one cached JIT build and falls back to
the existing PyTorch implementation if the compiler toolchain is unavailable.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
import threading

import torch

_EXT = None
_TRIED = False
_ERR: str | None = None
_EXTENSION_NAME = "tpq_cpu_kernels_v62"
_PACKED_MOE_WORKSPACE: tuple[torch.Tensor, torch.Tensor] | None = None
_PACKED_MOE_LOCK = threading.Lock()


def configure_cpu_threads() -> int:
    """为大核数双路 CPU 选择物理核与 SMT 之间的低延迟甜点。"""
    raw = os.environ.get("TPQ_CPU_THREADS", "auto").strip().lower()
    if raw in ("0", "false", "off", "none"):
        return torch.get_num_threads()
    if raw not in ("", "auto"):
        target = max(1, int(raw))
    else:
        logical = os.cpu_count() or torch.get_num_threads()
        try:
            import psutil

            physical = psutil.cpu_count(logical=False) or logical
        except ImportError:
            physical = logical
        if physical >= 32 and logical >= 2 * physical:
            target = max(1, physical * 3 // 4)
        else:
            target = min(logical, physical)
    torch.set_num_threads(target)
    return torch.get_num_threads()


def configure_numa_interleave() -> bool:
    """让双路 Linux CPU 的后续大块分配均匀落在所有 NUMA 节点。"""
    mode = os.environ.get("TPQ_CPU_NUMA", "auto").strip().lower()
    if (
        sys.platform != "linux"
        or mode in ("0", "false", "off", "none")
    ):
        return False
    try:
        library = ctypes.CDLL("libnuma.so.1", use_errno=True)
        library.numa_available.restype = ctypes.c_int
        library.numa_num_configured_nodes.restype = ctypes.c_int
        library.numa_set_interleave_mask.argtypes = [ctypes.c_void_p]
        if (
            library.numa_available() < 0
            or library.numa_num_configured_nodes() < 2
        ):
            return False
        all_nodes = ctypes.c_void_p.in_dll(
            library, "numa_all_nodes_ptr"
        )
        ctypes.set_errno(0)
        library.numa_set_interleave_mask(all_nodes)
        return ctypes.get_errno() == 0
    except (OSError, ValueError):
        return False


def _ensure_ninja_on_path() -> None:
    if shutil.which("ninja") is not None:
        return
    try:
        import ninja
    except ImportError:
        return
    bin_dir = getattr(ninja, "BIN_DIR", None)
    if not bin_dir:
        return
    executable = "ninja.exe" if os.name == "nt" else "ninja"
    if os.path.isfile(os.path.join(bin_dir, executable)):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _build(verbose: bool = False):
    global _EXT, _TRIED, _ERR
    if _EXT is not None or _TRIED:
        return _EXT
    _TRIED = True
    if os.environ.get("TPQ_CPU_FUSED", "1") == "0":
        _ERR = "TPQ_CPU_FUSED=0"
        return None
    try:
        _ensure_ninja_on_path()
        from torch.utils.cpp_extension import load

        source = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "csrc", "cpu_vq.cpp"
        )
        compile_flags = (
            ["/O2", "/openmp"]
            if os.name == "nt"
            else ["-O3", "-march=native", "-fopenmp"]
        )
        link_flags = [] if os.name == "nt" else ["-fopenmp"]
        _EXT = load(
            name=_EXTENSION_NAME,
            sources=[source],
            extra_cflags=compile_flags,
            extra_ldflags=link_flags,
            verbose=verbose,
        )
        _ERR = None
    except Exception as exc:  # a missing compiler must not break inference
        _EXT = None
        _ERR = f"{type(exc).__name__}: {exc}"
    return _EXT


def vq_gemv_cpu(
    x_rows: torch.Tensor,
    indices: torch.Tensor,
    codebooks: torch.Tensor,
) -> torch.Tensor | None:
    if (
        x_rows.is_cuda
        or indices.is_cuda
        or codebooks.is_cuda
        or indices.dtype not in (torch.uint8, torch.uint16)
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.vq_gemv(
        x_rows,
        indices,
        codebooks,
    )


def vq_gemv_list_cpu(
    x_rows: torch.Tensor,
    indices: list[torch.Tensor],
    codebook: torch.Tensor,
) -> torch.Tensor | None:
    if (
        x_rows.is_cuda
        or codebook.is_cuda
        or not indices
        or any(
            index.is_cuda
            or index.dtype not in (torch.uint8, torch.uint16)
            or index.dtype != indices[0].dtype
            for index in indices
        )
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.vq_gemv_list(
        x_rows,
        indices,
        codebook,
    )


def vq_gemv_packed_list_cpu(
    x_rows: torch.Tensor,
    payloads: list[torch.Tensor],
    codebook: torch.Tensor,
    rows: int,
    blocks: int,
    bits: int,
    *,
    allow_direct: bool = False,
) -> torch.Tensor | None:
    """Directly evaluate byte-packed VQ indices without a uint16 copy."""
    if (
        x_rows.is_cuda
        or codebook.is_cuda
        or not payloads
        or bits not in (8, 12, 14, 16)
        or any(
            payload.is_cuda or payload.dtype != torch.uint8
            for payload in payloads
        )
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.vq_gemv_packed_list(
        x_rows,
        payloads,
        codebook,
        int(rows),
        int(blocks),
        int(bits),
        bool(allow_direct),
    )


def moe_packed_topk_cpu(
    x_row: torch.Tensor,
    experts: list[tuple[object, object]],
    route_weights: torch.Tensor,
    limit: float,
    *,
    activation: str,
    activation_beta: float,
    activation_linear_beta: float | None,
) -> torch.Tensor | None:
    """Run mixed packed Top-K MoE through one native registered invocation.

    The payloads remain p8/p12/p14.  ATen's persistent CPU worker pool executes
    the dependent phases.  Only one process-wide float workspace is kept
    because decode executes CPU layers serially; it grows when required and is
    reused by every later layer and token.
    """
    if (
        x_row.is_cuda
        or x_row.ndim != 2
        or x_row.shape[0] != 1
        or not 0 < len(experts) <= 16
        or route_weights.is_cuda
        or route_weights.numel() != len(experts)
        or any(
            not hasattr(gu, "raw")
            or not hasattr(dn, "raw")
            for gu, dn in experts
        )
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    gu = [pair[0] for pair in experts]
    dn = [pair[1] for pair in experts]
    x_float = x_row.float().contiguous()
    weights = route_weights.float().contiguous()

    unique_gu: dict[tuple[int, int, int, int], object] = {}
    gu_score_count = 0
    for weight in gu:
        key = (
            weight.cb.data_ptr(),
            int(weight.blocks),
            int(weight.cb.shape[0]),
            int(weight.cb.shape[1]),
        )
        if key not in unique_gu:
            unique_gu[key] = weight
            gu_score_count += int(weight.blocks) * int(
                weight.cb.shape[0]
    )
    intermediate = int(dn[0].cols)
    hidden = int(x_float.shape[1])
    dn_score_count = sum(
        int(weight.blocks) * int(weight.cb.shape[0])
        for weight in dn
        if int(weight.rows) * int(weight.dim)
        >= int(weight.cb.shape[0]) * int(weight.dim)
        + int(weight.rows)
    )
    required = (
        gu_score_count
        + 4 * len(experts) * intermediate
        + dn_score_count
        + len(experts) * hidden
    )

    global _PACKED_MOE_WORKSPACE
    with _PACKED_MOE_LOCK:
        if (
            _PACKED_MOE_WORKSPACE is None
            or _PACKED_MOE_WORKSPACE[0].numel() < required
            or _PACKED_MOE_WORKSPACE[1].numel() < hidden
        ):
            _PACKED_MOE_WORKSPACE = (
                torch.empty(required, dtype=torch.float32),
                torch.empty(hidden, dtype=torch.float32),
            )
        workspace, result = _PACKED_MOE_WORKSPACE
        return extension.moe_packed_topk(
            x_float,
            [weight.raw for weight in gu],
            [weight.cb.float().contiguous() for weight in gu],
            [int(weight.rows) for weight in gu],
            [int(weight.blocks) for weight in gu],
            [int(weight.bits) for weight in gu],
            [weight.raw for weight in dn],
            [weight.cb.float().contiguous() for weight in dn],
            [int(weight.rows) for weight in dn],
            [int(weight.blocks) for weight in dn],
            [int(weight.bits) for weight in dn],
            weights,
            float(limit),
            str(activation).strip().lower(),
            float(activation_beta),
            (
                -1.0
                if activation_linear_beta is None
                else float(activation_linear_beta)
            ),
            workspace,
            result,
        )


def reset_packed_moe_phase_profile() -> None:
    extension = _build()
    if extension is not None:
        extension.reset_packed_moe_phase_profile()


def packed_moe_phase_profile() -> dict[str, float | int]:
    extension = _build()
    if extension is None:
        return {}
    values = extension.packed_moe_phase_profile()
    names = (
        "calls",
        "gu_score_seconds",
        "gu_lookup_seconds",
        "activation_seconds",
        "down_score_seconds",
        "down_compute_seconds",
        "reduce_seconds",
    )
    return {
        name: int(value) if name == "calls" else float(value)
        for name, value in zip(names, values)
    }


def kda_recurrent_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    workspace: torch.Tensor,
    output: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor | None:
    if (
        query.is_cuda
        or query.ndim != 2
        or query.dtype not in (torch.float32, torch.bfloat16)
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.kda_recurrent(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        gate.contiguous(),
        beta.contiguous(),
        a_log.contiguous(),
        dt_bias.contiguous(),
        state,
        workspace,
        output,
        float(lower_bound),
    )


def short_conv3_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> bool:
    if (
        query.is_cuda
        or query.ndim != 1
        or query.dtype not in (torch.float32, torch.bfloat16)
    ):
        return False
    extension = _build()
    if extension is None:
        return False
    return bool(
        extension.short_conv3(
            query,
            key,
            value,
            list(states),
            list(weights),
        )
    )


def gated_rmsnorm_cpu(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    if (
        value.is_cuda
        or value.ndim != 2
        or value.dtype not in (torch.float32, torch.bfloat16)
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.gated_rmsnorm(
        value,
        gate,
        weight,
        output,
        float(eps),
    )


def moe_mixed_cpu(
    x_row: torch.Tensor,
    gu_indices: list[torch.Tensor],
    gu_codebooks: list[torch.Tensor],
    dn_indices: list[torch.Tensor],
    dn_codebooks: list[torch.Tensor],
    route_weights: torch.Tensor,
    shared_w1_q: torch.Tensor,
    shared_w1_s: torch.Tensor,
    shared_w3_q: torch.Tensor,
    shared_w3_s: torch.Tensor,
    shared_w2_q: torch.Tensor,
    shared_w2_s: torch.Tensor,
    group_size: int,
    limit: float,
) -> torch.Tensor | None:
    if (
        x_row.is_cuda
        or x_row.ndim != 2
        or x_row.shape[0] != 1
        or not gu_indices
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.moe_mixed(
        x_row,
        gu_indices,
        gu_codebooks,
        dn_indices,
        dn_codebooks,
        route_weights,
        shared_w1_q,
        shared_w1_s,
        shared_w3_q,
        shared_w3_s,
        shared_w2_q,
        shared_w2_s,
        int(group_size),
        float(limit),
        False,
    )


def make_moe_layer_cpu(
    experts: tuple[tuple[object, object] | None, ...],
    shared_w1_q: torch.Tensor,
    shared_w1_s: torch.Tensor,
    shared_w3_q: torch.Tensor,
    shared_w3_s: torch.Tensor,
    shared_w2_q: torch.Tensor,
    shared_w2_s: torch.Tensor,
    gate_q: torch.Tensor,
    gate_s: torch.Tensor,
    gate_bias: torch.Tensor,
    gate_mask: torch.Tensor,
    group_size: int,
    limit: float,
    top_k: int,
    normalize_route: bool,
    routed_scaling: float,
):
    present = [expert for expert in experts if expert is not None]
    if not present:
        return None
    fallback_gu, fallback_dn = present[0]
    gu = [
        expert[0] if expert is not None else fallback_gu
        for expert in experts
    ]
    dn = [
        expert[1] if expert is not None else fallback_dn
        for expert in experts
    ]
    extension = _build()
    if extension is None:
        return None
    return extension.CpuMoeLayer(
        [weight.idx for weight in gu],
        [weight.cb for weight in gu],
        [weight.idx for weight in dn],
        [weight.cb for weight in dn],
        torch.tensor(
            [expert is not None for expert in experts],
            dtype=torch.bool,
        ),
        shared_w1_q,
        shared_w1_s,
        shared_w3_q,
        shared_w3_s,
        shared_w2_q,
        shared_w2_s,
        gate_q,
        gate_s,
        gate_bias,
        gate_mask,
        int(group_size),
        float(limit),
        int(top_k),
        bool(normalize_route),
        float(routed_scaling),
    )


def reset_moe_phase_profile_cpu() -> None:
    extension = _build()
    if extension is not None:
        extension.reset_moe_phase_profile()


def moe_phase_profile_cpu() -> torch.Tensor | None:
    extension = _build()
    if extension is None:
        return None
    return extension.moe_phase_profile()


def int4_gemv_cpu(
    x_row: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    cols: int,
    group_size: int,
) -> torch.Tensor | None:
    if (
        x_row.is_cuda
        or packed.is_cuda
        or scales.is_cuda
        or x_row.ndim != 2
        or x_row.shape[0] != 1
        or packed.dtype != torch.uint8
        or scales.dtype != torch.float16
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.int4_gemv(
        x_row,
        packed,
        scales,
        int(cols),
        int(group_size),
    )


def int4_gemv_many_cpu(
    x_row: torch.Tensor,
    packed: list[torch.Tensor],
    scales: list[torch.Tensor],
    group_size: int,
) -> list[torch.Tensor] | None:
    if (
        x_row.is_cuda
        or x_row.ndim != 2
        or x_row.shape[0] != 1
        or not packed
        or len(packed) != len(scales)
        or any(weight.is_cuda for weight in packed + scales)
        or any(weight.dtype != torch.uint8 for weight in packed)
        or any(scale.dtype != torch.float16 for scale in scales)
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.int4_gemv_many(
        x_row,
        packed,
        scales,
        int(group_size),
    )


def int4_grouped_gemv_cpu(
    x_groups: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    cols: int,
    group_size: int,
    rows_per_input: int,
) -> torch.Tensor | None:
    if (
        x_groups.is_cuda
        or x_groups.ndim != 2
        or packed.dtype != torch.uint8
        or scales.dtype != torch.float16
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.int4_grouped_gemv(
        x_groups,
        packed,
        scales,
        int(cols),
        int(group_size),
        int(rows_per_input),
    )


def o_proj_int4_cpu(
    x_groups: torch.Tensor,
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    a_cols: int,
    a_group_size: int,
    rows_per_input: int,
    b_packed: torch.Tensor,
    b_scales: torch.Tensor,
    b_cols: int,
    b_group_size: int,
) -> torch.Tensor | None:
    if (
        x_groups.is_cuda
        or x_groups.dtype != torch.float32
        or x_groups.ndim != 2
        or a_packed.dtype != torch.uint8
        or a_scales.dtype != torch.float16
        or b_packed.dtype != torch.uint8
        or b_scales.dtype != torch.float16
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.o_proj_int4(
        x_groups,
        a_packed,
        a_scales,
        int(a_cols),
        int(a_group_size),
        int(rows_per_input),
        b_packed,
        b_scales,
        int(b_cols),
        int(b_group_size),
    )


def hc_pre_norm_cpu(
    x: torch.Tensor,
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm: torch.Tensor,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        x.is_cuda
        or x.dtype != torch.float32
        or mixes.dtype != torch.float32
        or x.ndim != 4
        or x.shape[0] * x.shape[1] != 1
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.hc_pre_norm(
        x,
        mixes,
        scale,
        base,
        norm,
        int(sinkhorn_iters),
        float(rms_eps),
        float(hc_eps),
    )


def hc_post_cpu(
    out: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor | None:
    if (
        out.is_cuda
        or out.dtype != torch.float32
        or residual.dtype != torch.float32
        or residual.ndim != 4
        or residual.shape[0] * residual.shape[1] != 1
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.hc_post(
        out,
        residual,
        post,
        comb,
    )


def qkv_pre_cpu(
    q_rank_raw: torch.Tensor,
    kv_raw: torch.Tensor,
    q_norm: torch.Tensor,
    kv_norm: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        q_rank_raw.is_cuda
        or kv_raw.is_cuda
        or q_rank_raw.dtype != torch.float32
        or kv_raw.dtype != torch.float32
        or q_rank_raw.ndim != 2
        or q_rank_raw.shape[0] != 1
        or kv_raw.ndim != 2
        or kv_raw.shape[0] != 1
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.qkv_pre(
        q_rank_raw,
        kv_raw,
        q_norm,
        kv_norm,
        rope_cos,
        rope_sin,
        float(rms_eps),
    )


def q_post_cpu(
    query: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    rms_eps: float,
) -> torch.Tensor | None:
    if (
        query.is_cuda
        or query.dtype != torch.float32
        or query.ndim != 4
        or query.shape[0] * query.shape[1] != 1
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.q_post(
        query,
        rope_cos,
        rope_sin,
        float(rms_eps),
    )


def q_int4_post_cpu(
    q_rank: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    cols: int,
    group_size: int,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    heads: int,
    head_dim: int,
    rms_eps: float,
) -> torch.Tensor | None:
    if (
        q_rank.is_cuda
        or q_rank.dtype != torch.float32
        or packed.dtype != torch.uint8
        or scales.dtype != torch.float16
        or q_rank.ndim != 2
        or q_rank.shape[0] != 1
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.q_int4_post(
        q_rank,
        packed,
        scales,
        int(cols),
        int(group_size),
        rope_cos,
        rope_sin,
        int(heads),
        int(head_dim),
        float(rms_eps),
    )


def attention_decode_cpu(
    query: torch.Tensor,
    raw_values: torch.Tensor,
    raw_positions: torch.Tensor,
    selected_values: torch.Tensor,
    sink: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    scale: float,
) -> torch.Tensor | None:
    if (
        query.is_cuda
        or query.dtype != torch.float32
        or raw_values.dtype != torch.float32
        or selected_values.dtype != torch.float32
        or raw_positions.dtype != torch.long
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    return extension.attention_decode(
        query,
        raw_values,
        raw_positions,
        selected_values,
        sink,
        rope_cos,
        rope_sin,
        float(scale),
    )


def prebuild() -> bool:
    ok = _build(verbose=True) is not None
    print(
        "[tpq] CPU融合内核"
        + ("编译成功" if ok else f"不可用（{_ERR}），使用PyTorch回退")
    )
    return ok


def last_error() -> str | None:
    return _ERR
