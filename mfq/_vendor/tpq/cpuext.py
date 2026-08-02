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
_EXTENSION_NAME = "tpq_cpu_kernels_v86"
_PACKED_MOE_WORKSPACE: tuple[torch.Tensor, torch.Tensor] | None = None
_PACKED_THREE_WORKSPACE: tuple[torch.Tensor, ...] | None = None
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


def block_fp8_gemv_cpu(
    value: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    cols: int,
    block_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Direct compact E4M3FN block-scaled GEMV for one CPU token."""
    if (
        value.is_cuda
        or value.ndim != 2
        or value.shape != (1, int(cols))
        or value.dtype not in (torch.float32, torch.bfloat16)
        or weights.is_cuda
        or weights.dtype != torch.uint8
        or weights.ndim != 2
        or tuple(weights.shape) != (int(weights.shape[0]), int(cols))
        or scales.is_cuda
        or scales.dtype != torch.float32
        or int(block_size) != 128
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    rows = int(weights.shape[0])
    if output is None:
        output = torch.empty(rows, dtype=value.dtype)
    return extension.block_fp8_gemv(
        value,
        weights,
        scales,
        int(cols),
        int(block_size),
        output,
    )


def block_fp8_grouped_gemv_cpu(
    value: torch.Tensor,
    weight_ptrs: torch.Tensor,
    scale_ptrs: torch.Tensor,
    row_offsets: torch.Tensor,
    total_rows: int,
    cols: int,
    block_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Evaluate a logical row-concatenation of compact CPU FP8 weights."""
    if (
        value.is_cuda
        or value.ndim != 2
        or value.shape != (1, int(cols))
        or value.dtype not in (torch.float32, torch.bfloat16)
        or weight_ptrs.is_cuda
        or weight_ptrs.dtype != torch.int64
        or scale_ptrs.is_cuda
        or scale_ptrs.dtype != torch.int64
        or row_offsets.is_cuda
        or row_offsets.dtype != torch.int32
        or int(block_size) != 128
        or int(total_rows) <= 0
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    if output is None:
        output = torch.empty(int(total_rows), dtype=value.dtype)
    return extension.block_fp8_grouped_gemv(
        value,
        weight_ptrs.contiguous(),
        scale_ptrs.contiguous(),
        row_offsets.contiguous(),
        int(total_rows),
        int(cols),
        int(block_size),
        output,
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
        or not 8 <= bits <= 16
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


def _shared_projection_spec(weights: list[object]):
    """Return common layer metadata or ``None`` for an invalid mixed list."""
    first = weights[0]
    if any(
        int(weight.rows) != int(first.rows)
        or int(weight.blocks) != int(first.blocks)
        or int(weight.bits) != int(first.bits)
        or int(weight.dim) != int(first.dim)
        or tuple(weight.cb.shape) != tuple(first.cb.shape)
        or weight.cb.data_ptr() != first.cb.data_ptr()
        for weight in weights[1:]
    ):
        return None
    return first


def _shared_projection(
    weights: list[object],
    x_rows: torch.Tensor,
    *,
    allow_direct: bool = False,
) -> torch.Tensor | None:
    """Run one projection whose selected experts share layer metadata."""
    first = _shared_projection_spec(weights)
    if first is None:
        return None
    return vq_gemv_packed_list_cpu(
        x_rows,
        [weight.raw for weight in weights],
        first.cb.float().contiguous(),
        int(first.rows),
        int(first.blocks),
        int(first.bits),
        allow_direct=allow_direct,
    )


def _grouped_projection(
    weights: list[object],
    x_rows: torch.Tensor,
    output: torch.Tensor,
    *,
    allow_direct: bool = False,
) -> torch.Tensor | None:
    """Evaluate one projection while preserving per-group codebooks.

    Most projection archives share one layer codebook and take the fused
    single-call path.  Multi-codebook layouts group selected experts by the
    exact codebook pointer, invoke the same native packed GEMV for each group,
    and scatter into one persistent Top-K workspace.  Indices stay packed.
    """
    if not weights or output.shape[0] < len(weights):
        return None
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, weight in enumerate(weights):
        key = (
            int(weight.cb.data_ptr()),
            int(weight.rows),
            int(weight.blocks),
            int(weight.bits),
            int(weight.dim),
            int(weight.cb.shape[0]),
            int(weight.cb.shape[1]),
        )
        groups.setdefault(key, []).append(index)
    for positions in groups.values():
        group_weights = [weights[index] for index in positions]
        spec = _shared_projection_spec(group_weights)
        if spec is None:
            return None
        selection = torch.tensor(positions, dtype=torch.long)
        if x_rows.shape[0] == 1:
            inputs = x_rows
        else:
            inputs = x_rows.index_select(0, selection)
        values = vq_gemv_packed_list_cpu(
            inputs,
            [weight.raw for weight in group_weights],
            spec.cb.float().contiguous(),
            int(spec.rows),
            int(spec.blocks),
            int(spec.bits),
            allow_direct=allow_direct,
        )
        if values is None:
            return None
        # A Python list triggers advanced indexing and returns a temporary;
        # copying into it would leave the persistent workspace uninitialized.
        output[:, : int(spec.rows)].index_copy_(0, selection, values)
    return output[: len(weights), : int(weights[0].rows)]


def moe_packed_topk_cpu(
    x_row: torch.Tensor,
    experts: list[tuple[object, ...]],
    route_weights: torch.Tensor,
    limit: float,
    *,
    activation: str,
    activation_beta: float,
    activation_linear_beta: float | None,
) -> torch.Tensor | None:
    """Run mixed packed Top-K MoE through one native registered invocation.

    The payloads remain p8..p16. Legacy combined Gate+Up experts
    use the native fused entry below.  Three-projection archives are scheduled
    through the same registered call as Gate VQ -> Up VQ -> activation -> Down
    VQ, preserving every projection's own code dimension and codebook.
    """
    global _PACKED_MOE_WORKSPACE, _PACKED_THREE_WORKSPACE
    if (
        x_row.is_cuda
        or x_row.ndim != 2
        or x_row.shape[0] != 1
        or not 0 < len(experts) <= 16
        or route_weights.is_cuda
        or route_weights.numel() != len(experts)
        or len(experts[0]) not in (2, 3)
        or any(
            len(bundle) != len(experts[0])
            or any(not hasattr(weight, "raw") for weight in bundle)
            for bundle in experts
        )
    ):
        return None
    extension = _build()
    if extension is None:
        return None
    x_float = x_row.float().contiguous()
    weights = route_weights.float().contiguous()
    hidden = int(x_float.shape[1])

    if len(experts[0]) == 3:
        gate = [bundle[0] for bundle in experts]
        up = [bundle[1] for bundle in experts]
        dn = [bundle[2] for bundle in experts]
        intermediate = int(gate[0].rows)
        if (
            any(int(weight.rows) != intermediate for weight in up)
            or any(int(weight.rows) != hidden for weight in dn)
            or any(int(weight.cols) != hidden for weight in gate + up)
            or any(int(weight.cols) != intermediate for weight in dn)
        ):
            return None
        gate_spec = _shared_projection_spec(gate)
        up_spec = _shared_projection_spec(up)
        down_spec = _shared_projection_spec(dn)
        with _PACKED_MOE_LOCK:
            if (
                _PACKED_MOE_WORKSPACE is None
                or _PACKED_MOE_WORKSPACE[1].numel() < hidden
            ):
                _PACKED_MOE_WORKSPACE = (
                    torch.empty(1, dtype=torch.float32),
                    torch.empty(hidden, dtype=torch.float32),
                )
            result = _PACKED_MOE_WORKSPACE[1]
            if (
                gate_spec is not None
                and up_spec is not None
                and down_spec is not None
            ):
                return extension.moe_packed_three_projection(
                    x_float,
                    [weight.raw for weight in gate],
                    gate_spec.cb.float().contiguous(),
                    int(gate_spec.rows),
                    int(gate_spec.blocks),
                    int(gate_spec.bits),
                    [weight.raw for weight in up],
                    up_spec.cb.float().contiguous(),
                    int(up_spec.rows),
                    int(up_spec.blocks),
                    int(up_spec.bits),
                    [weight.raw for weight in dn],
                    down_spec.cb.float().contiguous(),
                    int(down_spec.rows),
                    int(down_spec.blocks),
                    int(down_spec.bits),
                    weights,
                    float(limit),
                    str(activation).strip().lower(),
                    float(activation_beta),
                    (
                        -1.0
                        if activation_linear_beta is None
                        else float(activation_linear_beta)
                    ),
                    result,
                )

            # Grouped projection codebooks (for example one codebook per
            # contiguous expert band) cannot use the one-codebook native fast
            # path above.  Retain one public operator call and one persistent
            # workspace while dispatching only the affected projection by
            # exact codebook group.
            required_shape = (len(experts), intermediate)
            down_shape = (len(experts), hidden)
            if (
                _PACKED_THREE_WORKSPACE is None
                or _PACKED_THREE_WORKSPACE[0].shape[0]
                < required_shape[0]
                or _PACKED_THREE_WORKSPACE[0].shape[1]
                < required_shape[1]
                or _PACKED_THREE_WORKSPACE[3].shape[0]
                < down_shape[0]
                or _PACKED_THREE_WORKSPACE[3].shape[1]
                < down_shape[1]
            ):
                _PACKED_THREE_WORKSPACE = (
                    torch.empty(required_shape, dtype=torch.float32),
                    torch.empty(required_shape, dtype=torch.float32),
                    torch.empty(required_shape, dtype=torch.float32),
                    torch.empty(down_shape, dtype=torch.float32),
                )
            gate_workspace, up_workspace, activated_workspace, down_workspace = (
                _PACKED_THREE_WORKSPACE
            )
            gate_values = _grouped_projection(
                gate,
                x_float,
                gate_workspace,
                allow_direct=True,
            )
            up_values = _grouped_projection(
                up,
                x_float,
                up_workspace,
                allow_direct=True,
            )
            if gate_values is None or up_values is None:
                return None
            activated = activated_workspace[
                : len(experts), :intermediate
            ]
            if limit != 0.0:
                gate_values.clamp_max_(float(limit))
                up_values.clamp_(-float(limit), float(limit))
            normalized_activation = str(activation).strip().lower()
            if normalized_activation == "situ":
                activated.copy_(gate_values)
                activated.div_(float(activation_beta)).tanh_()
                activated.mul_(float(activation_beta))
                activated.mul_(gate_values.sigmoid())
                if (
                    activation_linear_beta is not None
                    and float(activation_linear_beta) > 0.0
                ):
                    up_values.div_(float(activation_linear_beta)).tanh_()
                    up_values.mul_(float(activation_linear_beta))
                activated.mul_(up_values)
            elif normalized_activation in {"silu", "swiglu"}:
                activated.copy_(gate_values)
                activated.mul_(gate_values.sigmoid()).mul_(up_values)
            else:
                return None
            down_result = _grouped_projection(
                dn,
                activated,
                down_workspace,
                allow_direct=True,
            )
            if down_result is None:
                return None
            torch.mv(
                down_result.transpose(0, 1),
                weights,
                out=result[:hidden],
            )
            return result[:hidden]

    gu = [pair[0] for pair in experts]
    dn = [pair[1] for pair in experts]
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


def reset_three_projection_phase_profile() -> None:
    extension = _build()
    if extension is not None:
        extension.reset_three_projection_phase_profile()


def three_projection_phase_profile() -> dict[str, float | int]:
    extension = _build()
    if extension is None:
        return {}
    values = extension.three_projection_phase_profile()
    names = (
        "calls",
        "gate_seconds",
        "up_seconds",
        "activation_seconds",
        "down_seconds",
        "reduce_seconds",
    )
    return {
        name: int(value) if name == "calls" else float(value)
        for name, value in zip(names, values)
    }


def reset_block_fp8_gemv_profile() -> None:
    extension = _build()
    if extension is not None:
        extension.reset_block_fp8_gemv_profile()


def block_fp8_gemv_profile() -> dict[str, float | int]:
    extension = _build()
    if extension is None:
        return {}
    values = extension.block_fp8_gemv_profile()
    return {
        "calls": int(values[0]),
        "seconds": float(values[1]),
        "weight_elements": int(values[2]),
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
