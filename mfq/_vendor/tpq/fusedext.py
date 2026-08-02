"""TPQ 可选 CUDA 融合 kernel 的加载器。

包含六类融合 kernel（csrc/vq_gemv.cu）：
  - vq_gemv：VQ 分组 GEMV（码本查表+点积单 kernel，u8/u16 索引、码本/索引广播）；
  - hc_sinkhorn：Hyper-Connections 4×4 双随机归一化（softmax+20 轮一次 launch）；
  - rmsnorm：f32 行归一化；rope1：decode 单相位交错 RoPE。
  - dsv4_attn_decode：单 token 的 score/sink-softmax/value/RoPE。
  - dsv4_hc_pre：HC 的 RMS/GEMV/Sinkhorn/通道归约整段融合。

行为：
  - 导入时尝试用 torch.utils.cpp_extension.load 编译/复用缓存
    （已编译过走缓存，约 1-2s；未编译且工具链缺失时静默记为不可用）；
  - 可用时 available() 为 True，grouped.py / dsv4model.py 的钩子优先走融合路径；
  - 不可用（无 CUDA / 无 nvcc+MSVC+ninja / 编译失败）时自动回退
    torch 批量路径 —— 推理功能完全不依赖本模块。

手动预编译（推荐随安装执行一次）：
  python -c "from tpq import fusedext; fusedext.prebuild()"
环境变量：
  TPQ_FUSED=0  强制禁用（调试用）。
"""

from __future__ import annotations

import os
import shutil

import torch

_EXT = None
_ERR: str | None = None
_EXTENSION_NAME = "tpq_vq_gemv"


def _ensure_ninja_on_path() -> None:
    """让 PyTorch JIT 构建能找到 pip 安装的 Ninja 可执行文件。

    非交互 SSH 会话不一定继承 ``~/.local/bin``，即使 Python 已经可以
    导入 ninja 包。PyTorch 在构建前使用 ``shutil.which`` 查找可执行文件，
    因此在需要时补入该包声明的二进制目录。
    """
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
    if not os.path.isfile(os.path.join(bin_dir, executable)):
        return
    current = os.environ.get("PATH", "")
    paths = current.split(os.pathsep) if current else []
    normalized = {os.path.normcase(os.path.abspath(path)) for path in paths}
    if os.path.normcase(os.path.abspath(bin_dir)) not in normalized:
        os.environ["PATH"] = bin_dir + (os.pathsep + current if current else "")


def _build(verbose: bool = False):
    """编译（或命中缓存）并返回扩展模块；失败返回 None 并记录 last_error。"""
    global _EXT, _ERR
    if _EXT is not None:
        return _EXT
    if os.environ.get("TPQ_FUSED", "1") == "0":
        _ERR = "TPQ_FUSED=0 禁用"
        return None
    if not torch.cuda.is_available():
        _ERR = "无 CUDA"
        return None
    try:
        _ensure_ninja_on_path()
        # Windows 下新版 setuptools 的 distutils shim 不自动挂
        # _msvccompiler 子模块，而 torch._run_ninja_build 以属性方式访问它；
        # Linux 不得导入该 Windows 专用模块，否则会在编译 CUDA 扩展前失败。
        if os.name == "nt":
            import distutils._msvccompiler  # noqa: F401
        # 锁定当前卡的 arch（否则 torch 警告"all archs"且按全架构编译，很慢）。
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            try:
                _maj, _min = torch.cuda.get_device_capability(0)
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_maj}.{_min}"
            except Exception:
                pass
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "csrc", "vq_gemv.cu")
        _EXT = load(name=_EXTENSION_NAME, sources=[src],
                    extra_cuda_cflags=["-O3"], verbose=verbose)
        _ERR = None
    except Exception as e:  # noqa: BLE001 —— 任何编译/加载失败都回退
        _EXT = None
        _ERR = f"{type(e).__name__}: {e}"
    return _EXT


def available() -> bool:
    """融合 kernel 是否可用。"""
    return _EXT is not None


def last_error() -> str | None:
    """最近一次构建失败的原因（诊断用；可用时返回 None）。"""
    return _ERR


def prebuild() -> bool:
    """显式预编译入口，返回是否成功。"""
    ok = _build(verbose=True) is not None
    print("[fusedext] 融合 kernel " + ("编译成功并已缓存" if ok else
          f"不可用（{_ERR}），将使用 torch 批量路径"))
    return ok


_build()

if _EXT is not None:

    def vq_gemv_fused(x_rows: torch.Tensor, idx: torch.Tensor,
                      cb: torch.Tensor) -> torch.Tensor:
        """融合 VQ 分组 GEMV：x_rows [N|1, C] f32，idx u8/u16 [N,R,B]，
        cb f32 [N|1,K,D]（1 = 同层共享码本广播）→ [N,R] f32。"""
        return _EXT.vq_gemv(x_rows.contiguous(), idx.contiguous(), cb.contiguous())

    def short_conv3_fused(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> bool:
        """Update three BF16 depthwise short-convolution states in one launch."""
        if (
            not query.is_cuda
            or query.dtype != torch.bfloat16
            or key.dtype != torch.bfloat16
            or value.dtype != torch.bfloat16
            or any(item.dtype != torch.bfloat16 for item in states)
            or weights[0].dtype not in (
                torch.bfloat16,
                torch.float32,
            )
            or any(item.dtype != weights[0].dtype for item in weights)
        ):
            return False
        return bool(_EXT.kimi_short_conv3(
            query,
            key,
            value,
            states[0],
            states[1],
            states[2],
            weights[0],
            weights[1],
            weights[2],
        ))

    def kda_recurrent_fused(
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
        lower_bound: float = -5.0,
    ) -> torch.Tensor:
        """KDA decode update with persistent FP32 V-first state."""
        return _EXT.kimi_kda_recurrent(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            gate.contiguous(),
            beta.float().contiguous(),
            a_log.float().contiguous(),
            dt_bias.float().contiguous(),
            state,
            workspace,
            output,
            float(lower_bound),
        )

    def gated_rmsnorm_fused(
        value: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        eps: float,
    ) -> torch.Tensor | None:
        """Fuse decode RMSNorm, sigmoid gate and multiply."""
        if (
            not value.is_cuda
            or value.dtype != torch.bfloat16
            or gate.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
        ):
            return None
        return _EXT.kimi_gated_rmsnorm(
            value,
            gate,
            weight,
            output,
            float(eps),
        )

    def packed_moe_topk_fused(
        value: torch.Tensor,
        route_ids: torch.Tensor,
        weights: torch.Tensor,
        metadata: torch.Tensor,
        activation: str,
        beta: float,
        linear_beta: float,
        limit: float,
        hidden_workspace: torch.Tensor,
        out_workspace: torch.Tensor,
        result: torch.Tensor,
        p12_count: int = 0,
        projection_layout_tag: int = 0,
    ) -> torch.Tensor | None:
        """Run Top-K gated MLP directly from packed indices.

        ``p12_count`` is the grouped p12 prefix length.  ``-1`` scans an
        ungrouped route vector on device, which avoids a GPU→CPU sync in the
        full-resident multi-GPU path.
        """
        if (
            not value.is_cuda
            or value.dtype != torch.bfloat16
            or value.ndim != 2
            or value.shape[0] != 1
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or not 0 < route_ids.numel() <= 16
            or weights.dtype != torch.float32
            or weights.shape != route_ids.shape
            or metadata.dtype != torch.long
            or metadata.ndim != 2
            or metadata.shape[0] not in (10, 15)
            or not -1 <= p12_count <= route_ids.numel()
            or projection_layout_tag not in (0, 1, 2)
        ):
            return None
        activation_name = str(activation).strip().lower()
        activation_kind = {
            "situ": 0,
            "silu": 1,
            "swiglu": 1,
        }.get(activation_name)
        if activation_kind is None:
            return None
        return _EXT.packed_moe_topk(
            value.contiguous(),
            route_ids.contiguous(),
            weights.contiguous(),
            metadata.contiguous(),
            int(activation_kind),
            float(beta),
            float(linear_beta),
            float(limit),
            hidden_workspace,
            out_workspace,
            result,
            int(p12_count),
            int(projection_layout_tag),
        )

    def packed_route_slots_fused(
        route_ids: torch.Tensor,
        directory: torch.Tensor,
        selected: torch.Tensor,
        hit_mask: torch.Tensor,
    ) -> bool:
        """Gather dynamic Top-K expert slot metadata entirely on CUDA."""
        if (
            not route_ids.is_cuda
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or not 0 < route_ids.numel() <= 16
            or directory.dtype != torch.long
            or directory.ndim != 2
            or selected.dtype != torch.long
            or selected.shape
            != (directory.shape[1], route_ids.numel())
            or hit_mask.dtype != torch.bool
            or hit_mask.shape != route_ids.shape
        ):
            return False
        return bool(
            _EXT.packed_route_slots_out(
                route_ids.contiguous(),
                directory.contiguous(),
                selected,
                hit_mask,
            )
        )

    def moe_mlp_slots_fused(
        x_rows: torch.Tensor,
        gu_indices: list[torch.Tensor],
        gu_codebooks: list[torch.Tensor],
        dn_indices: list[torch.Tensor],
        dn_codebooks: list[torch.Tensor],
        weights: torch.Tensor,
        limit: float,
        hidden_workspace: torch.Tensor,
        out_workspace: torch.Tensor,
        result: torch.Tensor,
    ) -> torch.Tensor:
        """稳定专家槽 BF16 MLP；四个 kernel 完成 GU/SwiGLU/DN/加权。"""
        return _EXT.moe_mlp_slots(
            x_rows,
            gu_indices,
            gu_codebooks,
            dn_indices,
            dn_codebooks,
            weights,
            float(limit),
            hidden_workspace,
            out_workspace,
            result,
        )

    def moe_mlp_routed_slots_fused(
        x_rows: torch.Tensor,
        route_ids: torch.Tensor,
        weights: torch.Tensor,
        metadata: torch.Tensor,
        limit: float,
        hidden_workspace: torch.Tensor,
        out_workspace: torch.Tensor,
        result: torch.Tensor,
        accumulate: bool = False,
    ) -> torch.Tensor | None:
        """Full-resident EP MLP whose Top-K selection remains on CUDA."""
        if (
            os.environ.get("TPQ_EP_DEVICE_ROUTE", "1") == "0"
            or not x_rows.is_cuda
            or x_rows.dtype != torch.bfloat16
            or x_rows.shape[0] != 1
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or route_ids.numel() == 0
            or route_ids.numel() > 8
            or weights.dtype != torch.float32
            or weights.shape != route_ids.shape
            or metadata.dtype != torch.long
            or metadata.ndim != 2
            or metadata.shape[0] != 10
        ):
            return None
        return _EXT.moe_mlp_routed_slots(
            x_rows.contiguous(),
            route_ids.contiguous(),
            weights.contiguous(),
            metadata.contiguous(),
            float(limit),
            hidden_workspace,
            out_workspace,
            result,
            os.environ.get("TPQ_VQ_D4_SPECIALIZED", "1") != "0",
            bool(accumulate),
        )

    def moe_mlp_routed_vv_fused(
        x_rows: torch.Tensor,
        route_ids: torch.Tensor,
        weights: torch.Tensor,
        metadata: torch.Tensor,
        limit: float,
        hidden_workspace: torch.Tensor,
        out_workspace: torch.Tensor,
        result: torch.Tensor,
        accumulate: bool = False,
    ) -> torch.Tensor | None:
        """Run independent D4/K4096 experts with a shared-codebook kernel."""
        if (
            os.environ.get("TPQ_EP_DEVICE_ROUTE", "1") == "0"
            or not x_rows.is_cuda
            or x_rows.dtype != torch.bfloat16
            or x_rows.shape[0] != 1
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or route_ids.numel() == 0
            or route_ids.numel() > 8
            or weights.dtype != torch.float32
            or weights.shape != route_ids.shape
            or metadata.dtype != torch.long
            or metadata.ndim != 2
            or metadata.shape[0] != 10
        ):
            return None
        return _EXT.moe_mlp_routed_vv(
            x_rows.contiguous(),
            route_ids.contiguous(),
            weights.contiguous(),
            metadata.contiguous(),
            float(limit),
            hidden_workspace,
            out_workspace,
            result,
            bool(accumulate),
        )

    def moe_mlp_routed_codegemm_fused(
        x_rows: torch.Tensor,
        route_ids: torch.Tensor,
        weights: torch.Tensor,
        metadata: torch.Tensor,
        gu_sum: torch.Tensor,
        activation: torch.Tensor,
        dn_sum: torch.Tensor,
        result: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run the full-resident v256/D4 Psumbook expert kernel."""
        if (
            os.environ.get("TPQ_EP_DEVICE_ROUTE", "1") == "0"
            or not x_rows.is_cuda
            or x_rows.dtype != torch.bfloat16
            or x_rows.shape[0] != 1
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or route_ids.numel() == 0
            or route_ids.numel() > 8
            or weights.dtype != torch.float32
            or weights.shape != route_ids.shape
            or metadata.dtype != torch.long
            or metadata.ndim != 2
            or metadata.shape[0] != 10
        ):
            return None
        return _EXT.moe_mlp_routed_codegemm(
            x_rows.contiguous(),
            route_ids.contiguous(),
            weights.contiguous(),
            metadata.contiguous(),
            gu_sum,
            activation,
            dn_sum,
            result,
        )

    def pack_vq_tensor_shard_codegemm(
        source_gu: torch.Tensor,
        source_dn: torch.Tensor,
        target_gu: torch.Tensor,
        target_dn: torch.Tensor,
        global_intermediate: int,
        shard_start: int,
        local_intermediate: int,
    ) -> bool:
        """Pack one full-GPU tensor shard without changing its byte size."""
        if (
            source_gu.dtype != torch.uint8
            or source_dn.dtype != torch.uint8
            or target_gu.dtype != torch.uint8
            or target_dn.dtype != torch.uint8
        ):
            return False
        _EXT.pack_vq_tensor_shard_codegemm(
            source_gu,
            source_dn,
            target_gu,
            target_dn,
            int(global_intermediate),
            int(shard_start),
            int(local_intermediate),
        )
        return True

    def unpack_vq_codegemm(
        storage: torch.Tensor,
        rows: int,
        blocks: int,
    ) -> torch.Tensor:
        """Restore a temporary row-major index matrix for prefill."""
        return _EXT.unpack_vq_codegemm(
            storage,
            int(rows),
            int(blocks),
        )

    def expert_dispatch_pack_fused(
        x: torch.Tensor,
        route_ids: torch.Tensor,
        weights: torch.Tensor,
        x_out: torch.Tensor,
        route_ids_out: torch.Tensor,
        weights_out: torch.Tensor,
    ) -> bool:
        """一次 peer kernel 完成远端专家输入、ID 和权重分发。"""
        if (
            os.environ.get("TPQ_EP_FUSED_DISPATCH", "1") == "0"
            or not x.is_cuda
            or x.dtype not in (torch.float32, torch.bfloat16)
            or x.ndim != 2
            or x.shape[0] != 1
            or route_ids.dtype != torch.long
            or route_ids.ndim != 1
            or weights.dtype != torch.float32
            or weights.shape != route_ids.shape
            or x_out.dtype != torch.bfloat16
            or x_out.shape != x.shape
            or route_ids_out.dtype != torch.long
            or route_ids_out.shape != route_ids.shape
            or weights_out.dtype != torch.float32
            or weights_out.shape != weights.shape
        ):
            return False
        _EXT.expert_dispatch_pack(
            x.contiguous(),
            route_ids.contiguous(),
            weights.contiguous(),
            x_out,
            route_ids_out,
            weights_out,
        )
        return True

    def tp_peer_copy_fused(
        source: torch.Tensor,
        destination: torch.Tensor,
    ) -> bool:
        """Graph-safe rank dispatch without CUDA memcpy capture edges."""
        if (
            not source.is_cuda
            or not destination.is_cuda
            or source.dtype not in (
                torch.float32,
                torch.bfloat16,
                torch.long,
            )
            or source.dtype != destination.dtype
            or source.shape != destination.shape
        ):
            return False
        _EXT.tp_peer_copy(
            source.contiguous(),
            destination,
        )
        return True

    def tp_attention_peer_dispatch_fused(
        source_q: torch.Tensor,
        source_c: torch.Tensor,
        source_k: torch.Tensor,
        source_position: torch.Tensor,
        destination_q: torch.Tensor,
        destination_c: torch.Tensor,
        destination_k: torch.Tensor,
        destination_position: torch.Tensor,
    ) -> bool:
        """Copy all fixed Attention TP inputs with one graph kernel."""
        float_pairs = (
            (source_q, destination_q),
            (source_c, destination_c),
            (source_k, destination_k),
        )
        if any(
            not source.is_cuda
            or not destination.is_cuda
            or source.dtype != torch.float32
            or destination.dtype != torch.float32
            or source.shape != destination.shape
            for source, destination in float_pairs
        ):
            return False
        if (
            not source_position.is_cuda
            or not destination_position.is_cuda
            or source_position.dtype != torch.long
            or destination_position.dtype != torch.long
            or source_position.numel() != 1
            or destination_position.numel() != 1
        ):
            return False
        _EXT.tp_attention_peer_dispatch(
            source_q.contiguous(),
            source_c.contiguous(),
            source_k.contiguous(),
            source_position.contiguous(),
            destination_q,
            destination_c,
            destination_k,
            destination_position,
        )
        return True

    def tp_attention_source_pack_fused(
        source_q: torch.Tensor,
        source_c: torch.Tensor,
        source_k: torch.Tensor,
        destination_q: torch.Tensor,
        destination_c: torch.Tensor,
        destination_k: torch.Tensor,
        destination_position: torch.Tensor,
        position: int,
    ) -> bool:
        """Pack changing primary-rank Attention inputs with one kernel."""
        float_pairs = (
            (source_q, destination_q),
            (source_c, destination_c),
            (source_k, destination_k),
        )
        if any(
            not source.is_cuda
            or not destination.is_cuda
            or source.dtype != torch.float32
            or destination.dtype != torch.float32
            or source.shape != destination.shape
            or source.device != destination.device
            for source, destination in float_pairs
        ):
            return False
        if (
            not destination_position.is_cuda
            or destination_position.dtype != torch.long
            or destination_position.numel() != 1
            or destination_position.device != source_q.device
        ):
            return False
        _EXT.tp_attention_source_pack(
            source_q.contiguous(),
            source_c.contiguous(),
            source_k.contiguous(),
            destination_q,
            destination_c,
            destination_k,
            destination_position,
            int(position),
        )
        return True

    def hc_split_fused(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
                       hc: int, iters: int, eps: float):
        """融合 HC sinkhorn：mixes [..., 24] f32 CUDA + hc==4 时返回
        (pre, post, comb)（单次 kernel 完成 softmax + 全部归一化迭代）；
        不满足条件返回 None（调用方回退 tpq.dsv4.hc_split 的 torch 循环）。
        数值与 torch 版同序 fp32 计算，差异在 1e-7 量级。"""
        if (hc != 4 or not mixes.is_cuda or mixes.dtype != torch.float32
                or scale.dtype != torch.float32):
            return None
        out = _EXT.hc_sinkhorn(mixes, scale, base, iters, float(eps))
        pre = out[..., :4]
        post = out[..., 4:8]
        comb = out[..., 8:].unflatten(-1, (4, 4))
        return pre, post, comb

    def rmsnorm_fused(
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float,
        output: torch.Tensor | None = None,
    ):
        """融合 RMSNorm（f32 CUDA）：不满足条件返回 None（回退 torch 路径）。"""
        if not x.is_cuda or x.dtype != torch.float32 or w.dtype != torch.float32:
            return None
        return _EXT.rmsnorm(x, w, float(eps), output)

    def rmsnorm_bf16_fused(
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float,
        output: torch.Tensor | None = None,
    ):
        """One-launch BF16 RMSNorm with source BF16 or FP32 weights."""
        if (
            not x.is_cuda
            or x.dtype != torch.bfloat16
            or w.dtype not in (torch.bfloat16, torch.float32)
            or w.ndim != 1
            or w.numel() != x.shape[-1]
        ):
            return None
        return _EXT.rmsnorm_bf16(
            x,
            w,
            float(eps),
            output,
        )

    def attention_residual_bf16_fused(
        prefix: torch.Tensor,
        residual: torch.Tensor,
        projection: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        output: torch.Tensor | None = None,
        post_norm_weight: torch.Tensor | None = None,
        score_workspace: torch.Tensor | None = None,
        residual_inverse: torch.Tensor | None = None,
    ):
        if (
            not prefix.is_cuda
            or prefix.dtype != torch.bfloat16
            or residual.dtype != torch.bfloat16
            or projection.dtype != torch.bfloat16
            or norm_weight.dtype != torch.bfloat16
            or (
                post_norm_weight is not None
                and post_norm_weight.dtype != torch.bfloat16
            )
            or prefix.shape[0] != 1
            or residual.ndim != 3
            or not 0 < residual.shape[1] <= 31
            or (
                residual.shape[1] + 1
                > int(
                    os.environ.get(
                        "TPQ_RESIDUAL_SINGLE_MAX_ROWS",
                        "2",
                    )
                )
                and (
                    score_workspace is None
                    or not score_workspace.is_cuda
                    or score_workspace.dtype != torch.float32
                    or score_workspace.numel() < 32
                    or score_workspace.device != prefix.device
                )
            )
            or (
                residual_inverse is not None
                and (
                    not residual_inverse.is_cuda
                    or residual_inverse.dtype != torch.float32
                    or residual_inverse.numel() < residual.shape[1]
                    or residual_inverse.device != prefix.device
                )
            )
        ):
            return None
        return _EXT.attention_residual_bf16(
            prefix,
            residual,
            projection,
            norm_weight,
            post_norm_weight,
            float(eps),
            output,
            score_workspace,
            int(
                os.environ.get(
                    "TPQ_RESIDUAL_SINGLE_MAX_ROWS",
                    "2",
                )
            ),
            residual_inverse,
        )

    def gated_activation_bf16_fused(
        gate: torch.Tensor,
        up: torch.Tensor,
        activation: str,
        beta: float,
        linear_beta: float | None,
        limit: float = 0.0,
        output: torch.Tensor | None = None,
    ):
        normalized = activation.strip().lower()
        if (
            not gate.is_cuda
            or gate.dtype != torch.bfloat16
            or up.dtype != torch.bfloat16
            or gate.shape != up.shape
            or normalized not in {"silu", "swiglu", "situ"}
        ):
            return None
        return _EXT.gated_activation_bf16(
            gate.contiguous(),
            up.contiguous(),
            1 if normalized == "situ" else 0,
            float(beta),
            -1.0 if linear_beta is None else float(linear_beta),
            float(limit),
            output,
        )

    def glm_mla_bmm_decode_fused(
        input: torch.Tensor,
        weight: torch.Tensor,
        transpose_weight: bool,
        output: torch.Tensor | None = None,
    ):
        """Decode-only BF16 MLA batch GEMM through direct cuBLAS."""
        if (
            not input.is_cuda
            or input.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or input.ndim != 3
            or weight.ndim != 3
            or input.shape[1] != 1
        ):
            return None
        return _EXT.glm_mla_bmm_decode(
            input,
            weight,
            bool(transpose_weight),
            output,
        )

    def rope1_fused(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                    inverse: bool = False):
        """融合 RoPE（交错对）：仅 decode 单相位场景——x [..., rd] f32 CUDA 且
        cos/sin 各 rd/2 个元素（全部行同相位）时生效，否则 None（回退 torch）。
        数值与 tpq.dsv4.rope_apply 逐项一致。"""
        if (not x.is_cuda or x.dtype != torch.float32
                or cos.numel() * 2 != x.shape[-1] or sin.numel() * 2 != x.shape[-1]):
            return None
        return _EXT.rope1(x, cos.reshape(-1), sin.reshape(-1), bool(inverse))

    def glm_rope_qk_fused(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        """Fuse GLM MLA Q/K RoPE while preserving the reference cat layout."""
        if (
            os.environ.get("TPQ_GLM_ROPE_FUSED", "1") == "0"
            or not q.is_cuda
            or q.dtype != torch.float32
            or k.dtype != torch.float32
            or q.ndim != 3
            or k.ndim != 3
            or cos.ndim != 2
            or sin.ndim != 2
            or q.shape[1:] != k.shape[1:]
            or k.shape[0] != 1
            or cos.shape != sin.shape
            or cos.shape != (q.shape[1], q.shape[2] // 2)
        ):
            return None
        return _EXT.glm_rope_qk(q, k, cos, sin)

    def glm_latent_kv_decode_prepare_fused(
        c_raw: torch.Tensor,
        c_weight: torch.Tensor,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        ckv_buffer: torch.Tensor,
        krot_buffer: torch.Tensor,
        position: torch.Tensor,
        eps: float,
        output: torch.Tensor | None = None,
    ):
        """Fuse decode C RMSNorm, Q/K RoPE and BF16 latent-cache writes."""
        if (
            os.environ.get("TPQ_GLM_LATENT_PREP_FUSED", "1") == "0"
            or not c_raw.is_cuda
            or c_raw.dtype != torch.float32
            or c_weight.dtype != torch.float32
            or q_rot.dtype != torch.float32
            or k_rot.dtype != torch.float32
            or cos_cache.dtype != torch.float32
            or sin_cache.dtype != torch.float32
            or ckv_buffer.dtype != torch.bfloat16
            or krot_buffer.dtype != torch.bfloat16
            or not position.is_cuda
            or position.dtype != torch.long
            or position.numel() != 1
            or c_raw.shape[0] != 1
            or q_rot.ndim != 3
            or q_rot.shape[1] != 1
            or k_rot.shape != (1, 1, q_rot.shape[2])
        ):
            return None
        return _EXT.glm_latent_kv_decode_prepare(
            c_raw,
            c_weight,
            q_rot,
            k_rot,
            cos_cache,
            sin_cache,
            ckv_buffer,
            krot_buffer,
            position.contiguous(),
            float(eps),
            output,
        )

    def flashinfer_mla_batch1_plan_fused(
        int_workspace: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        length: int,
        page_size: int,
        heads: int,
        plan_info,
    ) -> bool:
        """Build the supported batch-1 MLA schedule directly on the GPU.

        FlashInfer may change the concrete planner layout with the head count.
        The public TPQ kernel implements FlashInfer's one- and two-CTA
        cluster layouts.  A future layout is not an error: callers must fall
        back to FlashInfer's own planner instead of aborting model execution.
        """
        normalized_plan = [int(value) for value in plan_info]
        if (
            os.environ.get(
                "TPQ_FLASHINFER_GPU_PLAN",
                "1",
            )
            == "0"
            or not int_workspace.is_cuda
            or int_workspace.dtype != torch.uint8
            or kv_indptr.dtype != torch.int32
            or kv_indices.dtype != torch.int32
            or kv_len_arr.dtype != torch.int32
            or length <= 0
            or page_size <= 0
            or heads <= 0
            or len(normalized_plan) != 18
            or normalized_plan[0] not in (1, 2)
        ):
            return False
        return bool(
            _EXT.flashinfer_mla_batch1_plan(
                int_workspace,
                kv_indptr,
                kv_indices,
                kv_len_arr,
                int(length),
                int(page_size),
                int(heads),
                normalized_plan,
            )
        )

    def latent_mla_attention_decode_fused(
        query_nope: torch.Tensor,
        query_rope: torch.Tensor,
        latent_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        position: torch.Tensor,
        scale_denominator: float,
        score_workspace: torch.Tensor,
        output: torch.Tensor | None = None,
    ):
        """Dynamic-length latent MLA selected by tensor capabilities."""
        if (
            not query_nope.is_cuda
            or query_nope.dtype != torch.bfloat16
            or query_rope.dtype != torch.bfloat16
            or latent_cache.dtype != torch.bfloat16
            or rope_cache.dtype != torch.bfloat16
            or position.dtype != torch.long
            or score_workspace.dtype != torch.float32
            or query_nope.ndim != 3
            or query_nope.shape[1] != 1
            or query_rope.ndim != 3
            or query_rope.shape[:2] != query_nope.shape[:2]
            or latent_cache.ndim != 2
            or rope_cache.ndim != 2
            or latent_cache.shape[0] != rope_cache.shape[0]
            or latent_cache.shape[1] != query_nope.shape[2]
            or rope_cache.shape[1] != query_rope.shape[2]
            or score_workspace.shape
            != (query_nope.shape[0], latent_cache.shape[0])
            or position.numel() != 1
            or scale_denominator <= 0.0
        ):
            return None
        return _EXT.latent_mla_attention_decode(
            query_nope.contiguous(),
            query_rope.contiguous(),
            latent_cache.contiguous(),
            rope_cache.contiguous(),
            position.contiguous(),
            float(scale_denominator),
            score_workspace,
            output,
        )

    def glm_merge_scores_fused(
        a: torch.Tensor,
        b: torch.Tensor,
        scale: float,
    ):
        """Fuse latent MLA score casts, scaling and addition."""
        if (
            os.environ.get("TPQ_GLM_SCORE_FUSED", "1") == "0"
            or not a.is_cuda
            or a.dtype != torch.bfloat16
            or b.dtype != torch.bfloat16
            or a.shape != b.shape
        ):
            return None
        return _EXT.glm_merge_scores(a, b, float(scale))

    def dsv4_attn_decode_fused(q: torch.Tensor, win_kv: torch.Tensor,
                               win_pos: torch.Tensor, comp_kv: torch.Tensor,
                               sink: torch.Tensor, cos: torch.Tensor,
                               sin: torch.Tensor, scale: float):
        """DSV4 B=1,T=1 attention 核；过长或 dtype/shape 不满足时返回 None。"""
        seq = win_kv.shape[1] + comp_kv.shape[1]
        if (not q.is_cuda or q.dtype != torch.float32 or q.shape[0] != 1
                or win_kv.dtype != torch.float32 or comp_kv.dtype != torch.float32
                or win_pos.dtype != torch.long or seq > 4096):
            return None
        return _EXT.dsv4_attn_decode(
            q.contiguous(), win_kv, win_pos, comp_kv, sink,
            cos.reshape(-1), sin.reshape(-1), float(scale)
        )

    def dsv4_hc_pre_fused(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor,
                          base: torch.Tensor, iters: int, eps: float):
        """融合 HC pre；返回形状与 dsv4.hc_pre 一致，不满足条件时返回 None。"""
        if (not x.is_cuda or x.dtype != torch.float32 or x.shape[-2] != 4
                or fn.dtype not in (torch.float32, torch.bfloat16)
                or scale.dtype != torch.float32 or base.dtype != torch.float32):
            return None
        y, post, comb = _EXT.dsv4_hc_pre(
            x, fn, scale, base, int(iters), float(eps)
        )
        lead = x.shape[:-2]
        return (
            y.view(*lead, x.shape[-1]),
            post.view(*lead, 4),
            comb.view(*lead, 4, 4),
        )

    def dsv4_hc_pre_norm_fused(
        x: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm: torch.Tensor,
        iters: int,
        eps: float,
        output_buffers: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ] | None = None,
    ):
        """BF16 HC pre + RMSNorm；可复用调用方固定输出缓冲。"""
        if (
            not x.is_cuda
            or x.dtype != torch.bfloat16
            or x.shape[-2] != 4
            or not all(
                isinstance(t, torch.Tensor)
                for t in (fn, scale, base, norm)
            )
            or fn.dtype not in (torch.float32, torch.bfloat16)
            or norm.dtype not in (torch.float32, torch.bfloat16)
            or scale.dtype != torch.float32
            or base.dtype != torch.float32
        ):
            return None
        if output_buffers is None:
            y, post, comb = _EXT.dsv4_hc_pre_norm(
                x, fn, scale, base, norm, int(iters), float(eps)
            )
        else:
            y, post, comb = _EXT.dsv4_hc_pre_norm_out(
                x,
                fn,
                scale,
                base,
                norm,
                *output_buffers,
                int(iters),
                float(eps),
            )
        lead = x.shape[:-2]
        return (
            y.view(*lead, x.shape[-1]),
            post.view(*lead, 4),
            comb.view(*lead, 4, 4),
        )

    def dsv4_hc_post_fused(
        out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        output: torch.Tensor | None = None,
    ):
        """BF16 HC post residual mix; accumulation stays FP32."""
        if (
            not out.is_cuda
            or out.dtype not in (torch.float32, torch.bfloat16)
            or residual.dtype != torch.bfloat16
            or post.dtype != torch.bfloat16
            or comb.dtype != torch.bfloat16
            or residual.shape[-2] != 4
        ):
            return None
        arguments = (
            out.contiguous(),
            residual.contiguous(),
            post.contiguous(),
            comb.contiguous(),
        )
        if output is None:
            return _EXT.dsv4_hc_post(*arguments)
        return _EXT.dsv4_hc_post_out(*arguments, output)

    def dsv4_hc_post_moe_fused(
        routed: torch.Tensor,
        shared: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        output: torch.Tensor | None = None,
    ):
        """Fuse BF16 routed/shared merge with Hyper-Connection post."""
        if (
            not routed.is_cuda
            or routed.dtype != torch.float32
            or shared.dtype != torch.bfloat16
            or residual.dtype != torch.bfloat16
            or post.dtype != torch.bfloat16
            or comb.dtype != torch.bfloat16
            or residual.shape[-2] != 4
            or routed.numel() * 4 != residual.numel()
            or shared.numel() * 4 != residual.numel()
        ):
            return None
        arguments = (
            routed.contiguous(),
            shared.contiguous(),
            residual.contiguous(),
            post.contiguous(),
            comb.contiguous(),
        )
        if output is None:
            return _EXT.dsv4_hc_post_moe(*arguments)
        return _EXT.dsv4_hc_post_moe_out(*arguments, output)

    def dsv4_route_post_fused(
        scores: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        top_k: int,
    ):
        """Fuse learned-router masked top-k selection and score gather."""
        if (
            not scores.is_cuda
            or scores.dtype != torch.float32
            or scores.dim() != 2
            or scores.shape[0] != 1
            or bias.dtype != torch.float32
            or mask.dtype != torch.bool
            or bias.numel() != scores.shape[1]
            or mask.numel() != scores.shape[1]
            or top_k <= 0
            or top_k > 16
        ):
            return None
        weights, indices = _EXT.dsv4_route_post(
            scores.contiguous(),
            bias.contiguous(),
            mask.contiguous(),
            int(top_k),
        )
        return weights, indices

    def route_topk_sigmoid_fused(
        logits: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        top_k: int,
        routed_scaling: float,
        output_buffers: tuple[
            torch.Tensor,
            torch.Tensor,
        ] | None = None,
    ):
        """Fuse sigmoid, corrected Top-K, gather and normalization."""
        if (
            os.environ.get(
                "TPQ_ROUTE_FUSED",
                os.environ.get("TPQ_GLM_ROUTE_FUSED", "1"),
            ) == "0"
            or not logits.is_cuda
            or logits.dtype != torch.float32
            or logits.dim() != 2
            or logits.shape[0] != 1
            or bias.dtype != torch.float32
            or mask.dtype != torch.bool
            or bias.numel() != logits.shape[1]
            or mask.numel() != logits.shape[1]
            or top_k <= 0
            or top_k > 16
        ):
            return None
        if output_buffers is None:
            weights, indices = _EXT.sigmoid_route(
                logits.contiguous(),
                bias.contiguous(),
                mask.contiguous(),
                int(top_k),
                float(routed_scaling),
            )
        else:
            weights, indices = output_buffers
            if (
                weights.dtype != torch.float32
                or indices.dtype != torch.long
                or weights.shape != (1, top_k)
                or indices.shape != (1, top_k)
                or weights.device != logits.device
                or indices.device != logits.device
            ):
                return None
            weights, indices = _EXT.sigmoid_route_out(
                logits.contiguous(),
                bias.contiguous(),
                mask.contiguous(),
                int(top_k),
                float(routed_scaling),
                weights,
                indices,
            )
        return weights, indices

    def linear_route_topk_sigmoid_fused(
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
        top_k: int,
        routed_scaling: float,
        output_buffers: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ):
        """Fuse a source-native FP32 router projection with sigmoid Top-K."""
        if (
            os.environ.get("TPQ_LINEAR_ROUTE_FUSED", "1") == "0"
            or not value.is_cuda
            or value.dtype not in (torch.bfloat16, torch.float32)
            or weight.dtype != torch.float32
            or weight.ndim != 2
            or value.shape != (1, weight.shape[1])
            or bias.dtype != torch.float32
            or mask.dtype != torch.bool
            or bias.numel() != weight.shape[0]
            or mask.numel() != weight.shape[0]
            or top_k <= 0
            or top_k > 16
        ):
            return None
        logits, weights, indices = output_buffers
        if (
            logits.dtype != torch.float32
            or logits.shape != (1, weight.shape[0])
            or weights.dtype != torch.float32
            or weights.shape != (1, top_k)
            or indices.dtype != torch.long
            or indices.shape != (1, top_k)
            or any(
                tensor.device != value.device
                for tensor in (
                    weight,
                    bias,
                    mask,
                    logits,
                    weights,
                    indices,
                )
            )
        ):
            return None
        weights, indices = _EXT.linear_sigmoid_route_out(
            value.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            mask.contiguous(),
            int(top_k),
            float(routed_scaling),
            logits,
            weights,
            indices,
        )
        return weights, indices

    # 旧公开名仅保留给外部脚本过渡；模型运行时统一经过 ops.route_topk。
    glm_route_fused = route_topk_sigmoid_fused

    def paged_gather_bf16_fused(
        page_ptrs: torch.Tensor,
        indices: torch.Tensor,
        page_items: int,
        dim: int,
    ):
        """Copy batch-1 BF16 paged entries without host synchronization."""
        if (
            os.environ.get("TPQ_PAGED_KV_FUSED", "1") == "0"
            or not page_ptrs.is_cuda
            or page_ptrs.dtype != torch.long
            or page_ptrs.ndim != 1
            or page_ptrs.numel() == 0
            or not indices.is_cuda
            or indices.dtype != torch.long
            or page_items <= 0
            or dim <= 0
        ):
            return None
        shape = (*indices.shape, dim)
        return _EXT.paged_gather_bf16(
            page_ptrs,
            indices.contiguous(),
            int(page_items),
            int(dim),
        ).view(shape)

    def hadamard_bf16_fused(x: torch.Tensor):
        """One-block FP32 Walsh-Hadamard transform with BF16 boundaries."""
        width = x.shape[-1] if x.ndim else 0
        if (
            os.environ.get("TPQ_INDEXER_HADAMARD_FUSED", "1") == "0"
            or not x.is_cuda
            or x.dtype != torch.bfloat16
            or width <= 0
            or width > 256
            or width & (width - 1)
        ):
            return None
        return _EXT.hadamard_bf16(x)

    def int4_gemv_fused(
        x: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        cols: int,
        group_size: int,
        output: torch.Tensor | None = None,
    ):
        """Direct packed INT4 decode GEMV; set TPQ_INT4_GEMV_FUSED=0 to fall back."""
        if (
            os.environ.get("TPQ_INT4_GEMV_FUSED", "1") == "0"
            or not x.is_cuda
            or x.dtype not in (torch.float32, torch.bfloat16)
            or x.ndim != 2
            or x.shape[0] != 1
            or packed.dtype != torch.uint8
            or scales.dtype != torch.float16
            or group_size != 64
            or cols <= 0
            or cols % 64
        ):
            return None
        output_rows = int(packed.shape[0])
        group_vector = (
            os.environ.get("TPQ_INT4_GROUP_VECTOR", "0") != "0"
            or (
                os.environ.get(
                    "TPQ_INT4_LM_HEAD_VECTOR",
                    "1",
                ) != "0"
                and output_rows == 154880
                and cols == 6144
            )
        )
        return _EXT.int4_gemv_packed_f32(
            x.contiguous(),
            packed.contiguous(),
            scales.contiguous(),
            int(cols),
            int(group_size),
            group_vector,
            output,
        )

    def block_fp8_gemv_fused(
        x: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        cols: int,
        block_size: int,
        output: torch.Tensor | None = None,
    ):
        """Decode directly from native E4M3 bytes and 128x128 scales."""
        if (
            os.environ.get("TPQ_FP8_GEMV_FUSED", "1") == "0"
            or not x.is_cuda
            or x.dtype not in (torch.float32, torch.bfloat16)
            or x.ndim != 2
            or x.shape != (1, cols)
            or weights.dtype != torch.uint8
            or weights.ndim != 2
            or weights.shape[1] != cols
            or scales.dtype != torch.float32
            or scales.ndim != 2
            or block_size != 128
        ):
            return None
        return _EXT.block_fp8_gemv_f32(
            x.contiguous(),
            weights.contiguous(),
            scales.contiguous(),
            int(cols),
            int(block_size),
            output,
        )

    def block_fp8_grouped_gemv_fused(
        x: torch.Tensor,
        weight_ptrs: torch.Tensor,
        scale_ptrs: torch.Tensor,
        row_offsets: torch.Tensor,
        total_rows: int,
        cols: int,
        block_size: int,
        output: torch.Tensor | None = None,
    ):
        """Run several compact block-FP8 projections in one CUDA launch."""
        if (
            os.environ.get("TPQ_FP8_GROUPED_GEMV", "1") == "0"
            or not x.is_cuda
            or x.dtype not in (torch.float32, torch.bfloat16)
            or x.ndim != 2
            or x.shape != (1, cols)
            or weight_ptrs.dtype != torch.int64
            or scale_ptrs.dtype != torch.int64
            or row_offsets.dtype != torch.int32
            or not weight_ptrs.is_cuda
            or not scale_ptrs.is_cuda
            or not row_offsets.is_cuda
            or weight_ptrs.ndim != 1
            or scale_ptrs.shape != weight_ptrs.shape
            or row_offsets.shape != (weight_ptrs.numel() + 1,)
            or total_rows <= 0
            or block_size != 128
        ):
            return None
        return _EXT.block_fp8_grouped_gemv_f32(
            x.contiguous(),
            weight_ptrs.contiguous(),
            scale_ptrs.contiguous(),
            row_offsets.contiguous(),
            int(total_rows),
            int(cols),
            int(block_size),
            output,
        )

    def int4_glm_qb_split_fused(
        x: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        cols: int,
        group_size: int,
        heads: int,
        nope_width: int,
        rope_width: int,
        nope_output: torch.Tensor | None = None,
        rope_output: torch.Tensor | None = None,
    ):
        """Decode Q-B directly into BF16 no-PE and FP32 RoPE rows."""
        if (
            os.environ.get("TPQ_GLM_QB_SPLIT", "1") == "0"
            or not x.is_cuda
            or x.dtype != torch.float32
            or x.shape != (1, cols)
            or packed.dtype != torch.uint8
            or scales.dtype != torch.float16
            or group_size != 64
        ):
            return None
        return _EXT.int4_glm_qb_split(
            x.contiguous(),
            packed.contiguous(),
            scales.contiguous(),
            int(cols),
            int(group_size),
            (
                os.environ.get(
                    "TPQ_GLM_QB_GROUP_VECTOR",
                    "1",
                )
                != "0"
                and (cols // group_size) % 4 == 0
            ),
            int(heads),
            int(nope_width),
            int(rope_width),
            nope_output,
            rope_output,
        )

    def int4_embedding_fused(
        packed: torch.Tensor,
        scales: torch.Tensor,
        row: int,
        cols: int,
        group_size: int,
        output: torch.Tensor | None = None,
    ):
        """Decode one packed INT4 embedding row directly into FP32."""
        if (
            os.environ.get("TPQ_INT4_EMBEDDING_FUSED", "1") == "0"
            or not packed.is_cuda
            or packed.dtype != torch.uint8
            or scales.dtype != torch.float16
            or packed.ndim != 2
            or scales.ndim != 2
            or group_size != 64
            or cols <= 0
            or cols % 64
            or row < 0
            or row >= packed.shape[0]
        ):
            return None
        return _EXT.int4_embedding_lookup(
            packed.contiguous(),
            scales.contiguous(),
            int(row),
            int(cols),
            int(group_size),
            output,
        )

    def int4_embedding_device_fused(
        packed: torch.Tensor,
        scales: torch.Tensor,
        row: torch.Tensor,
        cols: int,
        group_size: int,
        output: torch.Tensor | None = None,
    ):
        """Decode one embedding row selected by a CUDA int64 scalar."""
        if (
            os.environ.get("TPQ_INT4_EMBEDDING_FUSED", "1") == "0"
            or not packed.is_cuda
            or not row.is_cuda
            or packed.dtype != torch.uint8
            or scales.dtype != torch.float16
            or row.dtype != torch.long
            or row.numel() != 1
            or group_size != 64
        ):
            return None
        return _EXT.int4_embedding_lookup_device_row(
            packed.contiguous(),
            scales.contiguous(),
            row.contiguous(),
            int(cols),
            int(group_size),
            output,
        )

    def glm_norm_qkv_int4_fused(
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        q_packed: torch.Tensor,
        q_scales: torch.Tensor,
        kv_packed: torch.Tensor,
        kv_scales: torch.Tensor,
        cols: int,
        group_size: int,
        eps: float,
        output_buffers: tuple[
            torch.Tensor,
            torch.Tensor,
        ] | None = None,
    ):
        """Fuse GLM decode input RMSNorm with Q-A and KV-A INT4 GEMVs."""
        if (
            os.environ.get("TPQ_GLM_NORM_QKV_FUSED", "1") == "0"
            or not x.is_cuda
            or x.dtype != torch.float32
            or x.ndim != 2
            or x.shape != (1, cols)
            or norm_weight.dtype != torch.float32
            or norm_weight.shape != (cols,)
            or q_packed.dtype != torch.uint8
            or kv_packed.dtype != torch.uint8
            or q_scales.dtype != torch.float16
            or kv_scales.dtype != torch.float16
            or q_packed.ndim != 2
            or kv_packed.ndim != 2
            or q_packed.shape[1] * 2 != cols
            or kv_packed.shape[1] * 2 != cols
            or group_size != 64
            or cols <= 0
            or cols % 64
            or q_scales.shape != (
                q_packed.shape[0],
                cols // group_size,
            )
            or kv_scales.shape != (
                kv_packed.shape[0],
                cols // group_size,
            )
        ):
            return None
        return _EXT.glm_norm_qkv_int4(
            x.contiguous(),
            norm_weight.contiguous(),
            q_packed.contiguous(),
            q_scales.contiguous(),
            kv_packed.contiguous(),
            kv_scales.contiguous(),
            int(cols),
            int(group_size),
            float(eps),
            None,
            (
                output_buffers[0]
                if output_buffers is not None
                else None
            ),
            (
                output_buffers[1]
                if output_buffers is not None
                else None
            ),
        )

    def glm_residual_norm_qkv_int4_fused(
        residual: torch.Tensor,
        update: torch.Tensor,
        norm_weight: torch.Tensor,
        q_packed: torch.Tensor,
        q_scales: torch.Tensor,
        kv_packed: torch.Tensor,
        kv_scales: torch.Tensor,
        cols: int,
        group_size: int,
        eps: float,
    ):
        """Fuse a residual add into input RMSNorm plus Q-A/KV-A."""
        if (
            os.environ.get(
                "TPQ_GLM_RESIDUAL_NORM_QKV",
                "1",
            ) == "0"
            or not residual.is_cuda
            or residual.dtype != torch.float32
            or residual.ndim != 2
            or residual.shape != (1, cols)
            or update.dtype != torch.float32
            or update.shape != residual.shape
            or norm_weight.dtype != torch.float32
            or norm_weight.shape != (cols,)
            or q_packed.dtype != torch.uint8
            or kv_packed.dtype != torch.uint8
            or q_scales.dtype != torch.float16
            or kv_scales.dtype != torch.float16
            or q_packed.ndim != 2
            or kv_packed.ndim != 2
            or q_packed.shape[1] * 2 != cols
            or kv_packed.shape[1] * 2 != cols
            or group_size != 64
            or cols <= 0
            or cols % 64
            or q_scales.shape != (
                q_packed.shape[0],
                cols // group_size,
            )
            or kv_scales.shape != (
                kv_packed.shape[0],
                cols // group_size,
            )
        ):
            return None
        return _EXT.glm_norm_qkv_int4(
            residual.contiguous(),
            norm_weight.contiguous(),
            q_packed.contiguous(),
            q_scales.contiguous(),
            kv_packed.contiguous(),
            kv_scales.contiguous(),
            int(cols),
            int(group_size),
            float(eps),
            update.contiguous(),
            None,
            None,
        )

    def glm_residual_norm_router_fused(
        residual: torch.Tensor,
        update: torch.Tensor,
        norm_weight: torch.Tensor,
        router_weight: torch.Tensor,
        eps: float,
        norm_output: torch.Tensor | None = None,
        output_buffers: tuple[
            torch.Tensor,
            torch.Tensor,
        ] | None = None,
    ):
        """Fuse GLM decode residual add, RMSNorm and router GEMV."""
        if (
            os.environ.get(
                "TPQ_GLM_RESIDUAL_NORM_ROUTER",
                "1",
            ) == "0"
            or not residual.is_cuda
            or residual.dtype != torch.float32
            or residual.ndim != 2
            or residual.shape[0] != 1
            or update.dtype != torch.float32
            or update.shape != residual.shape
            or norm_weight.dtype != torch.float32
            or norm_weight.shape != (residual.shape[1],)
            or router_weight.dtype != torch.float32
            or router_weight.ndim != 2
            or router_weight.shape[1] != residual.shape[1]
        ):
            return None
        if norm_output is None:
            return _EXT.glm_residual_norm_router(
                residual.contiguous(),
                update.contiguous(),
                norm_weight.contiguous(),
                router_weight.contiguous(),
                float(eps),
            )
        if (
            norm_output.dtype != torch.float32
            or norm_output.shape != residual.shape
            or norm_output.device != residual.device
            or not norm_output.is_contiguous()
        ):
            return None
        return _EXT.glm_residual_norm_router_norm_out(
            residual.contiguous(),
            update.contiguous(),
            norm_weight.contiguous(),
            router_weight.contiguous(),
            float(eps),
            norm_output,
            (
                output_buffers[0]
                if output_buffers is not None
                else None
            ),
            (
                output_buffers[1]
                if output_buffers is not None
                else None
            ),
        )

    def residual_add3_fused(
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
    ):
        """Fuse ``residual + (routed + shared)`` with source dtype rounding."""
        if (
            not residual.is_cuda
            or residual.dtype not in {torch.float32, torch.bfloat16}
            or not residual.is_contiguous()
            or routed.dtype != residual.dtype
            or shared.dtype != residual.dtype
            or routed.shape != residual.shape
            or shared.shape != residual.shape
        ):
            return None
        return _EXT.residual_add3(
            residual,
            routed.contiguous(),
            shared.contiguous(),
        )

    def glm_moe_residual_add_fused(
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
    ):
        """Compatibility entry for the generic three-way residual operator."""
        if (
            os.environ.get("TPQ_GLM_MOE_RESIDUAL_ADD", "1") == "0"
            or residual.dtype != torch.float32
        ):
            return None
        return residual_add3_fused(residual, routed, shared)

    def glm_ep_reduce_residual_fused(
        contributions: list[torch.Tensor],
        residual: torch.Tensor,
    ):
        """Fuse up to 16 TP routed/shared contributions with the residual."""
        if (
            os.environ.get("TPQ_GLM_EP_FINAL_FUSED", "1") == "0"
            or not 1 <= len(contributions) <= 16
            or not contributions[0].is_cuda
            or any(
                item.dtype != torch.float32
                for item in contributions
            )
            or residual.dtype != torch.float32
            or any(
                item.numel() != contributions[0].numel()
                for item in contributions[1:]
            )
            or contributions[0].numel() != residual.numel()
        ):
            return None
        return _EXT.glm_ep_reduce_residual(
            [item.contiguous() for item in contributions],
            residual.contiguous(),
        )

    def tp_all_rank_reduce_fused(
        contributions: list[torch.Tensor],
        outputs: list[torch.Tensor],
    ):
        """Reduce canonical FP32 partials into fixed buffers on all ranks."""
        if (
            not 1 <= len(contributions) <= 16
            or not outputs
            or any(
                not item.is_cuda
                or item.dtype != torch.float32
                or not item.is_contiguous()
                for item in contributions
            )
            or any(
                not item.is_cuda
                or item.dtype not in {torch.float32, torch.bfloat16}
                or not item.is_contiguous()
                for item in outputs
            )
            or any(
                item.numel() != contributions[0].numel()
                for item in (*contributions[1:], *outputs)
            )
        ):
            return None
        return _EXT.tp_all_rank_reduce(contributions, outputs)

    def tp_hidden_add_batch_fused(
        left: list[torch.Tensor],
        left_events: list[torch.cuda.Event],
        right: list[torch.Tensor],
        right_events: list[torch.cuda.Event],
        outputs: list[torch.Tensor],
        output_events: list[torch.cuda.Event],
    ):
        if not (
            left
            and len(left)
            == len(left_events)
            == len(right)
            == len(right_events)
            == len(outputs)
            == len(output_events)
        ):
            return None
        return _EXT.tp_hidden_add_batch(
            left,
            [event.cuda_event for event in left_events],
            right,
            [event.cuda_event for event in right_events],
            outputs,
            [event.cuda_event for event in output_events],
        )

    def tp_hidden_rmsnorm_batch_fused(
        inputs: list[torch.Tensor],
        input_events: list[torch.cuda.Event],
        weights: list[torch.Tensor],
        eps: float,
        outputs: list[torch.Tensor],
        output_events: list[torch.cuda.Event],
    ):
        if not (
            inputs
            and len(inputs)
            == len(input_events)
            == len(weights)
            == len(outputs)
            == len(output_events)
        ):
            return None
        return _EXT.tp_hidden_rmsnorm_batch(
            inputs,
            [event.cuda_event for event in input_events],
            weights,
            float(eps),
            outputs,
            [event.cuda_event for event in output_events],
        )

    def tp_hidden_residual_mix_batch_fused(
        prefixes: list[torch.Tensor],
        prefix_events: list[torch.cuda.Event],
        residuals: list[torch.Tensor],
        residual_events: list[torch.cuda.Event],
        projections: list[torch.Tensor],
        norm_weights: list[torch.Tensor],
        post_norm_weights: list[torch.Tensor],
        workspaces: list[torch.Tensor],
        residual_inverses: list[torch.Tensor],
        eps: float,
        outputs: list[torch.Tensor],
        output_events: list[torch.cuda.Event],
    ):
        count = len(outputs)
        if not (
            count
            and len(prefixes)
            == len(prefix_events)
            == len(residuals)
            == len(residual_events)
            == len(projections)
            == len(norm_weights)
            == len(post_norm_weights)
            == len(workspaces)
            == len(residual_inverses)
            == len(output_events)
            == count
        ):
            return None
        return _EXT.tp_hidden_residual_mix_batch(
            prefixes,
            [event.cuda_event for event in prefix_events],
            residuals,
            [event.cuda_event for event in residual_events],
            projections,
            norm_weights,
            post_norm_weights,
            workspaces,
            residual_inverses,
            float(eps),
            int(os.environ.get("TPQ_RESIDUAL_SINGLE_MAX_ROWS", "2")),
            outputs,
            [event.cuda_event for event in output_events],
        )

    def launch_cuda_graphs_fused(
        devices: list[int],
        graphs: list[torch.cuda.CUDAGraph],
        streams: list[torch.cuda.Stream],
        done_events: list[torch.cuda.Event],
        source_event: torch.cuda.Event,
    ) -> None:
        """Launch all TP-rank graphs without per-rank Python context switches."""
        _EXT.launch_cuda_graphs(
            devices,
            [graph.raw_cuda_graph_exec() for graph in graphs],
            [stream.cuda_stream for stream in streams],
            [event.cuda_event for event in done_events],
            source_event.cuda_event,
        )

    def launch_cuda_graphs_reduce_fused(
        devices: list[int],
        graphs: list[torch.cuda.CUDAGraph],
        streams: list[torch.cuda.Stream],
        done_events: list[torch.cuda.Event],
        source_event: torch.cuda.Event,
        contributions: list[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Launch every rank and complete the Row-TP reduction in one call."""
        return _EXT.launch_cuda_graphs_reduce(
            devices,
            [graph.raw_cuda_graph_exec() for graph in graphs],
            [stream.cuda_stream for stream in streams],
            [event.cuda_event for event in done_events],
            source_event.cuda_event,
            contributions,
            residual,
        )

    def make_tp_graph_launch_batch(
        devices: list[int],
        graphs: list[torch.cuda.CUDAGraph],
        streams: list[torch.cuda.Stream],
        done_events: list[torch.cuda.Event],
        source_event: torch.cuda.Event,
    ):
        """Cache immutable CUDA handles once instead of per decode layer."""
        return _EXT.TPGraphLaunchBatch(
            devices,
            [graph.raw_cuda_graph_exec() for graph in graphs],
            [stream.cuda_stream for stream in streams],
            [event.cuda_event for event in done_events],
            source_event.cuda_event,
        )

    def make_tp_graph_sequence_batch(
        devices: list[int],
        graph_sequences: list[list[torch.cuda.CUDAGraph]],
        streams: list[torch.cuda.Stream],
        done_events: list[torch.cuda.Event],
        source_event: torch.cuda.Event,
    ):
        """Join fixed-address child graphs into one parent graph per rank."""
        if (
            not devices
            or len(devices) != len(graph_sequences)
            or len(devices) != len(streams)
            or len(devices) != len(done_events)
        ):
            raise ValueError(
                "TP graph sequences, devices and streams must be size-equal"
            )
        return _EXT.TPGraphLaunchBatch(
            devices,
            [
                [graph.raw_cuda_graph() for graph in sequence]
                for sequence in graph_sequences
            ],
            [stream.cuda_stream for stream in streams],
            [event.cuda_event for event in done_events],
            source_event.cuda_event,
        )

    def make_tp_graph_dag_batch(
        devices: list[int],
        graph_stages: list[list[list[torch.cuda.CUDAGraph]]],
        streams: list[torch.cuda.Stream],
        done_events: list[torch.cuda.Event],
        source_event: torch.cuda.Event,
    ):
        """Compose sequential stages while allowing children in one stage to overlap."""
        if (
            not devices
            or len(devices) != len(graph_stages)
            or len(devices) != len(streams)
            or len(devices) != len(done_events)
        ):
            raise ValueError(
                "TP graph DAGs, devices and streams must be size-equal"
            )
        return _EXT.TPGraphLaunchBatch(
            devices,
            [
                [
                    [graph.raw_cuda_graph() for graph in stage]
                    for stage in rank_stages
                ]
                for rank_stages in graph_stages
            ],
            [stream.cuda_stream for stream in streams],
            [event.cuda_event for event in done_events],
            source_event.cuda_event,
        )

    def make_tp_no_owner_moe_layer_plan(
        shared_batch,
        route_batch,
        expert_batch,
        final_batch,
        input_events,
        route_contribution_groups,
        route_output_groups,
        route_output_events,
        expert_contributions,
        packed_outputs,
        packed_output_events,
        routed_contributions,
        shared_contributions,
        shared_events,
        residuals,
        residual_events,
        routed_workspaces,
        shared_workspaces,
        outputs,
        output_events,
    ):
        """Cache a complete fixed-address all-rank MoE submission plan.

        The plan only combines host scheduling.  Router/latent, packed expert
        and hidden collectives remain explicit all-rank event boundaries.
        """
        return _EXT.TPNoOwnerMoELayerPlan(
            shared_batch,
            route_batch,
            expert_batch,
            final_batch,
            [event.cuda_event for event in input_events],
            route_contribution_groups,
            route_output_groups,
            [event.cuda_event for event in route_output_events],
            expert_contributions,
            packed_outputs,
            [event.cuda_event for event in packed_output_events],
            routed_contributions,
            shared_contributions,
            [event.cuda_event for event in shared_events],
            residuals,
            [event.cuda_event for event in residual_events],
            routed_workspaces,
            shared_workspaces,
            outputs,
            [event.cuda_event for event in output_events],
        )

    def make_tp_no_owner_decode_layer_plan(
        attention_batch,
        moe_plan,
        attention_contributions,
        attention_outputs,
        attention_output_events,
    ):
        """Cache Attention→MoE as one fixed all-rank host submission."""
        return _EXT.TPNoOwnerDecodeLayerPlan(
            attention_batch,
            moe_plan,
            attention_contributions,
            attention_outputs,
            [
                event.cuda_event
                for event in attention_output_events
            ],
        )

    def bf16_gemv_fused(
        value: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run registered-shape BF16 GEMV into a fixed BF16/FP32 buffer."""
        if (
            not value.is_cuda
            or not weight.is_cuda
            or not output.is_cuda
            or value.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16
            or output.dtype not in (torch.bfloat16, torch.float32)
            or value.ndim != 2
            or value.shape[0] != 1
            or weight.ndim != 2
            or weight.shape[1] != value.shape[1]
            or output.shape != (1, weight.shape[0])
            or value.device != weight.device
            or value.device != output.device
        ):
            return None
        return _EXT.bf16_gemv_out(value, weight, output)

    def int4_swiglu_fused(
        x: torch.Tensor,
        gate_packed: torch.Tensor,
        gate_scales: torch.Tensor,
        up_packed: torch.Tensor,
        up_scales: torch.Tensor,
        cols: int,
        group_size: int,
        output: torch.Tensor | None = None,
    ):
        """Fuse two packed INT4 decode GEMVs with their FP32 SwiGLU."""
        if (
            os.environ.get("TPQ_INT4_SWIGLU_FUSED", "1") == "0"
            or not x.is_cuda
            or x.dtype not in (torch.float32, torch.bfloat16)
            or x.ndim != 2
            or x.shape[0] != 1
            or gate_packed.dtype != torch.uint8
            or up_packed.dtype != torch.uint8
            or gate_scales.dtype != torch.float16
            or up_scales.dtype != torch.float16
            or gate_packed.shape != up_packed.shape
            or gate_scales.shape != up_scales.shape
            or group_size != 64
            or cols <= 0
            or cols % 64
        ):
            return None
        return _EXT.int4_swiglu_packed_f32(
            x.contiguous(),
            gate_packed.contiguous(),
            gate_scales.contiguous(),
            up_packed.contiguous(),
            up_scales.contiguous(),
            int(cols),
            int(group_size),
            os.environ.get(
                "TPQ_INT4_SWIGLU_GROUP_VECTOR",
                "0",
            ) != "0",
            output,
        )

else:

    def vq_gemv_fused(x_rows: torch.Tensor, idx: torch.Tensor,
                      cb: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(f"{_EXTENSION_NAME} 扩展不可用：{_ERR}")

    def kda_recurrent_fused(*args, **kwargs):
        return None

    def short_conv3_fused(*args, **kwargs):
        return False

    def gated_rmsnorm_fused(*args, **kwargs):
        return None

    def packed_moe_topk_fused(*args, **kwargs):
        return None

    def packed_route_slots_fused(*args, **kwargs):
        return False

    def moe_mlp_slots_fused(
        x_rows,
        gu_indices,
        gu_codebooks,
        dn_indices,
        dn_codebooks,
        weights,
        limit,
        hidden_workspace,
        out_workspace,
        result,
    ):
        return None

    def moe_mlp_routed_slots_fused(
        x_rows,
        route_ids,
        weights,
        metadata,
        limit,
        hidden_workspace,
        out_workspace,
        result,
        accumulate=False,
    ):
        return None

    def moe_mlp_routed_vv_fused(*args, **kwargs):
        return None

    def moe_mlp_routed_codegemm_fused(*args, **kwargs):
        return None

    def pack_vq_tensor_shard_codegemm(*args, **kwargs):
        return False

    def unpack_vq_codegemm(*args, **kwargs):
        return None

    def expert_dispatch_pack_fused(*args, **kwargs):
        return False

    def tp_peer_copy_fused(*args, **kwargs):
        return False

    def tp_attention_peer_dispatch_fused(*args, **kwargs):
        return False

    def tp_attention_source_pack_fused(*args, **kwargs):
        return False

    def hc_split_fused(mixes, scale, base, hc, iters, eps):
        return None

    def rmsnorm_fused(x, w, eps, output=None):
        return None

    def rmsnorm_bf16_fused(x, w, eps, output=None):
        return None

    def attention_residual_bf16_fused(*args, **kwargs):
        return None

    def gated_activation_bf16_fused(*args, **kwargs):
        return None

    def rope1_fused(x, cos, sin, inverse=False):
        return None

    def glm_rope_qk_fused(q, k, cos, sin):
        return None

    def glm_latent_kv_decode_prepare_fused(*args, **kwargs):
        return None

    def latent_mla_attention_decode_fused(*args, **kwargs):
        return None

    def flashinfer_mla_batch1_plan_fused(*args, **kwargs):
        return False

    def glm_mla_bmm_decode_fused(*args, **kwargs):
        return None

    def glm_merge_scores_fused(a, b, scale):
        return None

    def dsv4_attn_decode_fused(q, win_kv, win_pos, comp_kv, sink, cos, sin, scale):
        return None

    def dsv4_hc_pre_fused(x, fn, scale, base, iters, eps):
        return None

    def dsv4_hc_pre_norm_fused(
        x,
        fn,
        scale,
        base,
        norm,
        iters,
        eps,
        output_buffers=None,
    ):
        return None

    def dsv4_hc_post_fused(
        out,
        residual,
        post,
        comb,
        output=None,
    ):
        return None

    def dsv4_hc_post_moe_fused(
        routed,
        shared,
        residual,
        post,
        comb,
        output=None,
    ):
        return None

    def dsv4_route_post_fused(
        scores, bias, mask, top_k
    ):
        return None

    def route_topk_sigmoid_fused(
        logits,
        bias,
        mask,
        top_k,
        routed_scaling,
        output_buffers=None,
    ):
        return None

    def linear_route_topk_sigmoid_fused(*args, **kwargs):
        return None

    glm_route_fused = route_topk_sigmoid_fused

    def paged_gather_bf16_fused(
        page_ptrs, indices, page_items, dim
    ):
        return None

    def hadamard_bf16_fused(x):
        return None

    def int4_gemv_fused(
        x, packed, scales, cols, group_size, output=None
    ):
        return None

    def block_fp8_gemv_fused(*args, **kwargs):
        return None

    def block_fp8_grouped_gemv_fused(*args, **kwargs):
        return None

    def int4_glm_qb_split_fused(*args, **kwargs):
        return None

    def int4_embedding_fused(
        packed,
        scales,
        row,
        cols,
        group_size,
        output=None,
    ):
        return None

    def int4_embedding_device_fused(*args, **kwargs):
        return None

    def glm_norm_qkv_int4_fused(*args, **kwargs):
        return None

    def glm_residual_norm_qkv_int4_fused(*args, **kwargs):
        return None

    def glm_residual_norm_router_fused(*args, **kwargs):
        return None

    def residual_add3_fused(*args, **kwargs):
        return None

    def glm_moe_residual_add_fused(*args, **kwargs):
        return None

    def glm_ep_reduce_residual_fused(*args, **kwargs):
        return None

    def tp_all_rank_reduce_fused(*args, **kwargs):
        return None

    def tp_hidden_add_batch_fused(*args, **kwargs):
        return None

    def tp_hidden_rmsnorm_batch_fused(*args, **kwargs):
        return None

    def tp_hidden_residual_mix_batch_fused(*args, **kwargs):
        return None

    def launch_cuda_graphs_fused(*args, **kwargs):
        return None

    def launch_cuda_graphs_reduce_fused(*args, **kwargs):
        return None

    def make_tp_graph_launch_batch(*args, **kwargs):
        return None

    def make_tp_graph_sequence_batch(*args, **kwargs):
        return None

    def make_tp_graph_dag_batch(*args, **kwargs):
        return None

    def make_tp_no_owner_moe_layer_plan(*args, **kwargs):
        return None

    def make_tp_no_owner_decode_layer_plan(*args, **kwargs):
        return None

    def bf16_gemv_fused(*args, **kwargs):
        return None

    def int4_swiglu_fused(
        x,
        gate_packed,
        gate_scales,
        up_packed,
        up_scales,
        cols,
        group_size,
        output=None,
    ):
        return None


# 旧公开名只作为外部脚本的兼容别名；注册层只引用通用名称。
kimi_short_conv3_fused = short_conv3_fused
kimi_kda_recurrent_fused = kda_recurrent_fused
kimi_gated_rmsnorm_fused = gated_rmsnorm_fused
kimi_moe_packed_fused = packed_moe_topk_fused
