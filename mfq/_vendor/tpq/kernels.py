"""TPQ 数值内核：int4 / VQ 矩阵乘、RMSNorm、交错 RoPE。

两类量化权重的免还原/分块还原矩阵乘：
  - Int4Weight：u8 双半字节 [R, C//2] + f16 组缩放 [R, C//64]；matmul 按行块
    反量化到 f32 后 torch.mm，内存峰值 = 一个行块。
  - VQWeight：u8 码字索引 [R, C//dim] + 层共享码本 f32 [K, dim]；LUT 算法：
    y[r] = Σ_b s[b, idx[r, b]]，其中 s[b, c] = x[bd:bd+dim]·cb[c] 只需算 B×K 次，
    把 O(R·C) 的 matmul 降为 O(B·K + R·B) 的查表加（v 档约快 6 倍）。
RMSNorm / RoPE 与 CCCP/modelmath.py 逐行一致（单测对照过朴素实现）。
"""

from __future__ import annotations

import math

import torch

from .precision import compute_dtype

INT4_GROUP = 64


def rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSNorm：f32 算方差，乘权重后回原 dtype。
    CUDA + f32 输入走融合 kernel（1 次 launch 替代 ~6 次），其余回退 torch 表达式。"""
    dt = x.dtype
    if dt == torch.float32 and x.is_cuda:
        fn = _rms_fused()
        if fn is not None:
            r = fn(
                x,
                w if w.dtype == torch.float32 else w.float(),
                eps,
                output=output,
            )
            if r is not None:
                return r
    v = x.float().pow(2).mean(-1, keepdim=True)
    result = (
        w.float() * (x.float() * torch.rsqrt(v + eps))
    ).to(dt)
    if output is not None:
        output.copy_(result)
        return output
    return result


def _rms_fused():
    """fusedext.rmsnorm_fused 的懒导入（避免 kernels 被 CPU-only 场景导入时触发扩展编译）。"""
    global _RMS_FUSED
    if _RMS_FUSED is None:
        try:
            from .fusedext import rmsnorm_fused
            _RMS_FUSED = rmsnorm_fused
        except Exception:
            _RMS_FUSED = False
    return _RMS_FUSED or None


_RMS_FUSED = None


def _int4_gemv_fused():
    """Lazily resolve the direct packed INT4 decode kernel."""
    global _INT4_GEMV_FUSED
    if _INT4_GEMV_FUSED is None:
        try:
            from .fusedext import int4_gemv_fused
            _INT4_GEMV_FUSED = int4_gemv_fused
        except Exception:
            _INT4_GEMV_FUSED = False
    return _INT4_GEMV_FUSED or None


_INT4_GEMV_FUSED = None


def _glm_rope_qk_fused():
    """Lazily resolve the GLM Q/K RoPE fusion."""
    global _GLM_ROPE_QK_FUSED
    if _GLM_ROPE_QK_FUSED is None:
        try:
            from .fusedext import glm_rope_qk_fused
            _GLM_ROPE_QK_FUSED = glm_rope_qk_fused
        except Exception:
            _GLM_ROPE_QK_FUSED = False
    return _GLM_ROPE_QK_FUSED or None


_GLM_ROPE_QK_FUSED = None


def merge_attention_scores(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Merge latent-MLA score components without changing their GEMMs."""
    if a.is_cuda and a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16:
        global _GLM_MERGE_SCORES_FUSED
        if _GLM_MERGE_SCORES_FUSED is None:
            try:
                from .fusedext import glm_merge_scores_fused
                _GLM_MERGE_SCORES_FUSED = glm_merge_scores_fused
            except Exception:
                _GLM_MERGE_SCORES_FUSED = False
        if _GLM_MERGE_SCORES_FUSED:
            result = _GLM_MERGE_SCORES_FUSED(a, b, scale)
            if result is not None:
                return result
    return a.float() / scale + b.float() / scale


_GLM_MERGE_SCORES_FUSED = None


class RopeCache:
    """RoPE cos/sin 预计算（交错布局，[T, rope_dim//2]）。"""

    def __init__(self, rope_dim: int, theta: float, max_len: int = 8192):
        inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
        freqs = torch.outer(torch.arange(max_len, dtype=torch.float32), inv)
        self.cos = freqs.cos()
        self.sin = freqs.sin()

    def apply(self, q: torch.Tensor, k: torch.Tensor, pos0: int):
        """q: [H, T, D]；k: [1, T, D] → HF apply_rotary_pos_emb_interleave 的 cat 布局。"""
        T = q.shape[1]
        cos = self.cos[pos0:pos0 + T]
        sin = self.sin[pos0:pos0 + T]
        if q.is_cuda and q.dtype == torch.float32 and k.dtype == torch.float32:
            fn = _glm_rope_qk_fused()
            if fn is not None:
                result = fn(q, k, cos, sin)
                if result is not None:
                    return result
        q1, q2 = q[..., 0::2], q[..., 1::2]
        k1, k2 = k[..., 0::2], k[..., 1::2]
        qe = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        ke = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return qe, ke


def dequant_int4(packed: torch.Tensor, scales: torch.Tensor,
                 gs: int = INT4_GROUP, half: bool = False) -> torch.Tensor:
    """int4 行块反量化：packed u8 [r, C//2]，scales f16 [r, C//gs] → f32/f16 [r, C]。

    用 256 字节查找表一次 gather 出 (lo, hi) 两个半字节（连续写，最快路径），
    再按组就地乘缩放。比逐半字节位运算 + 跨步写快约 2 倍（本机实测）。
    half=True：LUT 与输出走 fp16（2080 张量核 matmul 提速 + 写出量减半；
    int4 网格本身 ~6% 误差，fp16 的 0.05% 精度远超所需，无额外损失）。
    """
    r = packed.shape[0]
    cols = packed.shape[1] * 2
    key = f"{packed.device}:h" if half else str(packed.device)
    lut = _LUTS.get(key)
    if lut is None:
        base = _INT4_LUT.to(torch.float16) if half else _INT4_LUT
        lut = base.to(packed.device)
        _LUTS[key] = lut
    w = lut[packed.long()].view(r, cols)
    if half:
        w.view(r, cols // gs, gs).mul_(scales.unsqueeze(-1))
    else:
        w.view(r, cols // gs, gs).mul_(scales.float().unsqueeze(-1))
    return w


def _make_lut() -> torch.Tensor:
    t = torch.arange(256, dtype=torch.int16)
    return (torch.stack((t & 15, t >> 4), 1).to(torch.float32) - 8)


_INT4_LUT = _make_lut()  # [256, 2]：字节 → (低半字节值, 高半字节值)，零点是 8
_LUTS: dict = {"cpu": _INT4_LUT}

# 码本半精度计算副本缓存：键 = (f32 码本 data_ptr, dtype)，值 = (低精度副本, f32 强引用)。
# 强引用防 data_ptr 复用后串码本（同 ExpertPool._cb_dev 的竞态教训）；
# 同层同档专家共享同一码本张量 → 全池天然去重，每 (层,档) 只多一份副本。
_CB_LO: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def cb_compute(cb: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """码本的计算 dtype 副本（fp32 原样返回；半精度按 ptr 去重缓存）。"""
    if dt == torch.float32 or cb.dtype == dt:
        return cb
    key = (cb.data_ptr(), str(dt))
    ent = _CB_LO.get(key)
    if ent is None:
        lo = cb.to(dt)
        ent = (lo, cb)
        _CB_LO[key] = ent
    return ent[0]


def _lut_on(device) -> torch.Tensor:
    """各设备缓存一份 LUT（GPU 推理路径避免跨设备索引）。"""
    key = str(device)
    lut = _LUTS.get(key)
    if lut is None:
        lut = _INT4_LUT.to(device)
        _LUTS[key] = lut
    return lut


class Int4Weight:
    """int4-g64 打包权重；matmul 按行块在线反量化（dense 低内存驻留方案）。
    half=True：反量化与 matmul 走 fp16（Turing 张量核，~2× fp32；权重仍 int4 驻留）。"""

    __slots__ = ("q", "s", "cols", "gs", "half")

    def __init__(self, q: torch.Tensor, s: torch.Tensor, cols: int, gs: int = INT4_GROUP,
                 half: bool = False):
        self.q = q          # u8 [R, C//2]
        self.s = s          # f16 [R, C//gs]
        self.cols = cols
        self.gs = gs
        self.half = half

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.q.shape[0], self.cols])

    @property
    def nbytes(self) -> int:
        return self.q.numel() + self.s.numel() * 2

    def dequant_rows(self, r0: int, r1: int) -> torch.Tensor:
        return dequant_int4(self.q[r0:r1], self.s[r0:r1], self.gs, half=self.half)

    def matmul_T(self, x: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """y = x @ W.T。x: [T, C] → [T, R] f32（half 时内部 fp16 计算、输出 f32）。

        行块大小自适应： transient 反量化块 ≤64MB（GPU 上 wq_b 级别大矩阵一次
        成型——原固定 512 行会把单个 GEMM 拆成 64 块 × 5 次 launch，WDDM 下
        launch 开销远超计算本身；显存代价仅一块临时缓冲）。
        """
        if (
            not x.is_cuda
            and x.dim() == 2
            and x.shape[0] == 1
            and self.q.dtype == torch.uint8
            and self.s.dtype == torch.float16
        ):
            from .cpuext import int4_gemv_cpu

            fused_cpu = int4_gemv_cpu(
                x, self.q, self.s, self.cols, self.gs
            )
            if fused_cpu is not None:
                return fused_cpu
        R = self.q.shape[0]
        if chunk is None:
            esz = 2 if self.half else 4
            chunk = max(512, min(R, (64 * 2**20) // max(self.cols * esz, 1)))
        if self.half:
            xh = x.half()
            out = torch.empty(x.shape[0], R, dtype=torch.float16, device=x.device)
            for r0 in range(0, R, chunk):
                r1 = min(r0 + chunk, R)
                out[:, r0:r1] = xh @ self.dequant_rows(r0, r1).t()
            return out.float()
        out = torch.empty(x.shape[0], R, dtype=torch.float32, device=x.device)
        for r0 in range(0, R, chunk):
            r1 = min(r0 + chunk, R)
            out[:, r0:r1] = x.float() @ self.dequant_rows(r0, r1).t()
        return out

    def matmul_T_decode_fused(
        self,
        x: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use direct packed INT4 GEMV for a compatible decode row."""
        if (
            x.is_cuda
            and x.dim() == 2
            and x.shape[0] == 1
            and x.dtype in (torch.float32, torch.bfloat16)
            and self.q.dtype == torch.uint8
            and self.s.dtype == torch.float16
            and self.gs == 64
        ):
            fn = _int4_gemv_fused()
            if fn is not None:
                fused = fn(
                    x,
                    self.q,
                    self.s,
                    self.cols,
                    self.gs,
                    output=output,
                )
                if fused is not None:
                    return fused
        return self.matmul_T(x)

    def row(self, r: int) -> torch.Tensor:
        """反量化单行 [C]（embed 查表用）。"""
        return self.dequant_rows(r, r + 1).squeeze(0)


class BlockFP8Weight:
    """原生 E4M3 权重与 128×128 FP32 反量化尺度。

    CCCP ``dense=fp8-native`` 直接保存 FP8 检查点字节，不先展开成
    BF16/F32。矩阵乘按行块临时反量化，常驻显存仍是 1 byte/weight。
    """

    __slots__ = ("q", "s", "cols", "block")

    def __init__(
        self,
        q: torch.Tensor,
        s: torch.Tensor,
        cols: int,
        block: int = 128,
    ):
        if q.dtype != torch.uint8 or s.dtype != torch.float32:
            raise TypeError("BlockFP8Weight requires uint8 data and f32 scales")
        self.q = q
        self.s = s
        self.cols = cols
        self.block = block

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.q.shape[0], self.cols])

    @property
    def nbytes(self) -> int:
        return self.q.numel() + self.s.numel() * 4

    def dequant_rows(
        self,
        r0: int,
        r1: int,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if r0 < 0 or r1 < r0 or r1 > self.q.shape[0]:
            raise IndexError((r0, r1))
        if dtype is None:
            dtype = (
                compute_dtype(self.q.device)
                if self.q.is_cuda
                else torch.float32
            )
        first_block = r0 // self.block
        last_block = (r1 + self.block - 1) // self.block
        scale_rows = self.s[first_block:last_block].repeat_interleave(
            self.block,
            dim=0,
        )
        offset = r0 - first_block * self.block
        scale_rows = scale_rows[offset : offset + (r1 - r0)]
        scales = scale_rows.repeat_interleave(
            self.block,
            dim=1,
        )[:, : self.cols]
        values = self.q[r0:r1].view(torch.float8_e4m3fn).to(dtype)
        return values * scales.to(dtype)

    def matmul_T(
        self,
        x: torch.Tensor,
        chunk: int | None = None,
    ) -> torch.Tensor:
        rows = self.q.shape[0]
        dtype = (
            compute_dtype(x.device)
            if x.is_cuda
            else torch.float32
        )
        if chunk is None:
            element_size = 2 if dtype != torch.float32 else 4
            chunk = max(
                self.block,
                min(
                    rows,
                    (64 * 2**20) // max(self.cols * element_size, 1),
                ),
            )
            chunk = max(
                self.block,
                (chunk // self.block) * self.block,
            )
        x_compute = x.to(dtype)
        out = torch.empty(
            x.shape[0],
            rows,
            dtype=torch.float32,
            device=x.device,
        )
        for r0 in range(0, rows, chunk):
            r1 = min(r0 + chunk, rows)
            out[:, r0:r1] = (
                x_compute @ self.dequant_rows(r0, r1, dtype).t()
            ).float()
        return out

    def matmul_T_decode_fused(
        self,
        x: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = self.matmul_T(x)
        if output is not None:
            output.copy_(result)
            return output
        return result

    def row(self, r: int) -> torch.Tensor:
        return self.dequant_rows(r, r + 1).squeeze(0)


class VQWeight:
    """VQ 索引态权重：u8 索引 [R, B] + 码本 [K, dim]，LUT 矩阵乘。"""

    __slots__ = ("idx", "cb", "cols", "dim")

    def __init__(self, idx: torch.Tensor, cb: torch.Tensor, cols: int):
        self.idx = idx              # u8 [R, B]，B = cols // dim
        self.cb = cb.float()        # f32 [K, dim]
        self.cols = cols
        self.dim = cb.shape[1]

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.idx.shape[0], self.cols])

    @property
    def nbytes(self) -> int:
        return self.idx.numel() * self.idx.element_size()

    def to(self, device, non_blocking: bool = False) -> "VQWeight":
        """搬移到指定设备（GPU 推理路径，可异步上传）。"""
        return VQWeight(self.idx.to(device, non_blocking=non_blocking),
                        self.cb.to(device, non_blocking=non_blocking), self.cols)

    def dequant(self) -> torch.Tensor:
        """还原为 f32 [R, C]（小矩阵或对照测试用）。"""
        return self.cb[self.idx.reshape(-1).long()].reshape(self.idx.shape[0], self.cols)

    def matmul_T(self, x: torch.Tensor) -> torch.Tensor:
        """LUT 版 y = x @ W.T。x: [T, C] → [T, R] f32。

        s[t, b, c] = x 第 b 块与码字 c 的点积（[T, B, K]），随后按索引查表求和。
        逐 t 循环 gather（峰值 [R,B]）；大码本（k4096）按 token 分块计算 s，
        峰值 [Tc,B,K] f32 封顶 ~256MB（全量 [T,B,K] 在长 prefill 会爆显存）。
        GPU 上内积走精度策略层的半精度（fp16/bf16 张量核，fp32 累加），
        查表求和用 sum(dtype=f32) 保 f32 累加精度——量化噪声比半精度舍入大
        两个数量级，输出分布不受影响（dspark_check 逐字一致验收过）。
        """
        T = x.shape[0]
        R, B = self.idx.shape
        K = self.cb.shape[0]
        dt = compute_dtype(x.device)
        cb = cb_compute(self.cb, dt)
        idxl = self.idx.long()
        barange = torch.arange(B, device=x.device)
        out = torch.empty(T, R, dtype=torch.float32, device=x.device)
        # 分块大小：让 [Tc, B, K] f32 ≤ 256MB
        tchunk = max(1, min(T, (256 * 2**20) // (B * K * 4)))
        for t0 in range(0, T, tchunk):
            t1 = min(t0 + tchunk, T)
            xb = x[t0:t1].to(dt).view(t1 - t0, B, self.dim)
            s = xb @ cb.t()                        # [Tc, B, K]（半精度 GEMM）
            for t in range(t1 - t0):
                # g[r, b] = s[t, b, idx[r, b]]；对 [B, K] 用 (行b, 列idx) 高级索引
                g = s[t][barange.unsqueeze(0), idxl]   # [R, B]
                out[t0 + t] = g.sum(1, dtype=torch.float32)   # f32 累加
        return out
