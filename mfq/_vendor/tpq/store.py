"""TPQ 模型仓库层：读取 TPQ 产出的 "tpq-1" 格式（纯文件 I/O，无 mmap）。

目录结构（GLM-5.2-tpq/）：
    tpq.json               清单（config + quant 元信息 + 文件映射）
    dense.safetensors       dense 权重：int4 对（name + name.qs）或 f32 小张量
    vq-codebooks.safetensors 可选专家级跨层码本池和U8分配表
    experts.L*.safetensors  每层专家：共享 cb.gu.{档}，或按连续专家分组的
                            cb.gu.{档}.g{组号} 码本（down 同理）+
                            e{N}.gu{档}(z)/e{N}.dn{档}(z) 索引（z = zlib 熵编码）

为什么不用 safetensors 的 mmap：Windows 下长进程累积大量映射（75 个专家文件 +
CUDA 显存预留）会间歇性触发 access violation（实测层 64 附近必现）。
这里自实现 safetensors 读取（8 字节头长 + JSON 头 + 原始字节），
普通文件读走页缓存，无映射累积问题，返回的张量全部自持内存。
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
import zlib
from collections import OrderedDict

import torch

from .expert_slots import ExpertSignature, GpuExpertArenas
from .kernels import BlockFP8Weight, Int4Weight, VQWeight
from .ramcache import active_ram_file

# 专家磁盘加载调参（2026-07-20 实测调优，见 tpq/README 缓存×速度基准）：
#   TPQ_LOAD_WORKERS：并行加载线程数（NVMe 随机读吃队列深度；默认 12）
#   TPQ_READ_BUF_MB ：每文件句柄读缓冲（默认 2MB——大缓冲无益：get_bytes 是
#       5-9MB 整块读，Python 对大 read 直写目标缓冲；且线程局部句柄 × 75 层文件
#       会放大内存占用，16MB×900 句柄实测 OOM 崩溃）
_LOAD_WORKERS = int(os.environ.get("TPQ_LOAD_WORKERS", "12"))
_READ_BUF = int(os.environ.get("TPQ_READ_BUF_MB", "2")) * 1024 * 1024
_EXEC = None
_PF_EXEC = None
_SAFEFILE_THREAD = threading.local()


def _unpack_u12(packed: torch.Tensor, count: int) -> torch.Tensor:
    """把 TPQ 的双 12-bit/3-byte 索引恢复为 u16。

    产物仅在磁盘/RAM blob 中紧凑保存；进入 GPU 专家 arena 前只解包一次，
    因而不会给每次专家计算增加位操作。
    """
    raw = packed.view(torch.uint8).reshape(-1).to(torch.int32)
    if raw.numel() % 3:
        raise ValueError(f"u12 packed bytes 必须为 3 的倍数，实际 {raw.numel()}")
    tri = raw.reshape(-1, 3)
    out = torch.empty(tri.shape[0] * 2, dtype=torch.uint16)
    out[0::2] = (tri[:, 0] | ((tri[:, 1] & 0x0F) << 8)).to(torch.uint16)
    out[1::2] = ((tri[:, 1] >> 4) | (tri[:, 2] << 4)).to(torch.uint16)
    if count > out.numel():
        raise ValueError(f"u12 索引数量不足: need={count}, have={out.numel()}")
    return out[:count]


def _unpack_u14(packed: torch.Tensor, count: int) -> torch.Tensor:
    """把 TPQ 的四 14-bit/7-byte 索引恢复为 u16。"""
    raw = packed.view(torch.uint8).reshape(-1).to(torch.int64)
    if raw.numel() % 7:
        raise ValueError(f"u14 packed bytes 必须为 7 的倍数，实际 {raw.numel()}")
    group = raw.reshape(-1, 7)
    word = torch.zeros(group.shape[0], dtype=torch.int64)
    for byte in range(7):
        word |= group[:, byte] << (8 * byte)
    out = torch.empty(group.shape[0] * 4, dtype=torch.uint16)
    mask = (1 << 14) - 1
    for index in range(4):
        out[index::4] = ((word >> (14 * index)) & mask).to(torch.uint16)
    if count > out.numel():
        raise ValueError(f"u14 索引数量不足: need={count}, have={out.numel()}")
    return out[:count]


def _unpack_u10(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Restore four little-endian 10-bit indices from every five bytes."""
    raw = packed.view(torch.uint8).reshape(-1).to(torch.int64)
    if raw.numel() % 5:
        raise ValueError(
            f"u10 packed bytes must be a multiple of 5, got {raw.numel()}"
        )
    group = raw.reshape(-1, 5)
    word = torch.zeros(group.shape[0], dtype=torch.int64)
    for byte in range(5):
        word |= group[:, byte] << (8 * byte)
    out = torch.empty(group.shape[0] * 4, dtype=torch.uint16)
    for index in range(4):
        out[index::4] = (
            (word >> (10 * index)) & 0x3FF
        ).to(torch.uint16)
    if count > out.numel():
        raise ValueError(
            f"u10 index count is too small: need={count}, "
            f"have={out.numel()}"
        )
    return out[:count]


def _unpack_u9(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Restore consecutive little-endian 9-bit indices.

    Nine-bit archives pack eight indices into nine bytes.  The vectorized
    reference path is used by correctness tests and CPU fallbacks; resident
    inference keeps the payload packed and extracts indices inside the native
    CPU/CUDA kernels.
    """
    raw = packed.view(torch.uint8).reshape(-1).to(torch.int32)
    if raw.numel() * 8 != int(count) * 9:
        raise ValueError(
            "u9 packed bytes do not match the requested index count: "
            f"bytes={raw.numel()}, count={count}"
        )
    bit_offsets = torch.arange(count, dtype=torch.int64) * 9
    byte_offsets = torch.bitwise_right_shift(bit_offsets, 3)
    shifts = torch.bitwise_and(bit_offsets, 7).to(torch.int32)
    # Every valid final index ends at or before the final payload bit.  One
    # zero pad byte keeps the two-byte gather branch-free at the boundary.
    padded = torch.cat((raw, torch.zeros(1, dtype=torch.int32)))
    words = (
        padded[byte_offsets]
        | torch.bitwise_left_shift(padded[byte_offsets + 1], 8)
    )
    return torch.bitwise_and(
        torch.bitwise_right_shift(words, shifts),
        0x1FF,
    ).to(torch.uint16)


def _unpack_odd_width(
    packed: torch.Tensor,
    count: int,
    bits: int,
) -> torch.Tensor:
    """Reference unpacker for row-aligned p11/p13/p15 payloads."""
    if bits not in (11, 13, 15):
        raise ValueError(f"unsupported odd packed width {bits}")
    raw = packed.view(torch.uint8).reshape(-1).to(torch.int64)
    if raw.numel() * 8 != int(count) * bits:
        raise ValueError(
            f"u{bits} packed bytes do not match index count: "
            f"bytes={raw.numel()}, count={count}"
        )
    bit_offsets = torch.arange(count, dtype=torch.int64) * bits
    byte_offsets = torch.bitwise_right_shift(bit_offsets, 3)
    shifts = torch.bitwise_and(bit_offsets, 7)
    # At most 22 bits are needed. Two pad bytes keep the final gather valid;
    # production CPU/CUDA paths read only the bytes actually required.
    padded = torch.cat((raw, torch.zeros(2, dtype=torch.int64)))
    words = (
        padded[byte_offsets]
        | torch.bitwise_left_shift(padded[byte_offsets + 1], 8)
        | torch.bitwise_left_shift(padded[byte_offsets + 2], 16)
    )
    return torch.bitwise_and(
        torch.bitwise_right_shift(words, shifts),
        (1 << bits) - 1,
    ).to(torch.uint16)


def _stored_index_bits(num_bytes: int, count: int) -> int:
    """Infer standard TPQ index width from exact payload length."""
    total_bits = int(num_bytes) * 8
    if count > 0 and total_bits % int(count) == 0:
        bits = total_bits // int(count)
        if 8 <= bits <= 16:
            return bits
    raise ValueError(
        f"cannot infer VQ index width: bytes={num_bytes}, count={count}"
    )


def _safe_arena_budget(
    *,
    requested_bytes: int,
    allocated_bytes: int,
    device_free_bytes: int,
    process_limit_bytes: int,
    reserve_bytes: int,
) -> int:
    """Cap a fixed expert arena before allocation.

    ``device_free_bytes`` already excludes current allocations, while the
    allocator's per-process limit does not.  Respect both ceilings so a large
    user request cannot first OOM and then fall back to an unnecessarily small
    half-sized cache.
    """
    device_room = max(0, int(device_free_bytes) - int(reserve_bytes))
    process_room = max(
        0,
        int(process_limit_bytes) - int(allocated_bytes) - int(reserve_bytes),
    )
    return max(
        0,
        min(int(requested_bytes), device_room, process_room),
    )


def _executor():
    """常驻加载线程池（避免每层每 token 重建线程的生成开销；线程局部句柄复用 fd）。"""
    global _EXEC
    if _EXEC is None:
        from concurrent.futures import ThreadPoolExecutor
        _EXEC = ThreadPoolExecutor(max_workers=_LOAD_WORKERS, thread_name_prefix="tpq-load")
    return _EXEC


def _pf_executor():
    """预取专用小池（4 线程）：与紧急加载池隔离——否则冷启动时 get_many 的
    紧急 miss 排在数百个预取任务后面（实测 1.57→0.07 tok/s 的饥饿事故）。"""
    global _PF_EXEC
    if _PF_EXEC is None:
        from concurrent.futures import ThreadPoolExecutor
        _PF_EXEC = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tpq-prefetch")
    return _PF_EXEC


_STAGE_EXEC = None


def _stage_executor():
    """后台 staging 专用单线程：预取的 RAM→VRAM 装槽+DMA 全部在此线程串行完成
    （单线程队列 = 槽位纪律天然有序，避开上次多线程并行装槽的错配事故），
    主线程推理不被 host memcpy 阻塞（真并行预加载）。"""
    global _STAGE_EXEC
    if _STAGE_EXEC is None:
        from concurrent.futures import ThreadPoolExecutor
        _STAGE_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tpq-stage")
    return _STAGE_EXEC


class PinnedStage:
    """专家上传的 pinned 分段暂存：host memcpy 进 pinned 槽 + 异步 DMA（真 ~10GB/s），
    替代页式上传（~3.7GB/s，驱动内部分段复制）。

    复用安全：每槽一个事件，复写前等该槽上次 DMA 完成；wait() 让计算流只等
    拷贝流尾部事件（前序 DMA 按流序天然先行）。索引张量为 u8（槽也按 u8 存取）。
    """

    def __init__(
        self,
        device,
        n_slots: int = 32,
        slot_mb: int = 12,
        *,
        measure: bool = False,
    ):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.slots = [torch.empty(slot_mb * 2**20, dtype=torch.uint8, pin_memory=True)
                      for _ in range(n_slots)]
        self.events = [torch.cuda.Event() for _ in range(n_slots)]
        self.last = torch.cuda.Event()
        # Direct uploads from permanently page-locked expert tensors bypass the
        # staging slots. Keep sources alive until their DMA event completes so
        # this remains safe for callers that do not otherwise retain the tensor.
        self._pinned_inflight: list[tuple[torch.cuda.Event, list[torch.Tensor]]] = []
        self._measure = bool(measure)
        self._timing: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.transfer_seconds = 0.0
        self.i = 0

    def upload_batch(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """成批上传：多对 (src CPU, dst GPU) 依次装槽 DMA，一次尾部事件。
        字节原生处理（u8/u16 索引通用）：槽按字节存取，dtype 由 dst 自带。
        装槽 host memcpy 串行（并行装槽曾致槽内容错配事故，见 EXPERIENCE §12）。

        跨流安全（NaN 事故的根因修复）：
          1) 批首 copy stream 等默认流当前队尾——被驱逐释放的显存块之后若被
             empty_like 复用为 DMA 目标，该块的最后读者（默认流 kernel）必定
             已在本次 wait 前提交，DMA 按序排在其后，杜绝「计算读旧块 vs
             DMA 写复用块」并发；
          2) 每个 dst record_stream(copy stream)——dst 被驱逐释放时 allocator
             会等 DMA 真正完成才把块给别人（wait_event 只排序不等于完成）。
        """
        if not pairs:
            return
        self._pinned_inflight = [
            item for item in self._pinned_inflight if not item[0].query()
        ]
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        timing_start = None
        timing_end = None
        if self._measure:
            timing_start = torch.cuda.Event(enable_timing=True)
            timing_end = torch.cuda.Event(enable_timing=True)
            timing_start.record(self.stream)
        views = []
        direct_sources = []
        for src, dst in pairs:
            if src.is_pinned():
                with torch.cuda.stream(self.stream):
                    dst.copy_(src, non_blocking=True)
                dst.record_stream(self.stream)
                direct_sources.append(src)
                views.append((src, dst, src.view(torch.uint8).view(-1)))
                continue
            slot = self.slots[self.i]
            ev = self.events[self.i]
            ev.synchronize()
            nb = src.nbytes
            assert nb <= slot.numel(), f"专家张量超槽位（{nb}B > {slot.numel()}B）"
            slot[:nb].copy_(src.view(torch.uint8).view(-1))
            with torch.cuda.stream(self.stream):
                dst.view(torch.uint8).view(-1).copy_(slot[:nb], non_blocking=True)
            dst.record_stream(self.stream)
            ev.record(self.stream)
            views.append((src, dst, slot[:nb]))
            self.i = (self.i + 1) % len(self.slots)
        if timing_end is None:
            self.last.record(self.stream)
        else:
            timing_end.record(self.stream)
            self.last = timing_end
            self._timing.append((timing_start, timing_end))
        if direct_sources:
            done = torch.cuda.Event()
            done.record(self.stream)
            self._pinned_inflight.append((done, direct_sources))
        if os.environ.get("TPQ_STAGE_VERIFY", "0") != "0":
            # 诊断：校验每对 DMA 落盘内容与源一致（定位错字节/错地址）
            torch.cuda.synchronize()
            for src, dst, view in views:
                s = src.view(torch.uint8).view(-1)
                d = dst.view(torch.uint8).view(-1).cpu()
                if not torch.equal(d, s):
                    neq = (d != s)
                    n = int(neq.sum())
                    first = int(neq.nonzero()[0])
                    slot_eq = torch.equal(view.cpu(), s)
                    print(f"[stage-verify] 不一致: {n}/{s.numel()} 字节 "
                          f"首个@{first} ({first / s.numel():.1%}) 槽位正确={slot_eq}",
                          flush=True)

    def upload(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """src（CPU u8 张量）→ dst（GPU u8 张量）：host memcpy + 异步 DMA。"""
        self.upload_batch([(src, dst)])

    def wait(self) -> None:
        """让当前流等待拷贝流尾部（本批全部 DMA 完成）。"""
        torch.cuda.current_stream().wait_event(self.last)

    def collect_timing(self, *, synchronize: bool = False) -> float:
        """回收已经完成的 copy-stream 批次，返回累计 staging+H2D 秒数。

        计时事件包住逐段 host staging 形成的提交间隙和实际 H2D，因此用于解释
        端到端关键路径；默认不阻塞，正式结果收集前可在外部全卡同步后调用。
        """
        if not self._measure:
            return 0.0
        if synchronize and self._timing:
            self._timing[-1][1].synchronize()
        completed = 0
        for _start, end in self._timing:
            if not end.query():
                break
            completed += 1
        for start, end in self._timing[:completed]:
            self.transfer_seconds += start.elapsed_time(end) / 1000.0
        if completed:
            del self._timing[:completed]
        return self.transfer_seconds


_DTYPES = {
    "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16, "I32": torch.int32,
    "I64": torch.int64, "F16": torch.float16, "F32": torch.float32,
    "F64": torch.float64, "BF16": torch.bfloat16, "BOOL": torch.bool,
    "U16": torch.uint16, "U32": torch.uint32, "U64": torch.uint64,
    # Safetensors has no raw-byte view mode.  These entries intentionally
    # expose FP8 payload bytes; the logical Dense decoder applies its scale.
    "F8_E4M3": torch.uint8, "F8_E5M2": torch.uint8,
    "F8_E8M0": torch.uint8,
}
_DTYPE_NBYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


class SafeFile:
    """极简 safetensors 读取器（纯文件 I/O，无 mmap；线程局部句柄支持并发读）。"""

    def __init__(self, path: str):
        self.path = path
        self._all_handles: set = set()
        self._handles_lock = threading.Lock()
        self._ram_blob = active_ram_file(path)
        if self._ram_blob is None:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n).decode("utf-8"))
        else:
            n = struct.unpack_from("<Q", self._ram_blob, 0)[0]
            header = json.loads(
                memoryview(self._ram_blob)[8:8 + n]
                .tobytes()
                .decode("utf-8")
            )
        self.meta = {k: v for k, v in header.items() if k != "__metadata__"}
        self.data_start = 8 + n

    def release_ram_blob(self) -> int:
        released = 0 if self._ram_blob is None else len(self._ram_blob)
        self._ram_blob = None
        return released

    @property
    def ram_blob_nbytes(self) -> int:
        return 0 if self._ram_blob is None else len(self._ram_blob)

    def close(self) -> None:
        with self._handles_lock:
            handles = tuple(self._all_handles)
            self._all_handles.clear()
        for handle in handles:
            handle.close()

    def _fh(self):
        """One current shard handle per reader thread.

        A handle per (thread, shard) leaks roughly ``workers * layers`` file
        descriptors during a full-model pass and exceeds the usual 1024-fd
        limit.  Workers process one layer at a time, so retaining only their
        current shard keeps seek independence and caps descriptors at the
        worker count.
        """
        slot = getattr(_SAFEFILE_THREAD, "slot", None)
        f = None if slot is None else slot[1]
        if slot is None or slot[0] is not self or f.closed:
            if slot is not None and not f.closed:
                previous = slot[0]
                with previous._handles_lock:
                    previous._all_handles.discard(f)
                f.close()
            f = open(self.path, "rb", buffering=_READ_BUF)
            with self._handles_lock:
                self._all_handles.add(f)
            _SAFEFILE_THREAD.slot = (self, f)
        return f

    def keys(self):
        return self.meta.keys()

    def get_bytes(self, name: str) -> bytes | memoryview:
        info = self.meta[name]
        start = self.data_start + info["data_offsets"][0]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        if self._ram_blob is not None:
            return memoryview(self._ram_blob)[start:start + size]
        f = self._fh()
        f.seek(start)
        return f.read(size)

    def get_tensor(self, name: str) -> torch.Tensor:
        """读张量：单次分配 + readinto 直填，省掉 bytes→bytearray 的整块拷贝
        （专家加载热路径：每 token ~5.9GB，省一次全量 memcpy）。"""
        info = self.meta[name]
        start = self.data_start + info["data_offsets"][0]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        if self._ram_blob is not None:
            buf = memoryview(self._ram_blob)[start:start + size]
        else:
            f = self._fh()
            f.seek(start)
            buf = bytearray(size)
            f.readinto(buf)
        t = torch.frombuffer(buf, dtype=_DTYPES[info["dtype"]])
        return t.reshape(info["shape"])


class SafeTensorCollection:
    """Lazy, read-only view over one or more safetensors shards.

    Legacy TPQ models keep non-expert tensors in one ``dense.safetensors``.
    Kimi K3 preserves source-native BF16 tensors in many ``dense/`` shards.
    This adapter gives both layouts the same ``keys``/``get_tensor`` contract
    and only opens a shard when one of its tensors is requested.
    """

    def __init__(
        self,
        root: str,
        files: list[str],
        *,
        audit_file: str | None = None,
    ):
        if not files:
            raise ValueError("at least one dense safetensors shard is required")
        self.root = root
        self.files = tuple(files)
        self._handles: dict[str, SafeFile] = {}
        self._locations: dict[str, str] = {}
        self._nbytes: dict[str, int] = {}
        self._logical_entries: dict[str, dict] = {}

        if audit_file is not None and os.path.exists(audit_file):
            with open(audit_file, "r", encoding="utf-8") as handle:
                audit = json.load(handle)
            aliases: dict[str, str] = {}
            for filename in self.files:
                normalized = filename.replace("\\", "/")
                aliases[normalized] = filename
                aliases[os.path.basename(normalized)] = filename
            for shard, item in audit.get("shards", {}).items():
                filename = aliases.get(str(shard).replace("\\", "/"))
                if filename is None:
                    continue
                for name in item.get("tensor_audit", {}):
                    self._locations[name] = filename
                    self._nbytes[name] = int(
                        item["tensor_audit"][name].get("bytes", 0)
                    )
            if audit.get("format") == "kimi-k3-dense-audit-v1":
                for name, entry in audit.get("entries", {}).items():
                    filename = aliases.get(
                        str(entry["shard"]).replace("\\", "/")
                    )
                    if filename is None:
                        raise ValueError(
                            f"dense audit shard is not in manifest: "
                            f"{entry['shard']}"
                        )
                    self._locations[name] = filename
                    self._nbytes[name] = int(entry["stored_bytes"])
                    self._logical_entries[name] = entry

        # Developer fixtures and legacy manifests may not have an audit.
        # Reading headers is cheap and does not touch tensor payloads.
        if not self._locations:
            for shard in self.files:
                safe = self._handle(shard)
                for name in safe.keys():
                    if name in self._locations:
                        raise ValueError(
                            f"duplicate dense tensor {name!r} in "
                            f"{self._locations[name]!r} and {shard!r}"
                        )
                    self._locations[name] = shard
                    info = safe.meta[name]
                    self._nbytes[name] = (
                        int(info["data_offsets"][1])
                        - int(info["data_offsets"][0])
                    )

    def _handle(self, shard: str) -> SafeFile:
        handle = self._handles.get(shard)
        if handle is None:
            handle = SafeFile(os.path.join(self.root, shard))
            self._handles[shard] = handle
        return handle

    def keys(self):
        return self._locations.keys()

    def get_tensor(self, name: str) -> torch.Tensor:
        handle = self._handle(self._locations[name])
        entry = self._logical_entries.get(name)
        if entry is None:
            return handle.get_tensor(name)
        kind = entry["storage_kind"]
        if kind == "source":
            return handle.get_tensor(entry.get("value_key") or name)
        if kind == "fp8":
            return self._decode_fp8(handle, entry)
        if kind == "d3-p12":
            return self._decode_d3(handle, entry)
        raise ValueError(f"unknown dense storage kind {kind!r}")

    def get_block_fp8(self, name: str) -> BlockFP8Weight | None:
        """Return audited block-FP8 without expanding it to BF16."""
        entry = self._logical_entries.get(name)
        if entry is None:
            # Source-exact archives retain an original E4M3 weight plus E8M0
            # exponent scale pair.  Recognize that pair from the safetensors
            # header and keep the weight byte-packed in RAM/VRAM.
            shard = self._locations.get(name)
            if shard is None or not name.endswith(".weight"):
                return None
            handle = self._handle(shard)
            info = handle.meta[name]
            scale_name = name[: -len("weight")] + "scale"
            if (
                info.get("dtype") != "F8_E4M3"
                or self._locations.get(scale_name) != shard
                or handle.meta[scale_name].get("dtype") != "F8_E8M0"
            ):
                return None
            raw = handle.get_tensor(name)
            scales = handle.get_tensor(scale_name)
            rows, columns = (int(value) for value in info["shape"])
        else:
            if entry["storage_kind"] != "fp8":
                return None
            handle = self._handle(self._locations[name])
            raw = handle.get_tensor(entry["value_key"])
            scales = handle.get_tensor(entry["scale_key"])
            rows, columns = (
                int(value) for value in entry["logical_shape"]
            )
        if scales.dtype == torch.uint8:
            scales = torch.pow(2.0, scales.float() - 127.0)
        if raw.shape != (rows, columns):
            raise ValueError(
                f"dense FP8 payload shape mismatch for {name!r}: "
                f"{tuple(raw.shape)} != {(rows, columns)}"
            )
        return BlockFP8Weight(raw, scales, columns, 128)

    @staticmethod
    def _logical_dtype(entry: dict) -> torch.dtype:
        dtype = str(entry["source_dtype"])
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported dense logical dtype {dtype!r}")
        return _DTYPES[dtype]

    @classmethod
    def _decode_fp8(cls, handle: SafeFile, entry: dict) -> torch.Tensor:
        raw = handle.get_tensor(entry["value_key"])
        scales = handle.get_tensor(entry["scale_key"])
        if scales.dtype == torch.uint8:
            scales = torch.pow(2.0, scales.float() - 127.0)
        rows, columns = (int(value) for value in entry["logical_shape"])
        block = 128
        dtype = cls._logical_dtype(entry)
        output = torch.empty((rows, columns), dtype=dtype)
        for row_start in range(0, rows, block):
            row_stop = min(row_start + block, rows)
            for column_start in range(0, columns, block):
                column_stop = min(column_start + block, columns)
                values = (
                    raw[row_start:row_stop, column_start:column_stop]
                    .view(torch.float8_e4m3fn)
                    .to(dtype)
                )
                output[
                    row_start:row_stop, column_start:column_stop
                ] = values * scales[
                    row_start // block,
                    column_start // block,
                ].to(dtype)
        return output

    @classmethod
    def _decode_d3(cls, handle: SafeFile, entry: dict) -> torch.Tensor:
        packed = handle.get_tensor(entry["index_key"]).to(torch.int32)
        if packed.ndim != 2 or packed.shape[1] % 3:
            raise ValueError(
                f"invalid row-aligned p12 dense tensor "
                f"{entry['logical_name']!r}"
            )
        rows, columns = (int(value) for value in entry["logical_shape"])
        groups = columns // 4
        tri = packed.reshape(rows, -1, 3)
        indices = torch.empty(
            (rows, tri.shape[1] * 2), dtype=torch.int64
        )
        indices[:, 0::2] = (
            tri[:, :, 0] | ((tri[:, :, 1] & 0x0F) << 8)
        )
        indices[:, 1::2] = (
            (tri[:, :, 1] >> 4) | (tri[:, :, 2] << 4)
        )
        indices = indices[:, :groups]
        codebook = handle.get_tensor(entry["codebook_key"]).float()
        return (
            codebook[indices.reshape(-1)]
            .reshape(rows, columns)
            .to(cls._logical_dtype(entry))
        )

    def nbytes(self, name: str) -> int:
        """Return payload bytes without reading the tensor body."""
        value = self._nbytes.get(name)
        if value:
            return value
        shard = self._locations[name]
        info = self._handle(shard).meta[name]
        return int(info["data_offsets"][1]) - int(info["data_offsets"][0])

    def resident_nbytes(self, name: str) -> int:
        """Bytes occupied after decoding one logical Dense tensor."""
        entry = self._logical_entries.get(name)
        if entry is None:
            return self.nbytes(name)
        if entry["storage_kind"] == "fp8":
            # The public BlockFP8 operator consumes the audited uint8 payload
            # and FP32 scales directly on CPU/CUDA.  Placement must therefore
            # budget the compact resident representation, not a hypothetical
            # BF16 expansion.
            return self.nbytes(name)
        elements = 1
        for value in entry["logical_shape"]:
            elements *= int(value)
        return elements * _DTYPE_NBYTES[str(entry["source_dtype"])]

    def ram_blob_paths(self) -> tuple[str, ...]:
        return tuple(
            handle.path
            for handle in self._handles.values()
            if handle.ram_blob_nbytes
        )

    def release_ram_blob(self) -> int:
        released = 0
        for handle in self._handles.values():
            released += handle.release_ram_blob()
        return released

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()


class Manifest:
    """Read a ``tpq.json`` manifest."""

    def __init__(self, root: str):
        manifest_path = os.path.join(root, "tpq.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if m.get("format") != "tpq-1":
            raise ValueError(f"不支持的 TPQ 格式: {m.get('format')!r}")
        self.root = root
        self.manifest_path = manifest_path
        self.format = str(m["format"])
        self.config = m["config"]
        self.quant = m["quant"]
        self.model_family = str(m.get("model_family", ""))
        dense_files = m.get("dense_files")
        if dense_files is None:
            dense_files = [m["dense_file"]]
            self.dense_root = root
        else:
            dense_path = (m.get("nonexpert") or {}).get("path", "dense")
            normalized_path = str(dense_path).strip("/\\")
            prefixed = normalized_path and all(
                str(value).replace("\\", "/").startswith(
                    normalized_path.replace("\\", "/") + "/"
                )
                for value in dense_files
            )
            # Standard Kimi archives store root-relative entries such as
            # ``dense/model-00001...`` while older fixtures may list only the
            # basename and rely on ``nonexpert.path``.
            self.dense_root = (
                root
                if prefixed
                else os.path.join(root, normalized_path)
            )
        self.dense_files = [str(value) for value in dense_files]
        # Keep the legacy attribute for startup estimators and old callers.
        self.dense_file = self.dense_files[0]
        audit_name = m.get("dense_audit_file")
        self.dense_audit_file = (
            os.path.join(root, audit_name) if audit_name else None
        )
        routed = m.get("routed_experts") or {}
        self.routed_layers = int(routed.get("layers", 0) or 0)
        self.routed_experts_per_layer = int(
            routed.get("experts_per_layer", 0) or 0
        )
        self.no_expert_drop = bool(routed.get("no_expert_drop", False))
        routed_layers = routed.get("layer_files") or {}
        # Projection-VQ has two released manifest layouts.  Kimi archives
        # describe each layer under ``routed_experts.layer_files`` while the
        # compact DeepSeek-V4 archive keeps the same information in the
        # top-level ``expert_files``/``layer_audit`` maps and stores the
        # projection descriptions in ``quant.layouts``.  Normalize both here
        # so model and operator code never branch on a model-family name.
        heterogeneous = (
            self.quant.get("heterogeneous_expert_tiering") or {}
        )
        heterogeneous_projection_vq = bool(
            self.quant.get("method") == "projection-vq"
            and self.quant.get("layouts")
            and heterogeneous.get("precision_levels")
            and heterogeneous.get("layer_expert_levels")
        )
        flat_projection_vq = bool(
            not routed_layers
            and m.get("expert_files")
            and self.quant.get("method") == "projection-vq"
            and (
                self.quant.get("projection_layouts")
                or heterogeneous_projection_vq
            )
        )
        self.projection_vq = bool(
            (
                routed_layers
                and (
                    self.quant.get("projection_layouts")
                    or heterogeneous_projection_vq
                )
            )
            or flat_projection_vq
        )
        self.heterogeneous_projection_vq = heterogeneous_projection_vq
        self.projection_layout_by_layer: dict[int, dict[str, str]] = {}
        self.projection_layout_by_expert: dict[
            int, tuple[dict[str, str], ...]
        ] = {}
        self.projection_precision_levels: dict[
            str, dict[str, str]
        ] = {}
        self.projection_level_by_expert: dict[
            int, tuple[str, ...]
        ] = {}
        if self.projection_vq:
            if flat_projection_vq:
                layer_audit = m.get("layer_audit") or {}
                self.expert_files = {
                    int(layer): str(filename)
                    for layer, filename in m["expert_files"].items()
                }
                self.expert_audit_files = {
                    int(layer): str(item["audit_path"])
                    for layer, item in layer_audit.items()
                    if isinstance(item, dict) and item.get("audit_path")
                }
                self.routed_layers = len(self.expert_files)
                self.routed_experts_per_layer = int(
                    self.config.get("n_experts", 0)
                )
                self.no_expert_drop = bool(
                    self.quant.get("no_expert_drop", False)
                )
            else:
                self.expert_files = {
                    int(layer): str(item["path"])
                    for layer, item in routed_layers.items()
                }
                self.expert_audit_files = {
                    int(layer): str(item["audit_path"])
                    for layer, item in routed_layers.items()
                }

            if heterogeneous_projection_vq:
                layout_specs = self.quant.get("layouts") or {}
                self.projection_precision_levels = {
                    str(level): {
                        str(projection): str(layout_name)
                        for projection, layout_name in layouts.items()
                    }
                    for level, layouts in heterogeneous[
                        "precision_levels"
                    ].items()
                }
                self.projection_level_by_expert = {
                    int(layer): tuple(str(level) for level in levels)
                    for layer, levels in heterogeneous[
                        "layer_expert_levels"
                    ].items()
                }
            elif flat_projection_vq:
                layout_specs = self.quant.get("layouts") or {}
                self.projection_layout_by_layer = {
                    int(layer): {
                        str(projection): str(layout_name)
                        for projection, layout_name in layouts.items()
                    }
                    for layer, layouts in self.quant[
                        "projection_layouts"
                    ].items()
                }
            else:
                layout_specs = self.quant["projection_layouts"]
                self.projection_layout_by_layer = {
                    int(layer): {
                        str(projection): str(layout_name)
                        for projection, layout_name in item[
                            "projection_layout"
                        ].items()
                    }
                    for layer, item in routed_layers.items()
                }
            self.vq_dims = {
                str(name): (
                    int(item["dim"]),
                    int(item["size"]),
                )
                for name, item in layout_specs.items()
            }
            self.projection_layout_specs = {
                str(name): dict(item)
                for name, item in layout_specs.items()
            }
            self.projection_codebook_group_sizes = {
                name: int(item["group_size"])
                for name, item in self.projection_layout_specs.items()
                if item.get("group_size") is not None
            }
            self.projection_codebook_group_counts = {
                name: int(item["groups"])
                for name, item in self.projection_layout_specs.items()
                if item.get("groups") is not None
            }
            if any(
                size <= 0
                for size in self.projection_codebook_group_sizes.values()
            ):
                raise ValueError(
                    "projection VQ codebook group_size must be positive"
                )
            self.index_packing = {
                str(name): str(packing)
                for name, packing in self.quant.get(
                    "index_packing", {}
                ).items()
            }
            self._validate_projection_layouts()
            if heterogeneous_projection_vq:
                self.projection_layout_by_expert = {
                    layer: tuple(
                        self.projection_precision_levels[level]
                        for level in levels
                    )
                    for layer, levels in self.projection_level_by_expert.items()
                }
        else:
            self.heterogeneous_projection_vq = False
            self.expert_files = {
                int(l): v for l, v in m["expert_files"].items()
            }
            self.expert_audit_files = {
                int(layer): value
                for layer, value in m.get(
                    "expert_audit_files", {}
                ).items()
            }
            self.projection_layout_specs = {}
            self.projection_codebook_group_sizes = {}
            self.projection_codebook_group_counts = {}
            self.index_packing = {}
            self.vq_dims = {
                k: tuple(v)
                for k, v in m["quant"]["vq"].items()
            }  # 档 -> (dim, k)
        layout = m["quant"].get("vq_codebook_layout") or {}
        layout_format = layout.get("format")
        if layout_format not in (
            None,
            "tpq-vq-codebook-layout-v1",
            "expert-assigned-codebook-v1",
        ):
            raise ValueError(f"不支持的 VQ 码本布局: {layout_format}")
        if (
            layout_format == "tpq-vq-codebook-layout-v1"
            and layout.get("assignment") != "contiguous-expert-id"
        ):
            raise ValueError(
                f"不支持的 VQ 码本分配规则: {layout.get('assignment')}"
            )
        if (
            layout_format == "expert-assigned-codebook-v1"
            and layout.get("assignment")
            != "per-expert-per-projection"
        ):
            raise ValueError(
                f"不支持的 VQ 码本分配规则: {layout.get('assignment')}"
            )
        self.vq_codebook_layout = layout
        self.vq_codebook_layout_format = layout_format
        self.vq_codebook_group_sizes = {
            str(kind): int(size)
            for kind, size in layout.get("group_size", {}).items()
        }
        if any(size <= 0 for size in self.vq_codebook_group_sizes.values()):
            raise ValueError("VQ 码本 group_size 必须大于 0")
        self.vq_codebook_file = (
            m["quant"].get("vq_codebook_file")
            or layout.get("codebook_file")
        )
        self.int4_group = m["quant"].get("int4_group", 64)
        self.zlib = m["quant"].get("zlib", False)
        # 每层每专家档位串（'v'/'w'/'x'/'d'=drop），量化/repack 时写入；缺省 = 全保留
        self.tiers_per_layer = {
            int(l): s
            for l, s in m.get("tiers_per_layer", {}).items()
        }
        if self.projection_vq:
            self.tiers_per_layer.update(
                {
                    int(layer): str(item.get("tier_string", ""))
                    for layer, item in routed_layers.items()
                    if item.get("tier_string")
                }
            )
        self._audit_tiers: dict[int, str] = {}

    def _validate_projection_layouts(self) -> None:
        projections = {"gate", "up", "down"}
        if self.heterogeneous_projection_vq:
            if not self.projection_precision_levels:
                raise ValueError(
                    "heterogeneous projection VQ has no precision_levels"
                )
            for level, layouts in self.projection_precision_levels.items():
                if set(layouts) != projections:
                    raise ValueError(
                        f"precision level {level!r} must define gate/up/down"
                    )
            expected = int(self.routed_experts_per_layer)
            for layer in self.expert_files:
                levels = self.projection_level_by_expert.get(layer)
                if levels is None:
                    raise ValueError(
                        f"L{layer} has no heterogeneous expert level map"
                    )
                if len(levels) != expected:
                    raise ValueError(
                        f"L{layer} expert level count {len(levels)} != "
                        f"n_experts {expected}"
                    )
                unknown = sorted(
                    set(levels) - set(self.projection_precision_levels)
                )
                if unknown:
                    raise ValueError(
                        f"L{layer} references unknown precision levels "
                        f"{unknown}"
                    )
            referenced = {
                layout
                for layouts in self.projection_precision_levels.values()
                for layout in layouts.values()
            }
        else:
            referenced = {
                layout
                for layouts in self.projection_layout_by_layer.values()
                for layout in layouts.values()
            }
        missing = sorted(referenced - set(self.projection_layout_specs))
        if missing:
            raise ValueError(
                f"projection VQ references undefined layouts: {missing}"
            )
        for layout in referenced:
            dim, size = self.vq_dims[layout]
            if dim <= 0 or size <= 0 or size & (size - 1):
                raise ValueError(
                    f"projection layout {layout} must have a positive dim "
                    "and power-of-two codebook size"
                )
            expected_bits = size.bit_length() - 1
            packing = self.index_packing.get(layout)
            if packing is None:
                continue
            bits = int(
                packing.removeprefix("packed-u").removeprefix("u")
            )
            if bits < expected_bits:
                raise ValueError(
                    f"projection layout {layout} packing {packing} does not "
                    f"have enough bits for codebook size {size}"
                )

    def projection_layouts(
        self,
        layer: int,
        expert_id: int | None = None,
    ) -> dict[str, str]:
        """Resolve Gate/Up/Down layouts without model-name dispatch."""
        layer = int(layer)
        if not self.heterogeneous_projection_vq:
            return self.projection_layout_by_layer[layer]
        if expert_id is None:
            raise ValueError(
                f"L{layer} uses heterogeneous layouts; expert_id is required"
            )
        levels = self.projection_level_by_expert[layer]
        if expert_id < 0 or expert_id >= len(levels):
            raise IndexError(f"L{layer} expert_id {expert_id} is out of range")
        return self.projection_precision_levels[levels[expert_id]]

    def tier_string(self, layer: int) -> str | None:
        value = self.tiers_per_layer.get(layer)
        if value is not None:
            return value
        cached = self._audit_tiers.get(layer)
        if cached is not None:
            return cached
        audit_name = self.expert_audit_files.get(layer)
        if audit_name is None:
            return None
        with open(
            os.path.join(self.root, audit_name),
            "r",
            encoding="utf-8",
        ) as handle:
            audit = json.load(handle)
        tiers = str(audit.get("tier_string", ""))
        if not tiers:
            return None
        self._audit_tiers[layer] = tiers
        return tiers

    def projection_operator_capability(
        self,
        layer: int,
        expert_id: int | None = None,
    ) -> dict[str, tuple]:
        """Return the exact public operator key for one expert layout."""
        if not self.projection_vq:
            return {}
        formats = {
            "u8": "p8",
            "u16": "p16",
        }
        formats.update(
            {f"packed-u{bits}": f"p{bits}" for bits in range(8, 17)}
        )
        if self.heterogeneous_projection_vq and expert_id is None:
            used_levels = set(self.projection_level_by_expert[int(layer)])
            layout_names = sorted(
                {
                    layout
                    for level in used_levels
                    for layout in self.projection_precision_levels[
                        level
                    ].values()
                }
            )
        else:
            layouts = self.projection_layouts(layer, expert_id)
            layout_names = [
                layouts[projection]
                for projection in ("gate", "up", "down")
            ]
        packed_formats = []
        code_dims = []
        codebook_sizes = []
        for layout in layout_names:
            packing = self.index_packing.get(layout)
            dim, size = self.vq_dims[layout]
            if packing is None:
                # A few early projection manifests described the exact
                # codebook but omitted the redundant packing table entry.
                # A power-of-two codebook has one unambiguous index width.
                bits = int(size).bit_length() - 1
                if size <= 0 or (1 << bits) != int(size):
                    raise ValueError(
                        f"L{layer} cannot infer packed width "
                        f"from non-power-of-two codebook {layout}"
                    )
                packing = (
                    "u8" if bits == 8
                    else "u16" if bits == 16
                    else f"packed-u{bits}"
                )
            if packing not in formats:
                raise ValueError(
                    f"L{layer} has no public packed format for "
                    f"{layout} -> {packing!r}"
                )
            packed_formats.append(formats[packing])
            code_dims.append(int(dim))
            codebook_sizes.append(int(size))
        return {
            "packed_formats": tuple(packed_formats),
            "code_dims": tuple(code_dims),
            "codebook_sizes": tuple(codebook_sizes),
        }

    def projection_operator_capabilities(
        self,
        layer: int,
    ) -> tuple[dict[str, tuple], ...]:
        """Return every unique public packed capability used by a layer."""
        per_expert = self.projection_layout_by_expert.get(int(layer))
        if per_expert is None:
            return (self.projection_operator_capability(layer),)
        unique: dict[tuple[tuple[str, tuple], ...], dict[str, tuple]] = {}
        for expert in range(len(per_expert)):
            capability = self.projection_operator_capability(layer, expert)
            key = tuple(sorted(capability.items()))
            unique.setdefault(key, capability)
        return tuple(unique.values())


class TPQStore:
    """TPQ dense and expert mmap store."""

    def __init__(self, root: str):
        self.root = root
        self.man = Manifest(root)
        self.cfg = self.man.config
        self._dense = SafeTensorCollection(
            self.man.dense_root,
            self.man.dense_files,
            audit_file=self.man.dense_audit_file,
        )
        self._dense_keys = set(self._dense.keys())
        self._expert_handles: dict[int, SafeFile] = {}
        self._expert_keys: dict[int, set[str]] = {}
        self._expert_open_lock = threading.RLock()
        self._cb_cache: dict[
            tuple[str, str, str], torch.Tensor
        ] = {}
        self._cb_lock = threading.RLock()
        self._vq_codebook_pool: SafeFile | None = None
        self._vq_codebook_pool_keys: set[str] = set()
        self._vq_assignments: dict[str, torch.Tensor] = {}
        if (
            self.man.vq_codebook_layout_format
            == "expert-assigned-codebook-v1"
        ):
            pool_name = (
                self.man.vq_codebook_file
                or "vq-codebooks.safetensors"
            )
            self._vq_codebook_pool = SafeFile(
                os.path.join(self.root, pool_name)
            )
            self._vq_codebook_pool_keys = set(
                self._vq_codebook_pool.keys()
            )
            assignment_keys = self.man.vq_codebook_layout.get(
                "assignment_keys",
                {
                    "gu": "assignment.v.gu",
                    "down": "assignment.v.down",
                },
            )
            for projection in ("gu", "down"):
                key = str(assignment_keys[projection])
                if key not in self._vq_codebook_pool_keys:
                    raise KeyError(f"多码本池缺少分配表: {key}")
                assignment = self._vq_codebook_pool.get_tensor(key)
                if (
                    assignment.dtype != torch.uint8
                    or tuple(assignment.shape)
                    != (92, int(self.cfg["n_experts"]))
                ):
                    raise ValueError(
                        f"多码本分配表形状/类型错误: "
                        f"{key} {assignment.dtype} "
                        f"{tuple(assignment.shape)}"
                    )
                self._vq_assignments[projection] = assignment
        self._mtp: SafeFile | None = None
        # 可选热度档案（模型目录 profile.json 或 TPQ_PROFILE_JSON）：层 → 按路由
        # 命中降序的专家号，供 ExpertPool 把最热专家永久钉进内存（LRU 对冷专家的
        # 一次性缓存会污染热集合，实测命中率仅 ~20%，钉住 top-32 ≈66% 路由质量）
        self.heat_ranks: dict[int, list[int]] | None = None
        pj = os.environ.get("TPQ_PROFILE_JSON") or os.path.join(root, "profile.json")
        if os.path.exists(pj):
            with open(pj, "r", encoding="utf-8") as f:
                pr = json.load(f)
            self.heat_ranks = {
                int(l): sorted((int(e) for e in cnt), key=lambda e: -cnt[str(e)])
                for l, cnt in pr.get("counts", {}).items()}
        # MTP 附件存在时，把第 78 层注册进专家体系（透明复用 ExpertPool/回退掩码）
        mtp_path = os.path.join(root, "mtp.safetensors")
        l78_path = os.path.join(root, "experts.L78.safetensors")
        if os.path.exists(mtp_path) and os.path.exists(l78_path):
            self._mtp = SafeFile(mtp_path)
            self.man.expert_files[78] = "experts.L78.safetensors"
            self.man.tiers_per_layer[78] = "v" * self.cfg["n_experts"]

    def has_mtp(self) -> bool:
        return self._mtp is not None

    def close(self) -> None:
        """Close lazily opened shard handles without modifying model files."""
        self._dense.close()
        for handle in self._expert_handles.values():
            handle.close()
        if self._vq_codebook_pool is not None:
            self._vq_codebook_pool.close()
        if self._mtp is not None:
            self._mtp.close()

    def release_ram_blobs(self) -> None:
        """Detach SafeFile views after all permanent GPU weights are ready."""
        self._dense.release_ram_blob()
        for handle in self._expert_handles.values():
            handle.release_ram_blob()
        if self._vq_codebook_pool is not None:
            self._vq_codebook_pool.release_ram_blob()
        if self._mtp is not None:
            self._mtp.release_ram_blob()

    def release_dense_ram_blob(self) -> tuple[int, tuple[str, ...]]:
        """Detach only Dense RAM images after CUDA weights become permanent.

        Packed expert views are intentionally untouched: RAM+GPU inference
        may still use them as the source of asynchronous expert DMA.
        """
        paths = self._dense.ram_blob_paths()
        return self._dense.release_ram_blob(), paths

    def get_mtp(self, name: str):
        """MTP dense 权重：attn.* 为 int4 对，router 等小张量 f32 原样。"""
        assert self._mtp is not None, "模型目录无 MTP 附件（mtp.safetensors）"
        keys = set(self._mtp.keys())
        if name + ".qs" in keys:
            q = self._mtp.get_tensor(name)
            s = self._mtp.get_tensor(name + ".qs")
            return Int4Weight(q, s, q.shape[1] * 2, self.man.int4_group)
        return self._mtp.get_tensor(name).float()

    # ---- dense ----
    def has(self, name: str) -> bool:
        return name in self._dense_keys

    def dense_names(self) -> list[str]:
        """全部 dense 权重名（不含 .qs 缩放键）。"""
        return sorted(
            name
            for name in self._dense_keys
            if not name.endswith(".qs")
            and not (
                name.endswith(".scale")
                and name[: -len("scale")] + "weight"
                in self._dense_keys
            )
        )

    def get_raw(self, name: str) -> torch.Tensor:
        return self._dense.get_tensor(name)

    def dense_nbytes(self, name: str) -> int:
        """Return one dense tensor's stored bytes without reading its payload."""
        return self._dense.nbytes(name)

    def dense_resident_nbytes(self, name: str) -> int:
        """Return decoded in-memory bytes used by placement planning."""
        return self._dense.resident_nbytes(name)

    def get_dense(self, name: str):
        """返回 f32 张量（小权重）或 Int4Weight（打包大权重）。"""
        if name + ".qs" in self._dense_keys:
            q = self._dense.get_tensor(name)
            s = self._dense.get_tensor(name + ".qs")
            if self.man.quant.get("dense") == "fp8-native":
                return BlockFP8Weight(q, s, q.shape[1])
            return Int4Weight(q, s, q.shape[1] * 2, self.man.int4_group)
        audited_fp8 = self._dense.get_block_fp8(name)
        if audited_fp8 is not None:
            return audited_fp8
        value = self._dense.get_tensor(name)
        if self.man.dense_audit_file is not None:
            # Audited mixed Dense formats already declare the exact logical
            # dtype per tensor. New format names must not fall through to the
            # legacy unconditional FP32 conversion.
            return value
        if self.man.quant.get("dense") in (
            "source-native-uncompressed",
            "mixed-source-fp8-d3-p12",
        ):
            return value
        return value.float()

    # ---- 专家 ----
    def _eh(self, layer: int) -> SafeFile:
        h = self._expert_handles.get(layer)
        if h is None:
            # get_many can request several experts from a newly opened layer
            # concurrently.  Publish the handle and its key set atomically.
            with self._expert_open_lock:
                h = self._expert_handles.get(layer)
                if h is None:
                    h = SafeFile(
                        os.path.join(
                            self.root,
                            self.man.expert_files[layer],
                        )
                    )
                    keys = set(h.keys())
                    self._expert_handles[layer] = h
                    self._expert_keys[layer] = keys
        return h

    def expert_kind(self, layer: int, eid: int) -> str:
        """探测专家档位：返回 VQ 档名（可带 z 后缀）或 ``drop``。

        ``p12`` 是 k=4096 索引的磁盘紧凑编码，不属于新的计算档位，所以
        对上层仍报告原档名，加载时再透明解包。
        """
        keys = self._expert_keys.get(layer)
        if keys is None:
            self._eh(layer)
            keys = self._expert_keys[layer]
        if self.man.projection_vq:
            layouts = self.man.projection_layouts(layer, eid)
            if all(
                f"e{eid}.{projection}.{layouts[projection]}" in keys
                for projection in ("gate", "up", "down")
            ):
                return "projection-vq"
            return "drop"
        for k in self.man.vq_dims:
            if (
                f"e{eid}.gu.{k}" in keys
                and f"e{eid}.down.{k}" in keys
            ):
                return k
            if f"e{eid}.gu{k}p14z" in keys:
                return k + "z"
            if f"e{eid}.gu{k}p14" in keys:
                return k
            if f"e{eid}.gu{k}p12z" in keys:
                return k + "z"
            if f"e{eid}.gu{k}p12" in keys:
                return k
            if f"e{eid}.gu{k}z" in keys:
                return k + "z"
            if f"e{eid}.gu{k}" in keys:
                return k
        return "drop"

    def available_mask(self, layer: int) -> torch.Tensor:
        """该层可用专家布尔掩码 [E]（drop 为 False），用于回退路由掩码。"""
        E = self.cfg["n_experts"]
        s = self.man.tier_string(layer)
        if s is not None:
            if len(s) != E:
                raise ValueError(
                    f"L{layer} tier_string 长度 {len(s)} != n_experts {E}"
                )
            return torch.tensor(
                [c.lower() != "d" for c in s],
                dtype=torch.bool,
            )
        return torch.ones(E, dtype=torch.bool)  # 清单无档位串 = 全保留（老产物）

    @staticmethod
    def _down_codebook_stem(
        keys: set[str],
        kind: str,
    ) -> str:
        standard = f"cb.down.{kind}"
        return (
            standard
            if any(
                key == standard or key.startswith(standard + ".")
                for key in keys
            )
            else f"cb.dn.{kind}"
        )

    def _codebook_reference(
        self,
        layer: int,
        kind: str,
        eid: int | None,
        projection: str,
    ) -> tuple[SafeFile, str, str]:
        """按专属→专家分配→连续分组→共享解析一个投影的码本。"""
        if projection not in ("gu", "down"):
            raise ValueError(projection)
        self._eh(layer)
        keys = self._expert_keys[layer]
        stem = (
            f"cb.gu.{kind}"
            if projection == "gu"
            else self._down_codebook_stem(keys, kind)
        )
        if (
            eid is not None
            and f"{stem}.e{eid}" in keys
        ):
            key = f"{stem}.e{eid}"
            return (
                self._expert_handles[layer],
                key,
                f"L{layer}.e{eid}",
            )
        layout = self.man.vq_codebook_layout
        if (
            self.man.vq_codebook_layout_format
            == "expert-assigned-codebook-v1"
            and kind == str(layout.get("kind", "v"))
            and eid is not None
        ):
            assignment = self._vq_assignments[projection]
            row = layer - 1
            if 0 <= row < assignment.shape[0]:
                codebook = int(assignment[row, eid])
                sentinel = int(
                    layout.get("missing_assignment_sentinel", 255)
                )
                if codebook != sentinel:
                    band_size = int(layout["layer_band_size"])
                    band = row // band_size
                    key = (
                        f"cb.{kind}.band{band:02d}."
                        f"{projection}.{codebook:03d}"
                    )
                    if key not in self._vq_codebook_pool_keys:
                        raise KeyError(
                            "专家分配表引用了不存在的码本: "
                            f"L{layer} e{eid} {projection} -> {key}"
                        )
                    assert self._vq_codebook_pool is not None
                    semantic = f"band{band:02d}.cb{codebook:03d}"
                    return self._vq_codebook_pool, key, semantic
                if not bool(layout.get("legacy_fallback", True)):
                    raise KeyError(
                        f"L{layer} e{eid} {projection} 无多码本分配"
                    )
        group_size = self.man.vq_codebook_group_sizes.get(kind)
        if group_size is not None and eid is not None:
            variant = f"g{eid // group_size:03d}"
            key = f"{stem}.{variant}"
            if key in keys:
                return (
                    self._expert_handles[layer],
                    key,
                    f"L{layer}.{variant}",
                )
        if stem not in keys:
            raise KeyError(
                f"L{layer} e{eid} {projection} 缺少可用码本: {stem}"
            )
        return (
            self._expert_handles[layer],
            stem,
            f"L{layer}.shared",
        )

    def codebook_variants(
        self,
        layer: int,
        kind: str,
        eid: int | None,
    ) -> tuple[str, str]:
        """返回GU/Down稳定语义键，用于RAM/GPU码本缓存。"""
        gu = self._codebook_reference(
            layer, kind, eid, "gu"
        )[2]
        down = self._codebook_reference(
            layer, kind, eid, "down"
        )[2]
        return gu, down

    def codebook_variant(
        self,
        layer: int,
        kind: str,
        eid: int | None,
    ) -> str:
        """旧调用兼容；仅当GU/Down选择相同时返回一个语义键。"""
        gu, down = self.codebook_variants(layer, kind, eid)
        return gu if gu == down else f"{gu}|{down}"

    def codebooks(
        self,
        layer: int,
        kind: str,
        eid: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """依次选择专家专属、自适应分配、连续分组、层共享码本。"""
        gu_handle, gu_key, gu_variant = self._codebook_reference(
            layer, kind, eid, "gu"
        )
        (
            down_handle,
            down_key,
            down_variant,
        ) = self._codebook_reference(
            layer, kind, eid, "down"
        )
        with self._cb_lock:
            gu_cache_key = (gu_handle.path, "gu", gu_key)
            cb_gu = self._cb_cache.get(gu_cache_key)
            if cb_gu is None:
                cb_gu = gu_handle.get_tensor(gu_key).float()
                self._cb_cache[gu_cache_key] = cb_gu
            down_cache_key = (
                down_handle.path,
                "down",
                down_key,
            )
            cb_down = self._cb_cache.get(down_cache_key)
            if cb_down is None:
                cb_down = down_handle.get_tensor(down_key).float()
                self._cb_cache[down_cache_key] = cb_down
        return cb_gu, cb_down

    def _projection_codebook_key(
        self,
        layer: int,
        projection: str,
        eid: int | None,
    ) -> str:
        layout = self.man.projection_layouts(layer, eid)[projection]
        key = f"cb.{projection}.{layout}"
        group_size = self.man.projection_codebook_group_sizes.get(layout)
        if group_size is None:
            return key
        if eid is None:
            raise ValueError(
                f"L{layer} {projection} layout {layout} requires expert_id"
            )
        group = int(eid) // group_size
        groups = self.man.projection_codebook_group_counts.get(layout)
        if group < 0 or (groups is not None and group >= groups):
            raise ValueError(
                f"L{layer} e{eid} {projection} codebook group {group} "
                f"is outside layout {layout}"
            )
        return f"{key}.g{group:03d}"

    def projection_codebooks(
        self,
        layer: int,
        eid: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """读取三投影独立码本，不把 gate/up 强行拼成同一码本。"""
        if not self.man.projection_vq:
            raise RuntimeError("当前模型不是三投影 VQ 格式")
        handle = self._eh(layer)
        result = []
        with self._cb_lock:
            for projection in ("gate", "up", "down"):
                key = self._projection_codebook_key(
                    layer,
                    projection,
                    eid,
                )
                cache_key = (handle.path, projection, key)
                codebook = self._cb_cache.get(cache_key)
                if codebook is None:
                    codebook = handle.get_tensor(key).float()
                    self._cb_cache[cache_key] = codebook
                result.append(codebook)
        return tuple(result)

    def projection_codebook_variants(
        self,
        layer: int,
        eid: int | None = None,
    ) -> tuple[str, str, str]:
        """返回稳定的三投影码本语义键，供 RAM/VRAM 缓存隔离。"""
        return tuple(
            f"L{layer}." + self._projection_codebook_key(
                layer,
                projection,
                eid,
            )[3:]
            for projection in ("gate", "up", "down")
        )

    def load_expert(self, layer: int, eid: int) -> tuple[VQWeight, VQWeight]:
        """加载一个专家的 (gu, dn) VQ 权重；zlib blob 就地解压。

        注意：z 后缀在量化端是按张量独立决定的（gu/dn 可能一个压缩一个原始），
        这里逐张量探测。
        """
        kind = self.expert_kind(layer, eid)
        if kind == "drop":
            raise KeyError(f"expert {layer}/{eid} 已被量化丢弃")
        base = kind.rstrip("z")
        dim, _ = self.man.vq_dims[base]
        cb_gu, cb_dn = self.codebooks(layer, base, eid)
        h = self._eh(layer)
        keys = self._expert_keys[layer]
        H = self.cfg.get("routed_hidden", self.cfg["hidden"])
        I = self.cfg["moe_inter"]

        def _idx(tag: str, rows: int, cols: int) -> torch.Tensor:
            count = rows * (cols // dim)
            standard_tag = "down" if tag == "dn" else tag
            standard_key = f"e{eid}.{standard_tag}.{base}"
            if standard_key in keys:
                stored = h.get_tensor(standard_key)
                raw = stored.view(torch.uint8).reshape(-1)
                bits = _stored_index_bits(raw.numel(), count)
                if bits == 8:
                    return raw.reshape(rows, cols // dim)
                if bits == 16:
                    return raw.view(torch.uint16).reshape(
                        rows,
                        cols // dim,
                    )
                if bits == 9:
                    unpacked = _unpack_u9(raw, count)
                elif bits == 10:
                    unpacked = _unpack_u10(raw, count)
                elif bits == 12:
                    unpacked = _unpack_u12(raw, count)
                elif bits == 14:
                    unpacked = _unpack_u14(raw, count)
                else:
                    unpacked = _unpack_odd_width(raw, count, bits)
                return unpacked.reshape(rows, cols // dim)
            p14zkey = f"e{eid}.{tag}{base}p14z"
            p14key = f"e{eid}.{tag}{base}p14"
            if p14zkey in keys:
                raw = zlib.decompress(h.get_bytes(p14zkey))
                packed = torch.frombuffer(raw, dtype=torch.uint8)
                return _unpack_u14(packed, count).reshape(rows, cols // dim)
            if p14key in keys:
                packed = h.get_tensor(p14key)
                return _unpack_u14(packed, count).reshape(rows, cols // dim)
            p12zkey = f"e{eid}.{tag}{base}p12z"
            p12key = f"e{eid}.{tag}{base}p12"
            if p12zkey in keys:
                raw = zlib.decompress(h.get_bytes(p12zkey))
                packed = torch.frombuffer(raw, dtype=torch.uint8)
                return _unpack_u12(packed, count).reshape(rows, cols // dim)
            if p12key in keys:
                packed = h.get_tensor(p12key)
                return _unpack_u12(packed, count).reshape(rows, cols // dim)
            zkey = f"e{eid}.{tag}{base}z"
            if zkey in keys:
                # get_bytes 纯文件读（无 mmap），zlib 解压后即为索引字节
                raw = zlib.decompress(h.get_bytes(zkey))
                # k>256 的档索引为 u16（如 w=8D-k4096），其余 u8
                idt = torch.uint16 if self.man.vq_dims[base][1] > 256 else torch.uint8
                a = torch.frombuffer(raw, dtype=idt)  # bytes 直视图，免拷贝
                return a.reshape(rows, cols // dim)
            return h.get_tensor(f"e{eid}.{tag}{base}")

        return (VQWeight(_idx("gu", 2 * I, H), cb_gu, H),
                VQWeight(_idx("dn", H, I), cb_dn, I))

    def load_expert_packed(
        self,
        layer: int,
        eid: int,
    ) -> tuple["PackedVQWeight", ...]:
        """Load an expert without expanding packed on-disk indices.

        Projection archives use row-aligned p9/p10/p12/p14 packing. Expanding
        these tensors to uint16 makes full-VRAM residency impractical. The
        packed representation is consumed directly by the common CPU/CUDA VQ
        operators; legacy ``load_expert`` remains unchanged.
        """
        if self.man.projection_vq:
            handle = self._eh(layer)
            layouts = self.man.projection_layouts(layer, eid)
            codebooks = self.projection_codebooks(layer, eid)
            hidden = int(
                self.cfg.get("routed_hidden", self.cfg["hidden"])
            )
            intermediate = int(self.cfg["moe_inter"])
            shapes = {
                "gate": (intermediate, hidden),
                "up": (intermediate, hidden),
                "down": (hidden, intermediate),
            }
            weights = []
            for projection, codebook in zip(
                ("gate", "up", "down"),
                codebooks,
            ):
                layout_name = layouts[projection]
                dim, _codebook_size = self.man.vq_dims[layout_name]
                rows, cols = shapes[projection]
                key = f"e{eid}.{projection}.{layout_name}"
                stored = (
                    handle.get_tensor(key)
                    .view(torch.uint8)
                    .reshape(-1)
                )
                count = rows * (cols // dim)
                bits = _stored_index_bits(stored.numel(), count)
                declared = self.man.index_packing.get(layout_name)
                packing_widths = {
                    f"packed-u{width}": width
                    for width in range(8, 17)
                }
                packing_widths.update({"u8": 8, "u16": 16})
                expected = packing_widths.get(declared)
                if expected is not None and bits != expected:
                    raise ValueError(
                        f"{key} 索引位宽与清单不符: "
                        f"{bits} != {expected}"
                    )
                weights.append(
                    PackedVQWeight(
                        stored,
                        codebook,
                        rows,
                        cols,
                        bits,
                    )
                )
            return tuple(weights)

        kind = self.expert_kind(layer, eid)
        if kind == "drop":
            raise KeyError(f"expert {layer}/{eid} 已被量化丢弃")
        base = kind.rstrip("z")
        dim, codebook_size = self.man.vq_dims[base]
        cb_gu, cb_dn = self.codebooks(layer, base, eid)
        handle = self._eh(layer)
        keys = self._expert_keys[layer]
        hidden = int(self.cfg.get("routed_hidden", self.cfg["hidden"]))
        intermediate = int(self.cfg["moe_inter"])

        def packed(
            tag: str,
            rows: int,
            cols: int,
        ) -> PackedVQWeight:
            standard_tag = "down" if tag == "dn" else tag
            standard_key = f"e{eid}.{standard_tag}.{base}"
            if standard_key in keys:
                storage = (
                    handle.get_tensor(standard_key)
                    .view(torch.uint8)
                    .reshape(-1)
                )
                bits = _stored_index_bits(
                    storage.numel(),
                    rows * (cols // dim),
                )
                return PackedVQWeight(
                    storage,
                    cb_gu if tag == "gu" else cb_dn,
                    rows,
                    cols,
                    bits,
                )
            for bits in (14, 12):
                stem = f"e{eid}.{tag}{base}p{bits}"
                if stem + "z" in keys:
                    raw = zlib.decompress(handle.get_bytes(stem + "z"))
                    storage = torch.frombuffer(raw, dtype=torch.uint8)
                    return PackedVQWeight(
                        storage,
                        cb_gu if tag == "gu" else cb_dn,
                        rows,
                        cols,
                        bits,
                    )
                if stem in keys:
                    storage = (
                        handle.get_tensor(stem)
                        .view(torch.uint8)
                        .reshape(-1)
                    )
                    return PackedVQWeight(
                        storage,
                        cb_gu if tag == "gu" else cb_dn,
                        rows,
                        cols,
                        bits,
                    )

            stem = f"e{eid}.{tag}{base}"
            dtype = torch.uint16 if codebook_size > 256 else torch.uint8
            if stem + "z" in keys:
                raw = zlib.decompress(handle.get_bytes(stem + "z"))
                indices = torch.frombuffer(raw, dtype=dtype)
            else:
                indices = handle.get_tensor(stem)
            return PackedVQWeight(
                indices.view(torch.uint8).reshape(-1),
                cb_gu if tag == "gu" else cb_dn,
                rows,
                cols,
                16 if dtype == torch.uint16 else 8,
            )

        return (
            packed("gu", 2 * intermediate, hidden),
            packed("dn", hidden, intermediate),
        )


class PackedVQWeight:
    """Byte-exact VQ indices plus logical matrix metadata.

    ``bits`` is 8/16 for ordinary indices and 9..15 for row-aligned
    packed indices.  Only the byte payload is staged to CUDA; no dequantized
    expert matrix and no expanded uint16 copy is created.
    """

    __slots__ = (
        "raw",
        "cb",
        "rows",
        "cols",
        "blocks",
        "dim",
        "bits",
        "source_bits",
        "layout",
    )

    def __init__(
        self,
        raw: torch.Tensor,
        cb: torch.Tensor,
        rows: int,
        cols: int,
        bits: int,
    ):
        if not 8 <= bits <= 16:
            raise ValueError(f"unsupported packed VQ width {bits}")
        self.raw = raw.contiguous().view(torch.uint8).reshape(-1)
        self.cb = cb.float()
        self.rows = int(rows)
        self.cols = int(cols)
        self.dim = int(cb.shape[1])
        if self.cols % self.dim:
            raise ValueError("VQ columns must be divisible by code dimension")
        self.blocks = self.cols // self.dim
        self.bits = int(bits)
        self.source_bits = int(bits)
        self.layout = "row-major"
        expected_bits = self.rows * self.blocks * self.bits
        if expected_bits % 8:
            raise ValueError(
                "packed VQ tensor must be byte aligned across complete rows"
            )
        expected = expected_bits // 8
        if self.raw.numel() != expected:
            raise ValueError(
                f"packed VQ payload mismatch: {self.raw.numel()} != {expected}"
            )

    @property
    def nbytes(self) -> int:
        return self.raw.numel()

    @property
    def dtype_tag(self) -> int:
        return {
            8: 0,
            16: 1,
            12: 2,
            14: 3,
            10: 4,
            9: 5,
            11: 6,
            13: 7,
            15: 8,
        }[
            self.bits
        ]

    def optimize_cpu_layout(self) -> bool:
        """Replace row-major indices with compact block-major traversal.

        The byte count and index width stay identical.  This is a CPU-only
        storage transform; CUDA transport continues to use the archive's
        original row-major representation unless explicitly requested.
        """
        if self.layout == "block-major":
            return True
        from .ops import vq_relayout_block_major

        packed = vq_relayout_block_major(
            self.raw,
            rows=self.rows,
            blocks=self.blocks,
            bits=self.bits,
            code_dim=self.dim,
            codebook_size=int(self.cb.shape[0]),
        )
        if packed is None:
            return False
        if packed.numel() != self.raw.numel():
            raise RuntimeError("compact VQ relayout changed payload size")
        self.raw = packed
        self.layout = "block-major"
        return True

    def optimize_cpu_row_tile(self, tile_rows: int = 8) -> bool:
        """Replace row-major indices with compact CPU row-tile traversal."""
        if self.layout == f"row-tile-{int(tile_rows)}":
            return True
        if self.layout != "row-major":
            return False
        from .ops import vq_relayout_row_tile

        packed = vq_relayout_row_tile(
            self.raw,
            rows=self.rows,
            blocks=self.blocks,
            bits=self.bits,
            code_dim=self.dim,
            codebook_size=int(self.cb.shape[0]),
            tile_rows=int(tile_rows),
        )
        if packed is None:
            return False
        if packed.numel() != self.raw.numel():
            raise RuntimeError("compact VQ row-tile changed payload size")
        self.raw = packed
        self.layout = f"row-tile-{int(tile_rows)}"
        return True

    def compile_cpu_u16_row_tile(self, tile_rows: int = 8) -> bool:
        """Compile packed bytes into an exact runtime-only CPU image."""
        if self.layout == f"u16-row-tile-{int(tile_rows)}":
            return True
        if self.layout != "row-major" or int(tile_rows) != 8:
            return False
        from .ops import vq_compile_u16_row_tile

        compiled = vq_compile_u16_row_tile(
            self.raw,
            rows=self.rows,
            blocks=self.blocks,
            bits=self.bits,
            code_dim=self.dim,
            codebook_size=int(self.cb.shape[0]),
            tile_rows=int(tile_rows),
        )
        if compiled is None:
            return False
        if compiled.dtype != torch.uint16 or compiled.numel() != (
            self.rows * self.blocks
        ):
            raise RuntimeError("CPU VQ compilation returned an invalid image")
        self.source_bits = int(self.bits)
        self.raw = compiled.contiguous().view(torch.uint8).reshape(-1)
        self.bits = 16
        self.layout = f"u16-row-tile-{int(tile_rows)}"
        return True

    def unpack(self) -> torch.Tensor:
        """Reference unpacker used by CPU tests and correctness probes."""
        count = self.rows * self.blocks
        if self.bits == 8:
            result = self.raw
        elif self.bits == 16:
            result = self.raw.view(torch.uint16)
        elif self.bits == 9:
            result = _unpack_u9(self.raw, count)
        elif self.bits == 10:
            result = _unpack_u10(self.raw, count)
        elif self.bits == 12:
            result = _unpack_u12(self.raw, count)
        elif self.bits == 14:
            result = _unpack_u14(self.raw, count)
        else:
            result = _unpack_odd_width(self.raw, count, self.bits)
        physical = result.reshape(-1)
        if self.layout == "row-major":
            return physical.reshape(self.rows, self.blocks)
        if self.layout == "block-major":
            return physical.reshape(self.blocks, self.rows).t().contiguous()
        if self.layout in ("row-tile-8", "u16-row-tile-8"):
            logical = torch.empty(
                self.rows,
                self.blocks,
                dtype=physical.dtype,
                device=physical.device,
            )
            for first_row in range(0, self.rows, 8):
                valid = min(8, self.rows - first_row)
                start = first_row * self.blocks
                stop = start + self.blocks * valid
                logical[first_row : first_row + valid].copy_(
                    physical[start:stop].reshape(self.blocks, valid).t()
                )
            return logical
        raise ValueError(f"unsupported packed VQ layout {self.layout!r}")


class PackedCpuExpertPool:
    """Generic CPU LRU that keeps VQ expert indices byte-packed.

    The pool deliberately mirrors only the small subset of ``ExpertPool`` used
    by CPU decode. Packed payloads stay compact in RAM; the common CPU VQ
    backend extracts indices while computing and never creates a resident
    uint16 expansion.
    """

    full_resident = False
    prefetch_default = True
    expanded_index_bytes = 0

    def __init__(self, store: TPQStore, budget_gb: float = 16.0):
        self.store = store
        self.device = torch.device("cpu")
        self.gpu = False
        self.budget = max(0, int(float(budget_gb) * 2**30))
        self.cache: OrderedDict[
            tuple[int, int],
            tuple[PackedVQWeight, ...],
        ] = OrderedDict()
        self.pinned: dict[
            tuple[int, int],
            tuple[PackedVQWeight, ...],
        ] = {}
        self.bytes = 0
        self.compact_full_resident = False
        self.hits = 0
        self.miss = 0
        self._pending: OrderedDict = OrderedDict()
        self._native_layers: dict[int, object | bool] = {}
        self.native_hits = 0
        self.native_fallbacks = 0
        self.block_major_entries = 0
        self.block_major_bytes = 0
        self.compiled_index_bytes = 0
        self.compiled_source_bytes = 0
        self.cpu_compile_mode = "off"

    @property
    def host_expert_bytes(self) -> int:
        """Return the compact resident/LRU payload footprint."""
        return int(self.bytes)

    @property
    def compact_resident_entries(self) -> int:
        return len(self.pinned)

    @staticmethod
    def _entry_bytes(entry) -> int:
        return sum(int(weight.nbytes) for weight in entry)

    def _put(self, key, entry) -> None:
        size = self._entry_bytes(entry)
        old = self.cache.pop(key, None)
        if old is not None:
            self.bytes -= self._entry_bytes(old)
        while self.cache and self.bytes + size > self.budget:
            _, victim = self.cache.popitem(last=False)
            self.bytes -= self._entry_bytes(victim)
        if size <= self.budget:
            self.cache[key] = entry
            self.bytes += size

    def preload_all(self, reserve_gb: float | None = None) -> bool:
        if os.environ.get("TPQ_FULL_RESIDENT", "1") == "0":
            return False
        import psutil
        from concurrent.futures import FIRST_COMPLETED, wait

        if reserve_gb is None:
            reserve_gb = float(
                os.environ.get("TPQ_RESIDENT_RESERVE_GB", "3.0")
            )
        native_expert_bytes = getattr(self.store, "expert_bytes", None)
        if native_expert_bytes is None:
            total = sum(
                os.path.getsize(os.path.join(self.store.root, filename))
                for filename in self.store.man.expert_files.values()
                if os.path.exists(os.path.join(self.store.root, filename))
            )
        else:
            total = int(native_expert_bytes)
        available = int(psutil.virtual_memory().available)
        compile_mode = os.environ.get(
            "TPQ_CPU_COMPILE", "off"
        ).strip().lower()
        if compile_mode not in {"0", "off", "false", "auto", "u16"}:
            raise ValueError("TPQ_CPU_COMPILE must be off, auto, or u16")
        compile_enabled = compile_mode in {"auto", "u16"}
        compiled_upper_bound = total * 2
        compiled_need = compiled_upper_bound + int(reserve_gb * 2**30)
        if compile_enabled and compiled_need > available:
            if compile_mode == "u16":
                raise MemoryError(
                    "forced CPU VQ compilation cannot fit: "
                    f"upper bound {compiled_upper_bound / 2**30:.1f}GiB + "
                    f"reserve {reserve_gb:.1f}GiB > available "
                    f"{available / 2**30:.1f}GiB"
                )
            compile_enabled = False
            print(
                "[tpq] CPU 在线编译自动回退紧凑索引："
                f"上界 {compiled_upper_bound / 2**30:.1f}GiB + "
                f"预留 {reserve_gb:.1f}GiB > 可用 "
                f"{available / 2**30:.1f}GiB",
                flush=True,
            )
        if total + int(reserve_gb * 2**30) > available:
            print(
                "[tpq] packed CPU专家无法全量常驻："
                f"文件约 {total / 2**30:.1f}GiB + 预留 {reserve_gb:.1f}GiB"
                f" > 可用 {available / 2**30:.1f}GiB，回退紧凑LRU",
                flush=True,
            )
            return False
        n_experts = int(self.store.cfg["n_experts"])
        keys = [
            (int(layer), expert)
            for layer in self.store.man.expert_files
            for expert in range(n_experts)
            if self.store.expert_kind(int(layer), expert) != "drop"
        ]
        started = time.time()
        print(
            f"[tpq] packed CPU专家全量常驻：{len(keys)} 个读取中…",
            flush=True,
        )
        resident_bytes = 0
        layout_mode = os.environ.get(
            "TPQ_CPU_PACKED_LAYOUT", "off"
        ).strip().lower()
        self.cpu_compile_mode = "u16" if compile_enabled else "off"
        executor = _executor()
        key_iterator = iter(keys)
        pending = {}
        window = max(4, int(os.environ.get("TPQ_CPU_LOAD_WINDOW", "32")))

        def submit_one() -> bool:
            try:
                key = next(key_iterator)
            except StopIteration:
                return False
            pending[executor.submit(self.store.load_expert_packed, *key)] = key
            return True

        for _ in range(min(window, len(keys))):
            submit_one()
        index = 0
        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                key = pending.pop(future)
                entry = future.result()
                index += 1
                if compile_enabled:
                    for weight in entry:
                        source_bytes = int(weight.nbytes)
                        if not weight.compile_cpu_u16_row_tile(8):
                            raise RuntimeError(
                                "CPU VQ compilation unavailable for "
                                f"layer/expert {key}"
                            )
                        self.compiled_source_bytes += source_bytes
                        self.compiled_index_bytes += int(weight.nbytes)
                elif layout_mode in {"tile8", "row-tile", "row_tile"}:
                    for projection, weight in enumerate(entry):
                        if projection >= 2:
                            continue
                        if weight.optimize_cpu_row_tile(8):
                            self.block_major_entries += 1
                            self.block_major_bytes += int(weight.nbytes)
                elif layout_mode not in {"0", "off", "false", "row"}:
                    for projection, weight in enumerate(entry):
                        if (
                            projection < 2
                            and weight.dim == 4
                            and int(weight.cb.shape[0]) <= 4096
                            and weight.optimize_cpu_layout()
                        ):
                            self.block_major_entries += 1
                            self.block_major_bytes += int(weight.nbytes)
                self.pinned[key] = entry
                resident_bytes += self._entry_bytes(entry)
                if index % 2000 == 0:
                    print(
                        f"[tpq] packed CPU专家常驻 {index}/{len(keys)}",
                        flush=True,
                    )
                submit_one()
        self.bytes = resident_bytes
        self.expanded_index_bytes = int(self.compiled_index_bytes)
        self.compact_full_resident = True
        if compile_enabled or layout_mode in {"tile8", "row-tile", "row_tile"}:
            # Relayout replaces each source tensor with an equal-size compact
            # tensor.  Release completed Future references and return the old
            # byte buffers to the OS instead of leaving one model-sized copy
            # in the glibc arena.
            pending.clear()
            future = None
            import ctypes
            import gc

            gc.collect()
            malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim(0)
        print(
            "[tpq] packed CPU专家常驻完成："
            f"{len(keys)} 个 / {resident_bytes / 2**30:.1f}GiB / "
            f"{time.time() - started:.1f}s；"
            f"cpu_compile={self.cpu_compile_mode}；"
            f"compiled_index_bytes={self.compiled_index_bytes}",
            flush=True,
        )
        return True

    def preload_pinned(self) -> None:
        # Heat-ranked residency remains an optimization for the staged CUDA
        # pool. The CPU packed LRU already uses the full automatic RAM budget.
        return None

    def pin_host_resident(self, budget_gb: float | None = None) -> float:
        return 0.0

    def build_gpu_arenas(self) -> float:
        return 0.0

    def get_many(self, keys: list[tuple[int, int]]) -> dict:
        from concurrent.futures import as_completed

        output = {}
        missing = []
        for key in keys:
            entry = self.pinned.get(key)
            if entry is not None:
                self.hits += 1
                output[key] = entry
                continue
            entry = self.cache.get(key)
            if entry is not None:
                self.hits += 1
                self.cache.move_to_end(key)
                output[key] = entry
                continue
            missing.append(key)
        futures = {}
        for key in missing:
            future = self._pending.pop(key, None)
            if future is None:
                future = _executor().submit(
                    self.store.load_expert_packed,
                    *key,
                )
            futures[future] = key
        for future in as_completed(futures):
            key = futures[future]
            entry = future.result()
            self.miss += 1
            self._put(key, entry)
            output[key] = entry
        return output

    def prefetch(self, keys: list[tuple[int, int]]) -> None:
        while len(self._pending) > 256:
            _, future = self._pending.popitem(last=False)
            future.cancel()
        for key in keys:
            if (
                key in self.pinned
                or key in self.cache
                or key in self._pending
            ):
                continue
            self._pending[key] = _pf_executor().submit(
                self.store.load_expert_packed,
                *key,
            )

    def run_native(
        self,
        layer: int,
        value: torch.Tensor,
        expert_ids: torch.Tensor,
        route_weights: torch.Tensor,
        *,
        activation: str,
        activation_beta: float,
        activation_linear_beta: float | None,
        limit: float = 0.0,
    ) -> torch.Tensor | None:
        """Run one full-resident layer through the common native directory.

        This removes the per-token Python expert-list reconstruction while
        preserving the exact packed tensors held by ``pinned``.  Mixed
        codebook routes deliberately return ``None`` and use the existing
        registered fallback.
        """
        if not self.compact_full_resident:
            return None
        cached = self.native_layer(int(layer))
        if cached is None:
            return None
        output = cached.forward(
            value.float().contiguous(),
            expert_ids,
            route_weights.float().contiguous(),
            float(limit),
            str(activation).strip().lower(),
            float(activation_beta),
            (
                -1.0
                if activation_linear_beta is None
                else float(activation_linear_beta)
            ),
        )
        if output.numel():
            self.native_hits += 1
            return output
        self.native_fallbacks += 1
        return None

    def native_layer(self, layer: int):
        """Return one cached format-driven compact resident executor."""
        if not self.compact_full_resident:
            return None
        cached = self._native_layers.get(int(layer))
        if cached is None:
            n_experts = int(self.store.cfg["n_experts"])
            entries = tuple(
                self.pinned.get((int(layer), expert))
                for expert in range(n_experts)
            )
            if any(entry is None or len(entry) != 3 for entry in entries):
                self._native_layers[int(layer)] = False
                return None
            from .cpuext import make_packed_three_layer_cpu

            cached = make_packed_three_layer_cpu(entries)
            self._native_layers[int(layer)] = cached or False
        if cached is False:
            return None
        return cached


class ExpertPool:
    """专家缓存池（两级）：计算设备缓存 + 可选内存前级缓存。

    CPU 模式：单级内存缓存（budget_gb），未命中从磁盘加载。
    GPU 模式：显存主缓存（budget_gb）+ 内存前级缓存（ram_gb，远大于显存），
    显存未命中优先从内存前级上传（PCIe 快），前级也未命中才读磁盘。
    两级均以 VQ 索引态驻留（v 档 ≈9.4MB/专家，w 档 ≈4.7MB），LRU 驱逐。
    """

    def __init__(self, store: TPQStore, budget_gb: float = 16.0, device: str = "cpu",
                 ram_gb: float = 0.0, pin_gb: float = 0.0):
        self.store = store
        self.device = torch.device(device)
        self.gpu = self.device.type != "cpu"
        self.budget = int(budget_gb * 2**30)
        self.ram_budget = int(ram_gb * 2**30)
        self.cache: OrderedDict[tuple[int, int], tuple[VQWeight, VQWeight]] = OrderedDict()
        self.ram: OrderedDict[tuple[int, int], tuple[VQWeight, VQWeight]] = OrderedDict()
        self.bytes = 0
        self.ram_bytes = 0
        self.hits = 0
        self.miss = 0
        self.stage = PinnedStage(self.device) if self.gpu else None
        self._stage_dirty = False   # 有预取 DMA 在飞（get/get_many 命中路径需先 wait）
        self._inflight: set = set()  # DMA 在飞的 cache key（落地前禁止驱逐：
        #   驱逐会释放目标显存被分配器复用，而在飞 DMA 继续写 → 随机覆写 KV/权重，
        #   曾致 prefill 后 hidden 全零、logits 全等 argmax 恒 0）
        self._pending: dict = {}    # 后台磁盘加载（预取软提示）：key → Future
        from collections import deque
        import threading
        self._recent: deque = deque(maxlen=256)   # 近期命中统计（0=hit 1=miss），预取自适应
        self._stage_lock = threading.RLock()      # staging/缓存突变互斥（后台 staging 线程安全）
        self._staging: dict = {}                  # 正在后台 staged 的 key → threading.Event
        self._gpu_arenas: GpuExpertArenas | None = None
        self._rebuilding_arenas = False
        # 热专家钉住区（永不驱逐）：按 profile 热度 top-N 准入，LRU 池只服务冷专家
        self.pinned: dict[tuple[int, int], tuple[VQWeight, VQWeight]] = {}
        self._pin_sets: dict[int, set[int]] = {}
        if pin_gb > 0 and store.heat_ranks:
            H = store.cfg.get("routed_hidden", store.cfg["hidden"])
            I = store.cfg["moe_inter"]
            est = 3 * I * H // 4  # v 档索引字节（上界）
            n_layers = max(1, len(store.man.expert_files))
            pin_n = int(pin_gb * 2**30 // (est * n_layers))
            if pin_n > 0:
                self._pin_sets = {l: set(r[:pin_n]) for l, r in store.heat_ranks.items()}
                print(f"[tpq] 热专家钉住: top-{pin_n}/层 ≈{pin_gb:.0f}GB", flush=True)

    def _hot(self, layer: int, eid: int) -> bool:
        s = self._pin_sets.get(layer)
        return s is not None and eid in s

    def preload_pinned(self) -> None:
        """启动时把钉住专家全部读入 RAM（消除逐 token 填充的冷启动拖尾）。

        只读盘入 RAM 钉住区，不上传显存（显存缓存由 decode 路径按需填充）。
        """
        keys = [(l, e) for l, es in self._pin_sets.items() for e in es
                if (l, e) not in self.pinned]
        if not keys:
            return
        import time as _time
        from concurrent.futures import as_completed
        t0 = _time.time()
        fmap = {_executor().submit(self.store.load_expert, *k): k for k in keys}
        n = 0
        for fut in as_completed(fmap):
            self.pinned[fmap[fut]] = fut.result()
            n += 1
            if n % 600 == 0:
                print(f"[tpq] 热专家预载 {n}/{len(keys)}", flush=True)
        gb = sum(v[0].nbytes + v[1].nbytes for v in self.pinned.values()) / 2**30
        print(f"[tpq] 热专家预载完成（{n} 个 / {gb:.1f}GB，{_time.time() - t0:.0f}s）",
              flush=True)

    def preload_all(self, reserve_gb: float | None = None) -> bool:
        """启动时尝试把全部专家常驻 RAM（钉住，永不驱逐，之后零磁盘读）。

        判定：专家文件总量 ×1.05 + 预留 ≤ 当前可用物理内存。
        满足 → 并行全量读入 pinned 区，返回 True；
        不满足 → 打印醒目警告（列出缺口与建议）并返回 False，调用方回退
        热专家钉住 + LRU 按需加载。可用 TPQ_FULL_RESIDENT=0 关闭本行为，
        TPQ_RESIDENT_RESERVE_GB 调整预留（默认 3GB）。
        """
        if os.environ.get("TPQ_FULL_RESIDENT", "1") == "0":
            return False
        import psutil
        import time as _time
        from concurrent.futures import as_completed
        if reserve_gb is None:
            reserve_gb = float(os.environ.get("TPQ_RESIDENT_RESERVE_GB", "3.0"))
        root = self.store.root
        total = 0
        for fn in self.store.man.expert_files.values():
            p = os.path.join(root, fn)
            if os.path.exists(p):
                total += os.path.getsize(p)
        total_gb = total / 2**30 * 1.05
        avail_gb = psutil.virtual_memory().available / 2**30
        if total_gb + reserve_gb > avail_gb:
            print(f"[tpq] 警告：无法全量常驻专家：专家 {total_gb:.1f}GB + 预留 {reserve_gb:.1f}GB"
                  f" > 可用内存 {avail_gb:.1f}GB（差 {total_gb + reserve_gb - avail_gb:.1f}GB）。\n"
                  f"       将按需加载（LRU + 磁盘读，多轮后命中率上升）。\n"
                  f"       想全常驻：关闭其他占内存程序 / 加内存条 / "
                  f"调小 TPQ_RESIDENT_RESERVE_GB。", flush=True)
            return False
        n_experts = self.store.cfg["n_experts"]
        keys = [(l, e) for l in (int(x) for x in self.store.man.expert_files)
                for e in range(n_experts)
                if self.store.expert_kind(l, e) != "drop"]
        t0 = _time.time()
        print(f"[tpq] 全量专家常驻：{len(keys)} 个 / ≈{total_gb:.1f}GB 读盘中…", flush=True)
        fmap = {_executor().submit(self.store.load_expert, *k): k for k in keys}
        n = 0
        for fut in as_completed(fmap):
            self.pinned[fmap[fut]] = fut.result()
            n += 1
            if n % 2000 == 0:
                print(f"[tpq] 全量常驻 {n}/{len(keys)}", flush=True)
        gb = sum(v[0].nbytes + v[1].nbytes for v in self.pinned.values()) / 2**30
        print(f"[tpq] 全量专家常驻完成（{n} 个 / {gb:.1f}GB，{_time.time() - t0:.0f}s），"
              f"之后推理零磁盘读", flush=True)
        return True

    def pin_host_resident(self, budget_gb: float | None = None) -> float:
        """把常驻 RAM 专家索引转换为真正的 CUDA page-locked 内存。

        普通 ``preload_all`` 仅表示 Python 强引用常驻，内存仍是 pageable；每次显存
        miss 都要先复制进 PinnedStage 槽，再做 DMA。这里替换 idx 为 pin_memory()
        副本后，上传可直接异步 DMA，省掉逐 token 的 CPU memcpy 与槽位等待。

        TPQ_HOST_PIN_GB 默认 auto：仅当转换后仍能保留足够可用 RAM 时全量启用；
        数字指定 GiB 上限，0 明确关闭。转换逐专家替换，峰值只多一个专家而非再
        复制整个模型。
        """
        if not self.gpu or not self.pinned:
            return 0.0
        if budget_gb is None:
            raw = os.environ.get("TPQ_HOST_PIN_GB", "auto").strip().lower()
            if raw in ("", "auto"):
                import psutil
                total = sum(g.nbytes + d.nbytes for g, d in self.pinned.values())
                avail = psutil.virtual_memory().available
                reserve = max(32 * 2**30, total // 2)
                if avail < total + reserve:
                    print("[tpq] 锁页专家内存自动关闭："
                          f"可用 {avail / 2**30:.1f}GB < 专家 {total / 2**30:.1f}GB"
                          f" + 安全余量 {reserve / 2**30:.1f}GB",
                          flush=True)
                    return 0.0
                budget = total
                print(f"[tpq] 锁页专家内存自动启用：{total / 2**30:.1f}GB",
                      flush=True)
            else:
                budget = max(0, int(float(raw) * 2**30))
        else:
            budget = max(0, int(budget_gb * 2**30))
        if budget == 0:
            return 0.0

        import time as _time
        t0 = _time.time()
        pinned_bytes = 0
        pinned_count = 0
        for key in self.pinned:
            gu, dn = self.pinned[key]
            nb = gu.nbytes + dn.nbytes
            if pinned_bytes + nb > budget:
                continue
            try:
                gu_idx = gu.idx if gu.idx.is_pinned() else gu.idx.pin_memory()
                dn_idx = dn.idx if dn.idx.is_pinned() else dn.idx.pin_memory()
            except RuntimeError as exc:
                print(f"[tpq] 锁页专家内存停止于 {pinned_bytes / 2**30:.1f}GB：{exc}",
                      flush=True)
                break
            self.pinned[key] = (
                VQWeight(gu_idx, gu.cb, gu.cols),
                VQWeight(dn_idx, dn.cb, dn.cols),
            )
            pinned_bytes += nb
            pinned_count += 1
            if pinned_count % 2000 == 0:
                print(f"[tpq] 锁页专家内存 {pinned_count}/{len(self.pinned)} "
                      f"({pinned_bytes / 2**30:.1f}GB)", flush=True)
        print(f"[tpq] 锁页专家内存完成：{pinned_count} 个 / "
              f"{pinned_bytes / 2**30:.1f}GB（{_time.time() - t0:.1f}s）",
              flush=True)
        return pinned_bytes / 2**30

    def build_gpu_arenas(self) -> float:
        """按显存缓存预算一次性分配稳定的专家索引槽。

        码本仍由 ``_cb_dev`` 按层共享；arena 只保存占显存主体的 GU/DN 索引。
        每种索引 shape/dtype 独立成池，并按模型中该签名的专家数量等比例分配，
        后续 miss 只覆盖槽位视图，不再调用 CUDA allocator。
        """
        if not self.gpu:
            return 0.0
        if self._gpu_arenas is not None:
            return self._gpu_arenas.nbytes / 2**30
        entries = list(self.pinned.values())
        if not entries:
            entries = list(self.ram.values())
        if not entries or self.budget <= 0:
            return 0.0

        # Fixed arenas allocate their full capacity immediately.  Unlike the
        # historical lazy LRU, a requested 20 GiB cache therefore cannot be
        # created blindly after 13+ GiB of dense BF16 weights are resident on a
        # 32 GiB card.  Clamp from the live allocator/device state first.
        allocated_bytes = torch.cuda.memory_allocated(self.device)
        device_free_bytes, device_total_bytes = torch.cuda.mem_get_info(
            self.device
        )
        device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        try:
            process_fraction = torch.cuda.get_per_process_memory_fraction(
                device_index
            )
        except (AttributeError, RuntimeError):
            process_fraction = 1.0
        process_limit_bytes = int(device_total_bytes * process_fraction)
        reserve_gb = float(os.environ.get("TPQ_VRAM_RUNTIME_GB", "3.0"))
        safe_budget = _safe_arena_budget(
            requested_bytes=self.budget,
            allocated_bytes=allocated_bytes,
            device_free_bytes=device_free_bytes,
            process_limit_bytes=process_limit_bytes,
            reserve_bytes=int(reserve_gb * 2**30),
        )
        if safe_budget < self.budget:
            requested_gb = self.budget / 2**30
            self.budget = safe_budget
            print(
                f"[tpq] 固定专家槽分配前封顶：{requested_gb:.1f}GB"
                f" → {safe_budget / 2**30:.1f}GB"
                f"（dense/已分配 {allocated_bytes / 2**30:.1f}GB"
                f" + 运行时余量 {reserve_gb:.1f}GB）",
                flush=True,
            )
        if self.budget <= 0:
            return 0.0

        from collections import Counter

        counts = Counter(ExpertSignature.of(entry) for entry in entries)
        total_model_bytes = sum(
            signature.slot_bytes * count
            for signature, count in counts.items()
        )
        if total_model_bytes <= 0:
            return 0.0
        scale = min(1.0, self.budget / total_model_bytes)
        minimum = max(1, int(self.store.cfg.get("top_k", 6)))
        allocated = {
            signature: min(count, max(minimum, int(count * scale)))
            for signature, count in counts.items()
        }

        def used_bytes() -> int:
            return sum(
                signature.slot_bytes * count
                for signature, count in allocated.items()
            )

        # 极小预算时先收缩到可分配范围；正常服务器预算远高于每签名 top-k。
        while used_bytes() > self.budget:
            candidates = [
                signature
                for signature, count in allocated.items()
                if count > 1
            ]
            if not candidates:
                return 0.0
            largest = max(
                candidates,
                key=lambda signature: signature.slot_bytes * allocated[signature],
            )
            allocated[largest] -= 1

        # 利用取整后的余量，优先补齐当前覆盖率最低的签名。
        while True:
            candidates = [
                signature
                for signature, count in counts.items()
                if allocated[signature] < count
                and used_bytes() + signature.slot_bytes <= self.budget
            ]
            if not candidates:
                break
            next_signature = min(
                candidates,
                key=lambda signature: allocated[signature] / counts[signature],
            )
            allocated[next_signature] += 1

        self._gpu_arenas = GpuExpertArenas(
            allocated.items(),
            self.device,
        )
        gb = self._gpu_arenas.nbytes / 2**30
        detail = ", ".join(
            f"{signature.gu_dtype}:{count}"
            for signature, count in allocated.items()
        )
        print(f"[tpq] 固定专家显存槽：{sum(allocated.values())} 个 / "
              f"{gb:.2f}GB（{detail}）", flush=True)
        return gb

    def _release_gpu_key(self, key) -> None:
        arenas = getattr(self, "_gpu_arenas", None)
        if arenas is not None:
            arenas.release(key)

    @property
    def gpu_storage_bytes(self) -> int:
        """当前已实际分配的专家索引显存（动态 cache 或固定 arena）。"""
        arenas = getattr(self, "_gpu_arenas", None)
        if arenas is None:
            return self.bytes
        dynamic = sum(
            gu.nbytes + dn.nbytes
            for key, (gu, dn) in self.cache.items()
            if not arenas.owns(key)
        )
        return arenas.nbytes + dynamic

    @property
    def gpu_arena_bytes(self) -> int:
        """固定专家 arena 当前真实占用；区别于可动态下调的逻辑预算。"""
        arenas = getattr(self, "_gpu_arenas", None)
        return 0 if arenas is None else arenas.nbytes

    def _drain_staging_for_arena_resize(self, timeout_s: float) -> None:
        """等待缩容前已经提交的后台 staging，等待期间不持有池锁。"""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            with self._stage_lock:
                events = list(self._staging.values())
            if not events:
                return
            for event in events:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0 or not event.wait(remaining):
                    with self._stage_lock:
                        pending = len(self._staging)
                    raise RuntimeError(
                        f"timed out draining {pending} staging expert batches "
                        "before arena resize"
                    )

    def resize_gpu_arenas(
        self,
        budget: int,
        *,
        staging_timeout_s: float = 30.0,
    ) -> tuple[int, int]:
        """在更小预算下物理重建固定专家槽，并返回 (旧字节数, 新字节数)。"""
        budget = max(0, int(budget))
        arenas = getattr(self, "_gpu_arenas", None)
        old_bytes = 0 if arenas is None else arenas.nbytes
        if not self.gpu or arenas is None or budget >= old_bytes:
            self.trim_to(budget)
            return old_bytes, old_bytes

        with self._stage_lock:
            self._rebuilding_arenas = True
        try:
            self._drain_staging_for_arena_resize(staging_timeout_s)
            torch.cuda.synchronize(self.device)
            with self._stage_lock:
                if self._stage_dirty:
                    self._wait_stage()
                if self._staging:
                    raise RuntimeError(
                        "cannot resize expert arenas while staging is active"
                    )
                # cache 中的 VQWeight.idx 是 arena GU/DN 张量的视图；必须先清空，
                # 再删除 arena 本体，才能让 caching allocator 看到真实可释放块。
                self.cache.clear()
                self.bytes = 0
                self._inflight.clear()
                self._gpu_arenas = None
                self.budget = budget
            # 局部变量同样持有旧 arena；不删除它，empty_cache 仍无法归还显存。
            del arenas
            torch.cuda.empty_cache()
            try:
                self.build_gpu_arenas()
            except torch.cuda.OutOfMemoryError:
                # 部分构造产生的临时张量在异常展开后已失去引用；清缓存并只降档一次，
                # 避免在显存压力下反复 OOM。
                self._gpu_arenas = None
                torch.cuda.empty_cache()
                retry_budget = max(2**29, budget // 2)
                if retry_budget >= budget:
                    raise
                self.budget = retry_budget
                self.build_gpu_arenas()
            new_bytes = self.gpu_arena_bytes
            if new_bytes > self.budget:
                raise RuntimeError(
                    "rebuilt expert arenas exceed budget: "
                    f"{new_bytes} > {self.budget}"
                )
            return old_bytes, new_bytes
        finally:
            with self._stage_lock:
                self._rebuilding_arenas = False

    def _touch_gpu_key(self, key) -> None:
        arenas = getattr(self, "_gpu_arenas", None)
        if arenas is not None:
            arenas.touch(key)

    def _lease_gpu_ent(self, key, cpu_ent, *, use_arena: bool = True):
        """返回 arena 目标视图；不支持的签名返回 None 走动态分配回退。"""
        arenas = getattr(self, "_gpu_arenas", None)
        if not use_arena or arenas is None or not arenas.supports(cpu_ent):
            return None
        lease, gu_idx, dn_idx = arenas.lease(key, cpu_ent)
        if lease.replaced is not None:
            old = self.cache.pop(lease.replaced, None)
            if old is not None:
                self.bytes -= old[0].nbytes + old[1].nbytes
            self._inflight.discard(lease.replaced)
        arenas.mark_inflight(key)
        self._inflight.add(key)
        gu, dn = cpu_ent
        return (
            (gu.idx, gu_idx),
            (dn.idx, dn_idx),
            (
                VQWeight(gu_idx, self._cb_dev(gu.cb), gu.cols),
                VQWeight(dn_idx, self._cb_dev(dn.cb), dn.cols),
            ),
        )

    def _evict(self, d: OrderedDict, size_ref: str, budget: int, need: int,
               skip_inflight: bool = False) -> None:
        scanned = 0
        while getattr(self, size_ref) + need > budget and d:
            if skip_inflight and scanned < len(d):
                key = next(iter(d))
                if key in self._inflight:
                    # DMA 在飞的条目不可驱逐（显存复用会被 DMA 覆写），顺延为最新
                    d.move_to_end(key)
                    scanned += 1
                    continue
            elif skip_inflight:
                break   # 全部在飞：宁可暂超预算也不驱逐（安全优先）
            key, (g, dd) = d.popitem(last=False)
            if d is self.cache:
                self.bytes -= g.nbytes + dd.nbytes
                self._release_gpu_key(key)
            else:
                self.ram_bytes -= g.nbytes + dd.nbytes

    def trim_to(self, budget: int) -> None:
        """动态收紧显存缓存预算并立即按 LRU 驱逐到预算内（VramWatch 止血用）。
        先等在飞 DMA 落地：否则驱逐会释放 DMA 目标显存，被覆写后数据随机损坏。"""
        with self._stage_lock:
            if self.gpu and self._stage_dirty:
                self._wait_stage()
            self.budget = max(0, int(budget))
            self._evict(self.cache, "bytes", self.budget, 0)

    def _put(self, key, ent) -> None:
        with self._stage_lock:
            nb = ent[0].nbytes + ent[1].nbytes
            old = self.cache.pop(key, None)
            if old is not None:
                self.bytes -= old[0].nbytes + old[1].nbytes
            self._evict(self.cache, "bytes", self.budget, nb, skip_inflight=self.gpu)
            self.cache[key] = ent
            self.bytes += nb

    def _put_ram(self, key, ent) -> None:
        if self.ram_budget <= 0:
            return
        with self._stage_lock:
            nb = ent[0].nbytes + ent[1].nbytes
            old = self.ram.pop(key, None)
            if old is not None:
                self.ram_bytes -= old[0].nbytes + old[1].nbytes
            self._evict(self.ram, "ram_bytes", self.ram_budget, nb)
            self.ram[key] = ent
            self.ram_bytes += nb

    def _cb_dev(self, cb: torch.Tensor) -> torch.Tensor:
        """层共享码本的设备副本（按 data_ptr 恒等缓存：消除每专家重复的码本小上传）。

        必须强引用 CPU 码本：并行加载时 codebooks() 竞态会产生重复码本张量，
        落选者被 LRU 驱逐释放后其 data_ptr 可能被后续分配复用——若只按裸 ptr
        缓存键，新层码本会命中旧指针，返回**别的层的码本**（GLM 实测 KL 8.9 /
        输出乱码复读的根因）。强引用使 ptr 永不复用，键恒有效。"""
        if not hasattr(self, "_cb_devs"):
            self._cb_devs = {}
        key = cb.data_ptr()
        ent = self._cb_devs.get(key)
        if ent is None:
            d = cb.to(self.device)
            ent = (d, cb)          # (设备副本, CPU 强引用防 ptr 复用)
            self._cb_devs[key] = ent
        return ent[0]

    def _stage_ent(self, key, cpu_ent) -> tuple:
        """CPU 专家 (VQWeight, VQWeight) 经 pinned 分段上传到 GPU（码本随行）。
        TPQ_STAGE_SYNC=1（诊断）：走默认流同步 .to() 直传，绕过 pinned/DMA 机制。
        全程持 _stage_lock：与后台 staging 线程互斥，槽位轮转才不乱。"""
        with self._stage_lock:
            leased = self._lease_gpu_ent(key, cpu_ent)
            if leased is not None:
                gu_pair, dn_pair, out = leased
                if os.environ.get("TPQ_STAGE_SYNC", "0") != "0":
                    gu_pair[1].copy_(gu_pair[0])
                    dn_pair[1].copy_(dn_pair[0])
                    self._gpu_arenas.clear_inflight(key)
                    self._inflight.discard(key)
                else:
                    self.stage.upload_batch([gu_pair, dn_pair])
                return out
            if os.environ.get("TPQ_STAGE_SYNC", "0") != "0":
                return tuple(VQWeight(vq.idx.to(self.device), self._cb_dev(vq.cb), vq.cols)
                             for vq in cpu_ent)
            out = []
            for vq in cpu_ent:
                idx_d = torch.empty_like(vq.idx, device=self.device)
                self.stage.upload(vq.idx, idx_d)
                out.append(VQWeight(idx_d, self._cb_dev(vq.cb), vq.cols))
            return tuple(out)

    def _stage_ents(self, keys: list, cpu_ents: list) -> list:
        """_stage_ent 的成批版：全部索引一次 upload_batch（少 stream/事件开销）。
        全程持 _stage_lock（见上）。"""
        if os.environ.get("TPQ_STAGE_SYNC", "0") != "0":
            return [self._stage_ent(k, e) for k, e in zip(keys, cpu_ents)]
        with self._stage_lock:
            pairs = []
            outputs = []
            from collections import Counter
            signature_counts = Counter(
                ExpertSignature.of(cpu_ent)
                for cpu_ent in cpu_ents
            )
            arenas = getattr(self, "_gpu_arenas", None)
            for key, cpu_ent in zip(keys, cpu_ents):
                use_arena = (
                    arenas is not None
                    and arenas.supports(cpu_ent)
                    and signature_counts[ExpertSignature.of(cpu_ent)]
                    <= arenas.capacity(cpu_ent)
                )
                leased = self._lease_gpu_ent(
                    key,
                    cpu_ent,
                    use_arena=use_arena,
                )
                if leased is not None:
                    gu_pair, dn_pair, out = leased
                    pairs.extend((gu_pair, dn_pair))
                    outputs.append(out)
                    continue
                ent = tuple(torch.empty_like(vq.idx, device=self.device) for vq in cpu_ent)
                pairs.extend((vq.idx, d) for vq, d in zip(cpu_ent, ent))
                outputs.append(tuple(
                    VQWeight(d, self._cb_dev(vq.cb), vq.cols)
                    for vq, d in zip(cpu_ent, ent)
                ))
            self.stage.upload_batch(pairs)
            return outputs

    def prefetch(self, keys: list[tuple[int, int]]) -> None:
        """跨层专家预取（软提示，不阻塞；预测错误无正确性影响，仅 LRU 轻微污染）。

        利用路由时序局部性（相邻 token 专家集实测重合 70-90%）：在计算第 L 层
        attention 期间，把上一 token 第 L 层的专家集预先装填：
          - RAM/pinned 已有的 → pinned 分段异步 DMA 上显存（真 ~10GB/s，与计算重叠）；
          - 仅在磁盘的 → 提交线程池后台加载，get_many 命中时等待结果。
        """
        if not keys:
            return
        # staging 积压闸门：后台线程消费不过来时直接放弃本轮预取——否则 _staging
        # 无界增长、staged 条目占满 inflight 无法驱逐、缓存超预算 OOM（GLM 实测）
        with self._stage_lock:
            if self._rebuilding_arenas:
                return
            staging_backlog = len(self._staging)
        if staging_backlog > 512:
            return
        # _pending 上限：预测错偏的 Future 永不消费会无限堆积（占内存），超cap丢弃最旧
        while len(self._pending) > 256:
            oldest = next(iter(self._pending))   # 插入序最旧
            self._pending.pop(oldest).cancel()   # 尽力取消；已运行的读盘结果随引用丢弃
        # 调试二分：TPQ_PREFETCH_STAGE=0 时只做磁盘预载，不做 RAM→VRAM 异步 DMA
        do_stage = os.environ.get("TPQ_PREFETCH_STAGE", "1") != "0"
        # 自适应 1：近期 miss 率过高（RAM 池装不下工作集）时，磁盘带宽全让给
        # get_many 的紧急 miss，不再为预测预取抢队列（冷启动负优化修复）；
        # 自适应 2：预取池积压 >64 时暂停提交（get_many 绝不阻塞在预取池 backlog 后）
        recent = self._recent
        disk_ok = (len(self._pending) < 64 and
                   (len(recent) < 64 or (sum(recent) / len(recent)) < 0.5))
        stage_keys, stage_ents = [], []
        for key in keys:
            # 无锁快速过滤；提交前 _stage_async 会在锁内再次校验并关闭竞争窗口。
            if key in self.cache or key in self._staging or key in self._pending:
                continue
            cpu_ent = self.pinned.get(key)
            if cpu_ent is None:
                cpu_ent = self.ram.get(key)
                if cpu_ent is not None:
                    self.ram.move_to_end(key)
            if cpu_ent is not None:
                if self.gpu and do_stage:
                    stage_keys.append(key)
                    stage_ents.append(cpu_ent)
            elif disk_ok:
                self._pending[key] = _pf_executor().submit(self.store.load_expert, *key)
        if stage_keys:
            self._stage_async(stage_keys, stage_ents)

    def _stage_async(self, keys: list, cpu_ents: list) -> None:
        """后台 staging（真并行预加载）：单线程队列里完成 装槽+DMA+入缓存，
        主线程推理零阻塞。get_many 经 _staging 事件查重（宁可等待也不重复加载）。"""
        import threading
        fresh_keys, fresh_ents = [], []
        dones = {}
        with self._stage_lock:
            for k, cpu_ent in zip(keys, cpu_ents):
                if k in self.cache or k in self._staging:
                    continue
                ev = threading.Event()
                self._staging[k] = ev
                dones[k] = ev
                fresh_keys.append(k)
                fresh_ents.append(cpu_ent)
        if not fresh_keys:
            return
        keys, cpu_ents = fresh_keys, fresh_ents

        def job():
            try:
                staged = self._stage_ents(keys, cpu_ents)  # 内部持锁（槽位纪律）
                with self._stage_lock:
                    self._inflight.update(keys)
                    self._stage_dirty = True
                    for k, ent in zip(keys, staged):
                        self._put(k, ent)
                self._wait_stage()          # 本批 DMA 落地即解 inflight，防积压超预算
            finally:
                with self._stage_lock:
                    for k, ev in dones.items():
                        self._staging.pop(k, None)       # 先入缓存再解除标记
                        ev.set()                          # 唤醒等待者（走缓存命中）

        _stage_executor().submit(job)

    def _wait_staging_key(self, key):
        """返回已落地缓存；只等待该 key 的后台 staging，不等待整个拷贝流。"""
        with self._stage_lock:
            ent = self.cache.get(key)
            ev = self._staging.get(key)
        if ev is not None:
            ev.wait()
        elif ent is not None:
            return ent
        with self._stage_lock:
            return self.cache.get(key)

    def _wait_stage(self) -> None:
        """有在飞 DMA 时让计算流等待拷贝流尾部（缓存命中路径的安全网）。
        落地后清除 inflight 标记（对应 cache 条目恢复可驱逐）。"""
        with self._stage_lock:
            if self._stage_dirty:
                self.stage.wait()
                self._stage_dirty = False
                arenas = getattr(self, "_gpu_arenas", None)
                if arenas is not None:
                    for key in self._inflight:
                        arenas.clear_inflight(key)
                self._inflight.clear()

    def get(self, layer: int, eid: int) -> tuple[VQWeight, VQWeight]:
        key = (layer, eid)
        ent = self._wait_staging_key(key)
        if ent is not None:
            self.hits += 1
            self._recent.append(0)
            self.cache.move_to_end(key)
            self._touch_gpu_key(key)
            return ent
        self.miss += 1
        self._recent.append(1)
        cpu_ent = self.pinned.get(key)
        if cpu_ent is None:
            cpu_ent = self.ram.get(key)
            if cpu_ent is None:
                fut = self._pending.pop(key, None)
                if fut is not None and fut.done():
                    cpu_ent = fut.result()      # 预取已完成：零等待
                else:
                    if fut is not None:
                        fut.cancel()            # 未完成不在此等待（防预取池 backlog 饥饿）
                    cpu_ent = self.store.load_expert(layer, eid)
                if self._hot(layer, eid):
                    self.pinned[key] = cpu_ent  # 热专家：永久钉住，不占 LRU 预算
                else:
                    self._put_ram(key, cpu_ent)
            else:
                self.ram.move_to_end(key)
        if self.gpu:
            ent = self._stage_ent(key, cpu_ent)
            self._stage_dirty = True
            self._wait_stage()
        else:
            ent = cpu_ent
        self._put(key, ent)
        return ent

    def get_many(self, keys: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[VQWeight, VQWeight]]:
        """批量取专家：未命中项并行磁盘加载（常驻线程池，NVMe 队列深度受益）
        + 异步上传显存。decode 每层 8 专家的读路径由此从串行 ~88ms 降到 ~20ms。
        """
        out: dict[tuple[int, int], tuple[VQWeight, VQWeight]] = {}
        missing: list[tuple[int, int]] = []
        ready_keys: list[tuple[int, int]] = []
        ready_cpu: list[tuple[VQWeight, VQWeight]] = []
        demand_upload = False
        unresolved: list[tuple[tuple[int, int], object | None]] = []
        # Decode top-k arrives as one batch. Snapshot ordinary GPU hits under
        # one lock instead of taking the same RLock once per expert.
        with self._stage_lock:
            for key in keys:
                ent = self.cache.get(key)
                event = self._staging.get(key)
                if ent is None or event is not None:
                    unresolved.append((key, event))
                    continue
                self.hits += 1
                self._recent.append(0)
                self.cache.move_to_end(key)
                self._touch_gpu_key(key)
                out[key] = ent
        for key, event in unresolved:
            ent = None
            if event is not None:
                event.wait()
                with self._stage_lock:
                    ent = self.cache.get(key)
                    if ent is not None:
                        self.hits += 1
                        self._recent.append(0)
                        self.cache.move_to_end(key)
                        self._touch_gpu_key(key)
            if ent is not None:
                out[key] = ent
                continue
            cpu_ent = self.pinned.get(key)
            if cpu_ent is None:
                cpu_ent = self.ram.get(key)
                if cpu_ent is not None:
                    self.ram.move_to_end(key)
            if cpu_ent is not None:
                self.hits += 1
                self._recent.append(0)
                if self.gpu:
                    ready_keys.append(key)
                    ready_cpu.append(cpu_ent)
                else:
                    self._put(key, cpu_ent)
                    out[key] = cpu_ent
            else:
                missing.append(key)
        if ready_keys:
            staged = self._stage_ents(ready_keys, ready_cpu)
            with self._stage_lock:
                self._inflight.update(ready_keys)
                self._stage_dirty = True
                for key, ent in zip(ready_keys, staged):
                    self._put(key, ent)
                    out[key] = ent
            demand_upload = True
        if missing:
            # 后台 staging 查重：正在 staged 的 key 等其完成事件走缓存命中，
            # 绝不重复读盘（等待 ≪ 磁盘加载；事件在入缓存后 set，无竞态窗）
            still = []
            for k in missing:
                ev = self._staging.get(k)
                if ev is not None:
                    ev.wait()
                    ent = self.cache.get(k)
                    if ent is not None:
                        self.hits += 1
                        self._recent.append(0)
                        self.cache.move_to_end(k)
                        self._touch_gpu_key(k)
                        out[k] = ent
                        continue
                still.append(k)
            missing = still
        if missing:
            from concurrent.futures import as_completed
            # 上传与加载重叠：哪个专家先读完就先上传显存，其余仍在后台读盘；
            # 预取已提交的加载直接复用其 Future（不重复读盘）
            futs = {}
            for k in missing:
                fut = self._pending.pop(k, None)
                if fut is not None and fut.done():
                    futs[k] = fut               # 预取已完成：零等待直接取结果
                else:
                    if fut is not None:
                        fut.cancel()            # 未完成的预取：尽力取消，绝不在此等待
                    #（预取池是 backlog 重灾区的慢池；紧急 miss 一律走 12 线程快池）
                    futs[k] = _executor().submit(self.store.load_expert, *k)
            fmap = {f: k for k, f in futs.items()}
            for fut in as_completed(fmap):
                key = fmap[fut]
                cpu_ent = fut.result()
                self.miss += 1
                self._recent.append(1)
                if self._hot(*key):
                    self.pinned[key] = cpu_ent  # 热专家：永久钉住
                else:
                    self._put_ram(key, cpu_ent)
                ent = self._stage_ent(key, cpu_ent) if self.gpu else cpu_ent
                if self.gpu:
                    self._inflight.add(key)
                    self._stage_dirty = True
                    demand_upload = True
                self._put(key, ent)
                out[key] = ent
        if self.gpu and demand_upload:
            with self._stage_lock:
                # 只在本批发起了 DMA 时等待；纯缓存命中不阻塞预取流。
                self._wait_stage()
        return out
