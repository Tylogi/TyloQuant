"""Runtime patches for memory-bounded TPQ expert residency."""

from __future__ import annotations

import os
import time
from collections import Counter
from concurrent.futures import as_completed


def _drop_file_cache(path: str) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _cgroup_reclaim_capacity() -> int | None:
    root = "/sys/fs/cgroup"
    try:
        with open(os.path.join(root, "memory.max"), encoding="ascii") as handle:
            raw_limit = handle.read().strip()
        if raw_limit == "max":
            return None
        limit = int(raw_limit)
        with open(os.path.join(root, "memory.current"), encoding="ascii") as handle:
            current = int(handle.read().strip())
        stats = {}
        with open(os.path.join(root, "memory.stat"), encoding="ascii") as handle:
            for line in handle:
                key, value = line.split()
                stats[key] = int(value)
        return max(0, limit - max(0, current - stats.get("file", 0)))
    except (OSError, ValueError):
        return None


def _patch_host_residency(store) -> None:
    if getattr(store.ExpertPool.preload_all, "_mfq_cgroup_resident", False):
        return
    original_preload_all = store.ExpertPool.preload_all

    def drop_dense_file_cache(self) -> None:
        _drop_file_cache(os.path.join(self.root, self.man.dense_file))

    def drop_expert_file_cache(self, layer: int) -> None:
        _drop_file_cache(os.path.join(self.root, self.man.expert_files[layer]))

    def preload_all(self, reserve_gb: float | None = None) -> bool:
        root = getattr(self.store, "root", None)
        expert_files = tuple(self.store.man.expert_files.values())
        if root is not None and expert_files and all(
            os.path.isfile(os.path.join(root, filename))
            for filename in expert_files
        ):
            return original_preload_all(self, reserve_gb)
        if os.environ.get("TPQ_FULL_RESIDENT", "1") == "0":
            return False
        import psutil

        if reserve_gb is None:
            reserve_gb = float(os.environ.get("TPQ_RESIDENT_RESERVE_GB", "3.0"))
        resident_bytes = sum(
            signature.slot_bytes * count
            for signature, count in self.store.expert_signature_counts().items()
        )
        total_gb = resident_bytes / 2**30 * 1.05
        available_gb = psutil.virtual_memory().available / 2**30
        cgroup_capacity = _cgroup_reclaim_capacity()
        if cgroup_capacity is not None:
            available_gb = min(available_gb, cgroup_capacity / 2**30)
        if total_gb + reserve_gb > available_gb:
            print(
                "[tpq] full expert residency unavailable: "
                f"{total_gb:.1f} GiB experts + {reserve_gb:.1f} GiB reserve "
                f"> {available_gb:.1f} GiB effective memory",
                flush=True,
            )
            return False

        self.store.drop_dense_file_cache()
        n_experts = self.store.cfg["n_experts"]
        layers = sorted(int(value) for value in self.store.man.expert_files)
        host_pin = os.environ.get("TPQ_HOST_PIN_GB", "auto").strip().lower()
        if host_pin in ("", "auto"):
            pin_on_load = self.gpu
        else:
            pin_on_load = (
                self.gpu
                and float(host_pin) * 2**30 >= resident_bytes
            )
        total_experts = sum(
            self.store.expert_kind(layer, expert) != "drop"
            for layer in layers
            for expert in range(n_experts)
        )
        started = time.time()
        loaded = 0
        print(
            f"[tpq] full expert residency: {total_experts} experts / "
            f"about {total_gb:.1f} GiB, loading per layer",
            flush=True,
        )

        def load_resident(key):
            gu, dn = self.store.load_expert(*key)
            if not pin_on_load:
                return gu, dn
            return (
                store.VQWeight(
                    gu.idx if gu.idx.is_pinned() else gu.idx.pin_memory(),
                    gu.cb,
                    gu.cols,
                ),
                store.VQWeight(
                    dn.idx if dn.idx.is_pinned() else dn.idx.pin_memory(),
                    dn.cb,
                    dn.cols,
                ),
            )

        for layer in layers:
            keys = [
                (layer, expert)
                for expert in range(n_experts)
                if self.store.expert_kind(layer, expert) != "drop"
            ]
            futures = {
                store._executor().submit(load_resident, key): key
                for key in keys
            }
            for future in as_completed(futures):
                self.pinned[futures[future]] = future.result()
                loaded += 1
            del futures
            self.store.drop_expert_file_cache(layer)
        resident_gib = sum(
            value[0].nbytes + value[1].nbytes for value in self.pinned.values()
        ) / 2**30
        print(
            f"[tpq] full expert residency complete: {loaded}/{total_experts}, "
            f"{resident_gib:.1f} GiB in {time.time() - started:.0f}s",
            flush=True,
        )
        return True

    preload_all._mfq_cgroup_resident = True
    if not hasattr(store.CCCPStore, "drop_dense_file_cache"):
        store.CCCPStore.drop_dense_file_cache = drop_dense_file_cache
    if not hasattr(store.CCCPStore, "drop_expert_file_cache"):
        store.CCCPStore.drop_expert_file_cache = drop_expert_file_cache
    store.ExpertPool.preload_all = preload_all
    store.ExpertPool._mfq_direct_host_pin = True

    original_pin_host_resident = store.ExpertPool.pin_host_resident

    def pin_host_resident(self, budget_gb: float | None = None) -> float:
        if self.pinned and all(
            gu.idx.is_pinned() and dn.idx.is_pinned()
            for gu, dn in self.pinned.values()
        ):
            resident_bytes = sum(
                gu.nbytes + dn.nbytes for gu, dn in self.pinned.values()
            )
            return resident_bytes / 2**30
        return original_pin_host_resident(self, budget_gb)

    store.ExpertPool.pin_host_resident = pin_host_resident


def _expert_signature(store, kind):
    import torch
    from tpq.expert_slots import ExpertSignature

    dim, codebook_size = store.man.vq_dims[kind.rstrip("z")]
    dtype = torch.uint16 if codebook_size > 256 else torch.uint8
    hidden = store.cfg["hidden"]
    intermediate = store.cfg["moe_inter"]
    return ExpertSignature(
        (2 * intermediate, hidden // dim),
        dtype,
        (hidden, intermediate // dim),
        dtype,
    )


def _patch_full_gpu_residency(store) -> None:
    if hasattr(store.ExpertPool, "preload_gpu_all"):
        return

    def expert_signature_counts(self):
        cached = getattr(self, "_expert_signature_count_cache", None)
        if cached is not None:
            return cached.copy()
        counts = Counter()
        for layer in sorted(self.man.expert_files):
            for expert in range(self.cfg["n_experts"]):
                kind = self.expert_kind(layer, expert)
                if kind != "drop":
                    counts[_expert_signature(self, kind)] += 1
        self._expert_signature_count_cache = counts
        return counts.copy()

    if not hasattr(store.CCCPStore, "expert_signature_counts"):
        store.CCCPStore.expert_signature_counts = expert_signature_counts

    original_build_gpu_arenas = store.ExpertPool.build_gpu_arenas

    def build_gpu_arenas(self):
        if (
            self._gpu_arenas is not None
            or self.budget <= 0
            or self.pinned
            or self.ram
        ):
            return original_build_gpu_arenas(self)

        counts = self.store.expert_signature_counts()
        representatives = {}
        for layer in sorted(self.store.man.expert_files):
            for expert in range(self.store.cfg["n_experts"]):
                kind = self.store.expert_kind(layer, expert)
                if kind == "drop":
                    continue
                signature = _expert_signature(self.store, kind)
                if signature not in representatives:
                    representatives[signature] = self.store.load_expert(
                        layer, expert
                    )
            if len(representatives) == len(counts):
                break

        previous = self.pinned
        synthetic = {}
        index = 0
        for signature, count in counts.items():
            entry = representatives[signature]
            for _ in range(count):
                synthetic[("__mfq_signature__", index)] = entry
                index += 1
        self.pinned = synthetic
        try:
            return original_build_gpu_arenas(self)
        finally:
            self.pinned = previous

    build_gpu_arenas._mfq_manifest_sized = True
    store.ExpertPool.build_gpu_arenas = build_gpu_arenas

    def preload_gpu_all(self) -> bool:
        if (
            not self.gpu
            or self._gpu_arenas is None
            or os.environ.get("TPQ_GPU_FULL_RESIDENT", "auto") == "0"
        ):
            return False
        counts = self.store.expert_signature_counts()
        for signature, count in counts.items():
            arena = self._gpu_arenas.arenas.get(signature)
            if arena is None or arena.book.count < count:
                return False

        layers = sorted(self.store.man.expert_files)
        expert_count = self.store.cfg["n_experts"]
        total_experts = sum(counts.values())
        started = time.time()
        loaded = 0
        print(
            f"[tpq] full GPU expert residency: {total_experts} experts / "
            f"{self._gpu_arenas.nbytes / 2**30:.2f} GiB, streaming per layer",
            flush=True,
        )
        for layer in layers:
            keys = [
                (layer, expert)
                for expert in range(expert_count)
                if self.store.expert_kind(layer, expert) != "drop"
            ]
            future_to_key = {
                store._executor().submit(self.store.load_expert, *key): key
                for key in keys
            }
            loaded_entries = {
                future_to_key[future]: future.result()
                for future in as_completed(future_to_key)
            }
            ordered_entries = [loaded_entries[key] for key in keys]
            staged = self._stage_ents(keys, ordered_entries)
            self._stage_dirty = True
            self._wait_stage()
            for key, entry in zip(keys, staged):
                self._put(key, entry)
            loaded += len(keys)
            del staged, ordered_entries, loaded_entries, future_to_key
            self.store.drop_expert_file_cache(layer)
        print(
            f"[tpq] full GPU expert residency complete: {loaded}/"
            f"{total_experts} in {time.time() - started:.0f}s; "
            "no RAM expert copies retained",
            flush=True,
        )
        return True

    preload_gpu_all._mfq_full_gpu_resident = True
    store.ExpertPool.preload_gpu_all = preload_gpu_all

    from tpq import dsv4model

    if hasattr(dsv4model.DSV4TPQModel, "_prepare_tp_packed_finalizer"):
        return

    def preload(self) -> None:
        if self.device.type == "cpu":
            return
        started = time.time()
        dense_bf16 = getattr(self, "_dense_bf16", ())
        if dense_bf16:
            print(
                "[tpq] dense BF16 resident: "
                + ",".join(sorted(dense_bf16)),
                flush=True,
            )
        for name in self.store.dense_names():
            self.w(name)
        self._prepare_tp_shared_mlp()
        self._prepare_tp_decode_metadata()
        self.store.drop_dense_file_cache()
        import torch

        vram = torch.cuda.memory_allocated(self.device) / 2**30
        print(
            f"[tpq] dense preload complete ({time.time() - started:.1f}s, "
            f"{vram:.1f} GiB VRAM)",
            flush=True,
        )
        if os.environ.get("TPQ_GROUPED", "1") != "0":
            from tpq import grouped

            path = (
                "fused CUDA kernel"
                if grouped._fused is not None
                else "torch batch path"
            )
            print(f"[tpq] grouped GEMM: {path}", flush=True)

        if getattr(self, "_packed_full_gpu", False):
            self.pool.preload()
            self._prefetch_auto = False
            return

        preload_gpu_all = getattr(self.pool, "preload_gpu_all", None)
        if preload_gpu_all is not None:
            self.pool.build_gpu_arenas()
            if preload_gpu_all():
                self._prefetch_auto = False
                return

        resident_all = self.pool.preload_all()
        self._prefetch_auto = not resident_all
        if resident_all:
            self.pool.pin_host_resident()
        else:
            self.pool.preload_pinned()
        if preload_gpu_all is None:
            self.pool.build_gpu_arenas()

    preload._mfq_gpu_streaming = True
    dsv4model.DSV4TPQModel.preload = preload


def apply_tpq_residency_patch() -> None:
    """Add cgroup-safe host loading and full-GPU streaming to TyloQuant."""

    from tpq import store

    _patch_host_residency(store)
    _patch_full_gpu_residency(store)
