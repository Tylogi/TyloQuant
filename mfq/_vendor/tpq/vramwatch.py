"""TPQ dynamic VRAM monitor running in a background thread to keep this process within physical VRAM.

Motivation: under WDDM, filling physical VRAM makes the driver page into shared system memory,
reducing bandwidth from ~600 GB/s HBM to PCIe levels and stalling whole synchronous passes. Measured shared-memory
paging moves several GB at a time and can reduce decode speed by multiples. A static reserve (the allocator hard limit
set during engine initialization) prevents this process from filling VRAM but cannot handle mid-run contention from
other processes, fragmentation, or switching to a smaller device such as 16 GB.

The monitor queries free VRAM every ``interval`` seconds and adjusts the expert VRAM cache budget with hysteresis:
  free < low_gb  -> reduce the budget by step_gb and immediately perform LRU eviction plus empty_cache;
  free > high_gb -> raise the budget by step_gb, without exceeding the initial cap, and let the cache refill on demand.
This automatically finds a usable operating point for any device (16/22/48 GB) and model tier.
Set TPQ_VRAM_WATCH=0 to disable it; override low/high/interval with same-named environment variables.
"""

from __future__ import annotations

import os
import threading
import time

import torch


class VramWatch:
    """Background hysteresis controller; pool is an ExpertPool with a budget attribute and trim_to method."""

    def __init__(self, pool, max_budget: int, device: int = 0,
                 low_gb: float | None = None, high_gb: float | None = None,
                 step_gb: float = 0.5, interval: float | None = None,
                 min_gb: float = 0.5, quiet: bool = False):
        self.pool = pool
        self.device = device
        self.max_budget = int(max_budget)
        self.min_budget = int(min_gb * 2**30)
        self.low = float(low_gb if low_gb is not None
                         else os.environ.get("TPQ_VRAM_WATCH_LOW_GB", "0.8"))
        self.high = float(high_gb if high_gb is not None
                          else os.environ.get("TPQ_VRAM_WATCH_HIGH_GB", "3.0"))
        self.step = int(step_gb * 2**30)
        self.interval = float(interval if interval is not None
                              else os.environ.get("TPQ_VRAM_WATCH_SEC", "3"))
        self.quiet = quiet
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self.trims = 0      # Cumulative emergency-trim count (for diagnostics/benchmark records)
        self.grows = 0

    def start(self) -> None:
        if self._th is not None or not torch.cuda.is_available():
            return
        self._th = threading.Thread(target=self._run, name="tpq-vramwatch",
                                    daemon=True)
        self._th.start()
        if not self.quiet:
            print(f"[tpq] 显存动态监测已启动（空闲<{self.low:.1f}GB 收紧 / "
                  f">{self.high:.1f}GB 放宽，{self.interval:.0f}s 周期）", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2)
            self._th = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                free = torch.cuda.mem_get_info(self.device)[0]
            except Exception:
                continue
            budget = self.pool.budget
            if free < self.low * 2**30 and budget > self.min_budget:
                new = max(self.min_budget, budget - self.step)
                self.pool.trim_to(new)
                torch.cuda.empty_cache()
                self.trims += 1
                if not self.quiet:
                    print(f"[vramwatch] 空闲 {free / 2**30:.2f}GB < {self.low}GB → "
                          f"显存缓存收紧至 {new / 2**30:.1f}GB", flush=True)
            elif free > self.high * 2**30 and budget < self.max_budget:
                new = min(self.max_budget, budget + self.step)
                self.pool.budget = new      # Raise only the limit; the cache refills naturally on demand
                self.grows += 1
                if not self.quiet:
                    print(f"[vramwatch] 空闲 {free / 2**30:.2f}GB > {self.high}GB → "
                          f"显存缓存放宽至 {new / 2**30:.1f}GB", flush=True)
