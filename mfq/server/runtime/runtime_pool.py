"""Managed MFQ runtime processes and model-aware request routing."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import socket
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from mfq.server.state.catalog import DiscoveredModel, ModelArtifactNotFoundError, ModelCatalog
from mfq.server.runtime.host_memory import (
    MemoryReleaseCredit,
    automatic_memory_reserve,
    host_memory_snapshot,
    metal_allocation_limit,
    process_physical_footprint,
    total_physical_memory,
)
from mfq.server.services.jobs import JobCancelledError, JobContext, JobExecutionError
from mfq.server.state.storage import InvalidJobStateError, JobNotFoundError
from mfq.server.protocol.models import (
    ErrorDetail,
    JobEventLevel,
    ModelLoadRequest,
    ModelUnloadRequest,
    ConfigureQsaKvOffloadRequest,
    QsaKvOffloadPolicy,
    QsaKvStorageSettings,
    KvMemorySettings,
    KvQuantizationSettings,
    ResponseFormat,
    RuntimeCapabilitiesResource,
    RuntimeConcurrencyLimits,
    RuntimeConcurrencyPolicy,
    RuntimeConcurrencyResource,
    RuntimeConcurrencySettings,
    RuntimeConcurrencyReloadRequest,
    RuntimeConcurrencyUpdate,
    UpdateRuntimeConcurrencyRequest,
    RuntimeInstanceList,
    RuntimeInstanceResource,
    RuntimeInstanceState,
    RuntimeMemoryResources,
    RuntimeMemoryPolicy,
    RuntimeInferencePolicy,
    RuntimeContextPolicy,
    RuntimeReloadRequest,
    RuntimeLogLevel,
    SamplingParams,
    ToolChoice,
    ToolDefinition,
    UpdateRuntimeInstanceRequest,
)
from mfq.server.runtime.backend import (
    BackendDelta,
    BackendError,
    ChatBackend,
    OpenAIChatBackend,
    closing_backend_stream,
    preflight_backend_request,
)
from mfq.server.runtime.client import HttpRuntimeClient, StdioRuntimeClient
from mfq.server.runtime.concurrency import resolve_concurrency
from mfq.server.runtime.prefix_disk_budget import maintain_prefix_disk_budget, inspect_prefix_cache, purge_prefix_cache, read_prefix_cache_tokens, prefix_cache_identity
from mfq.server.runtime.prefix_cache_text import decode_prefix_cache_tokens
from mfq.server.runtime.qsa_kv_policy import QSA_KV_BUFFER_BYTES, qsa_index_requirement
from mfq.server.runtime.kv_budget import checkpoint_kv_budget, check_ssd_capacity
from mfq.server.runtime.native import (
    append_native_prefill_chunk_override,
    find_native_runtime_resource,
    native_request_capacity,
    native_runtime_environment,
    native_tokenizer_arguments,
    supports_cuda_ram_expert_pipeline,
)
from mfq.server.services.hardware import (
    gpu_memory_status,
    hardware_identity,
    host_commit_capacity,
    host_memory_capacity,
    memory_topology,
)
from mfq.server.services.metrics import RequestMetrics, RuntimeMetricsSnapshot
from mfq.server.services.model_memory import checkpoint_attention_kv_bytes_per_token

logger = logging.getLogger(__name__)


class RuntimeManagementError(RuntimeError):
    pass


class RuntimeInstanceNotFoundError(RuntimeManagementError):
    pass


class RuntimeConflictError(RuntimeManagementError):
    pass


class _MemoryPlanContext:
    def __init__(self, context: JobContext, start: float = 0.0, end: float = 1.0, recovery: bool = False) -> None:
        self.context, self.start, self.end, self.recovery = context, start, end, recovery

    def __getattr__(self, name: str) -> Any:
        return getattr(self.context, name)

    @property
    def cancel_requested(self) -> bool:
        return False if self.recovery else self.context.cancel_requested

    def raise_if_cancelled(self) -> None:
        if not self.recovery:
            self.context.raise_if_cancelled()

    async def progress(self, value: float, **kwargs: Any) -> None:
        if not self.recovery:
            await self.context.progress(self.start + value * (self.end - self.start), **kwargs)


_AUTOMATIC_MEMORY_SOFT_RATIO = 0.90
_AUTOMATIC_MEMORY_HARD_RATIO = 0.95
_AUTOMATIC_MEMORY_TARGET_RATIO = 0.85
_RUNTIME_MAINTENANCE_WAIT_SECONDS = 30.0
_RUNTIME_LOG_MAX_CHARS = 4096
_RUNTIME_OUTPUT_DRAIN_TIMEOUT_SECONDS = 10.0
# WDDM charges device allocations to the process commit limit (RAM plus page
# file). Keep room for the service, WSL, and other workloads instead of handing
# all remaining commit headroom to resident experts.
_WINDOWS_COMMIT_RESERVE_BYTES = 16 << 30
# Pinned expert memory cannot be paged, and other workloads (a WSL VM, for
# example) keep growing after the worker starts; the reserve scales with the
# commit limit so large machines keep proportionate headroom.
_WINDOWS_COMMIT_RESERVE_FRACTION = 8
# CUDA contexts, KV staging, and kernels are committed but absent from the
# weight estimate; price them before deciding how many experts stay in RAM.
_WINDOWS_GPU_WORKSPACE_BYTES = 4 << 30
# Match the native expert cache's 1-GiB graph/driver allowance plus its
# 256-MiB allocator/graph construction guard, outside prompt workspace.
_CUDA_RUNTIME_OVERHEAD_BYTES = 1280 << 20
# Missing KV geometry: FP16 K/V for 64 layers, 8 KV heads, head dim 128.
_CUDA_FALLBACK_KV_BYTES_PER_TOKEN = 64 * 8 * 128 * 4
# Packed expert slots round up per arena; a cache meant to hold every expert
# asks for 1% of the routed bytes on top, at least this much.
_CUDA_CACHE_ALL_MIN_MARGIN_BYTES = 64 << 20
# Deadline failures keep their own result label so retries stay distinguishable
# from requests that failed for another reason.
_TIMEOUT_ERROR_CODES = frozenset({
    "backend_timeout",
    "runtime_maintenance_timeout",
    "runtime_queue_timeout",
})


@dataclass(frozen=True)
class _ExpertHostBudget:
    """Resident expert allowance together with the limit that produced it."""

    bytes: int
    physical_bytes: int
    commit_bytes: int | None = None
    commit_available_bytes: int | None = None
    commit_footprint_bytes: int | None = None

    @property
    def commit_limited(self) -> bool:
        return self.commit_bytes is not None and self.commit_bytes < self.physical_bytes


@dataclass(frozen=True)
class _CudaResidencyReserve:
    """Runtime headroom needed before choosing irrevocable GPU expert residency."""

    context: int
    kv_bytes: int
    kv_source: str
    workspace_bytes: int
    overhead_bytes: int = _CUDA_RUNTIME_OVERHEAD_BYTES

    @property
    def total_bytes(self) -> int:
        return self.kv_bytes + self.workspace_bytes + self.overhead_bytes


@dataclass(frozen=True)
class _CudaExpertCacheHostPlan:
    """How the host serves the routed experts behind a CUDA expert cache."""

    budget: _ExpertHostBudget
    pipeline: bool
    streaming: bool
    allow_ssd: bool
    failure: str | None


def _job_error(code: str, message: str, *, retryable: bool = False) -> JobExecutionError:
    return JobExecutionError(ErrorDetail(code=code, message=message, retryable=retryable))


def _request_deadline_exceeded(model: str) -> BackendError:
    """A client deadline that elapsed: absolute, so retrying cannot help."""

    return BackendError(
        "runtime_deadline_exceeded",
        f"request deadline expired before the runtime started it: {model}",
        retryable=False,
        status_code=504,
    )


@dataclass
class _SlotWaiter:
    priority: int
    granted: asyncio.Future[None]


class _RequestSlotGate:
    """Admission gate for one runtime's concurrent request slots.

    Waiters leave in descending priority order and FIFO inside one priority,
    which mirrors the native scheduler's stable sort on the request priority.
    Capacity changes replace the whole gate instead of mutating a live one, so a
    released slot never has to rediscover which generation it belonged to.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("request capacity must be positive")
        self.capacity = capacity
        self._granted = 0
        self._waiters: list[_SlotWaiter] = []

    @property
    def available(self) -> int:
        return max(0, self.capacity - self._granted)

    async def acquire(self, priority: int = 0) -> None:
        """Take one slot, waiting in priority order behind earlier arrivals."""

        if self._granted < self.capacity and not self._waiters:
            self._granted += 1
            return
        waiter = _SlotWaiter(priority, asyncio.get_running_loop().create_future())
        # Insert after everything of equal or higher priority so one priority
        # stays first-come-first-served.
        position = len(self._waiters)
        while position > 0 and self._waiters[position - 1].priority < priority:
            position -= 1
        self._waiters.insert(position, waiter)
        try:
            await waiter.granted
        except asyncio.CancelledError:
            if waiter.granted.done() and not waiter.granted.cancelled():
                # A release granted this slot just before the cancellation was
                # delivered; hand it to the next waiter instead of losing it.
                self.release()
            elif waiter in self._waiters:
                self._waiters.remove(waiter)
            raise

    def release(self) -> None:
        if self._granted <= 0:
            raise RuntimeError("request slot released without a matching acquire")
        self._granted -= 1
        while self._granted < self.capacity and self._waiters:
            waiter = self._waiters.pop(0)
            if waiter.granted.done():
                # Cancelling a task cancels its awaited future immediately, but the
                # waiter's own cleanup runs on a later tick; skip it here.
                continue
            self._granted += 1
            waiter.granted.set_result(None)


class _Runtime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: UUID
    artifact: DiscoveredModel
    process: SkipValidation[asyncio.subprocess.Process | subprocess.Popen[bytes]]
    backend: SkipValidation[ChatBackend]
    port: int
    context_size: int
    sampling_defaults: SamplingParams | None = None
    idle_ttl_seconds: int | None = None
    pinned: bool = False
    state: RuntimeInstanceState = RuntimeInstanceState.LOADING
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: datetime | None = None
    active_requests: int = 0
    queued_requests: int = 0
    serving_snapshot: dict[str, Any] = Field(default_factory=dict)
    serving_sampled_at: float = 0.0
    serving_refreshing: bool = False
    control_leases: int = 0
    read_control_leases: int = 0
    request_slots: SkipValidation[_RequestSlotGate] | None = None
    request_capacity: int = 1
    batch_capacity: int = 1
    concurrency_settings: RuntimeConcurrencySettings | None = None
    concurrency_changed: asyncio.Event = Field(default_factory=asyncio.Event)
    queue_wait_ms_total: float = 0.0
    queue_wait_ms_max: float = 0.0
    queue_wait_samples: int = 0
    rejections: int = 0
    # Prometheus counters and histograms live with the instance so a restart
    # publishes a new ``instance`` label instead of resetting old series.
    request_metrics: SkipValidation[RequestMetrics] = Field(default_factory=RequestMetrics)
    mtp_supported: bool = False
    mtp_available: bool = False
    qsa_kv_offload_supported: bool = False
    kv_quantization_supported: bool = False
    kv_memory_supported: bool = False
    error: ErrorDetail | None = None
    output_task: asyncio.Task[None] | None = None
    log_to_load_job: bool = False
    monitor_task: asyncio.Task[None] | None = None
    retirement_task: asyncio.Task[None] | None = None
    retirement_retry_task: asyncio.Task[None] | None = None
    retirement_failures: int = 0
    realtime_gateway: Any | None = None
    realtime_error: str | None = None
    reserved_bytes: int | None = None
    resident_bytes: int | None = None
    kv_bytes: int | None = None
    memory: RuntimeMemoryResources | None = None
    device_ids: list[str] = Field(default_factory=list)
    usage_refreshed_at: float = 0.0
    load_progress: float = 0.02
    context_capacity: int | None = None
    context_job: JobContext | bool | None = None
    prefix_dynamic_budget: bool = False
    prefix_concurrent_maintenance: bool = False
    prefix_disk_capacity: int | None = None
    prefix_pressure_bytes: int = 0
    prefix_ram_pressure_bytes: int = 0
    prefix_pending_writes: int = 0
    resident_dynamic_budget: bool = False
    resident_budget_limit: int = 0
    resident_weight_dynamic_budget: bool = False
    resident_weight_budget_limit: int = 0
    resident_memory_used: int = 0
    reclaimable_weight_bytes: int = 0


class _RuntimeLoadContext:
    """Job-context subset used by request-driven model activation."""

    def __init__(self) -> None:
        self._cleanup_callbacks: list[Callable[[], Awaitable[None] | None]] = []
        self.job_id = uuid4()
        self.cancel_requested = False

    def raise_if_cancelled(self) -> None:
        return None

    def add_cleanup(self, callback: Callable[[], Awaitable[None] | None]) -> None:
        self._cleanup_callbacks.append(callback)

    async def progress(
        self,
        value: float,
        *,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        del value, message, data

    async def log(self, message: str, **options: Any) -> None:
        del message, options

    async def cleanup(self) -> None:
        for callback in reversed(self._cleanup_callbacks):
            with suppress(Exception):
                result = callback()
                if result is not None:
                    await result


class _LeasedRealtimeConnector:
    """Hold a managed runtime resident for one upstream WebSocket lifetime."""

    def __init__(
        self,
        pool: RuntimePool,
        instance: _Runtime,
        connector: Any,
    ) -> None:
        self.pool = pool
        self.instance = instance
        self.connector = connector
        self.leased = False

    async def __aenter__(self) -> Any:
        async with self.pool._lock:
            if (
                self.pool._instances.get(self.instance.id) is not self.instance
                or self.instance.state
                not in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
            ):
                raise BackendError(
                    "model_not_ready",
                    "realtime runtime is no longer available",
                    retryable=True,
                    status_code=409,
                )
            self.instance.control_leases += 1
            self.leased = True
        try:
            return await self.connector.__aenter__()
        except BaseException:
            await self._release()
            raise

    async def __aexit__(self, *error: object) -> bool | None:
        try:
            return await self.connector.__aexit__(*error)
        finally:
            await self._release()

    async def _release(self) -> None:
        if not self.leased:
            return
        self.leased = False
        await self.pool._release_control_lease(self.instance)


class _CachedLoadFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_id: str
    detail: ErrorDetail
    failed_at: float


class RuntimePool:
    """Own local runtime processes while retaining an optional external fallback."""

    def __init__(
        self,
        catalog: ModelCatalog,
        executable: str | Path,
        *,
        fallback: ChatBackend | None = None,
        startup_timeout_seconds: float = 1800.0,
        max_instances: int = 2,
        max_requests_per_instance: int = 1,
        max_queued_requests_per_instance: int | None = None,
        queue_timeout_seconds: float = 0.0,
        max_runtime_memory_bytes: int | None = None,
        default_idle_ttl_seconds: int | None = None,
        load_failure_cooldown_seconds: float = 30.0,
        metric_interval_seconds: float = 2.0,
        backend: str = "metal",
        voice_component: Any | None = None,
        runtime_environment: dict[str, str] | None = None,
        startup_loads: Sequence[ModelLoadRequest] = (),
        automatic_memory_budget: bool = True,
        shared_cache_reclaimer: Callable[[], int] | None = None,
        transport: str = "http",
    ) -> None:
        if max_instances < 1:
            raise ValueError("max_instances must be positive")
        if max_requests_per_instance < 1:
            raise ValueError("max_requests_per_instance must be positive")
        if (
            max_queued_requests_per_instance is not None
            and max_queued_requests_per_instance < 0
        ):
            raise ValueError("max_queued_requests_per_instance must be non-negative")
        if queue_timeout_seconds < 0:
            raise ValueError("queue_timeout_seconds must be non-negative")
        if max_runtime_memory_bytes is not None and max_runtime_memory_bytes < 1:
            raise ValueError("max_runtime_memory_bytes must be positive")
        if default_idle_ttl_seconds is not None and default_idle_ttl_seconds < 0:
            raise ValueError("default_idle_ttl_seconds must be non-negative")
        if load_failure_cooldown_seconds < 0:
            raise ValueError("load_failure_cooldown_seconds must be non-negative")
        self.catalog = catalog
        self.executable = Path(executable).expanduser().resolve()
        self.fallback = fallback
        self.startup_timeout_seconds = startup_timeout_seconds
        self.max_instances = max_instances
        self.max_requests_per_instance = max_requests_per_instance
        self.max_queued_requests_per_instance = (
            max(32, max_requests_per_instance * 4)
            if max_queued_requests_per_instance is None
            else max_queued_requests_per_instance
        )
        self.queue_timeout_seconds = max(0.0, queue_timeout_seconds)
        automatic_budget = (
            self._default_runtime_memory_budget(backend)
            if automatic_memory_budget and max_runtime_memory_bytes is None
            else None
        )
        self.max_runtime_memory_bytes = max_runtime_memory_bytes or automatic_budget
        self.memory_policy = RuntimeMemoryPolicy()
        self.inference_policy = RuntimeInferencePolicy()
        self.context_policy = RuntimeContextPolicy()
        self._context_policy_lock = asyncio.Lock()
        self._policy_model_limits: dict[str, int] = {}
        self._policy_prefix_limits: dict[str, int] = {}
        self._memory_configuration_job: UUID | None = None
        self.automatic_memory_budget = automatic_budget is not None
        self.default_idle_ttl_seconds = default_idle_ttl_seconds
        self.load_failure_cooldown_seconds = load_failure_cooldown_seconds
        self.metric_interval_seconds = max(0.25, metric_interval_seconds)
        if backend not in {"cuda", "hip", "metal"}:
            raise ValueError(f"unsupported native backend: {backend}")
        if transport not in {"stdio", "http"}:
            raise ValueError(f"unsupported native runtime transport: {transport}")
        self.backend = backend
        self.transport = transport
        self.voice_component = voice_component
        self.runtime_environment = dict(runtime_environment or {})
        self._startup_loads = [request.model_copy(deep=True) for request in startup_loads]
        self.shared_cache_reclaimer = shared_cache_reclaimer
        self._shared_cache_reclaims = 0
        self._shared_cache_released_bytes = 0
        self._shared_cache_reclaim_failures = 0
        self._shared_cache_pressure_checked_at = 0.0
        self._host_vm_counters: tuple[int, int] | None = None
        self._host_vm_pressure_until = 0.0
        self._memory_release_credit = MemoryReleaseCredit()
        self.store = None
        self._instances: dict[UUID, _Runtime] = {}
        self._loading_model_names: set[str] = set()
        self._deleting_model_paths: set[Path] = set()
        self._load_events: dict[str, asyncio.Event] = {}
        self._load_errors: dict[str, ErrorDetail] = {}
        self._load_failures: dict[str, _CachedLoadFailure] = {}
        self._runtime_revival_at: dict[str, float] = {}
        self._loading_artifact_ids: dict[str, str] = {}
        self._load_requests: dict[str, ModelLoadRequest] = {}
        self._reserved_ports: set[int] = set()
        self._load_ports: dict[str, int] = {}
        self._load_bytes: dict[str, int] = {}
        self._session_routes: dict[UUID, UUID] = {}
        self._last_instance_id: UUID | None = None
        self._lock = asyncio.Lock()
        self._request_state_changed = asyncio.Condition(self._lock)
        self._prefix_budget_lock = asyncio.Lock()
        self._resident_budget_lock = asyncio.Lock()
        self._resident_load_limits: dict[str, int] = {}
        self._resident_load_owners: dict[str, asyncio.Task[Any] | None] = {}
        self._prefix_disk_status: dict[str, int] = {}
        self._prefix_disk_checked_at = 0.0
        self._prefix_budget_failures = 0
        self._realtime_activation_lock = asyncio.Lock()
        self._idle_reaper_task: asyncio.Task[None] | None = None
        self._idle_reaper_wakeup = asyncio.Event()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._lease_release_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    @staticmethod
    def _default_runtime_memory_budget(backend: str) -> int | None:
        """Return a conservative automatic residency ceiling for Metal."""

        if backend != "metal":
            return None
        total_bytes = total_physical_memory()
        if total_bytes is None:
            return None
        if total_bytes <= 0:
            return None
        reserve = automatic_memory_reserve(total_bytes)
        if total_bytes <= reserve:
            return max(1, total_bytes * 3 // 4)
        return total_bytes - reserve

    async def start(self) -> None:
        """Start lifecycle monitors and configured startup models."""

        async with self._lock:
            if self._closed:
                raise RuntimeManagementError("runtime pool is closed")
            for instance in self._instances.values():
                if instance.monitor_task is None:
                    instance.monitor_task = asyncio.create_task(
                        self._monitor(instance),
                        name=f"mfq-server-runtime-monitor-{instance.id}",
                    )
            if self._idle_reaper_task is None or self._idle_reaper_task.done():
                self._idle_reaper_wakeup.clear()
                self._idle_reaper_task = asyncio.create_task(
                    self._idle_reaper(),
                    name="mfq-server-runtime-idle-reaper",
                )
            if self.voice_component is not None and self.voice_component.ready():
                task = asyncio.create_task(
                    self.enable_realtime(),
                    name="mfq-server-enable-voice-output",
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_task_done)
            startup_loads = tuple(self._startup_loads)
            self._startup_loads.clear()

        for index, request in enumerate(startup_loads):
            context = _RuntimeLoadContext()
            try:
                await self.load(
                    context,  # type: ignore[arg-type]
                    request.model_dump(mode="python"),
                )
            except BaseException:
                async with self._lock:
                    self._startup_loads[:0] = startup_loads[index:]
                raise
            finally:
                await context.cleanup()

    def _background_task_done(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.discard(task)
        if not task.cancelled():
            # Retrieve unexpected failures so optional background work cannot
            # produce an unobserved-task warning during server shutdown.
            task.exception()

    def register_started(
        self,
        *,
        artifact: DiscoveredModel,
        process: subprocess.Popen[bytes],
        backend: ChatBackend,
        port: int,
        load_request: ModelLoadRequest,
    ) -> UUID:
        """Register a ready process started before the server event loop exists."""

        if self._closed:
            raise RuntimeManagementError("runtime pool is closed")
        status = process.poll()
        if status is not None:
            raise RuntimeManagementError(
                f"initial runtime process has already exited with status {status}"
            )
        if any(
            item.artifact.resource.name == artifact.resource.name
            for item in self._instances.values()
        ):
            raise RuntimeConflictError(f"model is already loaded: {artifact.resource.name}")
        active_count = sum(
            item.state != RuntimeInstanceState.FAILED for item in self._instances.values()
        )
        if active_count >= self.max_instances:
            raise RuntimeConflictError("managed runtime instance limit reached")
        concurrency = self.resolve_concurrency(load_request)
        request_capacity = native_request_capacity(
            backend=self.backend,
            routed_expert_bytes=artifact.routed_expert_bytes,
            requested=concurrency.max_requests,
        )
        canonical_request = load_request.model_copy(
            update={"model": artifact.resource.name,
                    "context_size": self.resolve_context_size(artifact, load_request.context_size)}
        )
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=process,
            backend=backend,
            port=port,
            context_size=canonical_request.context_size,
            device_ids=self._cuda_device_ids(canonical_request) if self.backend == "cuda" else [],
            sampling_defaults=canonical_request.sampling_defaults,
            state=RuntimeInstanceState.READY,
            last_used_at=datetime.now(timezone.utc),
            idle_ttl_seconds=(
                canonical_request.idle_ttl_seconds
                if canonical_request.idle_ttl_seconds is not None
                else self.default_idle_ttl_seconds
            ),
            pinned=canonical_request.pin,
            request_slots=_RequestSlotGate(request_capacity),
            request_capacity=request_capacity,
            batch_capacity=request_capacity,
            concurrency_settings=concurrency,
            reserved_bytes=self._estimated_load_bytes(
                artifact,
                canonical_request,
            ),
        )
        self._instances[instance.id] = instance
        self._load_requests[artifact.resource.name] = canonical_request
        self._last_instance_id = instance.id
        return instance.id

    async def load(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await self._load(context, payload)
        finally:
            owner = asyncio.current_task()
            async with self._lock:
                for name in list(self._resident_load_owners):
                    if self._resident_load_owners[name] is owner:
                        self._resident_load_limits.pop(name, None)
                        self._resident_load_owners.pop(name, None)

    async def _load(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = ModelLoadRequest.model_validate(payload)
        concurrency = self.resolve_concurrency(request)
        await self._maintain_prefix_disk_budget(force=not self._prefix_disk_status)
        if self.backend == "metal" and request.device_ids not in ([], ["metal"]):
            raise _job_error("unsupported_device", "the Metal runtime accepts device 'metal'")
        if self.backend == "cuda" and any(not value.isdecimal() for value in request.device_ids):
            raise _job_error("unsupported_device", "CUDA device IDs must be non-negative integers")
        if self.backend == "cuda" and len(set(request.device_ids)) != len(request.device_ids):
            raise _job_error("unsupported_device", "CUDA device IDs must be unique")
        if self.backend == "cuda" and request.device_ids:
            known = {device.device_id or str(index) for index, device in enumerate(hardware_identity().gpu_memory)}
            if known and any(device not in known for device in request.device_ids):
                raise _job_error("unsupported_device", "a selected CUDA GPU is unavailable")
        try:
            artifact = await self.catalog.resolve(request.model, request.artifact_uri)
        except ModelArtifactNotFoundError as error:
            raise _job_error(
                "model_artifact_not_found", f"model artifact was not found: {error}"
            ) from error
        async with self._lock:
            if artifact.path.resolve() in self._deleting_model_paths:
                raise _job_error("model_delete_busy", "Model files are being deleted; retry after deletion finishes")
        if not artifact.resource.loadable:
            raise _job_error(
                (
                    "model_conversion_required"
                    if artifact.resource.complete and artifact.resource.format == "hf"
                    else "model_artifact_incomplete"
                ),
                artifact.resource.error or "model artifact is incomplete",
            )
        from mfq.server.runtime.load_context import apply_load_context
        try:
            request = apply_load_context(request, artifact.resource.architecture, self.backend)
        except ValueError as error:
            raise _job_error("kv_ssd_required", "Disabling RAM preallocation requires SSD KV") from error
        if 'qsa_kv_offload' not in request.model_fields_set:
            saved_storage = self.context_policy.model_qsa_kv_offload.get(artifact.resource.name)
            if saved_storage is not None:
                request = request.model_copy(update={"qsa_kv_offload": QsaKvOffloadPolicy(**saved_storage.model_dump())})
        if 'kv_memory' not in request.model_fields_set:
            saved_memory = self.context_policy.model_kv_memory.get(artifact.resource.name)
            if saved_memory is not None:
                request = request.model_copy(update={"kv_memory": saved_memory})
        if request.kv_memory.allocation == "dynamic" and request.kv_memory.placement in {"ram", "auto_ssd"}:
            raise _job_error("kv_dynamic_unavailable", "Experimental dynamic KV requires VRAM or automatic without SSD")
        if request.kv_memory != KvMemorySettings() and (self.backend != "cuda" or
                not artifact.resource.architecture.casefold().startswith("qwen4_exp")):
            raise _job_error("kv_memory_unsupported", "Configurable KV placement currently requires the CUDA QSA backend")
        if 'kv_quantization' not in request.model_fields_set:
            saved_quantization = self.context_policy.model_kv_quantization.get(artifact.resource.name)
            if saved_quantization is not None:
                request = request.model_copy(update={"kv_quantization": saved_quantization})
        if request.kv_quantization.enabled and (self.backend not in {"cuda", "metal"} or not artifact.resource.architecture.casefold().startswith(
            ("qwen4_exp", "qwen3.5", "qwen3_5", "qwen35"))):
            raise _job_error("kv_quantization_unsupported", "TurboQuant KV is not supported by this model/backend")
        if request.yarn_enabled and (self.backend not in {"cuda", "metal"} or artifact.resource.yarn_context_capacity is None):
            raise _job_error("yarn_unsupported", "YaRN extension is not supported by this model/backend")
        resolved = self.resolve_context_size(artifact, request.context_size, yarn_enabled=request.yarn_enabled)
        request = request.model_copy(update={"context_size": resolved,
            "qsa_kv_offload": request.qsa_kv_offload.model_copy(update={"target_context": resolved})})
        if self.backend == "cuda" and artifact.resource.architecture.casefold().startswith("qwen4_exp"):
            await asyncio.to_thread(self._validate_kv_memory_budget, artifact, request)
        if request.qsa_kv_offload.enabled:
            await self._validate_qsa_kv_policy(artifact, request.qsa_kv_offload, yarn_enabled=request.yarn_enabled)
        if self.memory_policy.total_limit_bytes is not None or self.memory_policy.model_limit_bytes is not None or self.memory_policy.prefix_limit_bytes is not None:
            async with self._lock:
                current = [item for item in self._instances.values() if item.state != RuntimeInstanceState.FAILED]
                candidates = [item.artifact for item in current]
                new_model = all(item.artifact.resource.name != artifact.resource.name for item in current)
                limits, hot = self._plan_memory_policy(self.memory_policy, [*candidates, artifact] if new_model else candidates)
                resize = self.memory_policy.total_limit_bytes is None and self._memory_configuration_job is None and new_model and any(
                    (item.memory.resident_weight_bytes if item.memory and item.memory.resident_weight_bytes is not None else item.reserved_bytes or 0)
                    > limits.get(item.artifact.resource.name, self.max_runtime_memory_bytes or (1 << 63))
                    for item in current)
                if not resize and self._memory_configuration_job is None:
                    self._policy_model_limits = limits
                    self._policy_prefix_limits = hot
            if resize:
                await self.configure_memory_policy(_MemoryPlanContext(context, end=0.7), self.memory_policy.model_dump(), additional=artifact)
                context = _MemoryPlanContext(context, start=0.7)
        # Automatic residency is recalculated whenever an idle model is restored.
        replay_request = request.model_copy(deep=True)
        async with self._lock:
            if self._closed:
                raise RuntimeManagementError("runtime pool is closed")
            if self._memory_configuration_job is not None and self._memory_configuration_job != context.job_id:
                raise _job_error("memory_plan_busy", "memory settings are being applied", retryable=True)
            residency_ceiling = self._effective_runtime_memory_budget_locked()
            if self.backend == "metal" and artifact.routed_expert_bytes > 0 and residency_ceiling is not None:
                residency_ceiling = self._expert_load_headroom_locked(residency_ceiling, artifact.resource.name)
            if self.memory_policy.model_limit_bytes is not None:
                other_weights = sum((item.memory.resident_weight_bytes if item.memory and item.memory.resident_weight_bytes is not None else item.reserved_bytes or 0)
                    for item in self._instances.values() if item.artifact.resource.name != artifact.resource.name)
                other_weights += sum(size for name, size in self._load_bytes.items() if name != artifact.resource.name)
                available_weights = max(1, self.memory_policy.model_limit_bytes - other_weights)
                available_weights = min(available_weights, self._policy_model_limits.get(artifact.resource.name, available_weights))
                residency_ceiling = min(residency_ceiling, available_weights) if residency_ceiling is not None else available_weights
            elif artifact.resource.name in self._policy_model_limits:
                limit = self._policy_model_limits[artifact.resource.name]
                residency_ceiling = min(residency_ceiling, limit) if residency_ceiling is not None else limit
        if self.memory_policy.total_limit_bytes is not None or self.memory_policy.model_limit_bytes is not None or self.memory_policy.prefix_limit_bytes is not None:
            request = request.model_copy(update={"moe_gpu_cache_gb": None})
        request = self._apply_automatic_expert_residency(
            artifact,
            request,
            memory_ceiling=residency_ceiling,
        )
        if self.backend == "metal" and artifact.routed_expert_bytes > 0:
            gib = float(1 << 30)
            cache_gb = request.moe_gpu_cache_gb
            policy = (
                f"paged experts, cache={cache_gb:.2f} GiB"
                if cache_gb is not None and cache_gb > 0
                else "native automatic expert cache"
                if cache_gb is None and artifact.resource.format == "hf"
                else "fully resident experts"
            )
            ceiling_text = (
                f"{residency_ceiling / gib:.2f} GiB"
                if residency_ceiling is not None else "unlimited"
            )
            await context.log(
                f"Metal residency: {policy}; budget={ceiling_text}; "
                f"checkpoint={artifact.resource.total_bytes / gib:.2f} GiB; "
                f"row-streamed={artifact.always_streamed_bytes / gib:.2f} GiB; "
                f"experts={artifact.routed_expert_bytes / gib:.2f} GiB"
            )
        request_capacity = native_request_capacity(
            backend=self.backend,
            routed_expert_bytes=artifact.routed_expert_bytes,
            requested=self.resolve_concurrency(request).max_requests,
        )
        model_name = artifact.resource.name
        incoming_bytes = self._estimated_load_bytes(
            artifact,
            request,
        )
        async with self._lock:
            if self._closed:
                raise RuntimeManagementError("runtime pool is closed")
            if (
                self.max_runtime_memory_bytes is not None
                and incoming_bytes > self.max_runtime_memory_bytes
            ):
                raise _job_error(
                    "runtime_model_too_large",
                    f"model has an estimated resident set of {incoming_bytes} bytes "
                    f"but the runtime memory budget is "
                    f"{self.max_runtime_memory_bytes} bytes",
                )
        if self.backend == "metal":
            total = self._resident_total_limit()
            reservation = incoming_bytes + min(256 << 20, (total or incoming_bytes) // 20)
            try:
                await self._rebalance_resident_budget(additional_bytes=min(reservation, total) if total else reservation, prospective_model=model_name)
            except BackendError as error:
                if error.code == "resident_budget_exhausted":
                    async with self._lock:
                        ceiling = self._effective_runtime_memory_budget_locked() or 0
                        remaining = max(0, ceiling - self._committed_pool_bytes_locked())
                    raise _job_error("runtime_memory_limit", f"runtime memory budget reached: model needs {incoming_bytes:,} B; remaining capacity under the runtime budget: {remaining:,} B; non-offloadable weights and live KV cannot yield more memory", retryable=True) from error
                raise _job_error(error.code, error.message, retryable=error.retryable) from error
        await self._reclaim_shared_cache_for_budget(
            additional_bytes=incoming_bytes,
            prospective_model=model_name,
        )
        await self._trim_idle_prefix_caches_for_budget(
            additional_bytes=incoming_bytes,
            prospective_model=model_name,
        )
        evicted: list[_Runtime] = []
        async with self._lock:
            if self._closed:
                raise RuntimeManagementError("runtime pool is closed")
            if artifact.path.resolve() in self._deleting_model_paths:
                raise _job_error("model_delete_busy", "Model files are being deleted; retry after deletion finishes")
            existing = next(
                (
                    item
                    for item in self._instances.values()
                    if item.artifact.resource.name == model_name
                ),
                None,
            )
            if existing is not None and existing.state == RuntimeInstanceState.FAILED:
                self._detach_instance_locked(existing)
                existing = None
            revision_changed = existing is not None and not self._same_artifact_revision(
                existing.artifact,
                artifact,
            )
            replace_named_revision = (
                revision_changed
                and request.artifact_uri is None
                and request.model == model_name
            )
            if existing is not None and replace_named_revision:
                if (
                    existing.state != RuntimeInstanceState.READY
                    or existing.active_requests
                    or existing.queued_requests
                    or existing.control_leases
                ):
                    raise _job_error(
                        "runtime_revision_busy",
                        "the model artifact changed while its current runtime is busy",
                        retryable=True,
                    )
                evicted.append(existing)
                existing = None
            if existing is not None:
                raise _job_error(
                    "model_already_loaded",
                    f"model is already loaded by runtime instance {existing.id}",
                )
            if model_name in self._loading_model_names:
                raise _job_error(
                    "model_already_loading",
                    f"model is already loading: {model_name}",
                    retryable=True,
                )
            if self.memory_policy.model_limit_bytes is not None:
                committed_weights = sum((item.memory.resident_weight_bytes if item.memory and item.memory.resident_weight_bytes is not None else item.reserved_bytes or 0)
                    for item in self._instances.values() if item not in evicted)
                committed_weights += sum(self._load_bytes.values())
                if committed_weights + incoming_bytes > self.memory_policy.model_limit_bytes:
                    raise _job_error("runtime_memory_limit", "model weights exceed the configured aggregate budget; retry after other loads finish", retryable=True)
            claimed_instance_ids = {item.id for item in evicted}
            resident_names = {
                item.artifact.resource.name
                for item in self._instances.values()
                if item.state != RuntimeInstanceState.FAILED
                and item.id not in claimed_instance_ids
            }
            active_count = len(resident_names) + sum(
                name not in resident_names for name in self._loading_model_names
            )
            committed_bytes = max(
                0,
                self._committed_pool_bytes_locked()
                - sum(self._committed_runtime_bytes(item) for item in evicted),
            )
            memory_ceiling = self._effective_runtime_memory_budget_locked()
            while (
                active_count >= self.max_instances
                or (
                    memory_ceiling is not None
                    and committed_bytes + incoming_bytes > memory_ceiling
                )
            ):
                victim = self._lru_instance_for_unload_locked(
                    excluded_ids=claimed_instance_ids,
                )
                if victim is None:
                    memory_limited = (
                        memory_ceiling is not None
                        and committed_bytes + incoming_bytes > memory_ceiling
                    )
                    raise _job_error(
                        "runtime_memory_limit" if memory_limited else "runtime_instance_limit",
                        (
                            f"runtime memory budget reached: model needs {incoming_bytes:,} B; "
                            f"remaining capacity under the runtime budget: "
                            f"{max(0, memory_ceiling - committed_bytes):,} B"
                            if memory_limited
                            else "managed runtime instance limit reached; all instances "
                            "are pinned or busy"
                        ),
                        retryable=True,
                    )
                if memory_ceiling is not None and committed_bytes + incoming_bytes > memory_ceiling and (
                    self.memory_policy.total_limit_bytes is not None or getattr(victim, "resident_dynamic_budget", False)
                ):
                    self._resident_load_limits.pop(model_name, None)
                    raise _job_error("runtime_memory_limit", "the shared resident budget cannot accommodate the model without discarding active KV or non-offloadable weights", retryable=True)
                evicted.append(victim)
                claimed_instance_ids.add(victim.id)
                active_count = max(0, active_count - 1)
                committed_bytes = max(
                    0,
                    committed_bytes - self._committed_runtime_bytes(victim),
                )
            prefix_limit = self._reserve_prefix_capacity_locked(model_name, claimed_instance_ids, request)
            request = request.model_copy(update={"prefix_cache_hot_bytes": prefix_limit, "prefix_cache_max_bytes": prefix_limit})
            port = self._reserve_free_port_locked() if self.transport == "http" else 0
            for victim in evicted:
                self._mark_instance_unloading_locked(victim)
            load_event = asyncio.Event()
            self._loading_model_names.add(model_name)
            self._load_events[model_name] = load_event
            self._load_errors.pop(model_name, None)
            self._load_failures.pop(model_name, None)
            self._loading_artifact_ids[model_name] = artifact.resource.id
            if port:
                self._load_ports[model_name] = port
            self._load_bytes[model_name] = incoming_bytes

        if evicted:
            try:
                for victim in evicted:
                    await context.log(
                        f"Evicting idle runtime {victim.artifact.resource.name} "
                        f"before loading {model_name}"
                    )
                    await self._retire_instance(victim)
            except BaseException as error:
                for victim in evicted:
                    with suppress(Exception):
                        await self._retire_instance(victim)
                async with self._lock:
                    self._finish_model_load_locked(
                        model_name,
                        load_event,
                        self._runtime_load_error(error),
                    )
                raise

        try:
            if self.backend == "cuda" and memory_topology()["memory_architecture"] == "discrete":
                # Disposable prefixes yield before admitting new model weights.
                await self._rebalance_prefix_cache_budget(reclaim_ram=True)
            command, process_environment = self._launch_configuration(
                artifact,
                request,
                port=port,
            )
            await context.progress(0.02, message="Starting runtime process")
            process = await asyncio.create_subprocess_exec(
                *command,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                stdin=(
                    asyncio.subprocess.PIPE
                    if self.transport == "stdio"
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=(
                    asyncio.subprocess.PIPE
                    if self.transport == "stdio"
                    else asyncio.subprocess.STDOUT
                ),
                start_new_session=True,
                env=process_environment,
                limit=64 * 1024 * 1024,
            )
        except BaseException as error:
            async with self._lock:
                self._finish_model_load_locked(
                    model_name,
                    load_event,
                    self._runtime_load_error(error),
                )
            raise
        configured_video_library = process_environment.get("MFQ_AVFOUNDATION_VIDEO_LIBRARY")
        avfoundation_video_library = (
            Path(configured_video_library).expanduser().resolve()
            if configured_video_library
            else find_native_runtime_resource(
                self.executable,
                "libmfq_avfoundation_video.dylib",
            )
        )
        runtime_client = (
            StdioRuntimeClient(process)
            if self.transport == "stdio"
            else HttpRuntimeClient(f"http://127.0.0.1:{port}")
        )
        backend = OpenAIChatBackend(
            runtime_client,
            local_tensor_files=True,
            model_type=artifact.resource.architecture,
            avfoundation_video_library=(
                avfoundation_video_library
                if avfoundation_video_library is not None
                and avfoundation_video_library.is_file()
                else None
            ),
        )
        instance = _Runtime(
            id=uuid4(),
            artifact=artifact,
            process=process,
            backend=backend,
            port=port,
            context_size=request.context_size,
            device_ids=self._cuda_device_ids(request) if self.backend == "cuda" else [],
            sampling_defaults=request.sampling_defaults,
            idle_ttl_seconds=(
                request.idle_ttl_seconds
                if request.idle_ttl_seconds is not None
                else self.default_idle_ttl_seconds
            ),
            pinned=request.pin,
            request_slots=_RequestSlotGate(request_capacity),
            request_capacity=request_capacity,
            batch_capacity=request_capacity,
            concurrency_settings=concurrency,
            reserved_bytes=incoming_bytes,
            resident_budget_limit=int(process_environment.get("MFQ_SERVER_RESIDENT_BUDGET_BYTES", "0")),
            # Keep TTL and memory-pressure eviction out of the load
            # transaction until the ready instance has been published.
            control_leases=1,
            log_to_load_job=True,
        )
        try:
            async with self._lock:
                if self._closed:
                    raise RuntimeManagementError("runtime pool is closed")
                self._instances[instance.id] = instance
        except BaseException as error:
            try:
                await self._stop_process(instance)
            finally:
                async with self._lock:
                    self._finish_model_load_locked(
                        model_name,
                        load_event,
                        self._runtime_load_error(error),
                    )
            raise

        keep_process = False

        async def cleanup_failed_start() -> None:
            if keep_process:
                return
            try:
                await self._retire_instance(instance)
            finally:
                async with self._lock:
                    self._finish_model_load_locked(
                        model_name,
                        load_event,
                        ErrorDetail(
                            code="runtime_start_failed",
                            message=f"runtime failed while loading {model_name}",
                            retryable=True,
                        ),
                    )

        context.add_cleanup(cleanup_failed_start)
        instance.output_task = asyncio.create_task(
            self._pump_output(instance, context),
            name=f"mfq-server-runtime-output-{instance.id}",
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout_seconds
        while loop.time() < deadline:
            context.raise_if_cancelled()
            status = process.returncode
            if status is not None:
                if instance.output_task is not None:
                    await instance.output_task
                if instance.error is not None and instance.error.code.startswith("kv_"):
                    raise _job_error(instance.error.code, instance.error.message)
                raise _job_error(
                    "runtime_start_failed",
                    f"runtime exited during startup with status {status}",
                    retryable=True,
                )
            try:
                request_timeout = min(1.0, max(0.1, deadline - loop.time()))
                capabilities = await asyncio.wait_for(
                    backend.capabilities(), timeout=request_timeout
                )
            except (BackendError, asyncio.TimeoutError):
                await asyncio.sleep(0.25)
                continue
            if capabilities.model != artifact.resource.name:
                raise _job_error(
                    "runtime_identity_mismatch",
                    "runtime health returned an unexpected model identity",
                )
            instance.mtp_supported = capabilities.model_capabilities.features.mtp
            instance.mtp_available = capabilities.mtp_available
            self._apply_batch_capacity(instance, capabilities.batch_capacity)
            break
        else:
            raise _job_error(
                "runtime_start_timeout",
                "runtime did not become ready before the startup timeout",
                retryable=True,
            )

        pid = getattr(process, "pid", None)
        if isinstance(pid, int) and pid > 0:
            instance.resident_bytes = await asyncio.to_thread(
                self._process_resident_bytes,
                pid,
            )
        await self._refresh_instance_usage(instance)
        if request.qsa_kv_offload.enabled:
            native_status = await backend.runtime_status()
            if not instance.qsa_kv_offload_supported or native_status.get("qsa_kv_offload_enabled") != 1:
                raise _job_error("qsa_kv_offload_unavailable", "the native runtime did not enable QSA KV offload")
        if request.kv_quantization.enabled:
            native_status = await backend.runtime_status()
            if not instance.kv_quantization_supported or native_status.get("kv_quantization_bits") != request.kv_quantization.bits:
                raise _job_error("kv_quantization_unavailable", "the native runtime did not enable the requested TurboQuant KV precision")
        await self._rebalance_resident_budget(instance.id, activating=instance)
        instance.monitor_task = asyncio.create_task(
            self._monitor(instance), name=f"mfq-server-runtime-monitor-{instance.id}"
        )
        if self.voice_component is not None and self.voice_component.ready():
            await context.progress(0.96, message="Enabling voice output")
            async with self._realtime_activation_lock:
                await self._activate_realtime_instance(instance)
        await context.progress(1.0, message="Model ready")
        async with self._lock:
            if (
                self._instances.get(instance.id) is not instance
                or instance.state != RuntimeInstanceState.LOADING
                or process.returncode is not None
            ):
                raise _job_error(
                    "runtime_start_failed",
                    "runtime exited before model activation completed",
                    retryable=True,
                )
            self._load_requests[model_name] = replay_request.model_copy(
                update={"model": model_name}
            )
            instance.state = RuntimeInstanceState.READY
            instance.log_to_load_job = False
            instance.last_used_at = datetime.now(timezone.utc)
            self._last_instance_id = instance.id
            instance.control_leases = max(0, instance.control_leases - 1)
            self._finish_model_load_locked(model_name, load_event, None)
            keep_process = True
        return {
            "instance_id": str(instance.id),
            "model_id": artifact.resource.name,
            "model": artifact.resource.name,
            "artifact_id": artifact.resource.id,
            "context_size": request.context_size,
        }

    async def configure_inference_policy(self, policy: RuntimeInferencePolicy) -> dict[str, Any]:
        async with self._lock:
            if self.store is not None:
                await asyncio.to_thread(self.store.save_runtime_inference_policy, policy.model_dump())
            self.inference_policy = policy.model_copy()
        return self.inference_policy.model_dump()

    async def _validate_qsa_kv_policy(self, artifact: DiscoveredModel, policy: Any, *, yarn_enabled: bool | None = None) -> int:
        if self.backend not in {"metal", "cuda"}:
            raise _job_error("qsa_kv_offload_unsupported", "KV offload requires a Metal or CUDA QSA backend")
        try:
            return await asyncio.to_thread(qsa_index_requirement, artifact, policy.target_context,
                policy.budget_bytes if policy.enabled else None, maximum_context=self._context_limit(artifact, yarn_enabled), cuda=self.backend == "cuda")
        except (ValueError, OSError, KeyError) as error:
            raise _job_error("qsa_kv_configuration_invalid", str(error)) from error

    async def qsa_kv_offload_info(self, instance_id: UUID, target_context: int | None = None) -> dict[str, Any]:
        async with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise BackendError("runtime_instance_not_found", "runtime instance was not found", status_code=404)
            request = self._load_requests[instance.artifact.resource.name]
            supported = instance.qsa_kv_offload_supported
            architecture = instance.artifact.resource.architecture
        policy = request.qsa_kv_offload
        target = target_context or instance.context_size
        policy = policy.model_copy(update={"target_context": instance.context_size})
        required = await self._validate_qsa_kv_policy(instance.artifact, policy.model_copy(update={"target_context": target})) if supported else None
        return {"supported": supported, "architecture": architecture, "policy": policy.model_dump(),
            "required_index_bytes": required, "buffer_bytes": min(QSA_KV_BUFFER_BYTES, policy.budget_bytes // 4),
            "target_context": target, "reason": None if supported else "model_or_backend_unsupported"}

    def _qsa_kv_policy(self, model: str) -> QsaKvOffloadPolicy:
        request = self._load_requests.get(model)
        return request.qsa_kv_offload if request is not None else QsaKvOffloadPolicy()

    def _kv_quantization_policy(self, model: str) -> KvQuantizationSettings:
        request = self._load_requests.get(model)
        return request.kv_quantization if request is not None else KvQuantizationSettings()

    def _kv_memory_policy(self, model: str) -> KvMemorySettings:
        request = self._load_requests.get(model)
        return request.kv_memory if request is not None else KvMemorySettings()

    async def configure_qsa_kv_offload(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        settings = ConfigureQsaKvOffloadRequest.model_validate(payload)
        async with self._lock:
            instance = self._instances.get(settings.instance_id)
            if instance is None:
                raise _job_error("runtime_instance_not_found", "runtime instance was not found")
            if not instance.qsa_kv_offload_supported:
                raise _job_error("qsa_kv_offload_unsupported", "the loaded model or native backend does not support QSA KV offload")
            previous = self._load_requests[instance.artifact.resource.name].model_copy(deep=True)
            target = instance.context_size
        policy = QsaKvOffloadPolicy(**settings.model_dump(exclude={"instance_id"}), target_context=target)
        if settings.enabled:
            await self._validate_qsa_kv_policy(instance.artifact, policy)
        updated = ModelLoadRequest.model_validate({**previous.model_dump(), "qsa_kv_offload": policy.model_dump(),
            "context_size": target,
            "pin": instance.pinned, "idle_ttl_seconds": instance.idle_ttl_seconds})
        async with self._lock:
            if self._instances.get(instance.id) is not instance or instance.state != RuntimeInstanceState.READY or \
                instance.active_requests or instance.queued_requests or instance.control_leases or instance.context_size != target:
                raise _job_error("runtime_busy", "wait for model requests to finish before changing QSA KV storage", retryable=True)
            self._mark_instance_unloading_locked(instance)
            self._load_requests[instance.artifact.resource.name] = updated
        try:
            await context.progress(0.01, message="Reloading model with QSA KV storage settings")
            await self._retire_instance(instance)
            result = await self.load(context, updated.model_dump(mode="json"))
            return {**result, "qsa_kv_offload": policy.model_dump()}
        except BaseException:
            self._load_requests[instance.artifact.resource.name] = previous
            for item in list(self._instances.values()):
                if item.artifact.resource.name == instance.artifact.resource.name:
                    await self._retire_instance(item)
            try:
                await self.load(_MemoryPlanContext(context, recovery=True), previous.model_dump(mode="json"))
            except BaseException as error:
                await context.log(f"Could not restore the previous model configuration: {type(error).__name__}")
            raise

    def _apply_inference_policy(self, sampling: SamplingParams) -> SamplingParams:
        if self.inference_policy.mtp_enabled or not sampling.enable_mtp:
            return sampling
        return sampling.model_copy(update={"enable_mtp": False})

    def memory_policy_status(self) -> dict[str, Any]:
        effective_total = self._effective_runtime_memory_budget_locked()
        automatic_total = effective_total if self.memory_policy.total_limit_bytes is None else self._effective_runtime_memory_budget_locked(
            self.memory_policy.model_copy(update={"total_limit_bytes": None}))
        return {**memory_topology(), **self.memory_policy.model_dump(), "actual_prefix_directory": str(self._prefix_cache_directory()),
            "effective_total_limit_bytes": effective_total,
            "automatic_total_limit_bytes": automatic_total,
            "total_capacity_limit_bytes": self._resident_total_limit(),
            "effective_prefix_limit_bytes": self._prefix_total_limit(),
            "effective_prefix_ram_limit_bytes": self._prefix_ram_total_limit(), **self._prefix_disk_status}

    def _resident_total_limit(self, policy: RuntimeMemoryPolicy | None = None) -> int | None:
        requested = (policy or self.memory_policy).total_limit_bytes
        if requested is None:
            return self.max_runtime_memory_bytes
        return min(requested, self.max_runtime_memory_bytes) if self.max_runtime_memory_bytes is not None else requested

    async def _rebalance_resident_budget(self, preferred: UUID | None = None, *, additional_bytes: int = 0, prospective_model: str | None = None, activating: _Runtime | None = None, wait: bool = True) -> int | None:
        total = self._resident_total_limit()
        if total is None:
            await self._rebalance_weight_budget(activating=activating)
            return None
        if not wait and self._resident_budget_lock.locked():
            return None
        devices = await asyncio.to_thread(gpu_memory_status) if self.backend == "cuda" else []
        async with self._resident_budget_lock:
            async with self._lock:
                current = [item for item in self._instances.values() if item.state != RuntimeInstanceState.FAILED]
                if prospective_model is not None and (prospective_model in self._loading_model_names or any(item.artifact.resource.name == prospective_model for item in current)):
                    return None
                if self.memory_policy.total_limit_bytes is not None and any(
                    (item.state != RuntimeInstanceState.LOADING or item is activating)
                    and not getattr(item, "resident_dynamic_budget", False) for item in current
                ):
                    raise BackendError("resident_budget_unsupported", "this runtime does not support online shared resident budgets", status_code=501)
                idle = [item for item in current if (item.state == RuntimeInstanceState.READY
                    or item is activating and item.state == RuntimeInstanceState.LOADING)
                    and not item.active_requests and not item.queued_requests
                    and item.control_leases == getattr(item, "read_control_leases", 0) + (1 if item is activating else 0)
                    and getattr(item, "resident_dynamic_budget", False)]
                if not idle and not additional_bytes:
                    return None
                if self.memory_policy.total_limit_bytes is None and any((item.state != RuntimeInstanceState.LOADING or item is activating)
                    and not getattr(item, "resident_dynamic_budget", False) for item in current):
                    return None
                effective = self._effective_runtime_memory_budget_locked()
                if effective is not None:
                    total = min(total, effective)
                fixed = sum(getattr(item, "resident_budget_limit", 0) or self._committed_runtime_bytes(item)
                    for item in current if item not in idle)
                names = {item.artifact.resource.name for item in current}
                fixed += sum(size for name, size in self._resident_load_limits.items() if name not in names)
                available = total - fixed - additional_bytes
                if not idle:
                    if available < 0 and additional_bytes:
                        raise BackendError("resident_budget_exhausted", "the shared resident budget has no capacity for this load", status_code=413)
                    if prospective_model is not None:
                        additional_bytes += max(0, available)
                        self._resident_load_limits[prospective_model] = additional_bytes
                        self._resident_load_owners[prospective_model] = asyncio.current_task()
                    return additional_bytes or None
                reserve = min(256 << 20, total // 20)
                ordered = sorted(idle, key=lambda item: (item.id == preferred, item.last_used_at or item.started_at))
                targets = {item.id: item.resident_memory_used + reserve for item in idle}
                excess = max(0, sum(targets.values()) - available)
                trims: dict[UUID, int] = {}
                for item in ordered:
                    hot = item.memory.prefix_cache_bytes if item.memory else 0
                    hot = hot or 0
                    released = min(excess, hot)
                    if released:
                        trims[item.id] = hot - released
                        targets[item.id] -= released
                        excess -= released
                for item in ordered:
                    released = min(excess, item.reclaimable_weight_bytes)
                    targets[item.id] -= released
                    excess -= released
                if excess or available < 0:
                    raise BackendError("resident_budget_exhausted", "non-offloadable weights, live KV and execution headroom exceed the shared resident budget", status_code=413)
                if not additional_bytes:
                    targets[ordered[-1].id] += max(0, available - sum(targets.values()))
                elif prospective_model is not None:
                    additional_bytes += max(0, available - sum(targets.values()))
                # A global VRAM policy is an upper bound, not a shared physical
                # heap. Each single-device worker must stay within its own GPU.
                for device in devices:
                    free = device.get("memory_available_bytes")
                    if free is None:
                        continue
                    assigned = [item for item in idle if getattr(item, "device_ids", []) == [device["id"]]]
                    occupied = [item for item in current if item not in idle
                        and getattr(item, "device_ids", []) == [device["id"]]]
                    room = free + sum(item.resident_memory_used for item in assigned)
                    room -= sum(max(0, getattr(item, "resident_budget_limit", 0) - item.resident_memory_used)
                        for item in occupied)
                    for item in ordered:
                        if item in assigned:
                            target = min(targets[item.id], max(0, room))
                            if target <= 0:
                                raise BackendError("resident_budget_exhausted", "selected GPU has no free resident capacity", status_code=413)
                            targets[item.id] = target
                            room -= target
                if prospective_model is not None:
                    self._resident_load_limits[prospective_model] = additional_bytes
                    self._resident_load_owners[prospective_model] = asyncio.current_task()
                previous = {item.id: item.resident_budget_limit for item in idle}
                previous_weights = {item.id: getattr(item, "resident_weight_budget_limit", 0) for item in idle}
                weight_targets = {item.id: self._policy_model_limits.get(item.artifact.resource.name, 0)
                    if self.memory_policy.model_limit_bytes is not None else 0 for item in idle}
                weight_changes = {item.id for item in idle if getattr(item, "resident_weight_dynamic_budget", False)
                    and weight_targets[item.id] != previous_weights[item.id]}
                changed = [item for item in idle if targets[item.id] != previous[item.id] or item.id in trims or item.id in weight_changes]
                for item in changed:
                    item.control_leases += 1
            applied = []
            try:
                for item in ordered:
                    if item.id in trims:
                        result = await item.backend.trim_runtime_cache(trims[item.id])
                        if result.get("status") != "ok":
                            raise RuntimeError("prefix RAM could not be reclaimed for the shared resident budget")
                changes = sorted(changed, key=lambda item: (targets[item.id] > previous[item.id] and previous[item.id] != 0, targets[item.id]))
                for item in changes:
                    if targets[item.id] == previous[item.id] and item.id not in weight_changes:
                        continue
                    kwargs = {"weight_target_bytes": weight_targets[item.id]} if item.id in weight_changes else {}
                    result = await item.backend.set_resident_memory_budget(targets[item.id], **kwargs)
                    if result.get("status") != "ok":
                        raise RuntimeError("native shared resident budget was not applied")
                    applied.append(item)
                    item.resident_budget_limit = targets[item.id]
                    if item.id in weight_changes:
                        item.resident_weight_budget_limit = weight_targets[item.id]
            except BaseException:
                if prospective_model is not None:
                    self._resident_load_limits.pop(prospective_model, None)
                for item in reversed(applied):
                    try:
                        kwargs = {"weight_target_bytes": previous_weights[item.id]} if item.id in weight_changes else {}
                        await item.backend.set_resident_memory_budget(previous[item.id], **kwargs)
                        item.resident_budget_limit = previous[item.id]
                        item.resident_weight_budget_limit = previous_weights[item.id]
                    except Exception:
                        pass
                raise
            finally:
                for item in changed:
                    await self._release_control_lease(item)
            if wait:
                for item in changed:
                    await self._refresh_instance_usage(item)
            return additional_bytes or targets[ordered[-1].id]

    async def _rebalance_weight_budget(self, *, activating: _Runtime | None = None) -> None:
        async with self._resident_budget_lock:
            async with self._lock:
                candidates = [item for item in self._instances.values()
                    if getattr(item, "resident_weight_dynamic_budget", False)
                    and (item.state == RuntimeInstanceState.READY or item is activating)
                    and not item.active_requests and not item.queued_requests
                    and item.control_leases == getattr(item, "read_control_leases", 0) + (1 if item is activating else 0)]
                previous = {item.id: getattr(item, "resident_weight_budget_limit", 0) for item in candidates}
                targets = {item.id: self._policy_model_limits.get(item.artifact.resource.name, 0)
                    if self.memory_policy.model_limit_bytes is not None else 0 for item in candidates}
                changes = [item for item in candidates if targets[item.id] != previous[item.id]]
                changes.sort(key=lambda item: targets[item.id] == 0 or targets[item.id] > previous[item.id] and previous[item.id] != 0)
                for item in changes:
                    item.control_leases += 1
            applied = []
            try:
                for item in changes:
                    result = await item.backend.set_resident_memory_budget(item.resident_budget_limit, weight_target_bytes=targets[item.id])
                    if result.get("status") != "ok":
                        raise RuntimeError("native model weight budget was not applied")
                    item.resident_weight_budget_limit = targets[item.id]
                    applied.append(item)
            except BaseException:
                for item in reversed(applied):
                    try:
                        await item.backend.set_resident_memory_budget(item.resident_budget_limit, weight_target_bytes=previous[item.id])
                        item.resident_weight_budget_limit = previous[item.id]
                    except Exception:
                        pass
                raise
            finally:
                for item in changes:
                    await self._release_control_lease(item)
            for item in changes:
                await self._refresh_instance_usage(item)

    def _prefix_cache_directory(self) -> Path:
        if self.memory_policy.prefix_directory is not None:
            return Path(self.memory_policy.prefix_directory).expanduser().resolve()
        if self.store is not None:
            return self.store.path.parent / "prefix-cache"
        return Path.home() / ".cache" / "mfq" / "prefix"

    def _plan_memory_policy(self, policy: RuntimeMemoryPolicy, artifacts: Sequence[DiscoveredModel]) -> tuple[dict[str, int], dict[str, int]]:
        if self.max_runtime_memory_bytes is not None and any(value is not None and value > self.max_runtime_memory_bytes for value in (policy.total_limit_bytes, policy.model_limit_bytes, policy.prefix_limit_bytes)):
            raise _job_error("memory_budget_exceeds_capacity", "a manual budget cannot exceed detected inference memory")
        if not artifacts:
            return {}, {}
        totals = [max(0, item.resource.total_bytes - item.always_streamed_bytes) for item in artifacts]
        experts = [min(total, item.routed_expert_bytes) for item, total in zip(artifacts, totals)]
        dense = [total - expert for total, expert in zip(totals, experts)]
        ceiling = self._resident_total_limit(policy)
        prefixes = policy.prefix_limit_bytes if policy.prefix_limit_bytes is not None else 2 << 30
        limit = policy.model_limit_bytes
        if ceiling is not None:
            available = max(0, ceiling - prefixes - (2 << 30))
            if policy.total_limit_bytes is not None:
                prefixes = min(prefixes, max(0, ceiling - sum(dense)))
                available = max(0, ceiling - min(2 << 30, ceiling // 20))
            limit = min(limit, available) if limit is not None else available
        if limit is None:
            limit = sum(totals)
        floors = [min(size, (64 << 20) if policy.total_limit_bytes is not None or policy.model_limit_bytes is not None else (5 << 30)) for size in experts]
        minimum = sum(dense) + sum(floors)
        if limit < minimum:
            raise _job_error("memory_budget_too_small", f"these models require at least {minimum} weight bytes plus prefix and live KV headroom")
        excess = max(0, sum(totals) - limit)
        expert_total = sum(experts)
        allowed_experts = max(0, expert_total - excess)
        remainder = max(0, allowed_experts - sum(floors))
        extra_total = sum(expert - floor for expert, floor in zip(experts, floors))
        limits = {item.resource.name: base + floor + remainder * (expert - floor) // max(1, extra_total)
            for item, base, expert, floor in zip(artifacts, dense, experts, floors)}
        if policy.model_limit_bytes is None and policy.prefix_limit_bytes is None and policy.total_limit_bytes is None:
            limits = {}
        share, remainder = divmod(prefixes, len(artifacts))
        hot = {item.resource.name: share + (index < remainder) for index, item in enumerate(artifacts)}
        return limits, hot

    def _prefix_total_limit(self) -> int:
        return self.memory_policy.prefix_limit_bytes if self.memory_policy.prefix_limit_bytes is not None else 2 << 30

    def _prefix_ram_total_limit(self) -> int:
        if self.backend != "cuda" or memory_topology()["memory_architecture"] != "discrete":
            return 0
        limit = self.memory_policy.prefix_ram_limit_bytes
        return limit if limit is not None else 2 << 30

    def _prefix_reserved_capacity_locked(self, excluded_ids: set[UUID] | None = None) -> int:
        current = [item for item in self._instances.values() if item.state != RuntimeInstanceState.FAILED
            and item.id not in (excluded_ids or set())]
        used = sum(item.memory.prefix_cache_limit_bytes if item.memory and item.memory.prefix_cache_limit_bytes is not None
            else self._policy_prefix_limits.get(item.artifact.resource.name, 2 << 30) for item in current)
        names = {item.artifact.resource.name for item in current}
        return used + sum(self._policy_prefix_limits.get(name, 0) for name in self._loading_model_names if name not in names)

    def _prefix_hot_cap(self, request: ModelLoadRequest | None) -> int:
        if request is None:
            return self._prefix_total_limit()
        if not request.prefix_cache_enabled:
            return 0
        explicit = request.prefix_cache_hot_bytes
        if explicit is None:
            explicit = request.prefix_cache_max_bytes
        return min(self._prefix_total_limit(), explicit) if explicit is not None else self._prefix_total_limit()

    def _reserve_prefix_capacity_locked(self, model: str, excluded_ids: set[UUID],
                                       request: ModelLoadRequest | None = None) -> int:
        remaining = max(0, self._prefix_total_limit() - self._prefix_reserved_capacity_locked(excluded_ids))
        limit = min(remaining, self._policy_prefix_limits.get(model, self._prefix_total_limit()),
                    self._prefix_hot_cap(request))
        self._policy_prefix_limits[model] = limit
        return limit

    async def _maintain_prefix_disk_budget(self, *, force: bool = False) -> None:
        async with self._prefix_budget_lock:
            if not force and time.monotonic() - self._prefix_disk_checked_at < 5.0:
                return
            async with self._lock:
                evict = not self._loading_model_names and not any(item.active_requests or item.queued_requests
                    or item.control_leases or item.prefix_pending_writes for item in self._instances.values())
            if not evict and not force:
                return
            result = await asyncio.to_thread(maintain_prefix_disk_budget,
                self._prefix_cache_directory(), self.memory_policy.prefix_disk_limit_bytes, evict=False)
            if evict and result["prefix_cache_total_disk_bytes"] > result["prefix_cache_total_disk_max_bytes"]:
                async with self._lock:
                    current = list(self._instances.values())
                    evict = not self._loading_model_names and self._memory_configuration_job is None and not any(
                        item.active_requests or item.queued_requests or item.control_leases or item.prefix_pending_writes
                        or item.state not in {RuntimeInstanceState.READY, RuntimeInstanceState.FAILED} for item in current)
                    leased = current if evict else []
                    for item in leased:
                        item.control_leases += 1
                if evict:
                    cleanup = asyncio.create_task(asyncio.to_thread(maintain_prefix_disk_budget,
                        self._prefix_cache_directory(), self.memory_policy.prefix_disk_limit_bytes, evict=True))
                    try:
                        result = await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        await cleanup
                        raise
                    finally:
                        for item in leased:
                            await self._release_control_lease(item)
            self._prefix_disk_status = {key: value for key, value in result.items() if key.startswith("prefix_cache_")}
            self._prefix_disk_checked_at = time.monotonic()

    async def reclaim_prefix_disk_for_download(self, destination: Path) -> None:
        from mfq.server.runtime.disk_budget import DiskBudget, volume
        from mfq.server.runtime.prefix_disk_budget import _scan_blocks, _remove_block
        import shutil
        directory = self._prefix_cache_directory()
        if volume(directory) != volume(destination):
            return
        async with self._prefix_budget_lock:
            blocks = await asyncio.to_thread(_scan_blocks, directory)
            used = sum(block.size for block in blocks)
            deficit = await asyncio.to_thread(lambda: DiskBudget().reserved(destination) - shutil.disk_usage(destination).free)
            target = max(0, used - deficit)
            if target >= used:
                self._prefix_disk_checked_at = 0.0
                return
            identities = {name: await asyncio.to_thread(prefix_cache_identity, directory, name)
                for name in {block.namespace for block in blocks}}
            async with self._lock:
                if self._loading_model_names or self._memory_configuration_job is not None:
                    raise _job_error("cache_reclaim_busy", "Wait for model loading or cache configuration before reclaiming SSD prefix space")
                current = [item for item in self._instances.values() if item.state != RuntimeInstanceState.FAILED]
                if any(not getattr(item, "prefix_concurrent_maintenance", False) and
                    (item.active_requests or item.queued_requests or item.control_leases) for item in current):
                    raise _job_error("cache_reclaim_busy", "This runtime must finish its current request before reclaiming SSD prefix space")
                for item in current:
                    item.control_leases += 1
                    item.read_control_leases += 1
            try:
                loaded_paths = {str(item.artifact.path) for item in current}
                active = {name for name, identity in identities.items()
                    if identity.get('model_path') in loaded_paths or current and not identity.get('model_path')}
                active_used = sum(block.size for block in blocks if block.namespace in active)
                active_target = target * active_used // max(1, used)
                for item in current:
                    share = active_target // len(current)
                    hot = item.memory.prefix_cache_limit_bytes if item.memory and item.memory.prefix_cache_limit_bytes is not None else 0
                    result = await item.backend.set_prefix_cache_budget(hot, disk_target_bytes=share)
                    if result.get('status') != 'ok':
                        raise _job_error("cache_reclaim_failed", "Could not reduce the SSD prefix cache budget")
                    item.prefix_disk_capacity = share
                offline_target = target - active_target
                def evict_offline():
                    offline = [block for block in blocks if block.namespace not in active]
                    remaining = sum(block.size for block in offline)
                    for block in sorted(offline, key=lambda block: block.stat.st_mtime_ns):
                        if remaining <= offline_target:
                            break
                        removed = _remove_block(block)
                        if removed:
                            remaining -= removed[0]
                await asyncio.to_thread(evict_offline)
                deadline = time.monotonic() + 30
                while await asyncio.to_thread(lambda: shutil.disk_usage(destination).free < DiskBudget().reserved(destination)):
                    if time.monotonic() >= deadline:
                        raise _job_error("cache_reclaim_busy", "Some SSD prefix blocks are still in use; retry the download after the current request")
                    await asyncio.sleep(.05)
                self._prefix_disk_checked_at = 0.0
            finally:
                for item in current:
                    await self._release_control_lease(item, read_only=True)

    async def _rebalance_prefix_cache_budget(self, preferred: UUID | None = None, *, wait: bool = True,
                                          reclaim_ram: bool = False) -> None:
        if not wait and self._prefix_budget_lock.locked():
            return
        async with self._prefix_budget_lock:
            async with self._lock:
                current = [item for item in self._instances.values()
                    if item.state in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}]
                candidates = [item for item in current if getattr(item, "prefix_dynamic_budget", False) and item.active_requests == 0
                    and item.queued_requests == 0 and item.control_leases == getattr(item, "read_control_leases", 0) and item.memory is not None
                    and callable(getattr(item.backend, "set_prefix_cache_budget", None))]
                candidate_ids = {item.id for item in candidates}
                fixed = self._prefix_reserved_capacity_locked(candidate_ids)
                remaining = max(0, self._prefix_total_limit() - fixed)
                candidates.sort(key=lambda item: (item.id == preferred, item.last_used_at or item.started_at), reverse=True)
                targets = {}
                caps = {item.id: self._prefix_hot_cap(self._load_requests.get(item.artifact.resource.name))
                        for item in candidates}
                for item in candidates:
                    wanted = max(item.memory.prefix_cache_bytes or 0, item.prefix_pressure_bytes)
                    targets[item.id] = min(remaining, wanted, caps[item.id])
                    remaining -= targets[item.id]
                for item in candidates:
                    extra = min(remaining, caps[item.id] - targets[item.id])
                    targets[item.id] += extra
                    remaining -= extra
                disk_limit = self.memory_policy.prefix_disk_limit_bytes
                if disk_limit is None:
                    disk_limit = self._prefix_disk_status.get("prefix_cache_total_disk_max_bytes", 100 << 30)
                else:
                    disk_limit = min(disk_limit, self._prefix_disk_status.get("prefix_cache_total_disk_max_bytes", disk_limit))
                disk_targets = {}
                ram_candidates = [item for item in candidates if item.memory.prefix_cache_ram_limit_bytes is not None]
                ram_reserved = sum(item.memory.prefix_cache_ram_limit_bytes or 0 for item in current
                    if item.id not in candidate_ids and item.memory)
                ram_remaining = max(0, self._prefix_ram_total_limit() - ram_reserved)
                _, available_ram = host_memory_capacity()
                if available_ram is not None:
                    ram_used = sum(item.memory.prefix_cache_ram_bytes or 0 for item in ram_candidates)
                    ram_remaining = min(ram_remaining, max(0, available_ram + ram_used - (512 << 20)))
                if reclaim_ram or self._loading_model_names:
                    ram_remaining = 0
                ram_targets = {}
                ram_caps = {}
                for item in ram_candidates:
                    request = self._load_requests.get(item.artifact.resource.name)
                    per_model_ram = request.prefix_cache_ram_bytes if request else None
                    ram_caps[item.id] = per_model_ram if per_model_ram is not None else self._prefix_ram_total_limit()
                    wanted = max(item.memory.prefix_cache_ram_bytes or 0, getattr(item, "prefix_ram_pressure_bytes", 0))
                    ram_targets[item.id] = min(ram_remaining, wanted, ram_caps[item.id])
                    ram_remaining -= ram_targets[item.id]
                for item in ram_candidates:
                    extra = min(ram_remaining, ram_caps[item.id] - ram_targets[item.id])
                    ram_targets[item.id] += extra
                    ram_remaining -= extra
                for item in candidates:
                    request = self._load_requests.get(item.artifact.resource.name)
                    per_model = request.prefix_cache_disk_bytes if request else None
                    disk_targets[item.id] = min(disk_limit, per_model) if per_model is not None else disk_limit
                changes = [item for item in candidates if item.memory.prefix_cache_limit_bytes != targets[item.id]
                    or getattr(item, "prefix_disk_capacity", None) is not None and item.prefix_disk_capacity != disk_targets[item.id]
                    or item.id in ram_targets and item.memory.prefix_cache_ram_limit_bytes != ram_targets[item.id]]
                changes.sort(key=lambda item: (ram_targets.get(item.id, 0) > (item.memory.prefix_cache_ram_limit_bytes or 0),
                    targets[item.id] > (item.memory.prefix_cache_limit_bytes or 0)))
                concurrent = {item.id: getattr(item, "prefix_concurrent_maintenance", False) for item in changes}
                for item in changes:
                    item.control_leases += 1
                    if concurrent[item.id]:
                        item.read_control_leases += 1
            try:
                for item in changes:
                    disk_args = {"disk_target_bytes": disk_targets[item.id]} if getattr(item, "prefix_disk_capacity", None) is not None and item.prefix_disk_capacity != disk_targets[item.id] else {}
                    if item.id in ram_targets:
                        disk_args["ram_target_bytes"] = ram_targets[item.id]
                    result = await asyncio.wait_for(item.backend.set_prefix_cache_budget(targets[item.id], **disk_args), timeout=30.0)
                    if result.get("status") != "ok":
                        raise RuntimeError("native prefix budget was not applied")
                    async with self._lock:
                        if self._instances.get(item.id) is item:
                            self._policy_prefix_limits[item.artifact.resource.name] = targets[item.id]
                            item.memory = item.memory.model_copy(update={"prefix_cache_limit_bytes": targets[item.id]})
                            if item.id in ram_targets:
                                item.memory = item.memory.model_copy(update={"prefix_cache_ram_limit_bytes": ram_targets[item.id]})
                            item.prefix_pressure_bytes = 0
                            item.prefix_ram_pressure_bytes = 0
                            if "disk_target_bytes" in disk_args:
                                item.prefix_disk_capacity = disk_targets[item.id]
            finally:
                for item in changes:
                    await self._release_control_lease(item, read_only=concurrent[item.id])
            if wait:
                for item in changes:
                    await self._refresh_instance_usage(item)

    async def configure_memory_policy(self, context: JobContext, payload: dict[str, Any], *, additional: DiscoveredModel | None = None) -> dict[str, Any]:
        policy = RuntimeMemoryPolicy.model_validate(payload)
        if policy.prefix_directory is not None:
            directory = Path(policy.prefix_directory).expanduser()
            if not directory.is_absolute():
                raise _job_error("invalid_cache_directory", "prefix cache directory must be an absolute path")
            try:
                directory.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryFile(dir=directory):
                    pass
            except OSError as error:
                raise _job_error("cache_directory_unwritable", str(error)) from error
            policy = policy.model_copy(update={"prefix_directory": str(directory.resolve())})
        async with self._lock:
            if additional is None and policy == self.memory_policy:
                return {**policy.model_dump(), "models_replanned": 0}
            if self._memory_configuration_job is not None or self._loading_model_names:
                raise _job_error("memory_plan_busy", "a model or memory plan is already loading", retryable=True)
            current = list(self._instances.values())
            previous = self.memory_policy
            model_changed = policy.model_limit_bytes != previous.model_limit_bytes
            weight_online = all(getattr(item, "resident_weight_dynamic_budget", False) and getattr(item, "resident_dynamic_budget", False)
                and getattr(item, "prefix_dynamic_budget", False) for item in current)
            if model_changed and policy.prefix_directory == previous.prefix_directory and not weight_online:
                raise _job_error("resident_budget_unsupported", "this runtime does not support online model weight budgets; models were not reloaded")
            cache_only = policy.prefix_directory == previous.prefix_directory \
                and all(getattr(item, "prefix_dynamic_budget", False) for item in current) and (not model_changed or weight_online)
            if any(item.state != RuntimeInstanceState.READY or item.active_requests or item.queued_requests
                or item.control_leases > (getattr(item, "read_control_leases", 0) if cache_only else 0) for item in current):
                raise _job_error("runtime_busy", "wait for model requests to finish before applying memory budgets", retryable=True)
            limits, hot = self._plan_memory_policy(policy, [*[item.artifact for item in current], *([additional] if additional else [])])
            previous_limits, previous_hot = dict(self._policy_model_limits), dict(self._policy_prefix_limits)
            requests = [self._load_requests[item.artifact.resource.name].model_copy(update={
                "context_size": item.context_size, "pin": item.pinned, "idle_ttl_seconds": item.idle_ttl_seconds,
            }) for item in current]
            self._memory_configuration_job = context.job_id
        if policy.total_limit_bytes != previous.total_limit_bytes and not cache_only:
            async with self._lock:
                self._memory_configuration_job = None
                self._request_state_changed.notify_all()
            raise _job_error("resident_budget_unsupported", "changing the shared resident budget requires online native budgeting; model reloads are not used as a substitute")
        if cache_only:
            try:
                self.memory_policy = policy
                self._policy_model_limits, self._policy_prefix_limits = limits, hot
                await self._rebalance_prefix_cache_budget()
                await self._rebalance_resident_budget()
                await self._maintain_prefix_disk_budget(force=True)
                if self.store is not None:
                    await asyncio.to_thread(self.store.save_runtime_memory_policy, policy.model_dump())
                return {**policy.model_dump(), "models_replanned": 0}
            except BaseException as error:
                self.memory_policy = previous
                self._policy_model_limits, self._policy_prefix_limits = previous_limits, previous_hot
                await self._rebalance_prefix_cache_budget()
                await self._rebalance_resident_budget()
                if isinstance(error, BackendError):
                    raise _job_error(error.code, str(error), retryable=error.retryable) from error
                raise
            finally:
                async with self._lock:
                    self._memory_configuration_job = None
                    self._request_state_changed.notify_all()
        try:
            for item in current:
                await self._retire_instance(item)
            self.memory_policy = policy
            self._policy_model_limits, self._policy_prefix_limits = limits, hot
            ordered = sorted(requests, key=lambda request: next(item.artifact.routed_expert_bytes for item in current if item.artifact.resource.name == request.model))
            for index, request in enumerate(ordered):
                context.raise_if_cancelled()
                await self.load(_MemoryPlanContext(context, index / len(ordered), (index + 1) / len(ordered)), request.model_dump(mode="json"))
            if self.store is not None:
                await asyncio.to_thread(self.store.save_runtime_memory_policy, policy.model_dump())
            return {**policy.model_dump(), "models_replanned": len(requests)}
        except BaseException:
            for item in list(self._instances.values()):
                await self._retire_instance(item)
            self.memory_policy = previous
            self._policy_model_limits, self._policy_prefix_limits = previous_limits, previous_hot
            await context.log("Memory planning did not complete; restoring the previous model configuration.")
            for request in requests:
                try:
                    await self.load(_MemoryPlanContext(context, recovery=True), request.model_dump(mode="json"))
                except BaseException as error:
                    await context.log(f"Could not restore {request.model}: {type(error).__name__}")
            raise
        finally:
            async with self._lock:
                self._memory_configuration_job = None
                self._request_state_changed.notify_all()

    async def unload(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = ModelUnloadRequest.model_validate(payload)
        async with self._lock:
            instance = self._instances.get(request.instance_id)
            if instance is None:
                raise _job_error(
                    "runtime_instance_not_found",
                    f"runtime instance was not found: {request.instance_id}",
                )
            if (
                instance.active_requests
                or instance.queued_requests
                or instance.control_leases
            ) and not request.force:
                raise _job_error(
                    "runtime_busy",
                    "runtime has active requests, queued requests, or control operations",
                    retryable=True,
                )
            self._mark_instance_unloading_locked(instance)

        released = False

        async def cleanup_incomplete_unload() -> None:
            if released:
                return
            await self._retire_instance(instance)

        context.add_cleanup(cleanup_incomplete_unload)
        await context.progress(0.2, message="Stopping runtime")
        await self._retire_instance(instance)
        await context.progress(0.9, message="Releasing runtime")
        async with self._lock:
            released = True
        await context.progress(1.0, message="Model unloaded")
        return {"instance_id": str(instance.id), "unloaded": True}

    async def quantization_pressure(self) -> dict[str, bool]:
        async with self._lock:
            return {'busy': self._memory_configuration_job is not None or any(
                item.active_requests or item.queued_requests or item.control_leases > item.read_control_leases
                or item.state not in {RuntimeInstanceState.READY, RuntimeInstanceState.FAILED}
                for item in self._instances.values())}

    @property
    def concurrency_policy(self) -> RuntimeConcurrencyPolicy:
        """Expose configured defaults and ceilings, including an unlimited timeout."""
        defaults = RuntimeConcurrencySettings(
            max_requests=min(32, self.max_requests_per_instance),
            max_queued_requests=self.max_queued_requests_per_instance,
            queue_timeout_seconds=self.queue_timeout_seconds,
        )
        return RuntimeConcurrencyPolicy(
            defaults=defaults,
            limits=RuntimeConcurrencyLimits(
                max_requests=defaults.max_requests,
                max_queued_requests=defaults.max_queued_requests,
                queue_timeout_seconds=defaults.queue_timeout_seconds or None,
            ),
        )

    def resolve_concurrency(self, request: ModelLoadRequest) -> RuntimeConcurrencySettings:
        """Validate overrides before accepting a model load job."""
        return resolve_concurrency(self.concurrency_policy, request)

    def _instance_concurrency(self, instance: _Runtime) -> RuntimeConcurrencySettings:
        return instance.concurrency_settings or self.concurrency_policy.defaults

    def concurrency_resource(self, instance: _Runtime) -> RuntimeConcurrencyResource:
        """Describe requested policy, server ceilings and negotiated admission."""
        requested = self._instance_concurrency(instance)
        return RuntimeConcurrencyResource(
            requested=requested,
            limits=self.concurrency_policy.limits,
            effective=requested.model_copy(update={"max_requests": instance.request_capacity}),
        )

    async def update_concurrency(
        self, instance_id: UUID, request: UpdateRuntimeConcurrencyRequest,
    ) -> RuntimeConcurrencyUpdate:
        """Apply queue changes without replacing a worker or its active semaphore."""
        async with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise BackendError("runtime_instance_not_found", "runtime instance was not found", status_code=404)
            if instance.state not in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}:
                raise BackendError("runtime_busy", "runtime is changing state", status_code=409, retryable=True)
            current = self._instance_concurrency(instance)
            candidate = resolve_concurrency(self.concurrency_policy, request, current)
            reload_required = candidate.max_requests != current.max_requests
            applied = candidate.model_copy(update={"max_requests": current.max_requests})
            instance.concurrency_settings = applied
            previous_event = instance.concurrency_changed
            instance.concurrency_changed = asyncio.Event()
            previous_event.set()
            name = instance.artifact.resource.name
            if name in self._load_requests:
                self._load_requests[name] = self._load_requests[name].model_copy(update=applied.model_dump())
            return RuntimeConcurrencyUpdate(
                concurrency=self.concurrency_resource(instance), reload_required=reload_required,
            )

    async def configure_concurrency(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        """Explicitly restart one idle model, restoring its prior request on failure."""
        settings = RuntimeConcurrencyReloadRequest.model_validate(payload)
        context.raise_if_cancelled()
        async with self._lock:
            instance = self._instances.get(settings.instance_id)
            if instance is None:
                raise _job_error("runtime_instance_not_found", "runtime instance was not found")
            candidate = resolve_concurrency(self.concurrency_policy, settings, self._instance_concurrency(instance))
            if (instance.state != RuntimeInstanceState.READY or instance.active_requests
                    or instance.queued_requests or instance.control_leases or instance.context_job):
                raise _job_error("runtime_busy", "wait for model requests to finish before reloading", retryable=True)
            name = instance.artifact.resource.name
            previous = self._load_requests[name].model_copy(deep=True)
            updated = previous.model_copy(update={
                **candidate.model_dump(), "pin": instance.pinned, "idle_ttl_seconds": instance.idle_ttl_seconds,
            })
            self._mark_instance_unloading_locked(instance)
        try:
            await context.progress(0.01, message="Applying concurrency settings", data={"phase": "reloading"})
            await self._retire_instance(instance)
            return await self.load(context, updated.model_dump(mode="json"))
        except BaseException:
            for item in list(self._instances.values()):
                if item.artifact.resource.name == name:
                    await self._retire_instance(item)
            self._load_requests[name] = previous
            try:
                await self.load(_MemoryPlanContext(context, recovery=True), previous.model_dump(mode="json"))
            except BaseException as error:
                await context.log(f"Could not restore previous concurrency settings: {type(error).__name__}")
            raise

    async def _acquire_request_slot(
        self, instance: _Runtime, enqueued_at: float, model: str,
        *, priority: int = 0, deadline: float | None = None,
    ) -> None:
        """Keep queue rank while applying live timeouts and the client deadline."""
        assert instance.request_slots is not None

        def remaining_wait() -> float | None:
            timeout = self._instance_concurrency(instance).queue_timeout_seconds
            queue_left = enqueued_at + timeout - time.monotonic() if timeout else None
            client_left = deadline - asyncio.get_running_loop().time() if deadline is not None else None
            bounds = [value for value in (queue_left, client_left) if value is not None]
            remaining = min(bounds) if bounds else None
            if remaining is not None and remaining <= 0:
                instance.rejections += 1
                if client_left is not None and (queue_left is None or client_left <= queue_left):
                    raise _request_deadline_exceeded(model)
                raise BackendError(
                    "runtime_queue_timeout", f"request waited more than {timeout:g}s for a runtime slot: {model}",
                    retryable=True, status_code=503,
                )
            return remaining

        acquire = asyncio.create_task(instance.request_slots.acquire(priority))
        transferred = False
        try:
            while True:
                remaining = remaining_wait()
                changed = asyncio.create_task(instance.concurrency_changed.wait())
                try:
                    done, _ = await asyncio.wait(
                        (acquire, changed), timeout=remaining, return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    changed.cancel()
                    await asyncio.gather(changed, return_exceptions=True)
                if acquire in done:
                    await acquire
                    remaining_wait()
                    transferred = True
                    return
                if changed in done:
                    continue
        finally:
            if not transferred:
                acquire.cancel()
                await asyncio.gather(acquire, return_exceptions=True)
                if not acquire.cancelled() and acquire.exception() is None:
                    instance.request_slots.release()

    async def instances(self) -> RuntimeInstanceList:
        async with self._lock:
            values = list(self._instances.values())
            now = asyncio.get_running_loop().time()
            refresh = []
            for item in values:
                if (
                    item.state in {
                        RuntimeInstanceState.READY,
                        RuntimeInstanceState.BUSY,
                    }
                    and now - item.usage_refreshed_at >= self.metric_interval_seconds
                ):
                    item.usage_refreshed_at = now
                    refresh.append(item)
            route_counts: dict[UUID, int] = {}
            for instance_id in self._session_routes.values():
                route_counts[instance_id] = route_counts.get(instance_id, 0) + 1
        if refresh:
            await asyncio.gather(
                *(self._refresh_instance_usage(item) for item in refresh)
            )
        return RuntimeInstanceList(
            data=[
                RuntimeInstanceResource(
                    id=item.id,
                    model=item.artifact.resource.name,
                    state=item.state,
                    devices=[f"cuda:{device}" for device in item.device_ids] if item.device_ids else [self.backend],
                    active_sessions=route_counts.get(item.id, 0),
                    queued_requests=item.queued_requests,
                    request_capacity=item.request_capacity,
                    batch_capacity=item.batch_capacity,
                    concurrency=self.concurrency_resource(item),
                    queue_wait_ms_avg=self._concurrency_status(item)["queue_wait_ms_avg"],
                    queue_wait_ms_max=item.queue_wait_ms_max,
                    rejected_requests=item.rejections,
                    resident_bytes=item.resident_bytes,
                    kv_bytes=item.kv_bytes,
                    memory=item.memory,
                    context_size=item.context_size,
                    context_capacity=item.context_capacity,
                    started_at=item.started_at,
                    last_used_at=item.last_used_at,
                    idle_ttl_seconds=item.idle_ttl_seconds,
                    pinned=item.pinned,
                    mtp_supported=item.mtp_supported,
                    mtp_available=item.mtp_available,
                    qsa_kv_offload_supported=item.qsa_kv_offload_supported,
                    kv_quantization_supported=item.kv_quantization_supported,
                    kv_memory_supported=item.kv_memory_supported,
                    kv_memory=self._kv_memory_policy(item.artifact.resource.name),
                    kv_quantization=self._kv_quantization_policy(item.artifact.resource.name),
                    qsa_kv_offload=self._qsa_kv_policy(item.artifact.resource.name),
                    error=item.error,
                )
                for item in values
            ]
        )

    async def update_instance(
        self,
        instance_id: UUID,
        request: UpdateRuntimeInstanceRequest,
    ) -> RuntimeInstanceResource:
        async with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise BackendError(
                    "runtime_instance_not_found",
                    f"runtime instance was not found: {instance_id}",
                    status_code=404,
                )
            if instance.state not in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }:
                raise BackendError(
                    "model_not_ready",
                    f"model runtime is {instance.state.value}",
                    retryable=True,
                    status_code=409,
                )
            updates: dict[str, Any] = {}
            if "pinned" in request.model_fields_set:
                instance.pinned = bool(request.pinned)
                updates["pin"] = instance.pinned
            if "idle_ttl_seconds" in request.model_fields_set:
                instance.idle_ttl_seconds = request.idle_ttl_seconds
                updates["idle_ttl_seconds"] = request.idle_ttl_seconds
            model_name = instance.artifact.resource.name
            load_request = self._load_requests.get(model_name)
            if load_request is not None and updates:
                self._load_requests[model_name] = load_request.model_copy(
                    update=updates
                )
            active_sessions = sum(
                routed_id == instance.id
                for routed_id in self._session_routes.values()
            )
            resource = RuntimeInstanceResource(
                id=instance.id,
                model=model_name,
                state=instance.state,
                devices=[f"cuda:{device}" for device in instance.device_ids] if instance.device_ids else [self.backend],
                active_sessions=active_sessions,
                queued_requests=instance.queued_requests,
                request_capacity=instance.request_capacity,
                batch_capacity=instance.batch_capacity,
                concurrency=self.concurrency_resource(instance),
                queue_wait_ms_avg=self._concurrency_status(instance)["queue_wait_ms_avg"],
                queue_wait_ms_max=instance.queue_wait_ms_max,
                rejected_requests=instance.rejections,
                resident_bytes=instance.resident_bytes,
                kv_bytes=instance.kv_bytes,
                memory=instance.memory,
                context_size=instance.context_size,
                context_capacity=instance.context_capacity,
                started_at=instance.started_at,
                last_used_at=instance.last_used_at,
                idle_ttl_seconds=instance.idle_ttl_seconds,
                pinned=instance.pinned,
                mtp_supported=instance.mtp_supported,
                mtp_available=instance.mtp_available,
                qsa_kv_offload_supported=instance.qsa_kv_offload_supported,
                kv_quantization_supported=instance.kv_quantization_supported,
                kv_memory_supported=instance.kv_memory_supported,
                kv_memory=self._kv_memory_policy(instance.artifact.resource.name),
                kv_quantization=self._kv_quantization_policy(instance.artifact.resource.name),
                qsa_kv_offload=self._qsa_kv_policy(instance.artifact.resource.name),
                error=instance.error,
            )
            self._idle_reaper_wakeup.set()
        return resource

    async def _reject_full_queue(self, instance: _Runtime, model: str) -> NoReturn:
        instance.rejections += 1
        error = BackendError("runtime_queue_full", f"model runtime queue is full: {model}",
            retryable=True, status_code=429)
        if self.store is not None:
            try:
                await asyncio.to_thread(self.store.append_runtime_log, RuntimeLogLevel.WARNING,
                    "Request rejected: runtime_queue_full", instance_id=instance.id,
                    fields={"source": "runtime.admission", "model": instance.artifact.resource.name,
                        "code": error.code, "status_code": error.status_code})
            except Exception:
                logger.exception("Could not persist runtime admission error")
        raise error

    def _apply_batch_capacity(self, instance: _Runtime, reported: int) -> None:
        """Adopt the worker's real batch capacity without disturbing requests.

        The worker owns model/state compatibility, so its report is
        authoritative. Shrinking is only safe while nothing is in flight: a new
        gate would lose the slots held by running or waiting requests.
        """

        if reported < 1:
            return
        instance.batch_capacity = reported
        if (
            reported >= instance.request_capacity
            or instance.active_requests
            or instance.queued_requests
        ):
            return
        instance.request_capacity = reported
        instance.request_slots = _RequestSlotGate(reported)
        logger.info(
            "runtime concurrency reduced to %d for %s (worker reported capacity)",
            reported,
            instance.artifact.resource.name,
        )

    @staticmethod
    def _concurrency_status(instance: _Runtime) -> dict[str, Any]:
        samples = instance.queue_wait_samples
        return {
            "request_capacity": instance.request_capacity,
            "batch_capacity": instance.batch_capacity,
            "queue_wait_ms_avg": (
                instance.queue_wait_ms_total / samples if samples else 0.0
            ),
            "queue_wait_ms_max": instance.queue_wait_ms_max,
            "queue_wait_samples": samples,
            "rejected_requests": instance.rejections,
        }

    async def _wait_for_runtime_maintenance_locked(self, instance: _Runtime, *, global_only: bool = False) -> None:
        deadline = asyncio.get_running_loop().time() + _RUNTIME_MAINTENANCE_WAIT_SECONDS
        while True:
            if self._closed or self._instances.get(instance.id) is not instance or instance.state not in {
                RuntimeInstanceState.READY, RuntimeInstanceState.BUSY,
            } or instance.process.returncode is not None:
                raise BackendError("model_not_ready", f"model runtime is {instance.state.value}",
                    retryable=True, status_code=503)
            if self._memory_configuration_job is None and (
                global_only or instance.control_leases == instance.read_control_leases
            ):
                return
            try:
                await asyncio.wait_for(self._request_state_changed.wait(),
                    timeout=max(0.0, deadline - asyncio.get_running_loop().time()))
            except asyncio.TimeoutError:
                raise BackendError("runtime_maintenance_timeout", "runtime maintenance did not finish in time; retry the request",
                    retryable=True, status_code=503) from None

    async def preflight(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
    ) -> None:
        instance = await self._select(model, session_id=session_id)
        if instance is not None and instance.state == RuntimeInstanceState.LOADING:
            await self._wait_for_model_ready(instance.artifact.resource.name)
            instance = await self._select(model, session_id=session_id)
        if instance is None:
            instance = await self._ensure_model_loaded_with_revival(model)
        if instance is None:
            if self.fallback is None:
                raise BackendError(
                    "model_not_loaded",
                    f"model is not loaded: {model}",
                    status_code=404,
                )
            await preflight_backend_request(
                self.fallback,
                model=model,
                messages=messages,
                sampling=sampling,
                session_id=session_id,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )
            return
        async with self._lock:
            await self._wait_for_runtime_maintenance_locked(instance)
            queue_full = (
                instance.active_requests + instance.queued_requests
                >= instance.request_capacity
                + self._instance_concurrency(instance).max_queued_requests
            )
        if queue_full:
            await self._reject_full_queue(instance, model)

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
        priority: int = 0,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[BackendDelta]:
        # One deadline covers the server queue and the native request, so a
        # client timeout can never be extended by queueing.
        deadline = (
            None
            if timeout_seconds is None
            else asyncio.get_running_loop().time() + timeout_seconds
        )
        instance = await self._select(model, session_id=session_id)
        if instance is not None and instance.state == RuntimeInstanceState.LOADING:
            await self._wait_for_model_ready(instance.artifact.resource.name)
            instance = await self._select(model, session_id=session_id)
        if instance is None:
            instance = await self._ensure_model_loaded_with_revival(model)
        if instance is None:
            if self.fallback is None:
                raise BackendError("model_not_loaded", f"model is not loaded: {model}", status_code=404)
            async with closing_backend_stream(
                self.fallback.stream(
                    model=model,
                    messages=messages,
                    sampling=self._apply_inference_policy(sampling),
                    session_id=session_id,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    priority=priority,
                    timeout_seconds=timeout_seconds,
                )
            ) as fallback_stream:
                async for delta in fallback_stream:
                    yield delta
            return
        assert instance.request_slots is not None
        async with self._lock:
            await self._wait_for_runtime_maintenance_locked(instance, global_only=True)
        await self._rebalance_prefix_cache_budget(instance.id, wait=False)
        await self._rebalance_resident_budget(instance.id, wait=False)
        acquired = False
        active = False
        enqueued_at = time.monotonic()
        queue_wait_ms = 0.0
        async with self._lock:
            await self._wait_for_runtime_maintenance_locked(instance, global_only=True)
            queue_full = (
                instance.active_requests + instance.queued_requests
                >= instance.request_capacity
                + self._instance_concurrency(instance).max_queued_requests
            )
            if not queue_full:
                instance.queued_requests += 1
        if queue_full:
            await self._reject_full_queue(instance, model)
        queued = True
        request_id = None
        performance = None
        usage = None
        finish_reason = None
        try:
            await self._acquire_request_slot(
                instance,
                enqueued_at,
                model,
                priority=priority,
                deadline=deadline,
            )
            acquired = True
            queue_wait_ms = (time.monotonic() - enqueued_at) * 1000.0
            remaining_timeout = (
                None
                if deadline is None
                else deadline - asyncio.get_running_loop().time()
            )
            if remaining_timeout is not None and remaining_timeout <= 0.0:
                raise _request_deadline_exceeded(model)
            async with self._lock:
                await self._wait_for_runtime_maintenance_locked(instance)
                instance.queued_requests = max(0, instance.queued_requests - 1)
                queued = False
                instance.active_requests += 1
                active = True
                instance.state = RuntimeInstanceState.BUSY
                instance.last_used_at = datetime.now(timezone.utc)
                instance.queue_wait_samples += 1
                instance.queue_wait_ms_total += queue_wait_ms
                instance.queue_wait_ms_max = max(instance.queue_wait_ms_max, queue_wait_ms)
                if session_id is not None:
                    self._session_routes[session_id] = instance.id
                self._last_instance_id = instance.id
            instance.request_metrics.observe_queue_wait(queue_wait_ms / 1000.0)
            effective_sampling = self._apply_inference_policy(self._sampling_for_instance(instance, sampling))
            async with closing_backend_stream(
                instance.backend.stream(
                    model=instance.artifact.resource.name,
                    messages=messages,
                    sampling=effective_sampling,
                    session_id=session_id,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    priority=priority,
                    # Only the time left after queueing: the worker arms its own
                    # deadline, so an absolute bound stays absolute end to end.
                    timeout_seconds=remaining_timeout,
                )
            ) as backend_stream:
                async for delta in backend_stream:
                    request_id = delta.backend_request_id or request_id
                    performance = delta.performance or performance
                    usage = delta.usage or usage
                    finish_reason = delta.finish_reason or finish_reason
                    if queue_wait_ms and delta.performance is not None:
                        delta = replace(
                            delta,
                            performance=delta.performance.model_copy(
                                update={"server_queue_wait_ms": queue_wait_ms}
                            ),
                        )
                    yield delta
            if self.store is not None and performance is not None and request_id is not None:
                request = performance.model_dump(mode="json", exclude_unset=True)
                request.update(id=request_id, finish_reason=finish_reason,
                    completed_at=time.time(), queue_wait_ms=queue_wait_ms)
                if usage is not None:
                    request.update(prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens)
                try:
                    await asyncio.to_thread(self.store.append_runtime_metric,
                        {"last_request": request}, instance_id=instance.id,
                        model=instance.artifact.resource.name)
                except Exception:
                    logger.exception("Could not persist completed runtime request metrics")
        except GeneratorExit:
            # The consumer stopped reading: the client disconnected mid-stream.
            instance.request_metrics.record_request("cancelled", performance, usage)
            raise
        except asyncio.CancelledError:
            instance.request_metrics.record_request("cancelled", performance, usage)
            raise
        except BackendError as error:
            result = "timeout" if error.code in _TIMEOUT_ERROR_CODES else "error"
            instance.request_metrics.record_request(result, performance, usage)
            raise
        except Exception:
            instance.request_metrics.record_request("error", performance, usage)
            raise
        else:
            instance.request_metrics.record_request(
                "cancelled" if finish_reason == "cancelled" else "ok", performance, usage)
        finally:
            await self._release_runtime_lease(
                self._finish_request(instance, queued=queued, active=active, acquired=acquired),
                name=f"mfq-server-request-release-{instance.id}",
            )

    async def _finish_request(self, instance: _Runtime, *, queued: bool, active: bool, acquired: bool) -> None:
        async with self._lock:
            if queued:
                instance.queued_requests = max(0, instance.queued_requests - 1)
            if active:
                instance.active_requests = max(0, instance.active_requests - 1)
            if acquired:
                instance.request_slots.release()
            if active:
                if instance.state == RuntimeInstanceState.BUSY and instance.active_requests == 0:
                    instance.state = RuntimeInstanceState.READY
                instance.last_used_at = datetime.now(timezone.utc)
                if instance.active_requests == 0:
                    self._idle_reaper_wakeup.set()
                    self._prefix_disk_checked_at = 0.0

    @staticmethod
    def _sampling_for_instance(
        instance: _Runtime,
        requested: SamplingParams,
    ) -> SamplingParams:
        defaults = instance.sampling_defaults
        if defaults is None:
            return requested
        return defaults.model_copy(
            update={
                name: getattr(requested, name)
                for name in requested.model_fields_set
            }
        )

    @asynccontextmanager
    async def model_deletion_guard(self, artifact: DiscoveredModel) -> AsyncIterator[None]:
        path = artifact.path.resolve()
        async with self._lock:
            if path in self._deleting_model_paths or artifact.resource.name in self._loading_model_names or any(
                (item.artifact.path.resolve() == path or item.artifact.resource.name == artifact.resource.name)
                and item.state != RuntimeInstanceState.FAILED
                for item in self._instances.values()
            ):
                raise BackendError("model_delete_busy", "Unload the model and wait for loading to finish before deleting it", status_code=409)
            self._deleting_model_paths.add(path)
        try:
            yield
        finally:
            async with self._lock:
                self._deleting_model_paths.discard(path)

    async def fork_session(self, source_session_id: UUID, target_session_id: UUID) -> bool:
        async with self._lock:
            instance_id = self._session_routes.get(source_session_id)
            instance = self._instances.get(instance_id) if instance_id is not None else None
            if instance is not None and instance.state not in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }:
                self._session_routes.pop(source_session_id, None)
                instance = None
            if instance is not None:
                instance.control_leases += 1
        if instance is None:
            return (
                await self.fallback.fork_session(source_session_id, target_session_id)
                if self.fallback
                else False
            )
        try:
            succeeded = await instance.backend.fork_session(
                source_session_id,
                target_session_id,
            )
            if succeeded:
                async with self._lock:
                    if self._instances.get(instance.id) is instance:
                        self._session_routes[target_session_id] = instance.id
            return succeeded
        finally:
            await self._release_control_lease(instance)

    async def close_session(self, session_id: UUID) -> bool:
        async with self._lock:
            instance_id = self._session_routes.pop(session_id, None)
            instance = self._instances.get(instance_id) if instance_id is not None else None
            if instance is not None and instance.state not in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }:
                instance = None
            if instance is not None:
                instance.control_leases += 1
        if instance is None:
            return await self.fallback.close_session(session_id) if self.fallback else False
        try:
            return await instance.backend.close_session(session_id)
        finally:
            await self._release_control_lease(instance)

    async def cancel_response(self, session_id: UUID) -> bool:
        async with self._lock:
            instance_id = self._session_routes.get(session_id)
            instance = self._instances.get(instance_id) if instance_id is not None else None
            if instance is not None and instance.state not in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }:
                self._session_routes.pop(session_id, None)
                instance = None
            if instance is not None:
                instance.control_leases += 1
        backend = instance.backend if instance is not None else self.fallback
        if backend is None:
            return False
        cancel = getattr(backend, "cancel_response", None)
        try:
            return bool(await cancel(session_id)) if callable(cancel) else False
        finally:
            if instance is not None:
                await self._release_control_lease(instance)

    async def capabilities(
        self,
        instance_id: UUID | None = None,
    ) -> RuntimeCapabilitiesResource:
        async with self._runtime_control_lease(instance_id, read_only=True) as (_instance, backend):
            if backend is None:
                raise BackendError("model_not_loaded", "no runtime is available")
            return await backend.capabilities()

    async def serving_activity(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        async def refresh(item: _Runtime) -> None:
            if not item.active_requests or item.serving_refreshing or now - item.serving_sampled_at < 0.45:
                return
            item.serving_refreshing = True
            try:
                status = await asyncio.wait_for(item.backend.runtime_status(), timeout=0.25)
                live = status.get("live_requests")
                if isinstance(live, dict):
                    item.serving_snapshot = {key: value for key, value in live.items() if isinstance(value, dict)}
                    item.serving_sampled_at = time.monotonic()
            except Exception:
                pass
            finally:
                item.serving_refreshing = False
        await asyncio.gather(*(refresh(item) for item in tuple(self._instances.values())))
        now = time.monotonic()
        return [{"instance_id": str(item.id), "model": item.artifact.resource.name,
            "state": item.state.value, "active_requests": item.active_requests,
            "queued_requests": item.queued_requests,
            **self._concurrency_status(item),
            "sample_age_seconds": now - item.serving_sampled_at if item.serving_sampled_at else None,
            "requests": [{"id": key, **value} for key, value in item.serving_snapshot.items()]
                if item.active_requests else []}
            for item in self._instances.values() if item.state in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}]

    async def metrics_snapshot(self) -> list[RuntimeMetricsSnapshot]:
        """Return the serving runtimes as a Prometheus scrape sees them.

        Usage refresh shares the runtime-pool interval, so scraping never adds
        NVML work, SQLite reads, or more than one worker status call per
        interval.
        """

        async with self._lock:
            values = [
                item
                for item in self._instances.values()
                if item.state in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
            ]
            now = asyncio.get_running_loop().time()
            refresh = [
                item
                for item in values
                if now - item.usage_refreshed_at >= self.metric_interval_seconds
            ]
            for item in refresh:
                item.usage_refreshed_at = now
        if refresh:
            await asyncio.gather(
                *(self._refresh_instance_usage(item) for item in refresh)
            )
        return [
            self._metrics_snapshot_of(item)
            for item in values
            if item.state in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
        ]

    @staticmethod
    def _metrics_snapshot_of(instance: _Runtime) -> RuntimeMetricsSnapshot:
        memory = instance.memory
        return RuntimeMetricsSnapshot(
            instance=str(instance.id),
            model=instance.artifact.resource.name,
            request_capacity=instance.request_capacity,
            batch_capacity=instance.batch_capacity,
            active_requests=instance.active_requests,
            queued_requests=instance.queued_requests,
            rejected_requests=instance.rejections,
            resident_bytes=instance.resident_bytes,
            kv_bytes=instance.kv_bytes,
            kv_limit_bytes=memory.prefix_cache_limit_bytes if memory is not None else None,
            requests=instance.request_metrics,
        )

    async def runtime_status(
        self,
        instance_id: UUID | None = None,
    ) -> dict[str, Any]:
        await self._maintain_prefix_disk_budget()
        gpu_devices = await asyncio.to_thread(gpu_memory_status) if self.backend == "cuda" else []
        async with self._lock:
            (
                memory_pressure_level,
                memory_pressure_ratio,
                effective_memory_budget,
                committed_memory,
            ) = self._runtime_memory_pressure_locked()
            memory_status = {
                **memory_topology(),
                "mtp_service_enabled": self.inference_policy.mtp_enabled,
                "runtime_memory_budget_bytes": self._resident_total_limit(),
                "runtime_memory_effective_budget_bytes": effective_memory_budget,
                "runtime_memory_budget_mode": (
                    "automatic"
                    if self.automatic_memory_budget and self.memory_policy.total_limit_bytes is None
                    else "explicit"
                    if self._resident_total_limit() is not None
                    else "disabled"
                ),
                "runtime_memory_committed_bytes": committed_memory,
                "runtime_memory_headroom_bytes": (
                    max(0, effective_memory_budget - committed_memory)
                    if effective_memory_budget is not None
                    else None
                ),
                "runtime_memory_pressure_level": memory_pressure_level,
                "runtime_memory_pressure_ratio": memory_pressure_ratio,
                "runtime_memory_host_paging_pressure": (
                    time.monotonic() < self._host_vm_pressure_until
                ),
                "runtime_memory_shared_cache_reclaims": self._shared_cache_reclaims,
                "runtime_memory_shared_cache_released_bytes": (
                    self._shared_cache_released_bytes
                ),
                "runtime_memory_shared_cache_reclaim_failures": (
                    self._shared_cache_reclaim_failures
                ),
                "prefix_cache_total_hot_bytes": sum(item.memory.prefix_cache_bytes or 0 for item in self._instances.values() if item.memory),
                "prefix_cache_total_hot_max_bytes": self._prefix_total_limit(),
                "prefix_cache_total_ram_bytes": sum(item.memory.prefix_cache_ram_bytes or 0 for item in self._instances.values() if item.memory),
                "prefix_cache_total_ram_max_bytes": sum(item.memory.prefix_cache_ram_limit_bytes or 0 for item in self._instances.values() if item.memory),
                "prefix_cache_budget_failures": self._prefix_budget_failures,
                **self._prefix_disk_status,
            }
            if gpu_devices:
                memory_status["gpu_devices"] = gpu_devices
                if all(item.get("memory_available_bytes") is not None for item in gpu_devices):
                    memory_status["device_free_bytes"] = sum(item["memory_available_bytes"] for item in gpu_devices)
        async with self._runtime_control_lease(
            instance_id,
            allow_unready=True,
            read_only=True,
        ) as (instance, backend):
            if instance is not None and instance.context_job is not None:
                return {
                    **memory_status,
                    "instance_id": str(instance.id),
                    "runtime_state": "loading",
                    "model": instance.artifact.resource.name,
                    "max_context": instance.context_size,
                    "context_capacity": instance.context_capacity,
                    "active_requests": instance.active_requests,
                    "queued_requests": instance.queued_requests,
                    "reloading": True,
                }
            if instance is not None and instance.state not in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }:
                return {
                    **memory_status,
                    "instance_id": str(instance.id),
                    "runtime_state": instance.state.value,
                    "model": instance.artifact.resource.name,
                    "active_requests": instance.active_requests,
                    "queued_requests": instance.queued_requests,
                    "reloading": False,
                    "error": (
                        instance.error.model_dump(mode="json")
                        if instance.error is not None
                        else None
                    ),
                }
            if backend is None:
                return {
                    **memory_status,
                    "runtime_state": "idle",
                    "model": None,
                    "active_requests": 0,
                    "total_requests": 0,
                    "failed_requests": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "reloading": False,
                }
            status = dict(await backend.runtime_status())
            status.update(memory_status)
            if instance is not None:
                status["instance_id"] = str(instance.id)
                status["runtime_state"] = instance.state.value
                reported_capacity = status.get("batch_capacity")
                if isinstance(reported_capacity, int) and reported_capacity > 0:
                    self._apply_batch_capacity(instance, reported_capacity)
                status.update(self._concurrency_status(instance))
                if instance.sampling_defaults is not None:
                    status["sampling_defaults"] = (
                        instance.sampling_defaults.model_dump(mode="json")
                    )
                pid = getattr(instance.process, "pid", None)
                resident_bytes = (
                    await asyncio.to_thread(self._process_resident_bytes, pid)
                    if isinstance(pid, int) and pid > 0
                    else None
                )
                status["process_resident_bytes"] = resident_bytes
                kv_value = status.get(
                    "prefix_cache_hot_bytes",
                    status.get("prefix_cache_bytes"),
                )
                observed_resident_bytes = self._observed_runtime_bytes(
                    resident_bytes,
                    status,
                    separate_memory=self.backend == "cuda" and memory_topology()["memory_architecture"] == "discrete",
                )
                async with self._lock:
                    if self._instances.get(instance.id) is instance:
                        if observed_resident_bytes is not None:
                            instance.resident_bytes = observed_resident_bytes
                        if isinstance(kv_value, (int, float)) and kv_value >= 0:
                            instance.kv_bytes = int(kv_value)
                        instance.memory = self._memory_resources(status, instance.memory, device_ids=getattr(instance, "device_ids", ()))
                        capacity = status.get("context_capacity")
                        if isinstance(capacity, int) and capacity > 0:
                            instance.context_capacity = instance.artifact.resource.context_capacity or capacity
                            status["context_capacity"] = instance.context_capacity
            return status

    async def runtime_models(self) -> dict[str, Any]:
        async with self._lock:
            managed = [
                item
                for item in self._instances.values()
                if item.state in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
            ]
        data = []
        for instance in managed:
            data.append(
                {
                    "id": instance.artifact.resource.name,
                    "object": "model",
                    "model": instance.artifact.resource.name,
                    "state": instance.state.value,
                    "instance_id": str(instance.id),
                }
            )
        if not data and self.fallback is not None:
            return await self.fallback.runtime_models()
        return {"object": "list", "data": data}

    async def realtime_capabilities(self) -> dict[str, Any]:
        async with self._lock:
            instance = self._current_instance_locked()
            gateway = instance.realtime_gateway if instance is not None else None
            if gateway is not None:
                instance.control_leases += 1
        if instance is not None:
            if gateway is None:
                return {"available": False, "modes": []}
            try:
                return await gateway.capabilities()
            finally:
                await self._release_control_lease(instance)
        if self.fallback is None:
            return {"available": False, "modes": []}
        return await self.fallback.realtime_capabilities()

    async def voice_output_status(self) -> dict[str, Any]:
        async with self._lock:
            instance = self._current_instance_locked()
        architecture = instance.artifact.resource.architecture if instance is not None else ""
        supported = self._supports_voice_output(architecture)
        return {
            "active": bool(instance is not None and instance.realtime_gateway is not None),
            "supported_model_loaded": bool(instance is not None and supported),
            "model": instance.artifact.resource.name if instance is not None else None,
            "error": instance.realtime_error if instance is not None else None,
        }

    async def enable_realtime(self, instance_id: UUID | None = None) -> dict[str, Any]:
        async with self._realtime_activation_lock:
            async with self._lock:
                instance = (
                    self._instances.get(instance_id)
                    if instance_id is not None
                    else self._current_instance_locked()
                )
                if instance is not None and instance.state in {
                    RuntimeInstanceState.READY,
                    RuntimeInstanceState.BUSY,
                }:
                    instance.control_leases += 1
                else:
                    instance = None
            if instance is None:
                return {"active": False, "reason": "model_not_loaded"}
            try:
                return await self._activate_realtime_instance(instance)
            finally:
                await self._release_control_lease(instance)

    async def _activate_realtime_instance(
        self,
        instance: _Runtime,
    ) -> dict[str, Any]:
        if not self._supports_voice_output(instance.artifact.resource.architecture):
            return {"active": False, "reason": "model_not_supported"}
        if self.voice_component is None or not self.voice_component.ready():
            return {"active": False, "reason": "component_not_ready"}
        if instance.realtime_gateway is not None:
            return {"active": True, "model": instance.artifact.resource.name}
        base_url = getattr(instance.backend, "base_url", None)
        if not isinstance(base_url, str) or not base_url:
            return {"active": False, "reason": "backend_not_supported"}
        try:
            from mfq.runtime.minicpmo45_realtime import (
                RealtimeGateway,
                _backend_token2wav_steps,
            )

            def create_gateway() -> Any:
                steps = _backend_token2wav_steps(base_url, "", None)
                return RealtimeGateway(
                    base_url,
                    self.voice_component.root,
                    token2wav_steps=steps,
                )

            gateway = await asyncio.to_thread(create_gateway)
        except Exception as error:
            instance.realtime_error = str(error)
            return {"active": False, "reason": "activation_failed", "error": str(error)}
        async with self._lock:
            if self._instances.get(instance.id) is not instance:
                return {"active": False, "reason": "model_unloaded"}
            instance.realtime_gateway = gateway
            instance.realtime_error = None
        return {"active": True, "model": instance.artifact.resource.name}

    async def realtime_serve(self, client: Any, *, mode: str = "audio") -> bool:
        if mode != "audio":
            return False
        async with self._lock:
            instance = self._current_instance_locked()
            gateway = instance.realtime_gateway if instance is not None else None
            if gateway is not None:
                instance.control_leases += 1
        if gateway is None:
            return False
        try:
            await gateway.serve(client)
            return True
        finally:
            assert instance is not None
            await self._release_control_lease(instance)

    def resolve_context_size(self, artifact: DiscoveredModel, explicit: int | None = None,
        *, ignore_override: bool = False, yarn_enabled: bool | None = None) -> int:
        override = None if ignore_override else self.context_policy.model_overrides.get(artifact.resource.name)
        if explicit is not None:
            return min(explicit, self._context_limit(artifact, yarn_enabled) or explicit)
        if override is not None:
            return min(override, self._context_limit(artifact, yarn_enabled) or override)
        native = artifact.resource.context_capacity
        if native is not None:
            cap = self.context_policy.max_context_size
            return min(native, cap) if cap is not None else native
        return self.context_policy.fallback_context_size

    def _context_limit(self, artifact: DiscoveredModel, yarn_enabled: bool | None = None) -> int | None:
        if yarn_enabled is None:
            yarn_enabled = self.context_policy.model_yarn_enabled.get(artifact.resource.name, False)
        if yarn_enabled and self.backend in {"cuda", "metal"} and artifact.resource.yarn_context_capacity is not None:
            return artifact.resource.yarn_context_capacity
        return artifact.resource.context_capacity

    async def yarn_context_info(self, instance_id: UUID) -> dict[str, Any]:
        async with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise BackendError("model_not_loaded", "model is not loaded", status_code=404)
            resource = instance.artifact.resource
            supported = self.backend in {"cuda", "metal"} and resource.yarn_context_capacity is not None
            return {"supported": supported, "native_context": resource.context_capacity,
                "maximum_context": resource.yarn_context_capacity if supported else resource.context_capacity,
                "maximum_factor": resource.yarn_max_factor if supported else None,
                "enabled": self.context_policy.model_yarn_enabled.get(resource.name, False) and supported,
                "effective_factor": max(1.0, instance.context_size / resource.context_capacity) if resource.context_capacity else 1.0}

    async def configure_context_policy(self, policy: RuntimeContextPolicy) -> dict[str, Any]:
        async with self._context_policy_lock:
            policy = RuntimeContextPolicy.model_validate({**self.context_policy.model_dump(), **policy.model_dump(exclude_unset=True)})
            if self.store is not None:
                await asyncio.to_thread(self.store.save_runtime_context_policy, policy.model_dump())
            self.context_policy = policy
        return policy.model_dump()

    async def configure_context(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = RuntimeReloadRequest.model_validate(payload)
        context.raise_if_cancelled()
        try:
            if request.apply_global_cap:
                return await self._reload_global_context_cap(context)
            if request.qsa_kv_offload is not None or request.kv_quantization is not None or request.kv_memory is not None:
                return await self._configure_model_context(context, request)
            return await self.reload_runtime(request.context_size, request.instance_id, context=context, yarn_enabled=request.yarn_enabled)
        except BackendError as error:
            raise _job_error(error.code, str(error), retryable=error.retryable) from error

    async def _reload_global_context_cap(self, context: JobContext) -> dict[str, Any]:
        cap = self.context_policy.max_context_size
        if cap is None:
            return {"models_reloaded": 0}
        async with self._lock:
            candidates = [item for item in self._instances.values()
                if item.state in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY} and item.context_size > cap]
            if any(item.active_requests or item.queued_requests or item.control_leases or item.context_job for item in candidates):
                raise BackendError("runtime_busy", "wait for model requests to finish before reloading the global context cap", retryable=True, status_code=409)
        results = []
        for index, instance in enumerate(candidates):
            context.raise_if_cancelled()
            result = await self.reload_runtime(cap, instance.id,
                context=_MemoryPlanContext(context, index / len(candidates), (index + 1) / len(candidates)), save_policy=False)
            results.append(result)
        return {"models_reloaded": len(results), "max_context": cap, "models": results}

    async def _configure_model_context(self, context: JobContext, settings: RuntimeReloadRequest) -> dict[str, Any]:
        async with self._lock:
            instance = self._instances.get(settings.instance_id)
            if instance is None:
                raise _job_error("runtime_instance_not_found", "runtime instance was not found")
            previous = self._load_requests[instance.artifact.resource.name].model_copy(deep=True)
            name = instance.artifact.resource.name
            enabled = self.context_policy.model_yarn_enabled.get(name, False) if settings.yarn_enabled is None else settings.yarn_enabled
        if enabled and (self.backend not in {"cuda", "metal"} or instance.artifact.resource.yarn_context_capacity is None):
            raise _job_error("yarn_unsupported", "YaRN extension is not supported by this model/backend")
        target = self.resolve_context_size(instance.artifact, settings.context_size, ignore_override=True, yarn_enabled=enabled)
        warning = None if settings.context_size is None or target == settings.context_size else (
            "yarn_context_size_exceeded" if enabled else "context_size_exceeded")
        storage = settings.qsa_kv_offload or QsaKvStorageSettings(enabled=previous.qsa_kv_offload.enabled,
            budget_bytes=previous.qsa_kv_offload.budget_bytes, ram_budget_bytes=previous.qsa_kv_offload.ram_budget_bytes)
        quantization = settings.kv_quantization or previous.kv_quantization
        kv_memory = settings.kv_memory or previous.kv_memory
        if kv_memory.allocation == "dynamic" and kv_memory.placement in {"ram", "auto_ssd"}:
            raise _job_error("kv_dynamic_unavailable", "Experimental dynamic KV requires VRAM or automatic without SSD")
        if settings.kv_memory is not None:
            if not getattr(instance, "kv_memory_supported", False):
                raise _job_error("kv_memory_unsupported", "This runtime does not support configurable KV placement")
            storage = storage.model_copy(update={"enabled": False})
        if quantization.enabled and not getattr(instance, "kv_quantization_supported", False):
            raise _job_error("kv_quantization_unsupported", "the model or backend does not support TurboQuant KV")
        policy = QsaKvOffloadPolicy(**storage.model_dump(), target_context=target)
        if policy.enabled:
            if not instance.qsa_kv_offload_supported:
                raise _job_error("qsa_kv_offload_unsupported", "the model or backend does not support QSA KV offload")
            await self._validate_qsa_kv_policy(instance.artifact, policy, yarn_enabled=enabled)
        updated = previous.model_copy(update={"context_size": target, "yarn_enabled": enabled,
            "qsa_kv_offload": policy, "kv_quantization": quantization, "kv_memory": kv_memory,
            "pin": instance.pinned, "idle_ttl_seconds": instance.idle_ttl_seconds})
        if self.backend == "cuda" and getattr(instance, "kv_memory_supported", False):
            await asyncio.to_thread(self._validate_kv_memory_budget, instance.artifact, updated)
        async with self._lock:
            if self._instances.get(instance.id) is not instance or instance.state != RuntimeInstanceState.READY or \
                instance.active_requests or instance.queued_requests or instance.control_leases:
                raise _job_error("runtime_busy", "wait for model requests to finish before saving model context settings", retryable=True)
            self._mark_instance_unloading_locked(instance)
        try:
            await context.progress(0.01, message="Applying model context settings", data={"phase": "reloading", "context_size": target})
            await self._retire_instance(instance)
            result = await self.load(context, updated.model_dump(mode="json"))
            if result.get("context_size") != target:
                raise _job_error("context_reload_mismatch", "runtime context does not match the saved settings")
            async with self._context_policy_lock:
                overrides = dict(self.context_policy.model_overrides)
                if settings.context_size is None:
                    overrides.pop(name, None)
                else:
                    overrides[name] = target
                yarn = dict(self.context_policy.model_yarn_enabled)
                if enabled:
                    yarn[name] = True
                else:
                    yarn.pop(name, None)
                offload = dict(self.context_policy.model_qsa_kv_offload)
                offload[name] = storage
                quantization_map = dict(self.context_policy.model_kv_quantization)
                quantization_map[name] = quantization
                memory_map = dict(self.context_policy.model_kv_memory)
                memory_map[name] = kv_memory
                saved = self.context_policy.model_copy(update={"model_overrides": overrides,
                    "model_yarn_enabled": yarn, "model_qsa_kv_offload": offload, "model_kv_quantization": quantization_map,
                    "model_kv_memory": memory_map})
                if self.store is not None:
                    await asyncio.to_thread(self.store.save_runtime_context_policy, saved.model_dump())
                self.context_policy = saved
            return {**result, "max_context": target, "context_override": target if settings.context_size is not None else None,
                "yarn_enabled": enabled, "qsa_kv_offload": policy.model_dump(),
                "kv_quantization": quantization.model_dump(), "kv_memory": kv_memory.model_dump(), "warning": warning}
        except BaseException:
            for item in list(self._instances.values()):
                if item.artifact.resource.name == name:
                    await self._retire_instance(item)
            self._load_requests[name] = previous
            try:
                await self.load(_MemoryPlanContext(context, recovery=True), previous.model_dump(mode="json"))
            except BaseException as error:
                await context.log(f"Could not restore the previous model configuration: {type(error).__name__}")
            raise

    async def reload_runtime(
        self,
        context_size: int | None,
        instance_id: UUID | None = None,
        *, context: JobContext | None = None, yarn_enabled: bool | None = None, save_policy: bool = True,
    ) -> dict[str, Any]:
        async with self._runtime_control_lease(instance_id) as (instance, backend):
            if backend is None:
                raise BackendError("model_not_loaded", "no runtime is available")
            requested = context_size
            warning = None
            if instance is not None:
                enabled = self.context_policy.model_yarn_enabled.get(instance.artifact.resource.name, False) if yarn_enabled is None else yarn_enabled
                if enabled and (self.backend not in {"cuda", "metal"} or instance.artifact.resource.yarn_context_capacity is None):
                    raise BackendError("yarn_unsupported", "YaRN extension is not supported by this model/backend", status_code=400)
                context_size = requested if requested is not None else self.resolve_context_size(instance.artifact, ignore_override=True)
                limit = self._context_limit(instance.artifact, enabled) or instance.context_capacity
                if limit is not None and context_size > limit:
                    context_size = limit
                    warning = "yarn_context_size_exceeded" if enabled else "context_size_exceeded"
            elif context_size is None:
                context_size = self.context_policy.fallback_context_size
            if instance is not None and instance.context_job is not None:
                raise BackendError("runtime_busy", "context settings are already being applied", status_code=409)
            if instance is not None:
                instance.context_job = context or True
                instance.load_progress = 0.0
            try:
                if context is not None:
                    await context.progress(0.01, message="Applying context settings", data={"phase": "reloading", "context_size": context_size})
                result = await backend.reload_runtime(context_size)
                actual = result.get("max_context")
                if type(actual) is not int or actual != context_size:
                    raise BackendError("context_reload_mismatch", f"Requested context {context_size}, but runtime reported {actual}")
                if instance is None:
                    return result
                async with self._lock:
                    if self._instances.get(instance.id) is instance:
                        instance.context_size = actual
                        name = instance.artifact.resource.name
                        request = self._load_requests.get(name)
                        if request is not None:
                            self._load_requests[name] = request.model_copy(update={"context_size": instance.context_size,
                                "yarn_enabled": enabled,
                                "qsa_kv_offload": request.qsa_kv_offload.model_copy(update={"target_context": instance.context_size})})
                if save_policy:
                    async with self._context_policy_lock:
                        overrides = dict(self.context_policy.model_overrides)
                        if requested is None:
                            overrides.pop(instance.artifact.resource.name, None)
                        else:
                            overrides[instance.artifact.resource.name] = actual
                        yarn_settings = dict(self.context_policy.model_yarn_enabled)
                        if enabled:
                            yarn_settings[instance.artifact.resource.name] = True
                        else:
                            yarn_settings.pop(instance.artifact.resource.name, None)
                        policy = self.context_policy.model_copy(update={"model_overrides": overrides, "model_yarn_enabled": yarn_settings})
                        if self.store is not None:
                            await asyncio.to_thread(self.store.save_runtime_context_policy, policy.model_dump())
                        self.context_policy = policy
                if context is not None:
                    await context.progress(1.0, message="Context settings applied", data={"phase": "ready", "context_size": actual})
                return {**result, "instance_id": str(instance.id), "context_override": actual if requested is not None else None,
                    "yarn_enabled": enabled, "warning": warning}
            finally:
                if instance is not None:
                    instance.context_job = None

    async def clear_runtime_cache(
        self,
        instance_id: UUID | None = None,
    ) -> dict[str, Any]:
        async with self._runtime_control_lease(instance_id) as (_instance, backend):
            if backend is None:
                raise BackendError("model_not_loaded", "no runtime is available")
            return await backend.clear_runtime_cache()

    async def inspect_prefix_cache(self, namespace: str | None = None, *, offset: int = 0, limit: int = 100,
        query: str = '', search_in: str = 'all', text_filter: str = 'all', chain_filter: str = 'all', sort: str = 'recent') -> dict[str, Any]:
        directory = self._prefix_cache_directory()
        decode_text = None
        if namespace and query.strip() and search_in != 'id':
            identity = await asyncio.to_thread(prefix_cache_identity, directory, namespace)
            try:
                if identity.get('model_path') or identity.get('model_name'):
                    artifact = await self.catalog.resolve_path(identity['model_path']) if identity.get('model_path') else await self.catalog.resolve(identity['model_name'])
                    decode_text = lambda tokens: decode_prefix_cache_tokens(artifact.path, tokens)
            except (OSError, ValueError, KeyError, ImportError, ModelArtifactNotFoundError):
                pass
        result = await asyncio.to_thread(inspect_prefix_cache, directory, namespace, offset=offset, limit=limit,
            query=query, search_in=search_in, text_filter=text_filter, chain_filter=chain_filter, sort=sort, decode_text=decode_text)
        async with self._lock:
            result['can_clear'] = not self._loading_model_names and self._memory_configuration_job is None and not any(
                item.active_requests or item.queued_requests or item.control_leases > item.read_control_leases or item.prefix_pending_writes
                or item.state not in {RuntimeInstanceState.READY, RuntimeInstanceState.FAILED} for item in self._instances.values())
        return result

    async def prefix_cache_text(self, namespace: str, block: str, *, offset: int = 0, limit: int = 8192) -> dict[str, Any]:
        record = await asyncio.to_thread(read_prefix_cache_tokens, self._prefix_cache_directory(), namespace, block)
        if not record.get('available'):
            return record
        try:
            artifact = await self.catalog.resolve_path(record['model_path']) if record.get('model_path') else await self.catalog.resolve(record['model_name'])
            tokens = record['tokens']
            text = await asyncio.to_thread(decode_prefix_cache_tokens, artifact.path, tokens[offset:offset + limit])
            return {'available': True, 'text': text, 'total_tokens': len(tokens), 'offset': offset,
                'next_offset': offset + limit if offset + limit < len(tokens) else None}
        except (OSError, ValueError, KeyError, ImportError, ModelArtifactNotFoundError):
            return {'available': False, 'reason': 'tokenizer_unavailable'}

    async def purge_prefix_cache(self, namespace: str | None = None) -> dict[str, Any]:
        async with self._prefix_budget_lock:
            deadline = time.monotonic() + 2.0
            while True:
                async with self._lock:
                    instances = list(self._instances.values())
                    if self._loading_model_names or self._memory_configuration_job is not None or any(
                        item.active_requests or item.queued_requests or item.control_leases > item.read_control_leases or item.prefix_pending_writes
                        or item.state not in {RuntimeInstanceState.READY, RuntimeInstanceState.FAILED} for item in instances):
                        raise BackendError('runtime_busy', 'wait for inference and cache writes to finish before clearing cache', retryable=True, status_code=409)
                    if not any(item.control_leases for item in instances):
                        current = [item for item in instances if item.state == RuntimeInstanceState.READY]
                        self._memory_configuration_job = uuid4()
                        for item in current:
                            item.control_leases += 1
                        break
                    if time.monotonic() >= deadline:
                        raise BackendError('runtime_busy', 'wait for runtime status queries to finish before clearing cache', retryable=True, status_code=409)
                await asyncio.sleep(.02)
            try:
                for item in current:
                    await item.backend.refresh_prefix_cache_index()
                before = await asyncio.to_thread(inspect_prefix_cache, self._prefix_cache_directory(), namespace)
                if namespace is not None and not before['data']:
                    raise BackendError('cache_not_found', 'prefix cache group was not found', status_code=404)
                if namespace is None:
                    for item in current:
                        await item.backend.clear_runtime_cache()
                result = await asyncio.to_thread(purge_prefix_cache, self._prefix_cache_directory(), namespace)
                for item in current:
                    await item.backend.refresh_prefix_cache_index()
                    await self._refresh_instance_usage(item)
                remaining = await asyncio.to_thread(maintain_prefix_disk_budget, self._prefix_cache_directory(), self.memory_policy.prefix_disk_limit_bytes, evict=False)
                self._prefix_disk_status = {key: value for key, value in remaining.items() if key.startswith('prefix_cache_')}
                self._prefix_disk_checked_at = time.monotonic()
                if namespace is None:
                    result['removed_blocks'] = before['total_blocks'] - remaining['prefix_cache_total_disk_blocks']
                    result['released_bytes'] = before['total_bytes'] - remaining['prefix_cache_total_disk_bytes']
                return {'status': 'ok', **result, **self._prefix_disk_status}
            finally:
                async with self._lock:
                    self._memory_configuration_job = None
                    self._request_state_changed.notify_all()
                for item in current:
                    await self._release_control_lease(item)

    async def trim_runtime_cache(
        self,
        target_bytes: int = 0,
        instance_id: UUID | None = None,
    ) -> dict[str, Any]:
        if target_bytes < 0:
            raise ValueError("target_bytes must be non-negative")
        async with self._runtime_control_lease(instance_id, concurrent_prefix=True) as (instance, backend):
            if backend is None:
                raise BackendError("model_not_loaded", "no runtime is available")
            trim = getattr(backend, "trim_runtime_cache", None)
            if not callable(trim):
                raise BackendError(
                    "unsupported_operation",
                    "this runtime does not expose a tiered prefix cache",
                    status_code=501,
                )
            result = await trim(target_bytes)
        if instance is not None:
            await self._refresh_instance_usage(instance)
        return result

    def realtime_connect(self, *, mode: str = "audio") -> Any:
        instance = self._instances.get(self._last_instance_id) if self._last_instance_id else None
        if instance is not None and instance.state in {
            RuntimeInstanceState.READY,
            RuntimeInstanceState.BUSY,
        }:
            connector = instance.backend.realtime_connect(mode=mode)
            return _LeasedRealtimeConnector(self, instance, connector)
        if self.fallback is not None:
            return self.fallback.realtime_connect(mode=mode)
        raise BackendError("model_not_loaded", "no runtime is available")

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            self._request_state_changed.notify_all()
            instances = list(self._instances.values())
            idle_reaper = self._idle_reaper_task
            self._idle_reaper_task = None
            background_tasks = tuple(self._background_tasks)
            self._background_tasks.clear()
            self._idle_reaper_wakeup.set()
            for model, event in self._load_events.items():
                self._load_errors[model] = ErrorDetail(
                    code="runtime_pool_closed",
                    message="runtime pool closed while the model was loading",
                    retryable=True,
                )
                event.set()
            self._load_events.clear()
            self._loading_model_names.clear()
            self._loading_artifact_ids.clear()
            self._load_failures.clear()
            self._load_ports.clear()
            self._load_bytes.clear()
            self._reserved_ports.clear()
        if idle_reaper is not None and idle_reaper is not asyncio.current_task():
            await idle_reaper
        for task in background_tasks:
            if task is not asyncio.current_task() and not task.done():
                task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        await self._drain_control_lease_releases()
        first_error: Exception | None = None
        retirements = await asyncio.gather(
            *(self._retire_instance(instance) for instance in instances),
            return_exceptions=True,
        )
        first_error = next(
            (
                result
                for result in retirements
                if isinstance(result, Exception)
            ),
            None,
        )
        async with self._lock:
            self._instances.clear()
            self._session_routes.clear()
        if self.fallback is not None:
            try:
                await self.fallback.aclose()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    async def _select(self, model: str, *, session_id: UUID | None) -> _Runtime | None:
        stale_backend: ChatBackend | None = None
        async with self._lock:
            if session_id is not None:
                instance_id = self._session_routes.get(session_id)
                if instance_id is not None:
                    routed = self._instances.get(instance_id)
                    if (
                        routed is not None
                        and self._matches_model(routed, model)
                        and routed.state in {
                            RuntimeInstanceState.READY,
                            RuntimeInstanceState.BUSY,
                        }
                    ):
                        return routed
                    self._session_routes.pop(session_id, None)
                    if routed is not None and routed.state in {
                        RuntimeInstanceState.READY,
                        RuntimeInstanceState.BUSY,
                    }:
                        stale_backend = routed.backend
            matches = [
                item
                for item in self._instances.values()
                if self._matches_model(item, model)
                and item.state in {
                    RuntimeInstanceState.LOADING,
                    RuntimeInstanceState.READY,
                    RuntimeInstanceState.BUSY,
                }
            ]
        if len(matches) > 1:
            raise BackendError("ambiguous_model", f"multiple loaded runtimes match {model}")
        if stale_backend is not None and session_id is not None:
            close = getattr(stale_backend, "close_session", None)
            if callable(close):
                with suppress(Exception):
                    await close(session_id)
        return matches[0] if matches else None

    @staticmethod
    def _matches_model(instance: _Runtime, model: str) -> bool:
        resource = instance.artifact.resource
        return model == resource.name or model == resource.id

    @staticmethod
    def _same_artifact_revision(
        left: DiscoveredModel,
        right: DiscoveredModel,
    ) -> bool:
        return (
            left.resource.id == right.resource.id
            and left.resource.modified_at == right.resource.modified_at
            and left.path == right.path
        )

    def _begin_runtime_revival(self, model: str) -> bool:
        """Allow one immediate reload after an unexpected runtime exit."""

        now = time.monotonic()
        last = self._runtime_revival_at.get(model)
        if (
            last is not None
            and self.load_failure_cooldown_seconds > 0
            and now - last < self.load_failure_cooldown_seconds
        ):
            return False
        self._runtime_revival_at[model] = now
        return True

    async def _ensure_model_loaded_with_revival(
        self,
        model: str,
    ) -> _Runtime | None:
        try:
            return await self._ensure_model_loaded(model)
        except BackendError as error:
            if (
                error.code != "runtime_exited"
                or not error.retryable
                or not self._begin_runtime_revival(model)
            ):
                raise
            async with self._lock:
                self._load_failures.pop(model, None)
                self._load_errors.pop(model, None)
            return await self._ensure_model_loaded(model)

    async def _ensure_model_loaded(self, model: str) -> _Runtime | None:
        try:
            artifact = await self.catalog.resolve(model)
        except ModelArtifactNotFoundError:
            return None
        model_name = artifact.resource.name
        artifact_id = artifact.resource.id
        async with self._lock:
            if self._closed:
                return None
            ready = next(
                (
                    item
                    for item in self._instances.values()
                    if self._same_artifact_revision(item.artifact, artifact)
                    and item.state in {
                        RuntimeInstanceState.READY,
                        RuntimeInstanceState.BUSY,
                    }
                ),
                None,
            )
            event = self._load_events.get(model_name)
            request = self._load_requests.get(model_name)
            cached_failure = self._load_failures.get(model_name)
            if cached_failure is not None and (
                cached_failure.artifact_id != artifact_id
                or self.load_failure_cooldown_seconds == 0
                or time.monotonic() - cached_failure.failed_at
                >= self.load_failure_cooldown_seconds
            ):
                self._load_failures.pop(model_name, None)
                self._load_errors.pop(model_name, None)
                cached_failure = None
        if ready is not None:
            return ready
        if event is not None:
            await self._wait_for_model_ready(model_name)
            return await self._select_loaded_artifact(model_name, artifact)
        if cached_failure is not None:
            raise self._backend_load_error(cached_failure.detail)

        exact_request = (request or ModelLoadRequest(model=model_name)).model_copy(
            update={
                "model": model_name,
                "artifact_uri": f"mfq://{artifact_id}",
            }
        )
        context = _RuntimeLoadContext()
        try:
            await self.load(
                context,  # type: ignore[arg-type]
                exact_request.model_dump(mode="python"),
            )
        except JobExecutionError as error:
            if error.detail.code in {
                "model_already_loaded",
                "model_already_loading",
            }:
                await self._wait_for_model_ready(model_name)
                return await self._select_loaded_artifact(model_name, artifact)
            if error.detail.code == "model_artifact_not_found":
                return None
            if error.detail.code == "runtime_revision_busy":
                raise self._backend_load_error(error.detail) from error
            async with self._lock:
                self._load_errors[model_name] = error.detail
            raise self._backend_load_error(error.detail) from error
        except Exception as error:
            detail = self._runtime_load_error(error)
            async with self._lock:
                self._load_errors[model_name] = detail
            raise self._backend_load_error(detail) from error
        finally:
            await context.cleanup()
        return await self._select_loaded_artifact(model_name, artifact)

    async def _select_loaded_artifact(
        self,
        model_name: str,
        artifact: DiscoveredModel,
    ) -> _Runtime | None:
        selected = await self._select(model_name, session_id=None)
        if selected is None:
            return None
        if not self._same_artifact_revision(selected.artifact, artifact):
            raise BackendError(
                "model_name_conflict",
                f"loaded model name {model_name!r} refers to a different artifact",
                status_code=409,
            )
        return selected

    async def _wait_for_model_ready(self, model: str) -> bool:
        async with self._lock:
            ready = next(
                (
                    item
                    for item in self._instances.values()
                    if item.artifact.resource.name == model
                    and item.state in {
                        RuntimeInstanceState.READY,
                        RuntimeInstanceState.BUSY,
                    }
                ),
                None,
            )
            event = self._load_events.get(model)
            load_error = self._load_errors.get(model)
        if ready is not None:
            return True
        if event is None:
            if load_error is not None:
                raise self._backend_load_error(load_error)
            return False
        try:
            await asyncio.wait_for(
                event.wait(),
                timeout=max(1.0, self.startup_timeout_seconds),
            )
        except asyncio.TimeoutError:
            raise BackendError(
                "runtime_start_timeout",
                f"timed out waiting for model runtime: {model}",
                retryable=True,
                status_code=503,
            ) from None
        selected = await self._select(model, session_id=None)
        if selected is not None and selected.state in {
            RuntimeInstanceState.READY,
            RuntimeInstanceState.BUSY,
        }:
            return True
        async with self._lock:
            error = self._load_errors.get(model)
        if error is not None:
            raise self._backend_load_error(error)
        return False

    @asynccontextmanager
    async def benchmark_runtime(self, instance_id: UUID) -> AsyncIterator[tuple[Any, Any]]:
        deadline = asyncio.get_running_loop().time() + 2.0
        while True:
            async with self._lock:
                instance = self._instances.get(instance_id)
                if instance is None or instance.state != RuntimeInstanceState.READY:
                    raise BackendError("model_not_ready", "select a ready, loaded model for benchmarking")
                if self._memory_configuration_job is not None or instance.active_requests or instance.queued_requests:
                    raise BackendError("runtime_busy", "benchmarking requires an idle runtime", retryable=True)
                if not callable(getattr(instance.backend, "benchmark", None)):
                    raise BackendError("benchmark_unsupported", "this runtime does not expose native benchmark metrics")
                if not instance.control_leases:
                    instance.control_leases += 1
                    instance.active_requests += 1
                    instance.state = RuntimeInstanceState.BUSY
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise BackendError("runtime_busy", "runtime control operations have not finished; retry benchmarking", retryable=True)
            await asyncio.sleep(.02)
        try:
            yield instance, instance.backend
        finally:
            async with self._lock:
                instance.active_requests = max(0, instance.active_requests - 1)
                if instance.state == RuntimeInstanceState.BUSY and not instance.active_requests:
                    instance.state = RuntimeInstanceState.READY
                instance.last_used_at = datetime.now(timezone.utc)
            await self._release_control_lease(instance)

    @asynccontextmanager
    async def _runtime_control_lease(
        self,
        instance_id: UUID | None,
        *,
        allow_unready: bool = False,
        read_only: bool = False,
        concurrent_prefix: bool = False,
    ) -> AsyncIterator[tuple[_Runtime | None, ChatBackend | None]]:
        async with self._lock:
            if instance_id is None:
                instance = self._current_instance_locked()
            else:
                instance = self._instances.get(instance_id)
                if instance is None:
                    raise BackendError(
                        "runtime_instance_not_found",
                        f"runtime instance was not found: {instance_id}",
                        status_code=404,
                    )
                if not allow_unready and instance.state not in {
                    RuntimeInstanceState.READY,
                    RuntimeInstanceState.BUSY,
                }:
                    raise BackendError(
                        "model_not_ready",
                        f"model runtime is {instance.state.value}",
                        retryable=True,
                        status_code=409,
                    )
            backend = instance.backend if instance is not None else self.fallback
            leased = instance is not None and instance.state in {
                RuntimeInstanceState.READY,
                RuntimeInstanceState.BUSY,
            }
            if leased:
                read_only = read_only or concurrent_prefix and getattr(instance, "prefix_concurrent_maintenance", False)
                instance.control_leases += 1
                if read_only:
                    instance.read_control_leases += 1
        try:
            yield instance, backend
        finally:
            if leased and instance is not None:
                await self._release_control_lease(instance, read_only=read_only)

    async def _decrement_control_lease(self, instance: _Runtime, *, read_only: bool = False) -> None:
        async with self._lock:
            instance.control_leases = max(0, instance.control_leases - 1)
            if read_only:
                instance.read_control_leases = max(0, instance.read_control_leases - 1)
            self._request_state_changed.notify_all()

    def _finish_control_lease_release(self, task: asyncio.Task[None]) -> None:
        self._lease_release_tasks.discard(task)
        with suppress(asyncio.CancelledError):
            task.result()

    async def _release_control_lease(self, instance: _Runtime, *, read_only: bool = False) -> None:
        """Release a lease even if its caller is cancelled during cleanup."""

        await self._release_runtime_lease(
            self._decrement_control_lease(instance, read_only=read_only),
            name=f"mfq-server-runtime-lease-release-{instance.id}",
        )

    async def _release_runtime_lease(self, release: Coroutine[Any, Any, None], *, name: str) -> None:
        task = asyncio.create_task(release, name=name)
        self._lease_release_tasks.add(task)
        task.add_done_callback(self._finish_control_lease_release)
        await asyncio.shield(task)

    async def _drain_control_lease_releases(self) -> None:
        while self._lease_release_tasks:
            await asyncio.gather(
                *tuple(self._lease_release_tasks),
                return_exceptions=True,
            )

    def _current_instance_locked(self) -> _Runtime | None:
        instance = (
            self._instances.get(self._last_instance_id)
            if self._last_instance_id is not None
            else None
        )
        if instance is None or instance.state not in {
            RuntimeInstanceState.READY,
            RuntimeInstanceState.BUSY,
        }:
            instance = next(
                (
                    item
                    for item in self._instances.values()
                    if item.state in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
                ),
                None,
            )
        return instance

    def _lru_instance_for_unload_locked(
        self,
        *,
        excluded_ids: set[UUID] | None = None,
    ) -> _Runtime | None:
        excluded = excluded_ids or set()
        candidates = [
            item
            for item in self._instances.values()
            if item.id not in excluded
            and not item.pinned
            and item.state == RuntimeInstanceState.READY
            and item.active_requests == 0
            and item.queued_requests == 0
            and item.control_leases == 0
        ]
        if not candidates:
            return None
        victim = min(
            candidates,
            key=lambda item: (item.last_used_at or item.started_at, item.started_at),
        )
        return victim

    def _claim_lru_instance_for_unload_locked(self) -> _Runtime | None:
        victim = self._lru_instance_for_unload_locked()
        if victim is None:
            return None
        self._mark_instance_unloading_locked(victim)
        return victim

    def _claim_over_budget_instances_for_unload_locked(
        self,
        *,
        memory_ceiling: int | None = None,
        pending_releases: Sequence[_Runtime] = (),
    ) -> list[_Runtime]:
        if self.memory_policy.total_limit_bytes is not None or any(getattr(item, "resident_dynamic_budget", False) for item in self._instances.values()):
            return []
        if memory_ceiling is None:
            memory_ceiling = self._effective_runtime_memory_budget_locked()
        if memory_ceiling is None:
            return []
        committed = max(
            0,
            self._committed_pool_bytes_locked()
            - sum(self._committed_runtime_bytes(item) for item in pending_releases),
        )
        victims = []
        while committed > memory_ceiling:
            victim = self._claim_lru_instance_for_unload_locked()
            if victim is None:
                break
            victims.append(victim)
            committed = max(
                0,
                committed - self._committed_runtime_bytes(victim),
            )
        return victims

    async def _run_shared_cache_reclaimer(self) -> int:
        reclaimer = self.shared_cache_reclaimer
        if reclaimer is None:
            return 0
        try:
            released = max(0, int(await asyncio.to_thread(reclaimer)))
        except Exception:
            async with self._lock:
                self._shared_cache_reclaim_failures += 1
            return 0
        async with self._lock:
            if released > 0:
                self._shared_cache_reclaims += 1
                self._shared_cache_released_bytes += released
        return released

    async def _reclaim_shared_cache_for_budget(
        self,
        *,
        additional_bytes: int,
        prospective_model: str | None = None,
    ) -> int:
        """Release server-process caches before trimming or evicting models."""

        reclaimer = self.shared_cache_reclaimer
        if reclaimer is None:
            return 0
        async with self._lock:
            if self._closed:
                return 0
            if prospective_model is not None and (
                prospective_model in self._loading_model_names
                or any(
                    item.artifact.resource.name == prospective_model
                    and item.state != RuntimeInstanceState.FAILED
                    for item in self._instances.values()
                )
            ):
                return 0
            memory_ceiling = self._effective_runtime_memory_budget_locked()
            if memory_ceiling is None:
                return 0
            committed = self._committed_pool_bytes_locked() + additional_bytes
            if committed <= memory_ceiling:
                return 0
        return await self._run_shared_cache_reclaimer()

    async def _reclaim_shared_cache_under_pressure(self) -> int:
        if self.shared_cache_reclaimer is None:
            return 0
        async with self._lock:
            if self._closed:
                return 0
            level, _ratio, _ceiling, _committed = (
                self._runtime_memory_pressure_locked()
            )
            if level in {"normal", "disabled"}:
                return 0
            now = time.monotonic()
            if now - self._shared_cache_pressure_checked_at < max(
                1.0,
                self.metric_interval_seconds * 2,
            ):
                return 0
            self._shared_cache_pressure_checked_at = now
        return await self._run_shared_cache_reclaimer()

    async def _trim_idle_prefix_caches_for_budget(
        self,
        *,
        additional_bytes: int = 0,
        prospective_model: str | None = None,
        memory_target_bytes: int | None = None,
    ) -> None:
        if self.max_runtime_memory_bytes is None and memory_target_bytes is None:
            return
        async with self._lock:
            if self._closed:
                return
            memory_ceiling = self._effective_runtime_memory_budget_locked()
            if memory_ceiling is None:
                return
            if memory_target_bytes is not None:
                memory_ceiling = min(memory_ceiling, memory_target_bytes)
            if prospective_model is not None and (
                prospective_model in self._loading_model_names
                or any(
                    item.artifact.resource.name == prospective_model
                    and item.state != RuntimeInstanceState.FAILED
                    for item in self._instances.values()
                )
            ):
                return
            if prospective_model is not None:
                resident_names = {
                    item.artifact.resource.name
                    for item in self._instances.values()
                    if item.state != RuntimeInstanceState.FAILED
                }
                active_count = len(resident_names) + sum(
                    name not in resident_names for name in self._loading_model_names
                )
                if active_count >= self.max_instances:
                    return
            committed = self._committed_pool_bytes_locked() + additional_bytes
            if committed <= memory_ceiling:
                return
            candidates = sorted(
                (
                    item
                    for item in self._instances.values()
                    if item.state == RuntimeInstanceState.READY
                    and item.active_requests == 0
                    and item.queued_requests == 0
                    and item.control_leases == 0
                    and (item.kv_bytes or 0) > 0
                    and callable(getattr(item.backend, "trim_runtime_cache", None))
                ),
                key=lambda item: (item.last_used_at or item.started_at, item.started_at),
            )

        for instance in candidates:
            async with self._lock:
                if (
                    self._instances.get(instance.id) is not instance
                    or instance.state != RuntimeInstanceState.READY
                    or instance.active_requests != 0
                    or instance.queued_requests != 0
                    or instance.control_leases != 0
                ):
                    continue
                committed = self._committed_pool_bytes_locked() + additional_bytes
                excess = committed - memory_ceiling
                if excess <= 0:
                    return
                hot_bytes = instance.kv_bytes or 0
                target_bytes = max(0, hot_bytes - excess)
                trim = getattr(instance.backend, "trim_runtime_cache", None)
                if not callable(trim):
                    continue
                concurrent = getattr(instance, "prefix_concurrent_maintenance", False)
                instance.control_leases += 1
                if concurrent:
                    instance.read_control_leases += 1
            try:
                await asyncio.wait_for(trim(target_bytes), timeout=2.0)
            except Exception:
                pass
            finally:
                await self._release_control_lease(instance, read_only=concurrent)
            await self._refresh_instance_usage(instance)

    def _finish_model_load_locked(
        self,
        model: str,
        event: asyncio.Event,
        error: ErrorDetail | None,
    ) -> None:
        owns_event = self._load_events.get(model) is event
        if not owns_event:
            event.set()
            return
        artifact_id = self._loading_artifact_ids.pop(model, None)
        if error is None:
            self._load_errors.pop(model, None)
            self._load_failures.pop(model, None)
        elif model not in self._load_errors:
            self._load_errors[model] = error
        if error is not None and artifact_id is not None:
            self._load_failures[model] = _CachedLoadFailure(
                artifact_id=artifact_id,
                detail=error,
                failed_at=time.monotonic(),
            )
        self._loading_model_names.discard(model)
        self._load_events.pop(model, None)
        port = self._load_ports.pop(model, None)
        if port is not None:
            self._reserved_ports.discard(port)
        self._load_bytes.pop(model, None)
        self._resident_load_limits.pop(model, None)
        event.set()

    @staticmethod
    def _runtime_load_error(error: BaseException) -> ErrorDetail:
        if isinstance(error, JobExecutionError):
            return error.detail
        if isinstance(error, asyncio.CancelledError):
            return ErrorDetail(
                code="runtime_load_cancelled",
                message="runtime loading was cancelled",
                retryable=True,
            )
        return ErrorDetail(
            code="runtime_load_failed",
            message=str(error) or type(error).__name__,
            retryable=True,
        )

    @staticmethod
    def _backend_load_error(detail: ErrorDetail) -> BackendError:
        status_by_code = {
            "model_artifact_incomplete": 409,
            "model_conversion_required": 409,
            "runtime_model_too_large": 413,
            "runtime_revision_busy": 409,
            "unsupported_device": 422,
            "unsupported_runtime_option": 422,
            "runtime_identity_mismatch": 502,
        }
        return BackendError(
            detail.code,
            detail.message,
            retryable=detail.retryable,
            status_code=status_by_code.get(detail.code, 503),
        )

    @staticmethod
    def _committed_runtime_bytes(instance: _Runtime) -> int:
        if getattr(instance, "resident_dynamic_budget", False):
            if instance.active_requests or instance.queued_requests:
                return instance.resident_budget_limit or instance.resident_memory_used
            return instance.resident_memory_used + min(256 << 20, instance.resident_budget_limit // 20)
        estimates = [
            value
            for value in (instance.resident_bytes, instance.reserved_bytes)
            if value is not None and value > 0
        ]
        if estimates:
            # RSS can be temporarily smaller than the model's committed
            # capacity when mmap pages or an expert LRU have not been touched
            # yet. Conversely KV/runtime allocations can push live residency
            # beyond the load estimate. Admission must retain both ledgers.
            return max(estimates)
        return instance.artifact.resource.total_bytes

    def _committed_pool_bytes_locked(self) -> int:
        resident_names = {
            item.artifact.resource.name
            for item in self._instances.values()
            if item.state != RuntimeInstanceState.FAILED
        }
        return sum(
            self._committed_runtime_bytes(item)
            for item in self._instances.values()
            if item.state != RuntimeInstanceState.FAILED
        ) + sum(
            reserved_bytes
            for name, reserved_bytes in self._load_bytes.items()
            if name not in resident_names
        )

    def _resident_process_usage_locked(self) -> tuple[int, int]:
        physical, cpu = 0, 0
        for item in self._instances.values():
            if item.state == RuntimeInstanceState.FAILED:
                continue
            pid = getattr(getattr(item, "process", None), "pid", None)
            measured = process_physical_footprint(pid) if isinstance(pid, int) and pid > 0 else None
            used = measured if measured is not None else getattr(item, "resident_memory_used", 0) or getattr(item, "resident_bytes", 0) or 0
            memory = getattr(item, "memory", None)
            gpu = (memory.wired_bytes if memory.wired_bytes is not None else
                (memory.resident_weight_bytes or 0) + (memory.kv_bytes or 0)) if memory else 0
            physical += used
            cpu += max(0, used - gpu)
        return physical, cpu

    def _effective_runtime_memory_budget_locked(self, policy: RuntimeMemoryPolicy | None = None) -> int | None:
        ceiling = self._resident_total_limit(policy)
        if ceiling is None or self.backend != "metal":
            return ceiling
        physical, cpu = self._resident_process_usage_locked()
        metal = metal_allocation_limit()
        if metal is not None:
            ceiling = min(ceiling, metal + cpu)
        snapshot = host_memory_snapshot()
        if snapshot is None:
            return ceiling
        reserve = automatic_memory_reserve(snapshot.total)
        ceiling = min(ceiling, max(1, snapshot.total - reserve))
        own_cpu = cpu + (process_physical_footprint(os.getpid()) or 0)
        reclaimable = snapshot.reclaimable(active_ratio=0.25, own_cpu_bytes=own_cpu)
        released = self._memory_release_credit.update(physical, snapshot.reclaimable(active_ratio=0.0))
        dynamic_ceiling = physical + reclaimable + released - reserve
        return max(1, min(ceiling, dynamic_ceiling))

    def _runtime_memory_pressure_locked(
        self,
    ) -> tuple[str, float | None, int | None, int]:
        committed = self._committed_pool_bytes_locked()
        effective = self._effective_runtime_memory_budget_locked()
        if effective is None or (self.backend != "metal" and not self.automatic_memory_budget):
            return "disabled", None, effective, committed
        ratio = committed / max(1, effective)
        if (
            ratio >= _AUTOMATIC_MEMORY_HARD_RATIO
            or time.monotonic() < self._host_vm_pressure_until
        ):
            level = "hard"
        elif ratio >= _AUTOMATIC_MEMORY_SOFT_RATIO:
            level = "soft"
        else:
            level = "normal"
        return level, ratio, effective, committed

    def _observe_host_memory_pressure_locked(self) -> None:
        if self.backend != "metal":
            return
        snapshot = host_memory_snapshot()
        if snapshot is None or snapshot.compression_bytes is None or snapshot.swapout_bytes is None:
            self._host_vm_counters = None
            self._host_vm_pressure_until = 0.0
            return
        current = (snapshot.compression_bytes, snapshot.swapout_bytes)
        previous, self._host_vm_counters = self._host_vm_counters, current
        reserve = automatic_memory_reserve(snapshot.total)
        headroom = max(1 << 30, snapshot.total // 32)
        if snapshot.reclaimable(active_ratio=0.0) > reserve + headroom:
            self._host_vm_pressure_until = 0.0
            return
        if previous is None or any(now < before for now, before in zip(current, previous)):
            self._host_vm_pressure_until = 0.0
            return
        if current[0] - previous[0] >= 128 << 20 or current[1] - previous[1] >= 64 << 20:
            self._host_vm_pressure_until = time.monotonic() + max(5.0, self.metric_interval_seconds * 2)

    def _estimated_load_bytes(
        self,
        artifact: DiscoveredModel,
        request: ModelLoadRequest,
    ) -> int:
        total_bytes = max(
            0,
            artifact.resource.total_bytes - artifact.always_streamed_bytes,
        )
        cache_gb = request.moe_gpu_cache_gb
        streamed_bytes = artifact.routed_expert_bytes
        if streamed_bytes <= 0:
            return total_bytes
        if cache_gb is None:
            if self.backend != "metal" or artifact.resource.format != "hf":
                return total_bytes
            physical_bytes = total_physical_memory()
            if physical_bytes is None:
                # Conservatively reserve the complete model when the host
                # cannot reproduce the native Metal worker's automatic budget.
                return total_bytes
            cache_bytes = max(1 << 30, physical_bytes * 2 // 3)
        elif cache_gb <= 0:
            return total_bytes
        else:
            cache_bytes = int(cache_gb * (1 << 30))
        resident_expert_bytes = min(streamed_bytes, cache_bytes)
        return total_bytes - streamed_bytes + resident_expert_bytes

    def _expert_load_headroom_locked(self, ceiling: int, model: str) -> int:
        current = [item for item in self._instances.values()
            if item.state != RuntimeInstanceState.FAILED and item.artifact.resource.name != model]
        reserve = min(256 << 20, ceiling // 20)
        fixed = 0
        for item in current:
            if (getattr(item, "resident_dynamic_budget", False) and item.state == RuntimeInstanceState.READY
                and not item.active_requests and not item.queued_requests
                and item.control_leases == getattr(item, "read_control_leases", 0)):
                prefix = (item.memory.prefix_cache_bytes or 0) if item.memory else 0
                fixed += max(0, item.resident_memory_used - item.reclaimable_weight_bytes - prefix) + reserve
            else:
                fixed += getattr(item, "resident_budget_limit", 0) or self._committed_runtime_bytes(item)
        names = {item.artifact.resource.name for item in current} | {model}
        fixed += sum(max(self._load_bytes.get(name, 0), self._resident_load_limits.get(name, 0))
            for name in self._load_bytes.keys() | self._resident_load_limits.keys() if name not in names)
        return max(0, ceiling - fixed - reserve)

    def _cuda_device_ids(self, request: ModelLoadRequest) -> list[str]:
        if request.device_ids:
            return request.device_ids[:1]
        # Without explicit model parallelism the worker only uses visible GPU 0.
        visible = self.runtime_environment.get("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        return [visible.split(",")[0].strip()]

    def _cuda_discrete_device_capacity(self, request: ModelLoadRequest) -> int | None:
        hardware = hardware_identity()
        visible = self._cuda_device_ids(request)
        if hardware.memory_architecture != "discrete" or any(not value.isdecimal() for value in visible):
            return None
        devices = {device.device_id or str(index): device for index, device in enumerate(hardware.gpu_memory)}
        selected = [devices.get(value) for value in visible]
        if any(device is None or device.unified or device.capacity_bytes is None for device in selected):
            return None
        return sum(device.capacity_bytes for device in selected if device is not None)

    def _cuda_device_vram_limit(self, request: ModelLoadRequest) -> int | None:
        """VRAM the native worker may allocate on discrete GPUs (MFQ_CUDA_VRAM_LIMIT_GIB)."""
        capacity = self._cuda_discrete_device_capacity(request)
        if not capacity:
            return None
        configured = self.runtime_environment.get(
            "MFQ_CUDA_VRAM_LIMIT_GIB", os.environ.get("MFQ_CUDA_VRAM_LIMIT_GIB")
        )
        if configured:
            try:
                requested = int(float(configured) * (1 << 30))
            except (ValueError, OverflowError):
                raise _job_error("runtime_vram_budget_invalid", "VRAM limit must be a positive finite GiB value") from None
            if requested <= 0:
                raise _job_error("runtime_vram_budget_invalid", "VRAM limit must be a positive finite GiB value")
            limit = min(capacity, requested)
        else:
            limit = max(0, capacity - max(1 << 30, capacity // 16))
        # This is the worker's ceiling. Live free VRAM and reclaimable idle
        # caches are accounted together by admission and resident rebalancing;
        # capping this ceiling by raw free VRAM would prevent cache reuse and
        # model replacement while an existing worker still owns its storage.
        if limit <= 0:
            raise _job_error("runtime_vram_budget_exceeded", "No free VRAM remains for the model runtime", retryable=True)
        return limit

    def _apply_automatic_expert_residency(
        self,
        artifact: DiscoveredModel,
        request: ModelLoadRequest,
        *,
        memory_ceiling: int | None = None,
    ) -> ModelLoadRequest:
        """Choose expert residency within the physical device and runtime budgets."""

        ceiling = (
            memory_ceiling
            if memory_ceiling is not None
            else self.max_runtime_memory_bytes
        )
        if self.backend == "cuda" and request.moe_gpu_cache_gb is None:
            device_ceiling = self._cuda_device_vram_limit(request)
            if device_ceiling is not None:
                ceiling = min(ceiling, device_ceiling) if ceiling is not None else device_ceiling
        if (
            request.moe_gpu_cache_gb is not None
            or ceiling is None
            or artifact.resource.format not in {"hf", "mfq"}
            or artifact.routed_expert_bytes <= 0
        ):
            return request
        resident_total_bytes = max(
            0,
            artifact.resource.total_bytes - artifact.always_streamed_bytes,
        )
        reserve = (
            self._cuda_residency_runtime_reserve(artifact, request, ceiling=ceiling)
            if self.backend == "cuda" and artifact.resource.format == "mfq"
            else None
        )
        runtime_reserve = reserve.total_bytes if reserve is not None else 0
        fits = resident_total_bytes + runtime_reserve <= ceiling
        dense_bytes = max(0, resident_total_bytes - artifact.routed_expert_bytes)
        runtime_headroom = 0 if self.backend == "cuda" else min(
            4 << 30, max(1 << 30, ceiling // 20))
        # Native cache admission subtracts concrete KV/workspace requirements
        # and can lend expert storage to prefill; do not charge them twice.
        resident_expert_bytes = min(
            artifact.routed_expert_bytes,
            max(0, ceiling - runtime_headroom - dense_bytes),
        )
        environment = {**os.environ, **self.runtime_environment}
        # CUDA expert banks that the RAM-expert pipeline can serve decode (and
        # verify MTP drafts) as CUDA graphs only through the expert cache; plain
        # residency falls back to eager per-expert dispatch, measured 3-10x
        # slower with MTP on Qwen3.8-Flash-Next and Agnes V4-XS. When every
        # expert fits a discrete GPU, cache all of them rather than leaving
        # them resident, unless the pipeline was explicitly disabled.
        if (fits and reserve is not None and environment.get("MFQ_MOE_PIPELINE") != "0"
                and self._cuda_discrete_device_capacity(request) is not None
                and supports_cuda_ram_expert_pipeline(artifact.path)):
            # The native cache deducts its GPU KV charge from the requested
            # budget; ask for it on top so every expert still gets a slot.
            margin = max(_CUDA_CACHE_ALL_MIN_MARGIN_BYTES, artifact.routed_expert_bytes // 100)
            decision, cache_bytes = "cache_all", artifact.routed_expert_bytes + reserve.kv_bytes + margin
        elif fits:
            decision, cache_bytes = "resident", 0
        elif resident_expert_bytes > 0:
            decision, cache_bytes = "cache", resident_expert_bytes
        else:
            decision, cache_bytes = "no_cache_capacity", 0
        unavailable = None
        if reserve is not None and cache_bytes > 0:
            candidate = request.model_copy(update={"moe_gpu_cache_gb": cache_bytes / float(1 << 30)})
            try:
                unavailable = self._cuda_expert_cache_host_plan(artifact, candidate, environment).failure
            except (JobExecutionError, OSError) as error:
                unavailable = str(error)
            if unavailable is not None and resident_total_bytes <= ceiling:
                # Runtime headroom must not turn a load that launched resident
                # into a host-RAM launch failure.
                decision, cache_bytes = "resident", 0
        if reserve is not None:
            logger.info(
                "cuda_auto_expert_residency model=%s total_bytes=%d resident_total_bytes=%d "
                "context=%d kv_reserve_bytes=%d kv_source=%s workspace_reserve_bytes=%d "
                "overhead_reserve_bytes=%d runtime_reserve_bytes=%d ceiling_bytes=%d "
                "decision=%s cache_bytes=%d cache_gib=%.6f cache_unavailable=%r",
                artifact.resource.name, artifact.resource.total_bytes, resident_total_bytes,
                reserve.context, reserve.kv_bytes, reserve.kv_source, reserve.workspace_bytes,
                reserve.overhead_bytes, runtime_reserve, ceiling,
                decision, cache_bytes, cache_bytes / float(1 << 30), unavailable,
            )
        if decision == "resident":
            if artifact.resource.format == "hf":
                # Native-HF workers interpret an omitted cache budget as
                # "stream experts".  Make the opposite decision explicit
                # when the complete checkpoint fits; otherwise a 512-GiB
                # host needlessly starts a cold SSD LRU for a ~160-GiB MoE.
                return request.model_copy(update={"moe_gpu_cache_gb": 0.0})
            return request
        if cache_bytes <= 0:
            return request
        return request.model_copy(
            update={
                "moe_gpu_cache_gb": cache_bytes / float(1 << 30),
            }
        )

    def _cuda_residency_runtime_reserve(
        self, artifact: DiscoveredModel, request: ModelLoadRequest, *, ceiling: int | None = None
    ) -> _CudaResidencyReserve:
        """Estimate target GPU KV capacity, workspace headroom and fixed CUDA costs."""
        context = self.resolve_context_size(artifact, request.context_size, yarn_enabled=request.yarn_enabled)
        try:
            plan = checkpoint_kv_budget(artifact, context, request.kv_memory, request.kv_quantization)
        except (OSError, ValueError):
            plan = None
        if plan is None:
            # Unsupported/absent metadata must not turn into zero KV reserve.
            # One GiB minimum also covers small contexts and unknown fixed state.
            per_token = checkpoint_attention_kv_bytes_per_token(artifact)
            if per_token is not None:
                kv_bytes = max(1 << 30, context * per_token)
                source = "config_attention_fp16_min_1gib"
            else:
                # Without geometry this is only a conservative guess. Capped at
                # the ceiling it still rules out full residency, but a long
                # default context no longer reports more KV than the GPU holds.
                kv_bytes = max(1 << 30, context * _CUDA_FALLBACK_KV_BYTES_PER_TOKEN)
                kv_bytes = min(kv_bytes, ceiling) if ceiling is not None else kv_bytes
                source = "fallback_256kib_per_token_min_1gib_max_ceiling"
            kv_bytes = max(kv_bytes, request.kv_memory.gpu_budget_bytes or 0)
        elif (request.kv_memory.gpu_budget_bytes is not None
                or artifact.resource.architecture.casefold().startswith("qwen4_exp")):
            # Only qwen4_exp's tiered KvMemoryManager keeps a context-dependent
            # fraction of KV on the GPU (validated by _validate_kv_memory_budget).
            kv_bytes = plan.gpu_budget_bytes
            source = "checkpoint_kv_budget"
        else:
            # Other architectures preallocate full-context KV on the GPU.
            kv_bytes = plan.indexer_bytes + plan.raw_bytes
            source = "checkpoint_kv_full_context"
        configured = self.runtime_environment.get(
            "MFQ_CUDA_WORKSPACE_RESERVE_GIB", os.environ.get("MFQ_CUDA_WORKSPACE_RESERVE_GIB", "4")
        )
        try:
            gib = float(configured)
            workspace_bytes = int(gib * (1 << 30))
            if not math.isfinite(gib) or gib < 0 or workspace_bytes > (1 << 64) - 1:
                raise ValueError
        except (ValueError, OverflowError):
            raise _job_error(
                "runtime_workspace_budget_invalid", "CUDA workspace reserve must be a finite non-negative GiB value"
            ) from None
        # KvMemoryManager::finalize requires this free headroom even without an
        # expert cache; it is not a separate preallocated 4-GiB device buffer.
        return _CudaResidencyReserve(context, kv_bytes, source, workspace_bytes)

    @staticmethod
    def _observed_runtime_bytes(
        process_resident_bytes: int | None,
        status: dict[str, Any] | None,
        *, separate_memory: bool = False,
    ) -> int | None:
        candidates = [
            process_resident_bytes
            if not separate_memory and process_resident_bytes is not None and process_resident_bytes >= 0
            else 0
        ]
        if status is not None:
            mlx_active = status.get("mlx_active_bytes")
            mlx_cache = status.get("mlx_cache_bytes")
            if (
                isinstance(mlx_active, (int, float))
                and mlx_active >= 0
                and isinstance(mlx_cache, (int, float))
                and mlx_cache >= 0
            ):
                candidates.append(int(mlx_active + mlx_cache))
            for name in ("cuda_reserved_bytes", "cuda_allocated_bytes"):
                value = status.get(name)
                if isinstance(value, (int, float)) and value >= 0:
                    candidates.append(int(value))
        observed = max(candidates)
        return observed if observed > 0 else None

    def _unroute_instance_locked(self, instance: _Runtime) -> None:
        self._session_routes = {
            session_id: instance_id
            for session_id, instance_id in self._session_routes.items()
            if instance_id != instance.id
        }
        if self._last_instance_id == instance.id:
            self._last_instance_id = next(
                (
                    item.id
                    for item in self._instances.values()
                    if item.state in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
                ),
                None,
            )

    def _mark_instance_unloading_locked(self, instance: _Runtime) -> None:
        instance.state = RuntimeInstanceState.UNLOADING
        self._unroute_instance_locked(instance)
        self._request_state_changed.notify_all()

    def _detach_instance_locked(self, instance: _Runtime) -> None:
        self._instances.pop(instance.id, None)
        self._unroute_instance_locked(instance)
        self._request_state_changed.notify_all()

    async def _idle_reaper(self) -> None:
        while True:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._idle_reaper_wakeup.wait(),
                    timeout=self.metric_interval_seconds,
                )
            self._idle_reaper_wakeup.clear()
            async with self._lock:
                if self._closed:
                    return
                now_monotonic = asyncio.get_running_loop().time()
                refresh = [
                    item
                    for item in self._instances.values()
                    if item.state in {
                        RuntimeInstanceState.READY,
                        RuntimeInstanceState.BUSY,
                    }
                    and now_monotonic - item.usage_refreshed_at
                    >= self.metric_interval_seconds
                ]
                for item in refresh:
                    item.usage_refreshed_at = now_monotonic
            if refresh:
                await asyncio.gather(
                    *(self._refresh_instance_usage(item) for item in refresh)
                )

            try:
                await self._rebalance_prefix_cache_budget()
                await self._rebalance_resident_budget()
            except Exception:
                self._prefix_budget_failures += 1
            try:
                await self._maintain_prefix_disk_budget()
            except OSError:
                self._prefix_budget_failures += 1

            async with self._lock:
                self._observe_host_memory_pressure_locked()
            await self._reclaim_shared_cache_under_pressure()
            async with self._lock:
                pressure_level, _ratio, pressure_ceiling, _committed = (
                    self._runtime_memory_pressure_locked()
                )
                pressure_target = (
                    int(pressure_ceiling * _AUTOMATIC_MEMORY_TARGET_RATIO)
                    if pressure_ceiling is not None
                    and pressure_level in {"soft", "hard"}
                    else None
                )
            await self._trim_idle_prefix_caches_for_budget(
                memory_target_bytes=pressure_target,
            )

            victims: list[tuple[_Runtime, str]] = []
            now = datetime.now(timezone.utc)
            async with self._lock:
                if self._closed:
                    return
                pressure_level, _ratio, pressure_ceiling, _committed = (
                    self._runtime_memory_pressure_locked()
                )
                pressure_target = (
                    int(pressure_ceiling * _AUTOMATIC_MEMORY_TARGET_RATIO)
                    if pressure_ceiling is not None
                    and pressure_level == "hard"
                    else None
                )
                ttl_victims: list[_Runtime] = []
                for instance in list(self._instances.values()):
                    if (
                        instance.idle_ttl_seconds is None
                        or instance.pinned
                        or instance.state != RuntimeInstanceState.READY
                        or instance.active_requests != 0
                        or instance.queued_requests != 0
                        or instance.control_leases != 0
                        or instance.last_used_at is None
                    ):
                        continue
                    idle_seconds = (now - instance.last_used_at).total_seconds()
                    if idle_seconds < instance.idle_ttl_seconds:
                        continue
                    self._mark_instance_unloading_locked(instance)
                    ttl_victims.append(instance)
                    victims.append((instance, "idle_ttl"))
                victims.extend(
                    (instance, "host_memory_pressure" if time.monotonic() < self._host_vm_pressure_until else "memory_budget")
                    for instance in self._claim_over_budget_instances_for_unload_locked(
                        memory_ceiling=pressure_target,
                        pending_releases=ttl_victims,
                    )
                )
            async def retire_victim(
                instance: _Runtime,
                reason: str,
            ) -> None:
                try:
                    await self._retire_instance(instance)
                    if self.store is not None:
                        message = (
                            f"runtime unloaded after {instance.idle_ttl_seconds}s "
                            "of inactivity"
                            if reason == "idle_ttl"
                            else "idle LRU runtime unloaded after host compression or paging pressure"
                            if reason == "host_memory_pressure"
                            else "runtime unloaded to enforce the aggregate memory budget"
                        )
                        await asyncio.to_thread(
                            self.store.append_runtime_log,
                            RuntimeLogLevel.INFO,
                            message,
                            instance_id=instance.id,
                            fields={"source": "runtime.lifecycle", "reason": reason,
                                "model": instance.artifact.resource.name},
                        )
                except Exception as error:
                    if self.store is not None:
                        await asyncio.to_thread(
                            self.store.append_runtime_log,
                            RuntimeLogLevel.ERROR,
                            f"runtime unload failed: {error}",
                            instance_id=instance.id,
                            fields={"source": "runtime.lifecycle", "reason": reason,
                                "model": instance.artifact.resource.name},
                        )

            if victims:
                await asyncio.gather(
                    *(
                        retire_victim(instance, reason)
                        for instance, reason in victims
                    )
                )

    async def _retire_instance(self, instance: _Runtime) -> None:
        """Stop one registered runtime once and detach it only after success."""

        async with self._lock:
            if self._instances.get(instance.id) is not instance:
                return
            self._mark_instance_unloading_locked(instance)
            task = instance.retirement_task
            if task is None or task.done():
                task = asyncio.create_task(
                    self._stop_and_detach_instance(instance),
                    name=f"mfq-server-runtime-retire-{instance.id}",
                )
                instance.retirement_task = task
                task.add_done_callback(self._retirement_task_done)
        await asyncio.shield(task)

    @staticmethod
    def _retirement_task_done(task: asyncio.Task[None]) -> None:
        with suppress(asyncio.CancelledError):
            task.exception()

    async def _stop_and_detach_instance(self, instance: _Runtime) -> None:
        try:
            await self._stop_process(instance)
        except BaseException as error:
            async with self._lock:
                if self._instances.get(instance.id) is instance:
                    instance.retirement_failures += 1
                    instance.error = ErrorDetail(
                        code="runtime_unload_failed",
                        message=str(error) or type(error).__name__,
                        retryable=True,
                    )
                    self._schedule_retirement_retry_locked(instance)
            raise
        async with self._lock:
            if self._instances.get(instance.id) is instance:
                self._detach_instance_locked(instance)

    def _schedule_retirement_retry_locked(self, instance: _Runtime) -> None:
        if self._closed:
            return
        retry = instance.retirement_retry_task
        if retry is not None and not retry.done():
            return
        retry = asyncio.create_task(
            self._retry_retirement(instance),
            name=f"mfq-server-runtime-retire-retry-{instance.id}",
        )
        instance.retirement_retry_task = retry
        self._background_tasks.add(retry)
        retry.add_done_callback(self._background_task_done)

    async def _retry_retirement(self, instance: _Runtime) -> None:
        """Retry a failed teardown until it succeeds or the pool closes."""

        while True:
            failures = max(1, instance.retirement_failures)
            delay = min(30.0, 0.25 * (2 ** min(failures - 1, 7)))
            await asyncio.sleep(delay)
            async with self._lock:
                if (
                    self._closed
                    or self._instances.get(instance.id) is not instance
                    or instance.state != RuntimeInstanceState.UNLOADING
                ):
                    return
            try:
                await self._retire_instance(instance)
            except Exception:
                continue
            return

    @staticmethod
    def _supports_voice_output(architecture: str) -> bool:
        return "minicpmo" in architecture.casefold()

    @staticmethod
    async def _log_lines(stream: asyncio.StreamReader) -> AsyncIterator[bytes]:
        prefix = bytearray()
        limit = _RUNTIME_LOG_MAX_CHARS * 4
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                if prefix:
                    yield bytes(prefix)
                return
            fragments = chunk.split(b"\n")
            for fragment in fragments[:-1]:
                prefix.extend(fragment[:max(0, limit - len(prefix))])
                yield bytes(prefix)
                prefix.clear()
            prefix.extend(fragments[-1][:max(0, limit - len(prefix))])

    async def _pump_output(self, instance: _Runtime, context: JobContext) -> None:
        process = instance.process
        if isinstance(process, subprocess.Popen):
            return
        stream = process.stderr if self.transport == "stdio" else process.stdout
        if stream is None:
            return
        async for line in self._log_lines(stream):
            job_context = instance.context_job if isinstance(instance.context_job, JobContext) else context
            message = line.decode("utf-8", errors="replace").rstrip()
            normalized = message.casefold()
            for code in ("kv_memory_insufficient", "kv_vram_insufficient", "kv_dynamic_unavailable", "kv_budget_below_indexer", "kv_ssd_required", "kv_ssd_insufficient"):
                if code + ":" in message:
                    instance.error = ErrorDetail(code=code, message=message, retryable=False)
                    break
            if message.startswith("mfq-decode-metal:"):
                level = RuntimeLogLevel.ERROR
            elif message.startswith("runtime_kv_warning ") or re.search(r"(?:^|:\s+)warning:", normalized):
                level = RuntimeLogLevel.WARNING
            elif "error" in normalized:
                level = RuntimeLogLevel.ERROR
            else:
                level = RuntimeLogLevel.INFO
            if message and (instance.state == RuntimeInstanceState.LOADING or isinstance(instance.context_job, JobContext)):
                match = re.fullmatch(r"mfq_load_progress completed=(\d{1,10}) total=(\d{1,10})", message)
                value = None
                data = None
                if match:
                    completed, total = map(int, match.groups())
                    if 0 < total and 0 <= completed <= total:
                        value = 0.02 + 0.90 * completed / total
                        data = {"phase": "weights", "completed": completed, "total": total}
                elif message == "mfq_load_progress stage=finalizing":
                    value = 0.94
                    data = {"phase": "finalizing"}
                else:
                    compile_match = re.fullmatch(r"mfq_load_progress stage=(compiling|warming) completed=(\d{1,10}) total=(\d{1,10})", message)
                    if compile_match:
                        phase = compile_match.group(1)
                        completed, total = map(int, compile_match.groups()[1:])
                        if 0 < total and 0 <= completed <= total:
                            value = (0.94 + 0.01 * completed / total if phase == "compiling"
                                     else 0.95 + 0.04 * completed / total)
                            data = {"phase": phase, "completed": completed, "total": total}
                if value is not None and value > instance.load_progress:
                    instance.load_progress = value
                    with suppress(JobCancelledError, InvalidJobStateError, JobNotFoundError):
                        message_label = {"compiling": "Compiling inference kernels", "warming": "Preparing first inference",
                                         "weights": "Preparing model", "finalizing": "Finalizing runtime"}[data["phase"]]
                        await job_context.progress(value, message=message_label, data=data)
            if message and (instance.state == RuntimeInstanceState.LOADING or instance.log_to_load_job or isinstance(instance.context_job, JobContext)):
                with suppress(JobNotFoundError):
                    await job_context.log(message[:_RUNTIME_LOG_MAX_CHARS], level=JobEventLevel(level.value))
            if message and self.store is not None:
                await asyncio.to_thread(
                    self.store.append_runtime_log,
                    level,
                    message[:_RUNTIME_LOG_MAX_CHARS],
                    instance_id=instance.id,
                    fields={
                        "model": instance.artifact.resource.name,
                        "source": (
                            "runtime.stderr"
                            if self.transport == "stdio"
                            else "runtime.stdout"
                        )
                    },
                )

    async def _monitor(self, instance: _Runtime) -> None:
        process = instance.process
        status = (
            await asyncio.to_thread(process.wait)
            if isinstance(process, subprocess.Popen)
            else await process.wait()
        )
        error = ErrorDetail(
            code="runtime_exited",
            message=f"runtime process exited with status {status}",
            retryable=True,
        )
        async with self._lock:
            if (
                instance.state == RuntimeInstanceState.UNLOADING
                or self._closed
                or self._instances.get(instance.id) is not instance
            ):
                return
            instance.state = RuntimeInstanceState.FAILED
            self._request_state_changed.notify_all()
            instance.error = error
            model_name = instance.artifact.resource.name
            self._load_errors[model_name] = error
            self._load_failures[model_name] = _CachedLoadFailure(
                artifact_id=instance.artifact.resource.id,
                detail=error,
                failed_at=time.monotonic(),
            )
            self._session_routes = {
                session_id: instance_id
                for session_id, instance_id in self._session_routes.items()
                if instance_id != instance.id
            }
            if self._last_instance_id == instance.id:
                self._last_instance_id = next(
                    (
                        item.id
                        for item in self._instances.values()
                        if item.state in {
                            RuntimeInstanceState.READY,
                            RuntimeInstanceState.BUSY,
                        }
                    ),
                    None,
                )
        if self.store is not None:
            await asyncio.to_thread(
                self.store.append_runtime_log,
                RuntimeLogLevel.ERROR,
                error.message,
                instance_id=instance.id,
                fields={"source": "runtime.lifecycle", "exit_status": status,
                    "model": model_name},
            )
        await instance.backend.aclose()

    async def _refresh_instance_usage(self, instance: _Runtime) -> None:
        pid = getattr(instance.process, "pid", None)
        resident_task = (
            asyncio.create_task(
                asyncio.to_thread(self._process_resident_bytes, pid)
            )
            if isinstance(pid, int) and pid > 0
            else None
        )
        runtime_status = getattr(instance.backend, "runtime_status", None)
        status: dict[str, Any] | None = None
        try:
            if callable(runtime_status):
                try:
                    value = await asyncio.wait_for(runtime_status(), timeout=1.0)
                    if isinstance(value, dict):
                        status = value
                except Exception:
                    pass
            resident = await resident_task if resident_task is not None else None
        except BaseException:
            # Cancelling the idle reaper or a load transaction must not leave
            # its concurrently scheduled process-memory probe behind.
            if resident_task is not None:
                if not resident_task.done():
                    resident_task.cancel()
                await asyncio.gather(resident_task, return_exceptions=True)
            raise
        observed_resident = self._observed_runtime_bytes(resident, status,
            separate_memory=self.backend == "cuda" and memory_topology()["memory_architecture"] == "discrete")
        kv_bytes: int | None = None
        if status is not None:
            value = status.get("prefix_cache_hot_bytes", status.get("prefix_cache_bytes"))
            if isinstance(value, (int, float)) and value >= 0:
                kv_bytes = int(value)
        async with self._lock:
            if (
                self._instances.get(instance.id) is not instance
                or instance.state
                not in {RuntimeInstanceState.LOADING, RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
            ):
                return
            if observed_resident is not None:
                instance.resident_bytes = observed_resident
            if kv_bytes is not None:
                instance.kv_bytes = kv_bytes
            if status is not None:
                live = status.get("live_requests")
                if isinstance(live, dict):
                    instance.serving_snapshot = {key: value for key, value in live.items() if isinstance(value, dict)}
                    instance.serving_sampled_at = time.monotonic()
                instance.memory = self._memory_resources(status, instance.memory, device_ids=getattr(instance, "device_ids", ()))
                instance.prefix_dynamic_budget = status.get("prefix_cache_dynamic_budget") == 1
                instance.prefix_concurrent_maintenance = status.get("prefix_cache_concurrent_maintenance") == 1
                instance.resident_dynamic_budget = status.get("resident_memory_dynamic_budget") == 1
                instance.resident_weight_dynamic_budget = status.get("resident_weight_dynamic_budget") == 1
                if "qsa_kv_offload_supported" in status:
                    instance.qsa_kv_offload_supported = self.backend in {"metal", "cuda"} and status["qsa_kv_offload_supported"] == 1
                if "kv_quantization_supported" in status:
                    instance.kv_quantization_supported = self.backend in {"cuda", "metal"} and status["kv_quantization_supported"] == 1
                if "kv_memory_supported" in status:
                    instance.kv_memory_supported = self.backend == "cuda" and status["kv_memory_supported"] == 1
                for field, metric in (("resident_budget_limit", "resident_memory_budget_bytes"),
                    ("resident_weight_budget_limit", "resident_weight_budget_bytes"),
                    ("resident_memory_used", "resident_memory_used_bytes"), ("reclaimable_weight_bytes", "resident_reclaimable_weight_bytes")):
                    value = status.get(metric)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                        setattr(instance, field, int(value))
                disk_capacity = status.get("prefix_cache_disk_max_bytes")
                if isinstance(disk_capacity, (int, float)) and not isinstance(disk_capacity, bool) and math.isfinite(disk_capacity) and disk_capacity >= 0:
                    instance.prefix_disk_capacity = int(disk_capacity)
                pending = status.get("prefix_cache_pending_writes")
                if isinstance(pending, (int, float)) and not isinstance(pending, bool) and math.isfinite(pending) and pending >= 0:
                    instance.prefix_pending_writes = int(pending)
                pressure = status.get("prefix_cache_hot_pressure_bytes")
                if isinstance(pressure, (int, float)) and not isinstance(pressure, bool) and math.isfinite(pressure) and pressure >= 0:
                    instance.prefix_pressure_bytes = int(pressure)
                ram_pressure = status.get("prefix_cache_ram_pressure_bytes")
                if isinstance(ram_pressure, (int, float)) and not isinstance(ram_pressure, bool) and math.isfinite(ram_pressure) and ram_pressure >= 0:
                    instance.prefix_ram_pressure_bytes = int(ram_pressure)
                capacity = status.get("context_capacity")
                if isinstance(capacity, int) and capacity > 0:
                    instance.context_capacity = instance.artifact.resource.context_capacity or capacity

    @staticmethod
    def _memory_resources(
        status: dict[str, Any], previous: RuntimeMemoryResources | None = None,
        *, device_ids: Sequence[str] = (),
    ) -> RuntimeMemoryResources:
        """Retain last telemetry when a busy native worker skips its try-lock.

        Do not use RSS/allocator totals as weight bytes, or cumulative I/O as
        SSD payload size. Those counters describe different resources.
        """
        values = previous.model_dump() if previous is not None else {}
        for target, source in (
            ("resident_weight_bytes", "resident_weight_bytes"),
            ("resident_buffer_bytes", "resident_buffer_bytes"),
            ("ram_expert_bytes", "ram_expert_payload_bytes"),
            ("ram_kv_bytes", "qsa_kv_ram_bytes"),
            ("ram_kv_limit_bytes", "qsa_kv_ram_limit_bytes"),
            ("prefix_cache_ram_bytes", "prefix_cache_ram_bytes"),
            ("prefix_cache_ram_limit_bytes", "prefix_cache_ram_max_bytes"),
            ("prefix_cache_ram_blocks", "prefix_cache_ram_blocks"),
            ("wired_bytes", "metal_wired_bytes"),
            ("wired_limit_bytes", "metal_wired_limit_bytes"),
            ("ssd_expert_bytes", "ssd_expert_payload_bytes"),
            ("ssd_ple_bytes", "ssd_ple_payload_bytes"),
            ("ssd_kv_bytes", "qsa_kv_ssd_bytes"),
            ("streaming_kv_resident_bytes", "qsa_kv_resident_bytes"),
            ("streaming_kv_budget_bytes", "qsa_kv_budget_bytes"),
            ("streaming_kv_pending_bytes", "qsa_kv_pending_bytes"),
            ("ssd_kv_read_bytes", "qsa_kv_ssd_read_bytes"),
            ("ssd_kv_written_bytes", "qsa_kv_ssd_written_bytes"),
            ("ssd_kv_reads", "qsa_kv_ssd_reads"),
            ("ssd_kv_hits", "qsa_kv_ram_hits"),
        ):
            value = status.get(source)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                values[target] = int(value)
        for target, source in (("ram_experts", "ram_expert_enabled"), ("ram_kv", "qsa_kv_ram_enabled"), ("ssd_experts", "ssd_expert_enabled"), ("ssd_ple", "ssd_ple_enabled"), ("ssd_kv", "qsa_kv_offload_enabled"), ("wired_available", "metal_wired_available")):
            value = status.get(source)
            if isinstance(value, (bool, int, float)) and value in (0, 1):
                values[target] = bool(value)
        if status.get("kv_memory_ram_bytes", 0) > 0:
            values["ram_kv_bytes"] = values.get("ram_kv_bytes", 0) + int(status["kv_memory_ram_bytes"])
            values["ram_kv_limit_bytes"] = values["ram_kv_bytes"]
            values["ram_kv"] = True
        if "qsa_kv_ssd_enabled" in status:
            values["ssd_kv"] = status["qsa_kv_ssd_enabled"] == 1
        if "kv_memory_ssd_enabled" in status:
            values["ssd_kv"] = status["kv_memory_ssd_enabled"] == 1
        hot = status.get("prefix_cache_hot_bytes", status.get("prefix_cache_bytes"))
        live = status.get("kv_cache_bytes")
        groups = status.get("kv_cache_contexts")
        sessions = status.get("prefix_cache_sessions")
        hot_sessions = status.get("prefix_cache_resident_sessions")
        blocks = status.get("prefix_cache_hot_blocks", status.get("prefix_cache_snapshots"))
        def measured(value: Any) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        if measured(hot):
            values["prefix_cache_bytes"] = int(hot)
        limit = status.get("prefix_cache_max_bytes")
        if measured(limit):
            values["prefix_cache_limit_bytes"] = int(limit)
        if measured(live) and measured(hot):
            values["kv_bytes"] = int(live) + int(hot)
        if measured(groups) and measured(hot):
            if hot == 0:
                values["context_count"] = int(groups)
            elif measured(hot_sessions):
                values["context_count"] = int(groups) + int(hot_sessions)
            elif "prefix_cache_hot_blocks" not in status and measured(sessions):
                # Legacy snapshots are wholly resident. Paged bindings may
                # refer to disk-only blocks, so their total is not a RAM count.
                values["context_count"] = int(groups) + int(sessions)
            else:
                values["context_count"] = None
        if measured(blocks):
            values["prefix_cache_blocks"] = int(blocks)
        indices = sorted({int(match[1]) for key in status if (match := re.fullmatch(r"gpu_(\d+)_resident_bytes", key))})
        if indices:
            devices = []
            for index in indices:
                fields = {field: int(status[f"gpu_{index}_{field}"])
                    for field in ("resident_bytes", "resident_weight_bytes", "resident_buffer_bytes", "kv_bytes",
                        "prefix_cache_bytes", "memory_total_bytes", "memory_available_bytes")
                    if measured(status.get(f"gpu_{index}_{field}"))}
                devices.append({"device_id": device_ids[index] if index < len(device_ids) else str(index), **fields})
            values["devices"] = devices
        return RuntimeMemoryResources(**values)

    async def _stop_process(self, instance: _Runtime) -> None:
        instance.state = RuntimeInstanceState.UNLOADING
        instance.realtime_gateway = None
        process = instance.process
        first_error: Exception | None = None
        if (instance.memory and instance.memory.prefix_cache_ram_limit_bytes is not None
                and getattr(instance, "prefix_dynamic_budget", False)):
            # Finish RAM -> SSD demotion while the native worker can still
            # process controls; Windows terminate does not run destructors.
            with suppress(Exception):
                await asyncio.wait_for(instance.backend.set_prefix_cache_budget(
                    0, ram_target_bytes=0), timeout=30.0)
        try:
            if isinstance(process, subprocess.Popen):
                await asyncio.to_thread(self._stop_popen, process)
            elif process.returncode is None:
                stopped = False
                process_error: Exception | None = None
                try:
                    process.terminate()
                except ProcessLookupError:
                    stopped = True
                except Exception as error:
                    process_error = error
                if not stopped and process_error is None:
                    try:
                        await asyncio.wait_for(process.wait(), timeout=10.0)
                        stopped = True
                    except asyncio.TimeoutError:
                        pass
                    except ProcessLookupError:
                        stopped = True
                    except Exception as error:
                        process_error = error
                if not stopped:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        stopped = True
                    except Exception as error:
                        if process_error is None:
                            process_error = error
                    else:
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                            stopped = True
                        except ProcessLookupError:
                            stopped = True
                        except Exception as error:
                            if process_error is None:
                                process_error = error
                if stopped:
                    process_error = None
                if process_error is not None:
                    raise process_error
        except ProcessLookupError:
            pass
        except Exception as error:
            first_error = error
        current = asyncio.current_task()
        output = instance.output_task
        if output is not None and output is not current and not output.done() and process.returncode is not None:
            try:
                await asyncio.wait_for(output, timeout=_RUNTIME_OUTPUT_DRAIN_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                if self.store is not None:
                    with suppress(Exception):
                        await asyncio.to_thread(self.store.append_runtime_log, RuntimeLogLevel.WARNING,
                            "Runtime log drain timed out; output may be incomplete", instance_id=instance.id,
                            fields={"source": "runtime.log-drain", "model": instance.artifact.resource.name})
            except Exception as error:
                if first_error is None:
                    first_error = error
        for task in (instance.monitor_task, instance.output_task):
            if task is None or task is current or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as error:
                if first_error is None:
                    first_error = error
        try:
            await instance.backend.aclose()
        except Exception as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error

    @staticmethod
    def _stop_popen(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        first_error: Exception | None = None
        try:
            process.terminate()
        except ProcessLookupError:
            return
        except Exception as error:
            first_error = error
        if first_error is None:
            try:
                process.wait(timeout=10.0)
                return
            except subprocess.TimeoutExpired:
                pass
            except ProcessLookupError:
                return
            except Exception as error:
                first_error = error
        try:
            process.kill()
        except ProcessLookupError:
            return
        except Exception as error:
            if first_error is None:
                first_error = error
        else:
            try:
                process.wait(timeout=5.0)
                return
            except ProcessLookupError:
                return
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _reserve_free_port_locked(self) -> int:
        for _ in range(128):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = int(sock.getsockname()[1])
            if port not in self._reserved_ports:
                self._reserved_ports.add(port)
                return port
        raise RuntimeManagementError("unable to reserve a unique runtime port")

    def _cuda_expert_host_budget(self, artifact: DiscoveredModel, request: ModelLoadRequest) -> int:
        return self._cuda_expert_host_budget_plan(artifact, request).bytes

    @staticmethod
    def _cuda_expert_preload_bytes(artifact: DiscoveredModel) -> int:
        """Host bytes held by the complete preload of the routed expert set."""
        # Preload keeps a host image of every routed expert, including the ones
        # uploaded to the GPU cache, so the requirement is not the cold subset.
        # The native guard is moe_preload_fits_host() on the same definition.
        return max(0, artifact.routed_expert_bytes)

    def _cuda_expert_host_budget_plan(
        self, artifact: DiscoveredModel, request: ModelLoadRequest
    ) -> _ExpertHostBudget:
        """Price the resident expert tier against physical RAM and commit."""
        total, available = host_memory_capacity()
        budget = max(0, min(available or 0, max(0, (total or 0) - (4 << 30))) - (2 << 30))
        # KV has priority over the evictable expert tier. Price the target KV
        # before filling RAM, so SSD experts do not consume its reservation.
        try:
            plan = checkpoint_kv_budget(artifact,
                self.resolve_context_size(artifact, request.context_size, yarn_enabled=request.yarn_enabled),
                request.kv_memory, request.kv_quantization)
        except ValueError:
            # Missing/unsupported KV metadata is validated by its own path.
            plan = None
        if plan is not None:
            reserve = plan.ram_required_bytes
            if not request.kv_memory.ram_preallocated:
                reserve = min(reserve, request.kv_memory.ram_budget_bytes or (4 << 30))
            budget = max(0, budget - reserve)
        architecture = memory_topology()["memory_architecture"]
        if architecture == "unified":
            budget = max(0, budget - self._estimated_load_bytes(artifact, request))
        physical_budget = budget
        commit_bytes = commit_available = footprint = None
        if architecture == "discrete":
            commit_limit, commit_available = host_commit_capacity()
            if commit_available is not None:
                # Windows charges the worker's VRAM allocations to the same
                # commit limit as its host weights. Experts that do not fit
                # fall back to the existing SSD streaming path.
                footprint = self._estimated_load_bytes(artifact, request)
                if plan is not None:
                    footprint += plan.gpu_budget_bytes
                footprint += _WINDOWS_GPU_WORKSPACE_BYTES
                # Workspaces, transfer staging and KV growth fill VRAM up to the
                # worker's limit, and WDDM commits whatever it allocates
                # (measured: 96 GB used on a 96 GB card with a 40 GB expert cache).
                device_limit = self._cuda_device_vram_limit(request)
                if device_limit is not None:
                    footprint = max(footprint, device_limit)
                commit_bytes = max(
                    0, commit_available - footprint - max(
                        _WINDOWS_COMMIT_RESERVE_BYTES,
                        (commit_limit or 0) // _WINDOWS_COMMIT_RESERVE_FRACTION,
                    )
                )
                budget = min(budget, commit_bytes)
        configured = self.runtime_environment.get("MFQ_MOE_HOST_CACHE_BYTES", os.environ.get("MFQ_MOE_HOST_CACHE_BYTES"))
        if configured is not None:
            try:
                value = int(configured)
            except ValueError as error:
                raise _job_error("runtime_ram_expert_budget_invalid", "expert RAM cache budget must be a non-negative byte count") from error
            if value < 0:
                raise _job_error("runtime_ram_expert_budget_invalid", "expert RAM cache budget must be a non-negative byte count")
            budget = min(budget, value)
        return _ExpertHostBudget(
            budget, physical_budget, commit_bytes, commit_available, footprint
        )

    def _cuda_expert_cache_host_plan(
        self, artifact: DiscoveredModel, request: ModelLoadRequest, environment: Mapping[str, str]
    ) -> _CudaExpertCacheHostPlan:
        """Price the host side of a CUDA expert cache; residency and launch share this verdict."""
        budget = self._cuda_expert_host_budget_plan(artifact, request)
        # Preload needs room for the complete routed set, not only for the
        # experts missing from the GPU cache; otherwise Windows commit (and
        # physical RAM on Linux) is exhausted after load.
        complete = budget.bytes >= self._cuda_expert_preload_bytes(artifact)
        allow_ssd = environment.get("MFQ_MOE_SSD_EXPERTS", "1") != "0"
        if supports_cuda_ram_expert_pipeline(artifact.path):
            streaming = not complete or environment.get("MFQ_MOE_PRELOAD_ALL") == "0"
            failure = None if allow_ssd or not streaming else (
                "remaining experts do not fit in RAM; enable SSD expert streaming or reduce the model size")
            return _CudaExpertCacheHostPlan(budget, True, streaming, allow_ssd, failure)
        failure = None if complete else (
            "this expert container requires complete RAM residency; use a canonical MFQ checkpoint for SSD streaming")
        return _CudaExpertCacheHostPlan(budget, False, False, allow_ssd, failure)

    def _kv_ram_limit(self, artifact: DiscoveredModel, request: ModelLoadRequest) -> int | None:
        _, available = host_memory_capacity()
        limit = request.kv_memory.ram_budget_bytes
        if available is not None:
            resident_weights = max(0, artifact.resource.total_bytes - artifact.always_streamed_bytes)
            host_experts = max(0, resident_weights - self._estimated_load_bytes(artifact, request))
            if (self.backend == "cuda" and (request.moe_gpu_cache_gb or 0) > 0
                    and supports_cuda_ram_expert_pipeline(artifact.path)):
                budget = self._cuda_expert_host_budget(artifact, request)
                host_experts = min(host_experts, budget)
                if budget >= self._cuda_expert_preload_bytes(artifact):
                    # Preload keeps the complete routed set in host memory,
                    # including experts mirrored in the GPU cache.
                    host_experts = self._cuda_expert_preload_bytes(artifact)
            available = max(0, available - host_experts - (2 << 30))
            limit = available if limit is None else min(limit, available)
        return limit

    def _validate_kv_memory_budget(self, artifact: DiscoveredModel, request: ModelLoadRequest):
        try:
            plan = checkpoint_kv_budget(artifact,
                self.resolve_context_size(artifact, request.context_size, yarn_enabled=request.yarn_enabled),
                request.kv_memory, request.kv_quantization)
            if request.kv_memory.placement == "auto_ssd":
                # Dynamic RAM may need to release every cold block. Verify that
                # SSD can hold the complete L1 overflow before loading weights.
                ram = self._kv_ram_limit(artifact, request)
                required = plan.ram_required_bytes if not request.kv_memory.ram_preallocated or ram is None else max(0, plan.ram_required_bytes - ram)
                check_ssd_capacity(self._prefix_cache_directory(), required + min(64 << 20, plan.raw_bytes))
            elif request.kv_memory.ram_budget_bytes is not None and request.kv_memory.ram_budget_bytes < plan.ram_required_bytes:
                raise ValueError("kv_memory_insufficient: RAM KV budget cannot hold the target cold KV; enable SSD KV or reduce context length")
            return plan
        except ValueError as error:
            code = str(error).split(":", 1)[0]
            raise _job_error(code if code.startswith("kv_") else "kv_memory_unsupported", str(error)) from error

    def _launch_configuration(
        self,
        artifact: DiscoveredModel,
        request: ModelLoadRequest,
        *,
        port: int,
    ) -> tuple[list[str], dict[str, str]]:
        request_capacity = native_request_capacity(
            backend=self.backend,
            routed_expert_bytes=artifact.routed_expert_bytes,
            requested=self.resolve_concurrency(request).max_requests,
        )
        command = [
            str(self.executable),
            "--model",
            str(artifact.path),
            "--transport",
            self.transport,
        ]
        if self.transport == "http":
            command.extend(["--host", "127.0.0.1", "--port", str(port)])
        command.extend([
            "--ctx-size",
            str(self.resolve_context_size(artifact, request.context_size, yarn_enabled=request.yarn_enabled)),
            "--model-name",
            artifact.resource.name,
        ])
        append_native_prefill_chunk_override(command, request.prefill_chunk_size)
        yarn_enabled = self.context_policy.model_yarn_enabled.get(artifact.resource.name, False) if request.yarn_enabled is None else request.yarn_enabled
        if (self.backend in {"cuda", "metal"} and yarn_enabled
            and artifact.resource.yarn_context_capacity is not None):
            command.append("--yarn-context-extension")
        if self.backend in {"cuda", "hip", "metal"} and request_capacity > 1:
            command.extend(
                [
                    "--continuous-batching",
                    str(request_capacity),
                ]
            )
        command.extend(native_tokenizer_arguments(artifact.path, self.backend))
        if request.moe_gpu_cache_gb is not None:
            command.extend(["--moe-gpu-cache-gb", str(request.moe_gpu_cache_gb)])

        process_environment = native_runtime_environment(
            self.executable, self.backend, model=artifact.path
        )
        process_environment.update(self.runtime_environment)
        from mfq.server.runtime.disk_budget import DiskBudget
        process_environment['MFQ_DISK_BUDGET_DIR'] = str(DiskBudget().root)
        if self.backend == "cuda":
            process_environment.setdefault("MFQ_MEMORY_ARCHITECTURE", memory_topology()["memory_architecture"])
        if (self.backend == "cuda" and artifact.resource.format == "mfq"
                and request.moe_gpu_cache_gb is not None and request.moe_gpu_cache_gb > 0):
            plan = self._cuda_expert_cache_host_plan(artifact, request, process_environment)
            budget = plan.budget
            host_budget = budget.bytes
            if budget.commit_limited:
                logger.info(
                    "cuda_expert_host_budget limited_by=commit budget_bytes=%d "
                    "physical_budget_bytes=%d commit_budget_bytes=%d "
                    "commit_available_bytes=%d device_footprint_bytes=%d",
                    host_budget, budget.physical_bytes, budget.commit_bytes,
                    budget.commit_available_bytes, budget.commit_footprint_bytes,
                )
            if plan.failure is not None:
                raise _job_error("runtime_ram_expert_budget_exceeded", plan.failure)
            if plan.pipeline:
                defaults = {
                    "MFQ_MOE_HOST_CACHE_BYTES": str(host_budget),
                    "MFQ_MOE_PIPELINE": "1",
                    "MFQ_MOE_TWO_STAGE_FFN": "1",
                    "MFQ_MOE_PRELOAD_ALL": "1",
                    "MFQ_MOE_ASSERT_RESIDENT": "1",
                    "MFQ_MOE_RAM_PCIE": "1",
                    "MFQ_MOE_RAM_PCIE_FRACTION": "1",
                    "MFQ_MOE_DECODE_HOST_PIN": "1",
                    "MFQ_MOE_TRANSFER_CACHE": "1",
                    "MFQ_MOE_MAPPED_COPY": "0",
                    "MFQ_MOE_PREFILL_LAYER": "1",
                    "MFQ_MOE_PREFILL_LAYER_ASYNC": "1",
                    "MFQ_MOE_RESIDENCY_INTERVAL": "4",
                }
                device_limit = self._cuda_device_vram_limit(request)
                if device_limit is not None:
                    # WDDM can page overcommitted allocations into host RAM
                    # without reporting OOM. The native limit also trims idle
                    # pool pages before a new allocation reaches that boundary.
                    defaults["MFQ_CUDA_VRAM_LIMIT_GIB"] = str(device_limit / float(1 << 30))
                for name, value in defaults.items():
                    process_environment.setdefault(name, value)
                process_environment["MFQ_MOE_HOST_CACHE_BYTES"] = str(host_budget)
                process_environment["MFQ_MOE_SSD_EXPERTS"] = "1" if plan.allow_ssd else "0"
                if plan.streaming:
                    process_environment["MFQ_MOE_PRELOAD_ALL"] = "0"
                    process_environment["MFQ_MOE_ASSERT_RESIDENT"] = "0"
                    process_environment["MFQ_MOE_NVQ_DENSE"] = "0"
                    # Bulk expert images bypass the OS cache; sparse PLE rows
                    # retain batched mapped reads. Native LRUs own copied data.
                    if process_environment.setdefault("MFQ_MODEL_FILE_IO", "direct") == "direct":
                        process_environment.setdefault("MFQ_PLE_FILE_MAP", "1")
            else:
                # Older expert containers use the resident CPU loader, followed
                # by the same bounded GPU cache; they must not read SSD per step.
                process_environment.setdefault("MFQ_DISABLE_MOE_SSD_RANGES", "1")
        process_environment["MFQ_KV_TURBOQUANT_BITS"] = str(request.kv_quantization.bits if request.kv_quantization.enabled else 0)
        if self.backend == "cuda" and request.context_management is not None:
            defaults = request.context_management
            process_environment["MFQ_KV_VRAM_PREALLOCATED"] = "1" if defaults.automatic or defaults.vram_preallocated else "0"
        if self.backend == "cuda":
            process_environment["MFQ_KV_PLACEMENT"] = request.kv_memory.placement
            process_environment["MFQ_KV_ALLOCATION"] = request.kv_memory.allocation
            process_environment["MFQ_KV_RAM_PREALLOCATED"] = "1" if request.kv_memory.ram_preallocated else "0"
            process_environment.pop("MFQ_KV_GPU_BUDGET_BYTES", None)
            if request.kv_memory.gpu_budget_bytes is not None:
                process_environment["MFQ_KV_GPU_BUDGET_BYTES"] = str(request.kv_memory.gpu_budget_bytes)
            ram_limit = self._kv_ram_limit(artifact, request)
            if ram_limit is not None:
                process_environment["MFQ_KV_RAM_LIMIT_BYTES"] = str(ram_limit)
        process_environment.pop("MFQ_QSA_KV_BUDGET_BYTES", None)
        process_environment.pop("MFQ_QSA_KV_TARGET_CONTEXT", None)
        process_environment.pop("MFQ_QSA_KV_RAM_BUDGET_BYTES", None)
        if request.qsa_kv_offload.enabled:
            if self.backend not in {"metal", "cuda"}:
                raise _job_error("qsa_kv_offload_unsupported", "KV offload requires a Metal or CUDA QSA backend")
            qsa_index_requirement(artifact, self.resolve_context_size(artifact, request.context_size, yarn_enabled=request.yarn_enabled),
                request.qsa_kv_offload.budget_bytes, maximum_context=self._context_limit(artifact, request.yarn_enabled), cuda=self.backend == "cuda")
            process_environment["MFQ_QSA_KV_BUDGET_BYTES"] = str(request.qsa_kv_offload.budget_bytes)
            if self.backend == "cuda":
                ram_budget = request.qsa_kv_offload.ram_budget_bytes
                if memory_topology()["memory_architecture"] == "unified":
                    ram_budget = 0
                elif ram_budget is None:
                    _, available = host_memory_capacity()
                    ram_budget = min(4 << 30, max(0, available // 4)) if available is not None else 0
                process_environment["MFQ_QSA_KV_RAM_BUDGET_BYTES"] = str(ram_budget)
        if self.backend in {"cuda", "metal"} and artifact.resource.name in self._resident_load_limits:
            process_environment["MFQ_SERVER_RESIDENT_BUDGET_BYTES"] = str(self._resident_load_limits[artifact.resource.name])
        cache_directory = str(self._prefix_cache_directory())
        process_environment["MFQ_RUNTIME_PREFIX_CACHE_DIR"] = cache_directory
        process_environment["MFQ_SERVER_PREFIX_CACHE_DIR"] = cache_directory
        disk_limit = self.memory_policy.prefix_disk_limit_bytes
        if disk_limit is None:
            disk_limit = self._prefix_disk_status.get("prefix_cache_total_disk_max_bytes", 100 << 30)
        cache_environment = {
            "MFQ_RUNTIME_MAX_KV_SESSIONS": request.prefix_cache_max_sessions,
            "MFQ_RUNTIME_MAX_KV_SNAPSHOTS_PER_SESSION": (
                request.prefix_cache_max_snapshots_per_session
            ),
            "MFQ_RUNTIME_KV_SESSION_BYTES": request.prefix_cache_max_bytes,
            "MFQ_RUNTIME_DISABLE_PREFIX_CACHE": (
                None if request.prefix_cache_enabled else 1
            ),
            "MFQ_RUNTIME_PREFIX_CACHE_DISK_BYTES": min(disk_limit, request.prefix_cache_disk_bytes) if request.prefix_cache_disk_bytes is not None else disk_limit,
            "MFQ_RUNTIME_PREFIX_CACHE_HOT_BYTES": (
                request.prefix_cache_hot_bytes
                if request.prefix_cache_hot_bytes is not None
                else request.prefix_cache_max_bytes
            ),
            "MFQ_RUNTIME_PREFIX_CACHE_BLOCK_TOKENS": request.prefix_cache_block_tokens,
            "MFQ_RUNTIME_PREFIX_CACHE_RAM_BYTES": min(max(0, self._prefix_ram_total_limit() - sum(
                item.memory.prefix_cache_ram_limit_bytes or 0 for item in self._instances.values() if item.memory)),
                request.prefix_cache_ram_bytes if request.prefix_cache_ram_bytes is not None else self._prefix_ram_total_limit()),
            "MFQ_RUNTIME_PREFIX_CACHE_PENDING_BYTES": request.prefix_cache_pending_bytes,
        }
        for name, value in cache_environment.items():
            if value is not None:
                process_environment[name] = str(value)
                if self.backend == "metal":
                    process_environment[name.replace("MFQ_RUNTIME_", "MFQ_SERVER_", 1)] = str(value)
        if self.backend == "cuda" and request.device_ids:
            process_environment["CUDA_VISIBLE_DEVICES"] = ",".join(request.device_ids)
        return command, process_environment

    @staticmethod
    def _process_resident_bytes(pid: int) -> int | None:
        try:
            output = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(pid)],
                text=True,
                timeout=1,
            ).strip()
            return int(output) * 1024 if output else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
