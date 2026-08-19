"""Managed MFQ runtime processes and model-aware request routing."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from mfq.server.backend import BackendDelta, BackendError, ChatBackend, OpenAIChatBackend
from mfq.server.catalog import DiscoveredModel, ModelArtifactNotFoundError, ModelCatalog
from mfq.server.jobs import JobContext, JobExecutionError
from mfq.server.models import (
    ErrorDetail,
    ModelLoadRequest,
    ModelUnloadRequest,
    ResponseFormat,
    RuntimeCapabilitiesResource,
    RuntimeInstanceList,
    RuntimeInstanceResource,
    RuntimeInstanceState,
    RuntimeLogLevel,
    SamplingParams,
    ToolChoice,
    ToolDefinition,
)


class RuntimeManagementError(RuntimeError):
    pass


class RuntimeInstanceNotFoundError(RuntimeManagementError):
    pass


class RuntimeConflictError(RuntimeManagementError):
    pass


def _job_error(code: str, message: str, *, retryable: bool = False) -> JobExecutionError:
    return JobExecutionError(ErrorDetail(code=code, message=message, retryable=retryable))


@dataclass
class _ManagedRuntime:
    id: UUID
    artifact: DiscoveredModel
    process: asyncio.subprocess.Process | subprocess.Popen[bytes]
    backend: ChatBackend
    port: int
    context_size: int
    sampling_defaults: SamplingParams | None = None
    state: RuntimeInstanceState = RuntimeInstanceState.LOADING
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: datetime | None = None
    active_requests: int = 0
    queued_requests: int = 0
    request_slots: asyncio.Semaphore | None = None
    error: ErrorDetail | None = None
    output_task: asyncio.Task[None] | None = None
    monitor_task: asyncio.Task[None] | None = None


class ManagedRuntimePool:
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
        metric_interval_seconds: float = 2.0,
        backend: str = "metal",
    ) -> None:
        if max_instances < 1:
            raise ValueError("max_instances must be positive")
        if max_requests_per_instance < 1:
            raise ValueError("max_requests_per_instance must be positive")
        self.catalog = catalog
        self.executable = Path(executable).expanduser().resolve()
        self.fallback = fallback
        self.startup_timeout_seconds = startup_timeout_seconds
        self.max_instances = max_instances
        self.max_requests_per_instance = max_requests_per_instance
        self.metric_interval_seconds = max(0.25, metric_interval_seconds)
        if backend not in {"cuda", "metal"}:
            raise ValueError(f"unsupported native backend: {backend}")
        self.backend = backend
        self.store = None
        self._instances: dict[UUID, _ManagedRuntime] = {}
        self._loading_model_names: set[str] = set()
        self._session_routes: dict[UUID, UUID] = {}
        self._last_instance_id: UUID | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def start(self) -> None:
        """Start lifecycle monitors for runtimes registered before the event loop."""

        async with self._lock:
            if self._closed:
                raise RuntimeManagementError("runtime pool is closed")
            for instance in self._instances.values():
                if instance.monitor_task is None:
                    instance.monitor_task = asyncio.create_task(
                        self._monitor(instance),
                        name=f"mfq-server-runtime-monitor-{instance.id}",
                    )

    def register_started(
        self,
        *,
        artifact: DiscoveredModel,
        process: subprocess.Popen[bytes],
        backend: ChatBackend,
        port: int,
        context_size: int,
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
        instance = _ManagedRuntime(
            id=uuid4(),
            artifact=artifact,
            process=process,
            backend=backend,
            port=port,
            context_size=context_size,
            state=RuntimeInstanceState.READY,
            last_used_at=datetime.now(timezone.utc),
            request_slots=asyncio.Semaphore(self.max_requests_per_instance),
        )
        self._instances[instance.id] = instance
        self._last_instance_id = instance.id
        return instance.id

    async def load(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = ModelLoadRequest.model_validate(payload)
        if self.backend == "metal" and request.device_ids not in ([], ["metal"]):
            raise _job_error("unsupported_device", "the Metal runtime accepts device 'metal'")
        if self.backend == "cuda" and any(not value.isdecimal() for value in request.device_ids):
            raise _job_error("unsupported_device", "CUDA device IDs must be non-negative integers")
        if request.idle_ttl_seconds is not None or request.pin:
            raise _job_error(
                "unsupported_model_policy",
                "runtime pinning and idle eviction are not implemented yet",
            )
        try:
            artifact = await self.catalog.resolve(request.model, request.artifact_uri)
        except ModelArtifactNotFoundError as error:
            raise _job_error(
                "model_artifact_not_found", f"model artifact was not found: {error}"
            ) from error
        if not artifact.resource.loadable:
            raise _job_error(
                "model_artifact_incomplete",
                artifact.resource.error or "model artifact is incomplete",
            )
        async with self._lock:
            if self._closed:
                raise RuntimeManagementError("runtime pool is closed")
            model_name = artifact.resource.name
            existing = next(
                (
                    item
                    for item in self._instances.values()
                    if item.artifact.resource.name == model_name
                ),
                None,
            )
            if existing is not None or model_name in self._loading_model_names:
                raise _job_error(
                    "model_already_loaded",
                    (
                        f"model is already loaded by runtime instance {existing.id}"
                        if existing is not None
                        else f"model is already loading: {model_name}"
                    ),
                )
            active_count = sum(
                item.state != RuntimeInstanceState.FAILED for item in self._instances.values()
            ) + len(self._loading_model_names)
            if active_count >= self.max_instances:
                raise _job_error("runtime_instance_limit", "managed runtime instance limit reached")
            port = self._free_port()
            self._loading_model_names.add(model_name)

        command = [
            str(self.executable),
            "--mfq",
            str(artifact.path),
            "--server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(request.context_size),
            "--model-name",
            artifact.resource.name,
        ]
        if self.backend == "metal":
            command.extend(["--prefill-chunk-size", str(request.prefill_chunk_size)])
        if request.moe_gpu_cache_gb is not None:
            command.extend(["--moe-gpu-cache-gb", str(request.moe_gpu_cache_gb)])
        process_environment = os.environ.copy()
        cache_environment = {
            "MFQ_SERVER_MAX_KV_SESSIONS": request.prefix_cache_max_sessions,
            "MFQ_SERVER_MAX_KV_SNAPSHOTS_PER_SESSION": (
                request.prefix_cache_max_snapshots_per_session
            ),
            "MFQ_SERVER_KV_SESSION_BYTES": request.prefix_cache_max_bytes,
        }
        for name, value in cache_environment.items():
            if value is not None:
                process_environment[name] = str(value)
        if self.backend == "cuda" and request.device_ids:
            process_environment["CUDA_VISIBLE_DEVICES"] = ",".join(request.device_ids)

        try:
            await context.progress(0.02, message="Starting runtime process")
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                env=process_environment,
            )
        except BaseException:
            async with self._lock:
                self._loading_model_names.discard(model_name)
            raise
        configured_video_library = os.environ.get("MFQ_AVFOUNDATION_VIDEO_LIBRARY")
        avfoundation_video_library = (
            Path(configured_video_library).expanduser().resolve()
            if configured_video_library
            else self.executable.with_name("libmfq_avfoundation_video.dylib")
        )
        backend = OpenAIChatBackend(
            f"http://127.0.0.1:{port}",
            local_tensor_files=True,
            avfoundation_video_library=(
                avfoundation_video_library
                if avfoundation_video_library.is_file()
                else None
            ),
        )
        instance = _ManagedRuntime(
            id=uuid4(),
            artifact=artifact,
            process=process,
            backend=backend,
            port=port,
            context_size=request.context_size,
            sampling_defaults=request.sampling_defaults,
            request_slots=asyncio.Semaphore(self.max_requests_per_instance),
        )
        try:
            async with self._lock:
                self._loading_model_names.discard(model_name)
                self._instances[instance.id] = instance
        except BaseException:
            async with self._lock:
                self._loading_model_names.discard(model_name)
            await self._stop_process(instance)
            raise

        keep_process = False

        async def cleanup_failed_start() -> None:
            if keep_process:
                return
            await self._stop_process(instance)

        context.add_cleanup(cleanup_failed_start)
        instance.output_task = asyncio.create_task(
            self._pump_output(instance, context),
            name=f"mfq-server-runtime-output-{instance.id}",
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout_seconds
        next_progress = 0.05
        while loop.time() < deadline:
            context.raise_if_cancelled()
            status = process.returncode
            if status is not None:
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
            except (BackendError, TimeoutError):
                await asyncio.sleep(0.25)
                if next_progress < 0.9:
                    next_progress = min(0.9, next_progress + 0.005)
                    await context.progress(next_progress, message="Loading model")
                continue
            if capabilities.model != artifact.resource.name:
                raise _job_error(
                    "runtime_identity_mismatch",
                    "runtime health returned an unexpected model identity",
                )
            break
        else:
            raise _job_error(
                "runtime_start_timeout",
                "runtime did not become ready before the startup timeout",
                retryable=True,
            )

        instance.state = RuntimeInstanceState.READY
        instance.last_used_at = datetime.now(timezone.utc)
        keep_process = True
        async with self._lock:
            self._last_instance_id = instance.id
        instance.monitor_task = asyncio.create_task(
            self._monitor(instance), name=f"mfq-server-runtime-monitor-{instance.id}"
        )
        await context.progress(1.0, message="Model ready")
        return {
            "instance_id": str(instance.id),
            "model_id": artifact.resource.name,
            "model": artifact.resource.name,
            "artifact_id": artifact.resource.id,
            "context_size": request.context_size,
        }

    async def unload(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = ModelUnloadRequest.model_validate(payload)
        async with self._lock:
            instance = self._instances.get(request.instance_id)
        if instance is None:
            raise _job_error(
                "runtime_instance_not_found",
                f"runtime instance was not found: {request.instance_id}",
            )
        if instance.active_requests and not request.force:
            raise _job_error("runtime_busy", "runtime has active requests", retryable=True)
        instance.state = RuntimeInstanceState.UNLOADING
        await context.progress(0.2, message="Stopping runtime")
        await self._stop_process(instance)
        await context.progress(0.9, message="Releasing runtime")
        async with self._lock:
            self._instances.pop(instance.id, None)
            self._session_routes = {
                session_id: instance_id
                for session_id, instance_id in self._session_routes.items()
                if instance_id != instance.id
            }
            if self._last_instance_id == instance.id:
                self._last_instance_id = None
        await context.progress(1.0, message="Model unloaded")
        return {"instance_id": str(instance.id), "unloaded": True}

    async def instances(self) -> RuntimeInstanceList:
        async with self._lock:
            values = list(self._instances.values())
            route_counts: dict[UUID, int] = {}
            for instance_id in self._session_routes.values():
                route_counts[instance_id] = route_counts.get(instance_id, 0) + 1
        return RuntimeInstanceList(
            data=[
                RuntimeInstanceResource(
                    id=item.id,
                    model=item.artifact.resource.name,
                    state=item.state,
                    devices=[self.backend],
                    active_sessions=route_counts.get(item.id, 0),
                    queued_requests=item.queued_requests,
                    resident_bytes=None,
                    kv_bytes=None,
                    context_size=item.context_size,
                    started_at=item.started_at,
                    last_used_at=item.last_used_at,
                    error=item.error,
                )
                for item in values
            ]
        )

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
    ) -> AsyncIterator[BackendDelta]:
        instance = await self._select(model, session_id=session_id)
        if instance is None:
            if self.fallback is None:
                raise BackendError("model_not_loaded", f"model is not loaded: {model}")
            async for delta in self.fallback.stream(
                model=model,
                messages=messages,
                sampling=sampling,
                session_id=session_id,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            ):
                yield delta
            return
        if instance.state not in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}:
            raise BackendError("model_not_ready", f"model runtime is {instance.state.value}")
        assert instance.request_slots is not None
        acquired = False
        instance.queued_requests += 1
        try:
            await instance.request_slots.acquire()
            acquired = True
        finally:
            instance.queued_requests = max(0, instance.queued_requests - 1)
        instance.active_requests += 1
        instance.state = RuntimeInstanceState.BUSY
        instance.last_used_at = datetime.now(timezone.utc)
        if session_id is not None:
            async with self._lock:
                self._session_routes[session_id] = instance.id
                self._last_instance_id = instance.id
        try:
            async for delta in instance.backend.stream(
                model=instance.artifact.resource.name,
                messages=messages,
                sampling=sampling,
                session_id=session_id,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            ):
                yield delta
        finally:
            instance.active_requests = max(0, instance.active_requests - 1)
            if acquired:
                instance.request_slots.release()
            if instance.state == RuntimeInstanceState.BUSY and instance.active_requests == 0:
                instance.state = RuntimeInstanceState.READY
            instance.last_used_at = datetime.now(timezone.utc)

    async def fork_session(self, source_session_id: UUID, target_session_id: UUID) -> bool:
        async with self._lock:
            instance_id = self._session_routes.get(source_session_id)
            instance = self._instances.get(instance_id) if instance_id is not None else None
        if instance is None:
            return (
                await self.fallback.fork_session(source_session_id, target_session_id)
                if self.fallback
                else False
            )
        succeeded = await instance.backend.fork_session(source_session_id, target_session_id)
        if succeeded:
            async with self._lock:
                self._session_routes[target_session_id] = instance.id
        return succeeded

    async def close_session(self, session_id: UUID) -> bool:
        async with self._lock:
            instance_id = self._session_routes.pop(session_id, None)
            instance = self._instances.get(instance_id) if instance_id is not None else None
        if instance is None:
            return await self.fallback.close_session(session_id) if self.fallback else False
        return await instance.backend.close_session(session_id)

    async def capabilities(self) -> RuntimeCapabilitiesResource:
        backend = await self._current_backend()
        if backend is None:
            raise BackendError("model_not_loaded", "no runtime is available")
        return await backend.capabilities()

    async def runtime_status(self) -> dict[str, Any]:
        backend = await self._current_backend()
        if backend is None:
            return {
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
        async with self._lock:
            instance = (
                self._instances.get(self._last_instance_id)
                if self._last_instance_id is not None
                else None
            )
        if instance is not None:
            status["instance_id"] = str(instance.id)
            status["runtime_state"] = instance.state.value
            if instance.sampling_defaults is not None:
                status["sampling_defaults"] = instance.sampling_defaults.model_dump(mode="json")
            status["process_resident_bytes"] = await asyncio.to_thread(
                self._process_resident_bytes, instance.process.pid
            )
        return status

    async def runtime_models(self) -> dict[str, Any]:
        async with self._lock:
            managed = list(self._instances.values())
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
        backend = await self._current_backend()
        if backend is None:
            return {"available": False, "modes": []}
        return await backend.realtime_capabilities()

    async def reload_runtime(self, context_size: int) -> dict[str, Any]:
        backend = await self._current_backend()
        if backend is None:
            raise BackendError("model_not_loaded", "no runtime is available")
        result = await backend.reload_runtime(context_size)
        async with self._lock:
            if self._last_instance_id in self._instances:
                self._instances[self._last_instance_id].context_size = context_size
        return result

    async def clear_runtime_cache(self) -> dict[str, Any]:
        backend = await self._current_backend()
        if backend is None:
            raise BackendError("model_not_loaded", "no runtime is available")
        return await backend.clear_runtime_cache()

    def realtime_connect(self, *, mode: str = "audio") -> Any:
        instance = self._instances.get(self._last_instance_id) if self._last_instance_id else None
        if instance is not None:
            return instance.backend.realtime_connect(mode=mode)
        if self.fallback is not None:
            return self.fallback.realtime_connect(mode=mode)
        raise BackendError("model_not_loaded", "no runtime is available")

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            instances = list(self._instances.values())
        for instance in instances:
            await self._stop_process(instance)
        async with self._lock:
            self._instances.clear()
            self._loading_model_names.clear()
            self._session_routes.clear()
        if self.fallback is not None:
            await self.fallback.aclose()

    async def _select(self, model: str, *, session_id: UUID | None) -> _ManagedRuntime | None:
        async with self._lock:
            if session_id is not None:
                instance_id = self._session_routes.get(session_id)
                if instance_id is not None:
                    return self._instances.get(instance_id)
            matches = [
                item
                for item in self._instances.values()
                if model == item.artifact.resource.name
            ]
        if len(matches) > 1:
            raise BackendError("ambiguous_model", f"multiple loaded runtimes match {model}")
        return matches[0] if matches else None

    async def _current_backend(self) -> ChatBackend | None:
        async with self._lock:
            instance = (
                self._instances.get(self._last_instance_id)
                if self._last_instance_id is not None
                else None
            )
            if instance is None:
                instance = next(
                    (
                        item
                        for item in self._instances.values()
                        if item.state in {RuntimeInstanceState.READY, RuntimeInstanceState.BUSY}
                    ),
                    None,
                )
        return instance.backend if instance is not None else self.fallback

    async def _pump_output(self, instance: _ManagedRuntime, context: JobContext) -> None:
        process = instance.process
        if isinstance(process, subprocess.Popen):
            return
        stream = process.stdout
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            message = line.decode("utf-8", errors="replace").rstrip()
            if message and instance.state == RuntimeInstanceState.LOADING:
                await context.log(message[:4096])
            if message and self.store is not None:
                await asyncio.to_thread(
                    self.store.append_runtime_log,
                    (
                        RuntimeLogLevel.ERROR
                        if "error" in message.casefold()
                        else RuntimeLogLevel.INFO
                    ),
                    message[:4096],
                    instance_id=instance.id,
                    fields={"source": "runtime.stdout"},
                )

    async def _monitor(self, instance: _ManagedRuntime) -> None:
        process = instance.process
        status = (
            await asyncio.to_thread(process.wait)
            if isinstance(process, subprocess.Popen)
            else await process.wait()
        )
        if instance.state == RuntimeInstanceState.UNLOADING or self._closed:
            return
        instance.state = RuntimeInstanceState.FAILED
        instance.error = ErrorDetail(
            code="runtime_exited",
            message=f"runtime process exited with status {status}",
            retryable=True,
        )
        if self.store is not None:
            await asyncio.to_thread(
                self.store.append_runtime_log,
                RuntimeLogLevel.ERROR,
                instance.error.message,
                instance_id=instance.id,
                fields={"source": "runtime.lifecycle", "exit_status": status},
            )
        await instance.backend.aclose()

    async def _stop_process(self, instance: _ManagedRuntime) -> None:
        instance.state = RuntimeInstanceState.UNLOADING
        process = instance.process
        if isinstance(process, subprocess.Popen):
            await asyncio.to_thread(self._stop_popen, process)
        elif process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        current = asyncio.current_task()
        for task in (instance.monitor_task, instance.output_task):
            if task is not None and task is not current and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        await instance.backend.aclose()

    @staticmethod
    def _stop_popen(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

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
