"""Registered background job execution for MFQ Server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel

from mfq.server.models import (
    CreateJobRequest,
    ErrorDetail,
    EvaluationKind,
    EvaluationResultResource,
    JobEventLevel,
    JobEventType,
    JobKindList,
    JobKindResource,
    JobList,
    JobResource,
    JobStatus,
)
from mfq.server.storage import InvalidJobStateError, JobNotFoundError, SessionStore


class JobCancelledError(asyncio.CancelledError):
    """Raised by a cooperative handler after a cancellation request."""


class JobHandler(Protocol):
    async def __call__(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]: ...


class JobKindNotRegisteredError(LookupError):
    pass


class JobExecutionError(RuntimeError):
    """A handler failure with a stable public error payload."""

    def __init__(self, detail: ErrorDetail) -> None:
        super().__init__(detail.message)
        self.detail = detail


@dataclass(frozen=True)
class TypedJobHandler:
    callback: JobHandler
    payload_model: type[BaseModel]

    async def __call__(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.callback(context, payload)

    def payload_schema(self) -> dict[str, Any]:
        return self.payload_model.model_json_schema()


@dataclass
class JobContext:
    store: SessionStore
    job_id: UUID
    cancel_event: asyncio.Event
    _cleanup_callbacks: list[Callable[[], Awaitable[None] | None]] = field(default_factory=list)

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancel_requested:
            raise JobCancelledError()

    def add_cleanup(self, callback: Callable[[], Awaitable[None] | None]) -> None:
        self._cleanup_callbacks.append(callback)

    async def progress(
        self,
        value: float,
        *,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> JobResource:
        self.raise_if_cancelled()
        return await asyncio.to_thread(
            self.store.update_job_progress,
            self.job_id,
            value,
            message=message,
            data=data,
        )

    async def log(
        self,
        message: str,
        *,
        level: JobEventLevel = JobEventLevel.INFO,
        data: dict[str, Any] | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.store.append_job_event,
            self.job_id,
            JobEventType.LOG,
            level,
            message=message,
            data=data,
        )

    async def artifact(
        self,
        *,
        name: str,
        uri: str,
        media_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        data: dict[str, Any] = {"name": name, "uri": uri}
        if media_type is not None:
            data["media_type"] = media_type
        if metadata:
            data["metadata"] = metadata
        await asyncio.to_thread(
            self.store.append_job_event,
            self.job_id,
            JobEventType.ARTIFACT,
            JobEventLevel.INFO,
            message=name,
            data=data,
        )
        lineage_metadata = dict(metadata or {})
        source_uris = lineage_metadata.pop("source_uris", [])
        parameters = lineage_metadata.pop("parameters", None)
        await asyncio.to_thread(
            self.store.record_artifact_lineage,
            artifact_uri=uri,
            artifact_name=name,
            producer_job_id=self.job_id,
            source_uris=source_uris,
            parameters=parameters,
            metadata=lineage_metadata,
        )

    async def validate_artifact(self, uri: str) -> None:
        await asyncio.to_thread(
            self.store.record_artifact_validation,
            uri,
            self.job_id,
        )

    async def evaluation(
        self,
        *,
        kind: EvaluationKind,
        model_id: str,
        metrics: dict[str, Any],
        parameters: dict[str, Any],
        comparison_parameters: dict[str, Any],
        dataset_id: UUID | None = None,
        dataset_manifest: dict[str, Any] | None = None,
        runtime_identity: dict[str, Any] | None = None,
    ) -> EvaluationResultResource:
        canonical = json.dumps(
            comparison_parameters,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return await asyncio.to_thread(
            self.store.record_evaluation,
            job_id=self.job_id,
            kind=kind,
            model_id=model_id,
            metrics=metrics,
            parameters=parameters,
            dataset_id=dataset_id,
            dataset_manifest=dataset_manifest or {},
            hardware_identity={
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
            },
            runtime_identity=runtime_identity or {},
            comparison_key=hashlib.sha256(canonical).hexdigest(),
        )

    async def cleanup(self) -> None:
        for callback in reversed(self._cleanup_callbacks):
            with suppress(Exception):
                result = callback()
                if result is not None:
                    await result


class JobManager:
    """Run only explicitly registered job kinds with bounded concurrency."""

    def __init__(
        self,
        store: SessionStore,
        handlers: Mapping[str, JobHandler] | None = None,
        *,
        max_concurrency: int = 2,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self.store = store
        self.handlers: dict[str, JobHandler] = dict(handlers or {})
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tasks: dict[UUID, asyncio.Task[None]] = {}
        self._cancel_events: dict[UUID, asyncio.Event] = {}
        self._started = False
        self._closing = False
        self._lifecycle_lock = asyncio.Lock()

    def register(self, kind: str, handler: JobHandler) -> None:
        if self._started:
            raise RuntimeError("job handlers must be registered before the manager starts")
        if kind in self.handlers:
            raise ValueError(f"job kind is already registered: {kind}")
        self.handlers[kind] = handler

    def kinds(self) -> JobKindList:
        return JobKindList(
            data=[
                JobKindResource(kind=kind, payload_schema=self._handler_schema(handler))
                for kind, handler in sorted(self.handlers.items())
            ]
        )

    @staticmethod
    def _handler_schema(handler: JobHandler) -> dict[str, Any]:
        schema = getattr(handler, "payload_schema", None)
        if callable(schema):
            value = schema()
            if isinstance(value, dict):
                return value
        return {"type": "object", "additionalProperties": True}

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return
            self._started = True
            self._closing = False
            await asyncio.to_thread(self.store.recover_interrupted_jobs)
            queued = await asyncio.to_thread(self.store.list_queued_job_ids)
            for job_id in queued:
                self._schedule(job_id)

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if not self._started:
                return
            self._closing = True
            tasks = list(self._tasks.values())
            for task in tasks:
                task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task
            self._tasks.clear()
            self._cancel_events.clear()
            self._started = False

    async def submit(self, request: CreateJobRequest) -> JobResource:
        await self.start()
        if request.kind not in self.handlers:
            raise JobKindNotRegisteredError(request.kind)
        job = await asyncio.to_thread(self.store.create_job, request.kind, request.payload)
        self._schedule(job.id)
        return job

    async def cancel(self, job_id: UUID) -> JobResource:
        job = await asyncio.to_thread(self.store.request_job_cancel, job_id)
        event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
        return job

    async def retry(self, job_id: UUID) -> JobResource:
        previous = await asyncio.to_thread(self.store.get_job, job_id)
        if previous.status not in {
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.INTERRUPTED,
        }:
            raise InvalidJobStateError(f"cannot retry job in state {previous.status.value}")
        return await self.submit(
            CreateJobRequest(kind=previous.kind, payload=dict(previous.payload))
        )

    async def archive(self, job_id: UUID) -> None:
        await asyncio.to_thread(self.store.archive_job, job_id)

    async def archive_completed(self) -> int:
        return await asyncio.to_thread(self.store.archive_completed_jobs)

    def _schedule(self, job_id: UUID) -> None:
        if self._closing or job_id in self._tasks:
            return
        event = asyncio.Event()
        self._cancel_events[job_id] = event
        task = asyncio.create_task(self._run(job_id, event), name=f"mfq-server-job-{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._forget(job_id))

    def _forget(self, job_id: UUID) -> None:
        self._tasks.pop(job_id, None)
        self._cancel_events.pop(job_id, None)

    async def _run(self, job_id: UUID, cancel_event: asyncio.Event) -> None:
        context = JobContext(self.store, job_id, cancel_event)
        try:
            async with self._semaphore:
                job = await asyncio.to_thread(self.store.get_job, job_id)
                if job.status != JobStatus.QUEUED:
                    return
                handler = self.handlers.get(job.kind)
                if handler is None:
                    await asyncio.to_thread(
                        self.store.fail_job,
                        job_id,
                        ErrorDetail(
                            code="job_kind_unavailable",
                            message=f"job kind is not registered: {job.kind}",
                            retryable=True,
                        ),
                    )
                    return
                job = await asyncio.to_thread(self.store.claim_job, job_id)
                if job.status != JobStatus.RUNNING:
                    return
                result = await handler(context, dict(job.payload))
                context.raise_if_cancelled()
                await asyncio.to_thread(self.store.complete_job, job_id, result)
        except (JobCancelledError, asyncio.CancelledError):
            cancel_event.set()
            with suppress(InvalidJobStateError, JobNotFoundError):
                if self._closing:
                    await asyncio.to_thread(self.store.interrupt_job, job_id)
                else:
                    await asyncio.to_thread(self.store.cancel_job, job_id)
            raise
        except JobExecutionError as error:
            with suppress(InvalidJobStateError, JobNotFoundError):
                await asyncio.to_thread(self.store.fail_job, job_id, error.detail)
        except Exception as error:
            detail = ErrorDetail(
                code="job_execution_failed",
                message=str(error) or type(error).__name__,
                retryable=False,
                details={"exception_type": type(error).__name__},
            )
            with suppress(InvalidJobStateError, JobNotFoundError):
                await asyncio.to_thread(self.store.fail_job, job_id, detail)
        finally:
            await context.cleanup()

    async def list_jobs(
        self,
        *,
        status: JobStatus | None,
        kind: str | None,
        limit: int,
        offset: int,
    ) -> JobList:
        jobs = await asyncio.to_thread(
            self.store.list_jobs,
            status=status,
            kind=kind,
            limit=limit,
            offset=offset,
        )
        return JobList(data=jobs)
