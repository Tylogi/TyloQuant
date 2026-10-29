"""Strict MFQ tool jobs backed by argv-only subprocesses."""

from __future__ import annotations

import asyncio
import hashlib
import fnmatch
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from mfq.formats.header import MFQ_SUFFIXES
from mfq.formats.companions import with_engram_shards

from mfq.server.api.network import download_environment, hub_download_url
from mfq.server.protocol.models import ErrorDetail
from mfq.server.protocol.quantization import Wt2ReferencePayload
from mfq.server.services.jobs import JobContext, JobExecutionError, TypedJobHandler
from mfq.server.services.download_directory import DownloadDirectorySettings
from mfq.server.runtime.disk_budget import DiskBudget, volume
from mfq.server.services.inference_benchmark import InferenceBenchmarkPayload, WikitextQualityPayload, run_inference_benchmark
from mfq.server.services.accuracy_benchmark import AccuracyBenchmarkPayload, run_accuracy_benchmark
from mfq.server.services.evaluation_datasets import OfficialDatasetDownloadPayload, download_official_dataset, materialize_wt2, WT2_TEXT_SHA256
from mfq.server.state.catalog import ModelArtifactNotFoundError, ModelCatalog
from mfq.server.state.storage import StorageError

_TOOL_OUTPUT_TAIL_LINES = 4096
_TOOL_TERMINATE_TIMEOUT_SECONDS = 10.0
_TOOL_KILL_TIMEOUT_SECONDS = 5.0


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VoiceOutputInstallPayload(_Payload):
    pass


class ModelScopeDownloadPayload(_Payload):
    repo_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    destination: str = Field(min_length=1, max_length=1024)
    download_directory: str | None = Field(default=None, min_length=1, max_length=4096)
    revision: str = Field(default="master", min_length=1, max_length=255)
    repo_type: Literal["model", "dataset"] = "model"
    include: list[str] = Field(default_factory=list, max_length=4096)
    exclude: list[str] = Field(default_factory=list, max_length=64)
    max_workers: int = Field(default=8, ge=1, le=16)
    direct: bool = False
    expected_bytes: int | None = Field(default=None, ge=0)


class HuggingFaceDownloadPayload(_Payload):
    repo_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    destination: str = Field(min_length=1, max_length=1024)
    download_directory: str | None = Field(default=None, min_length=1, max_length=4096)
    revision: str = Field(default="main", min_length=1, max_length=255)
    repo_type: Literal["model", "dataset", "space"] = "model"
    include: list[str] = Field(default_factory=list, max_length=4096)
    exclude: list[str] = Field(default_factory=list, max_length=64)
    max_workers: int = Field(default=8, ge=1, le=16)
    direct: bool = False
    expected_bytes: int | None = Field(default=None, ge=0)


class QuantizePayload(_Payload):
    input: str = Field(min_length=1, max_length=1024)
    output: str = Field(min_length=1, max_length=1024)
    source_format: Literal["auto", "hf", "gguf", "mfq"] = "auto"
    recipe: str | None = Field(default=None, max_length=1024)
    scheme: str | None = Field(default=None, max_length=1024)
    imatrix: str | None = Field(default=None, max_length=1024)
    calibrate_imatrix: bool = Field(
        default=False,
        title="Calibrate imatrix before quantization",
        description="Collect and freeze an imatrix, then pass it to this quantization run.",
    )
    imatrix_model: str | None = Field(
        default=None,
        max_length=1024,
        title="Imatrix full-precision HF model",
        description="Defaults to the quantization input when that input is an HF model.",
    )
    imatrix_corpus: str | None = Field(
        default=None,
        max_length=1024,
        title="Imatrix calibration corpus",
    )
    imatrix_output: str | None = Field(
        default=None,
        max_length=1024,
        title="Imatrix output",
        description="Workspace-relative path for the newly calibrated imatrix.",
    )
    imatrix_backend: Literal["auto", "cuda", "metal"] = "auto"
    imatrix_device: str = Field(default="", max_length=64)
    imatrix_attention: Literal["sdpa", "eager"] = "sdpa"
    imatrix_objective: Literal["naq", "linear"] = "naq"
    imatrix_window_length: int = Field(default=16_384, ge=2)
    imatrix_batch_size: int = Field(default=1, ge=1)
    imatrix_train_tokens: int = Field(default=1_572_864, ge=2)
    imatrix_seed: int = Field(default=20260810, ge=0)
    imatrix_accumulation_dtype: Literal["auto", "float32", "float64"] = "auto"
    tokenizer: str | None = Field(default=None, max_length=1024)
    sampling_profile: str | None = Field(default=None, max_length=1024)
    bits: int | None = Field(default=None, ge=1, le=8)
    groupsize: int | None = Field(default=None, ge=1, le=4096)
    sub_bits: int | None = Field(default=None, ge=1, le=16)
    q8_mode: Literal["nint8", "nint8-0"] = "nint8-0"
    backend: Literal["auto", "cuda", "cpu"] = "auto"
    device: str = Field(default="cuda", max_length=64)
    full_precision: bool = False
    text_only: bool = False
    exclude_mtp: bool = False
    staged_blobs: bool = Field(
        default=False,
        title="Retain staged tensor blobs",
        description=(
            "Use the resumable two-phase HF/MFQ writer instead of the default "
            "low-disk streaming writer."
        ),
    )
    resume: bool = False
    overwrite: bool = False
    split_max_size: str | None = Field(default=None, pattern=r"^[1-9][0-9]*[MmGg]$")
    split_max_tensors: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_precision(self) -> QuantizePayload:
        if self.full_precision and any(
            value is not None for value in (self.recipe, self.scheme, self.bits, self.imatrix)
        ):
            raise ValueError("full_precision cannot be combined with a quantization recipe")
        if self.full_precision and self.calibrate_imatrix:
            raise ValueError("full_precision cannot calibrate an imatrix")
        if self.full_precision and self.staged_blobs:
            raise ValueError("staged_blobs applies to quantized outputs, not full_precision")
        if self.calibrate_imatrix:
            if self.imatrix is not None:
                raise ValueError("choose an existing imatrix or calibrate a new one, not both")
            missing = [
                name for name in ("imatrix_corpus", "imatrix_output") if getattr(self, name) is None
            ]
            if missing:
                raise ValueError("imatrix calibration requires " + ", ".join(missing))
        if self.split_max_size is not None and self.split_max_tensors is not None:
            raise ValueError("split_max_size and split_max_tensors are mutually exclusive")
        return self


class ImatrixCalibrationPayload(_Payload):
    model: str = Field(
        min_length=1,
        max_length=1024,
        title="Full-precision HF model",
    )
    corpus: str = Field(min_length=1, max_length=1024, title="Calibration corpus")
    output: str = Field(min_length=1, max_length=1024, title="Output imatrix")
    backend: Literal["auto", "cuda", "metal"] = "auto"
    device: str = Field(default="", max_length=64)
    attention: Literal["sdpa", "eager"] = "sdpa"
    objective: Literal["naq", "linear"] = "naq"
    window_length: int = Field(default=16_384, ge=2)
    batch_size: int = Field(default=1, ge=1)
    train_tokens: int = Field(default=1_572_864, ge=2)
    seed: int = Field(default=20260810, ge=0)
    accumulation_dtype: Literal["auto", "float32", "float64"] = "auto"
    work_dir: str | None = Field(default=None, max_length=1024)
    keep_hidden: bool = False
    layerwise: bool = True


class ImportArtifactPayload(_Payload):
    media_id: UUID
    destination: str = Field(
        min_length=1,
        max_length=1024,
        title="Workspace destination",
    )
    kind: Literal["imatrix"] = "imatrix"
    overwrite: bool = False



class ContainerValidationPayload(_Payload):
    model: str = Field(min_length=1, max_length=255)


class PerplexityPayload(_Payload):
    model: str = Field(min_length=1, max_length=255)
    dataset_file: str = Field(min_length=1, max_length=1024)
    dataset: str = Field(default="custom", min_length=1, max_length=255)
    dataset_id: UUID | None = None
    context_size: int = Field(default=512, ge=32)
    chunks: int | None = Field(default=None, ge=1)
    parallel: int = Field(default=1, ge=1, le=64)
    ubatch_size: int | None = Field(default=None, ge=1)
    kl_reference: str | None = Field(default=None, max_length=1024)
    kl_manifest: str | None = Field(default=None, max_length=1024)
    score_count: int | None = Field(default=None, ge=1)
    logits_file: str | None = Field(default=None, max_length=1024)
    logits_manifest: str | None = Field(default=None, max_length=1024)
    model_label: str | None = Field(default=None, max_length=255)
    moe_gpu_cache_gb: float | None = Field(default=None, ge=0.0)


class KernelBenchmarkPayload(_Payload):
    model: str = Field(min_length=1, max_length=255)
    tensor: str = Field(min_length=1, max_length=1024)
    repetitions: int = Field(default=20, ge=1, le=10000)
    experts: list[int] = Field(default_factory=list, max_length=256)
    swiglu: bool = False


@dataclass(frozen=True)
class ToolJobPaths:
    work_root: Path
    python: Path
    modelscope: Path | None
    huggingface: Path | None
    runtime: Path | None
    perplexity: Path | None
    diagnostics: Path | None = None
    standalone_cli: bool = False
    internal_modelscope: bool = False
    internal_huggingface: bool = False


class ToolJobHandlers:
    def __init__(
        self,
        catalog: ModelCatalog,
        paths: ToolJobPaths,
        *,
        voice_component: Any | None = None,
        activate_voice_output: Callable[[], Awaitable[dict[str, Any]]] | None = None,
        runtime_manager: Any | None = None,
    ) -> None:
        self.catalog = catalog
        self.paths = paths
        self.root = paths.work_root.expanduser().resolve()
        self.model_root = catalog.roots[0] if catalog.roots else self.root / "models"
        self.download_directories = DownloadDirectorySettings(self.root, self.model_root)
        self._download_locks: dict[Path, asyncio.Lock] = {}
        self.voice_component = voice_component
        self.activate_voice_output = activate_voice_output
        self.runtime_manager = runtime_manager
        from mfq.server.services.quantization import QuantizationWorkbench
        self.quantization_workbench = QuantizationWorkbench(self)

    def _download_prefix_bytes(self, directory: Path) -> int:
        if self.runtime_manager is None:
            return 0
        cache = self.runtime_manager._prefix_cache_directory()
        if volume(cache) != volume(directory):
            return 0
        from mfq.server.runtime.prefix_disk_budget import _scan_blocks
        return sum(block.size for block in _scan_blocks(cache))

    def download_directory_status(self) -> dict[str, str | int | bool | None]:
        status = self.download_directories.status()
        directory = Path(status['actual_directory'])
        if status['disk_free_bytes'] is not None:
            reserved = DiskBudget().reserved(directory)
            prefix = self._download_prefix_bytes(directory) if self.download_directories.evict_prefix_cache else 0
            status.update(download_reserved_bytes=reserved, reclaimable_prefix_bytes=prefix,
                disk_available_bytes=max(0, status['disk_free_bytes'] + prefix - reserved))
        return status

    def _mfq_command(self, *arguments: str) -> list[str]:
        command = [str(self.paths.python)]
        if not self.paths.standalone_cli:
            command.extend(["-m", "mfq.cli"])
        command.extend(arguments)
        return command

    def handlers(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "dataset.download": TypedJobHandler(self.download_evaluation_dataset, OfficialDatasetDownloadPayload),
            "reference.wikitext2": TypedJobHandler(self.wikitext_reference, Wt2ReferencePayload),
            "artifact.import": TypedJobHandler(self.import_artifact, ImportArtifactPayload),
            "calibrate.imatrix": TypedJobHandler(self.calibrate_imatrix, ImatrixCalibrationPayload),
            "model.validate": TypedJobHandler(self.validate_container, ContainerValidationPayload),
            "model.quantize": TypedJobHandler(self.quantize, QuantizePayload),
        }
        result.update(self.quantization_workbench.handlers())
        if self.voice_component is not None:
            result["component.voice_output.install"] = TypedJobHandler(
                self.install_voice_output, VoiceOutputInstallPayload
            )
        if self.paths.modelscope is not None or self.paths.internal_modelscope:
            result["download.modelscope"] = TypedJobHandler(
                self.download_modelscope, ModelScopeDownloadPayload
            )
        if self.paths.huggingface is not None or self.paths.internal_huggingface:
            result["download.huggingface"] = TypedJobHandler(
                self.download_huggingface, HuggingFaceDownloadPayload
            )
        if self.paths.perplexity is not None:
            result["evaluate.wikitext2"] = TypedJobHandler(self.wikitext_quality, WikitextQualityPayload)
        if self.runtime_manager is not None:
            result["benchmark.inference"] = TypedJobHandler(self.inference_benchmark, InferenceBenchmarkPayload)
            result["evaluate.accuracy"] = TypedJobHandler(self.accuracy_benchmark, AccuracyBenchmarkPayload)
        if self.paths.runtime is not None:
            result["benchmark.kernel"] = TypedJobHandler(
                self.kernel_benchmark, KernelBenchmarkPayload
            )
        return result

    async def install_voice_output(
        self, context: JobContext, payload: dict[str, Any]
    ) -> dict[str, Any]:
        VoiceOutputInstallPayload.model_validate(payload)
        result = await self.voice_component.install(context)
        activation: dict[str, Any] = {"active": False, "reason": "model_not_loaded"}
        if self.activate_voice_output is not None:
            await context.progress(0.999, message="Enabling voice output")
            try:
                activation = await self.activate_voice_output()
            except Exception as error:
                activation = {"active": False, "reason": "activation_failed", "error": str(error)}
                await context.log(f"Voice component installed but activation failed: {error}")
        return {**result, "activation": activation}

    @staticmethod
    def _resolve_imatrix_backend(backend: str) -> str:
        return ("metal" if sys.platform == "darwin" else "cuda") if backend == "auto" else backend

    async def _collect_imatrix(
        self,
        context: JobContext,
        *,
        model: Path,
        corpus: Path,
        output: Path,
        backend: str,
        device: str,
        attention: str,
        objective: str,
        window_length: int,
        batch_size: int,
        train_tokens: int,
        seed: int,
        accumulation_dtype: str,
        work_dir: Path | None = None,
        keep_hidden: bool = False,
        progress_start: float = 0.01,
        progress_end: float = 0.99,
        apply_chat_template: bool = True,
        layerwise: bool = True,
    ) -> dict[str, Any]:
        resolved_backend = self._resolve_imatrix_backend(backend)
        resolved_device = device or ("mps" if resolved_backend == "metal" else "cuda:0")
        output.parent.mkdir(parents=True, exist_ok=True)
        argv = self._mfq_command(
            "calibrate",
            "imatrix",
            "--model",
            str(model),
            "--corpus",
            str(corpus),
            "--render-mode",
            "auto" if apply_chat_template else "plain",
            "--output",
            str(output),
            "--backend",
            resolved_backend,
            "--device",
            resolved_device,
            "--attention",
            attention,
            "--objective",
            objective,
            "--window-length",
            str(window_length),
            "--batch-size",
            str(batch_size),
            "--train-tokens",
            str(train_tokens),
            "--seed",
            str(seed),
            "--accumulation-dtype",
            accumulation_dtype,
        )
        if work_dir is not None:
            work_dir.parent.mkdir(parents=True, exist_ok=True)
            argv.extend(["--work-dir", str(work_dir)])
        if keep_hidden:
            argv.append("--keep-hidden")
        if not layerwise:
            argv.append("--resident")

        loading_data = {}
        def progress(line: str):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return None
            if not isinstance(event, dict):
                return None
            if event.get('event') == 'source_loading':
                loading_data['source_loading'] = event
                return progress_start, 'Insufficient free memory; using layerwise loading' if event.get('fallback_reason') else 'Source loading mode selected', loading_data.copy()
            if event.get("event") == "imatrix_layer":
                layer = int(event.get("layer", 0)) + 1
                layers = max(1, int(event.get("layers", 1)))
                fraction = layer / layers
                return (
                    progress_start + (progress_end - progress_start) * fraction,
                    f"Imatrix layer {layer}/{layers}",
                    loading_data.copy(),
                )
            if event.get("event") == "imatrix_saved":
                return progress_end, "Imatrix saved", loading_data.copy()
            return None

        lines = await self._run(
            context,
            argv,
            progress_parser=progress,
            final_progress=progress_end,
            final_message="Finalizing imatrix",
        )
        if not output.is_file():
            raise self._failure("imatrix_output_missing", "calibration produced no imatrix")
        saved = self._last_json_event(lines, "imatrix_saved") or {}
        prepared = self._last_json_event(lines, "calibration_corpus_prepared") or {}
        loading = self._last_json_event(lines, "source_loading") or {}
        return {
            "backend": resolved_backend,
            "device": resolved_device,
            "entries": int(saved.get("entries", 0)),
            "tokens": int(saved.get("tokens", train_tokens)),
            "total_bytes": output.stat().st_size,
            "chat_rendering": prepared.get("chat_rendering", {}),
            "loading": loading,
        }

    async def calibrate_imatrix(
        self, context: JobContext, payload: dict[str, Any]
    ) -> dict[str, Any]:
        request = ImatrixCalibrationPayload.model_validate(payload)
        model = self._input(request.model)
        corpus = self._input(request.corpus)
        output = self._output(request.output)
        work_dir = self._output(request.work_dir) if request.work_dir is not None else None
        result = await self._collect_imatrix(
            context,
            model=model,
            corpus=corpus,
            output=output,
            backend=request.backend,
            device=request.device,
            attention=request.attention,
            objective=request.objective,
            window_length=request.window_length,
            batch_size=request.batch_size,
            train_tokens=request.train_tokens,
            seed=request.seed,
            accumulation_dtype=request.accumulation_dtype,
            work_dir=work_dir,
            keep_hidden=request.keep_hidden,
            layerwise=request.layerwise,
        )
        uri = self._artifact_uri(output)
        await context.artifact(
            name=output.name,
            uri=uri,
            media_type="application/x-mfq-imatrix",
            metadata={
                "source_uris": [self._artifact_uri(model), self._artifact_uri(corpus)],
                "parameters": request.model_dump(mode="json")
                | {"resolved_backend": result["backend"], "resolved_device": result["device"]},
                "files": 1,
                "total_bytes": result["total_bytes"],
                "entries": result["entries"],
                "tokens": result["tokens"],
            },
        )
        return {**result, "artifact": uri}

    async def import_artifact(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = ImportArtifactPayload.model_validate(payload)
        try:
            media, source = await asyncio.to_thread(context.store.get_media_path, request.media_id)
        except Exception as error:
            raise self._failure(
                "import_media_not_found", "uploaded imatrix was not found"
            ) from error
        output = self._output(request.destination)
        if output.exists() and not request.overwrite:
            raise self._failure(
                "artifact_exists", f"artifact already exists: {request.destination}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f"{output.name}.{os.getpid()}.import")
        try:
            await asyncio.to_thread(shutil.copyfile, source, temporary)
            from mfq.quantize.imatrix import load_importance_matrix

            matrix = await asyncio.to_thread(load_importance_matrix, temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        uri = self._artifact_uri(output)
        await context.artifact(
            name=output.name,
            uri=uri,
            media_type="application/x-mfq-imatrix",
            metadata={
                "source_uris": [f"media://{request.media_id}"],
                "parameters": request.model_dump(mode="json"),
                "files": 1,
                "total_bytes": output.stat().st_size,
                "entries": len(matrix.entries),
                "datasets": list(matrix.datasets),
            },
        )
        return {
            "artifact": uri,
            "entries": len(matrix.entries),
            "total_bytes": output.stat().st_size,
        }


    async def download_modelscope(
        self, context: JobContext, payload: dict[str, Any]
    ) -> dict[str, Any]:
        request = ModelScopeDownloadPayload.model_validate(payload)
        destination = self._download_output(request.destination, request.download_directory)
        if self.paths.internal_modelscope:
            argv = self._mfq_command(
                "_hub-download",
                "--provider",
                "modelscope",
                "--repo-id",
                request.repo_id,
            )
        else:
            executable = self._required_executable(self.paths.modelscope, "ModelScope")
            argv = [str(executable), "download", request.repo_id]
        argv.extend([
            "--repo-type",
            request.repo_type,
            "--revision",
            request.revision,
            "--local-dir",
            str(destination),
            "--max-workers",
            str(request.max_workers),
        ])
        if request.include:
            if self.paths.internal_modelscope:
                for pattern in request.include:
                    argv.extend(["--include", pattern])
            else:
                argv.extend(["--include", *request.include])
        if request.exclude:
            if self.paths.internal_modelscope:
                for pattern in request.exclude:
                    argv.extend(["--exclude", pattern])
            else:
                argv.extend(["--exclude", *request.exclude])
        env, proxy = await self._download_environment('modelscope', request)
        await context.log("Download connection: proxy" if proxy else "Download connection: direct")
        await context.progress(0.01, message="Starting ModelScope download")
        return await self._download_and_register(context, argv, env, destination, request, 'modelscope')

    async def download_huggingface(
        self, context: JobContext, payload: dict[str, Any]
    ) -> dict[str, Any]:
        request = HuggingFaceDownloadPayload.model_validate(payload)
        destination = self._download_output(request.destination, request.download_directory)
        if self.paths.internal_huggingface:
            argv = self._mfq_command(
                "_hub-download",
                "--provider",
                "huggingface",
                "--repo-id",
                request.repo_id,
            )
        else:
            executable = self._required_executable(self.paths.huggingface, "Hugging Face")
            argv = [str(executable), "download", request.repo_id]
        argv.extend([
            "--repo-type",
            request.repo_type,
            "--revision",
            request.revision,
            "--local-dir",
            str(destination),
            "--max-workers",
            str(request.max_workers),
        ])
        for pattern in request.include:
            argv.extend(["--include", pattern])
        for pattern in request.exclude:
            argv.extend(["--exclude", pattern])
        await context.progress(0.01, message="Starting Hugging Face download")
        env, proxy = await self._download_environment('huggingface', request)
        await context.log("Download connection: proxy" if proxy else "Download connection: direct")
        return await self._download_and_register(context, argv, env, destination, request, 'huggingface')

    async def _download_and_register(
        self, context: JobContext, argv: list[str], env: dict[str, str], destination: Path,
        request: ModelScopeDownloadPayload | HuggingFaceDownloadPayload,
        provider: Literal['modelscope', 'huggingface'],
    ) -> dict[str, Any]:
        lock = self._download_locks.setdefault(destination, asyncio.Lock())
        if lock.locked():
            await context.log('Waiting for the other model download in this folder')
        async with lock:
            await self._run_download(context, argv, env, destination, request.expected_bytes, selection=request)
            return await self._download_result(
                context, destination, request.repo_id, provider=provider,
                revision=request.revision, parameters=request.model_dump(mode='json'))

    async def _run_download(
        self, context: JobContext, argv: list[str], env: dict[str, str], destination: Path,
        expected_bytes: int | None, *,
        selection: ModelScopeDownloadPayload | HuggingFaceDownloadPayload | None = None,
    ) -> None:
        def selected(path: Path) -> bool:
            if selection is None:
                return True
            relative = path.relative_to(destination)
            name = relative.as_posix()
            # Both SDKs keep resumable files under hidden download directories;
            # HF appends the content hash and .incomplete after the filename.
            if any(part.startswith('.') for part in relative.parts) or path.suffix in {'.partial', '.incomplete'}:
                match = re.search(r'\.(?:mfq|safetensors|gguf|bin)(?=\.|$)', name, re.IGNORECASE)
                if match:
                    name = name[:match.end()]
                else:
                    name = name.removesuffix('.partial').removesuffix('.incomplete')
                candidates = [name, *(name[index + 1:] for index, char in enumerate(name) if char == '/')]
            else:
                candidates = [name]
            return any((not selection.include or any(fnmatch.fnmatchcase(value, pattern) for pattern in selection.include))
                       and not any(fnmatch.fnmatchcase(value, pattern) for pattern in selection.exclude)
                       for value in candidates)

        def observed() -> tuple[int, int, int]:
            size = files = allocated = 0
            seen = set()
            for path in destination.rglob("*"):
                try:
                    if path.is_symlink() or not path.is_file() or path.suffix in {".lock", ".metadata"}:
                        continue
                    if not selected(path):
                        continue
                    status = path.stat()
                    identity = (status.st_dev, status.st_ino)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    size += status.st_size
                    if hasattr(status, 'st_blocks'):
                        allocated += min(status.st_size, status.st_blocks * 512)
                    elif os.name == 'nt':
                        import ctypes
                        high = ctypes.c_ulong()
                        measure = ctypes.windll.kernel32.GetCompressedFileSizeW
                        measure.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
                        measure.restype = ctypes.c_ulong
                        low = measure(str(path), ctypes.byref(high))
                        if low != 0xffffffff or ctypes.windll.kernel32.GetLastError() == 0:
                            allocated += min(status.st_size, (high.value << 32) | low)
                    if path.suffix not in {".incomplete", ".partial"} and not any(part.startswith(".") for part in path.relative_to(destination).parts):
                        files += 1
                except FileNotFoundError:
                    continue
            return size, files, allocated

        _, _, initial = await asyncio.to_thread(observed)
        if expected_bytes is None or expected_bytes <= 0:
            raise self._failure("download_size_unknown", "Resolve the model files before downloading so disk space can be reserved")
        margin = max(512 << 20, expected_bytes // 100)
        required = max(0, expected_bytes - initial)
        reclaimable = await asyncio.to_thread(self._download_prefix_bytes, destination) if self.download_directories.evict_prefix_cache else 0
        budget = DiskBudget()
        pending_reservation = asyncio.create_task(asyncio.to_thread(budget.reserve, destination,
            required + margin, reclaimable=reclaimable, exclusive=True))
        try:
            reservation = await asyncio.shield(pending_reservation)
        except asyncio.CancelledError:
            with suppress(OSError, ValueError):
                abandoned = await pending_reservation
                await asyncio.to_thread(abandoned.close)
            raise
        except OSError as error:
            raise self._failure("insufficient_disk_space", "Not enough unreserved disk space for this download") from error
        except ValueError as error:
            raise self._failure("download_destination_busy", "This destination already has an active download") from error

        async def monitor() -> None:
            previous, _, _ = await asyncio.to_thread(observed)
            previous_time = time.monotonic()
            speed = 0.0
            while True:
                await asyncio.sleep(1.0)
                size, files, allocated = await asyncio.to_thread(observed)
                now = time.monotonic()
                current_speed = max(0, size - previous) / max(0.001, now - previous_time)
                speed = current_speed if speed == 0 else speed * 0.5 + current_speed * 0.5
                previous, previous_time = size, now
                try:
                    await asyncio.to_thread(reservation.update, max(0, expected_bytes - allocated) + margin)
                except OSError as error:
                    raise self._failure('insufficient_disk_space', 'Not enough disk space to continue this download') from error
                await context.progress(min(0.98, size / expected_bytes) if expected_bytes else 0.01,
                    data={"downloaded_bytes": size, "total_bytes": expected_bytes,
                          "bytes_per_second": speed, "files_completed": files})

        task = transfer = None
        try:
            if reclaimable and self.runtime_manager is not None:
                await self.runtime_manager.reclaim_prefix_disk_for_download(destination)
            if await asyncio.to_thread(lambda: shutil.disk_usage(destination).free < budget.reserved(destination)):
                raise self._failure("insufficient_disk_space", "Prefix cache could not release enough disk space for this download")
            task = asyncio.create_task(monitor())
            child_env = {**env, "MFQ_DISK_BUDGET_DIR": str(budget.root)}
            if os.name != 'nt':
                child_env['MFQ_DOWNLOAD_RESERVATION_FD'] = str(reservation.stream.fileno())
            transfer = asyncio.create_task(self._run(context, argv, env=child_env))
            finished, _ = await asyncio.wait((task, transfer), return_when=asyncio.FIRST_COMPLETED)
            for completed in finished:
                await completed
        finally:
            try:
                for pending in (task, transfer):
                    if pending is not None:
                        pending.cancel()
                await asyncio.gather(*(pending for pending in (task, transfer) if pending is not None), return_exceptions=True)
            finally:
                await asyncio.to_thread(reservation.close)
                if self.runtime_manager is not None:
                    self.runtime_manager._prefix_disk_checked_at = 0.0

    async def quantize(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = QuantizePayload.model_validate(payload)
        source = self._input(request.input)
        output = self._output(request.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        imatrix: Path | None = None
        source_uris = [self._artifact_uri(source)]
        calibrated: dict[str, Any] | None = None
        if request.calibrate_imatrix:
            model = self._input(request.imatrix_model) if request.imatrix_model else source
            corpus = self._input(request.imatrix_corpus or "")
            imatrix = self._output(request.imatrix_output or "")
            await context.progress(0.01, message="Calibrating imatrix")
            calibrated = await self._collect_imatrix(
                context,
                model=model,
                corpus=corpus,
                output=imatrix,
                backend=request.imatrix_backend,
                device=request.imatrix_device,
                attention=request.imatrix_attention,
                objective=request.imatrix_objective,
                window_length=request.imatrix_window_length,
                batch_size=request.imatrix_batch_size,
                train_tokens=request.imatrix_train_tokens,
                seed=request.imatrix_seed,
                accumulation_dtype=request.imatrix_accumulation_dtype,
                progress_start=0.01,
                progress_end=0.44,
            )
            imatrix_uri = self._artifact_uri(imatrix)
            source_uris.extend((self._artifact_uri(model), self._artifact_uri(corpus), imatrix_uri))
            await context.artifact(
                name=imatrix.name,
                uri=imatrix_uri,
                media_type="application/x-mfq-imatrix",
                metadata={
                    "source_uris": [self._artifact_uri(model), self._artifact_uri(corpus)],
                    "parameters": {
                        "backend": calibrated["backend"],
                        "device": calibrated["device"],
                        "attention": request.imatrix_attention,
                        "objective": request.imatrix_objective,
                        "window_length": request.imatrix_window_length,
                        "batch_size": request.imatrix_batch_size,
                        "train_tokens": request.imatrix_train_tokens,
                        "seed": request.imatrix_seed,
                        "accumulation_dtype": request.imatrix_accumulation_dtype,
                    },
                    "files": 1,
                    "total_bytes": calibrated["total_bytes"],
                    "entries": calibrated["entries"],
                    "tokens": calibrated["tokens"],
                },
            )
        elif request.imatrix is not None:
            imatrix = self._input(request.imatrix)
            source_uris.append(self._artifact_uri(imatrix))
        argv = self._mfq_command(
            "quantize",
            str(source),
            str(output),
            "--source-format",
            request.source_format,
            "--q8-mode",
            request.q8_mode,
            "--backend",
            request.backend,
            "--device",
            request.device,
        )
        for name in ("recipe", "scheme", "tokenizer", "sampling_profile"):
            value = getattr(request, name)
            if value is not None:
                argv.extend([f"--{name.replace('_', '-')}", str(self._input(value))])
        if imatrix is not None:
            argv.extend(["--imatrix", str(imatrix)])
        for name in ("bits", "groupsize", "sub_bits", "split_max_tensors"):
            value = getattr(request, name)
            if value is not None:
                argv.extend([f"--{name.replace('_', '-')}", str(value)])
        if request.split_max_size is not None:
            argv.extend(["--split-max-size", request.split_max_size])
        for enabled, flag in (
            (request.full_precision, "--full-precision"),
            (request.text_only, "--text-only"),
            (request.exclude_mtp, "--exclude-mtp"),
            (request.staged_blobs, "--staged-blobs"),
            (request.resume, "--resume"),
            (request.overwrite, "--overwrite"),
        ):
            if enabled:
                argv.append(flag)
        quantize_start = 0.46 if calibrated is not None else 0.01
        await context.progress(quantize_start, message="Starting quantization")

        def quantize_progress(line: str) -> tuple[float, str] | None:
            match = re.search(r"(?:^|\s)(\d{1,3})%", line)
            if match is None:
                return None
            fraction = min(1.0, int(match.group(1)) / 100.0)
            return quantize_start + (0.98 - quantize_start) * fraction, line[:200]

        await self._run(
            context,
            argv,
            progress_parser=quantize_progress,
            final_progress=0.99,
            final_message="Finalizing quantization output",
        )
        outputs = sorted(output.parent.glob(f"{output.stem}*.mfq"))
        if not outputs:
            raise self._failure("quantization_output_missing", "quantization produced no MFQ file")
        total = sum(path.stat().st_size for path in outputs)
        await context.artifact(
            name=output.name,
            uri=self._artifact_uri(output),
            metadata={
                "source_uris": list(dict.fromkeys(source_uris)),
                "parameters": request.model_dump(mode="json"),
                "files": len(outputs),
                "total_bytes": total,
            },
        )
        result = {
            "files": len(outputs),
            "total_bytes": total,
            "artifact": self._artifact_uri(output),
        }
        if imatrix is not None:
            result["imatrix"] = self._artifact_uri(imatrix)
        if calibrated is not None:
            result["imatrix_calibration"] = calibrated
        return result

    async def validate_container(
        self, context: JobContext, payload: dict[str, Any]
    ) -> dict[str, Any]:
        request = ContainerValidationPayload.model_validate(payload)
        artifact = await self._model(request.model)
        diagnostics = self._required_executable(
            self.paths.diagnostics or self.paths.runtime, "MFQ diagnostics"
        )
        output = await self._run(
            context,
            [str(diagnostics), "--model", str(artifact.path), "--check-mfq-container"],
        )
        with suppress(ValueError, StorageError):
            await context.validate_artifact(self._artifact_uri(artifact.path))
        return {
            "model_id": artifact.resource.name,
            "artifact_id": artifact.resource.id,
            "summary": output[-1] if output else "ok",
        }

    async def inference_benchmark(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        return await run_inference_benchmark(context, payload, self.runtime_manager)

    async def download_evaluation_dataset(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        return await download_official_dataset(context, payload, self.root)

    async def accuracy_benchmark(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = AccuracyBenchmarkPayload.model_validate(payload)
        dataset = await asyncio.to_thread(context.store.get_dataset, request.dataset_id)
        path = self._input(dataset.artifact_uri.removeprefix("workspace://"))
        return await run_accuracy_benchmark(context, payload, self.runtime_manager, path, self.root)

    async def wikitext_quality(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = WikitextQualityPayload.model_validate(payload)
        artifact = await self._model(request.model)
        if self.runtime_manager is not None:
            instances = await self.runtime_manager.instances()
            if any(item.model == artifact.resource.name and item.state.value in {"ready", "busy", "loading"} for item in instances.data):
                raise self._failure("quality_model_loaded", "quality evaluation loads a separate process; unload this model first to avoid duplicating resident weights")
        dataset = await asyncio.to_thread(context.store.get_dataset, request.dataset_id)
        if dataset.kind != "wikitext2":
            raise self._failure("wrong_dataset_kind", "WT2 quality requires a WikiText-2 dataset")
        source = self._input(dataset.artifact_uri.removeprefix("workspace://"))
        text_path = await asyncio.to_thread(materialize_wt2, source, dataset)
        path = self._artifact_uri(text_path).removeprefix("workspace://")
        manifest_path = self._input(request.reference_manifest.removeprefix("workspace://"))
        if manifest_path.stat().st_size > 1024 * 1024:
            raise self._failure("invalid_reference_manifest", "reference manifest exceeds 1 MiB")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset", {}).get("sha256") != WT2_TEXT_SHA256:
            raise self._failure("reference_dataset_mismatch", "reference logits do not belong to this dataset")
        geometry = manifest.get("evaluation", {})
        if geometry.get("n_ctx") != request.context_size or geometry.get("n_seq") != request.parallel:
            raise self._failure("reference_geometry_mismatch", "ctx and parallel sequences must match the reference logits manifest")
        return await self.perplexity(context, {"model": request.model, "dataset_file": path,
            "dataset": "wikitext2", "dataset_id": str(request.dataset_id),
            "context_size": request.context_size, "chunks": request.chunks, "parallel": request.parallel,
            "kl_reference": request.reference_logits.removeprefix("workspace://"),
            "kl_manifest": request.reference_manifest.removeprefix("workspace://"),
            "moe_gpu_cache_gb": request.moe_gpu_cache_gb})

    async def wikitext_reference(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = Wt2ReferencePayload.model_validate(payload)
        source = await asyncio.to_thread(self.quantization_workbench.source, request.model, reference=True)
        if source.format != 'hf' or not source.imatrix_supported:
            raise self._failure('reference_architecture_unsupported', 'WT2 source generation currently supports original Qwen3.5 and Gemma4 HF architectures, including supported native QAT weights')
        dataset = await asyncio.to_thread(context.store.get_dataset, request.dataset_id)
        if dataset.kind != 'wikitext2':
            raise self._failure('wrong_dataset_kind', 'WT2 generation requires the official WikiText-2 dataset')
        text = await asyncio.to_thread(materialize_wt2, self._input(dataset.artifact_uri.removeprefix('workspace://')), dataset)
        output = self.quantization_workbench.resolve(request.output)
        manifest = output.with_name(output.name + '.manifest.json')
        if output.exists() or manifest.exists():
            raise self._failure('output_exists', 'reference output or manifest already exists')
        backend = self._resolve_imatrix_backend(request.backend)
        argv = self._mfq_command('wt2-reference', '--model', source.path, '--dataset', str(text),
            '--output', str(output), '--context-size', str(request.context_size), '--chunks', str(request.chunks),
            '--backend', backend, '--device', 'mps' if backend == 'metal' else 'cuda:0')
        if not request.layerwise:
            argv.append('--resident')
        if backend == 'cuda':
            argv.extend(['--runtime', str(self._required_executable(self.paths.runtime, 'MFQ native CUDA runtime'))])

        loading_data = {}
        def progress(line):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return None
            if event.get('event') == 'wt2_layer':
                return .05 + .7 * event['layer'] / event['layers'], f"Reference layer {event['layer']}/{event['layers']}", loading_data.copy()
            if event.get('event') == 'wt2_head':
                return .75 + .23 * event['chunk'] / event['chunks'], f"Reference head {event['chunk']}/{event['chunks']}", loading_data.copy()
            if event.get('event') == 'source_loading':
                loading_data['source_loading'] = event
                return .02, 'Insufficient free memory; using layerwise loading' if event.get('fallback_reason') else 'Source loading mode selected', loading_data.copy()
            return None

        async with self.quantization_workbench._lock:
            lines = await self._run(context, argv, progress_parser=progress, final_progress=.99)
        result = self._last_json_event(lines, 'wt2_reference_saved')
        if not result or not output.is_file() or not manifest.is_file():
            raise self._failure('reference_output_missing', 'reference generator produced no complete logits/manifest pair')
        for path in (output, manifest):
            await context.artifact(name=path.name, uri=path.as_uri(), media_type='application/octet-stream' if path == output else 'application/json',
                metadata={'path': str(path), 'total_bytes': path.stat().st_size})
        return {**result, 'dataset_id': str(request.dataset_id), 'artifact_kind': 'wt2-reference'}

    async def perplexity(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = PerplexityPayload.model_validate(payload)
        artifact = await self._model(request.model)
        executable = self._required_executable(self.paths.perplexity, "MFQ perplexity")
        dataset = self._input(request.dataset_file)
        argv = [
            str(executable),
            "--model",
            str(artifact.path),
            "--dataset",
            request.dataset,
            "--ctx-size",
            str(request.context_size),
            "--parallel",
            str(request.parallel),
        ]
        if request.chunks is not None:
            argv.extend(["--chunks", str(request.chunks)])
        if request.ubatch_size is not None:
            argv.extend(["--ubatch-size", str(request.ubatch_size)])
        if request.kl_reference is not None:
            argv.extend(["--kl-base", str(self._input(request.kl_reference))])
        else:
            argv.extend(["--file", str(dataset)])
        if request.kl_manifest is not None:
            argv.extend(["--kl-manifest", str(self._input(request.kl_manifest))])
        if request.score_count is not None:
            argv.extend(["--kl-score-count", str(request.score_count)])
        if request.logits_file is not None:
            logits = self._output(request.logits_file)
            logits.parent.mkdir(parents=True, exist_ok=True)
            argv.extend(["--logits-file", str(logits)])
            if request.logits_manifest is not None:
                manifest = self._output(request.logits_manifest)
                argv.extend(["--logits-manifest", str(manifest)])
        else:
            logits = None
        if request.model_label is not None:
            argv.extend(["--model-label", request.model_label])
        if request.moe_gpu_cache_gb is not None:
            argv.extend(["--moe-gpu-cache-gb", str(request.moe_gpu_cache_gb)])
        lines = await self._run(
            context,
            argv,
            progress_pattern=re.compile(r"^\[(\d+)\]"),
            progress_total=request.chunks,
        )
        result: dict[str, Any] = {
            "model_id": artifact.resource.name,
            "artifact_id": artifact.resource.id,
            "dataset": request.dataset,
            "context_size": request.context_size,
        }
        final = next((line for line in reversed(lines) if "Final estimate:" in line), None)
        kl = next((line for line in reversed(lines) if line.startswith("cpp_kl_result ")), None)
        if final:
            match = re.search(r"PPL = ([0-9.eE+-]+) \+/- ([0-9.eE+-]+)", final)
            if match:
                result.update(perplexity=float(match.group(1)), uncertainty=float(match.group(2)))
        if kl:
            for key, value in re.findall(r"([a-z_]+)=([^ ]+)", kl):
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
        if request.kl_reference is not None:
            if not kl or any(not isinstance(result.get(key), (int, float)) or not math.isfinite(result[key]) for key in ("kld", "same_top", "scored_tokens")):
                raise self._failure("quality_metrics_missing", "runtime did not return valid KLD/Top1 metrics")
            if not 0 <= result["same_top"] <= 1 or result["scored_tokens"] <= 0:
                raise self._failure("quality_metrics_invalid", "runtime returned invalid Top1 or scored token count")
            result["top1_agreement"] = result["same_top"]
        if logits is not None:
            await context.artifact(
                name=logits.name,
                uri=self._artifact_uri(logits),
                metadata={
                    "source_uris": [f"mfq://{artifact.resource.id}"],
                    "parameters": request.model_dump(mode="json"),
                    "dataset": request.dataset,
                    "context_size": request.context_size,
                },
            )
            result["logits"] = self._artifact_uri(logits)
        metrics = {
            key: value
            for key, value in result.items()
            if key not in {"model_id", "artifact_id", "dataset", "context_size", "logits"}
        }
        manifest = self._file_manifest(dataset)
        manifest.update(name=request.dataset)
        if request.dataset_id:
            registered = await asyncio.to_thread(context.store.get_dataset, request.dataset_id)
            manifest.update(source_sha256=registered.sha256, source_byte_size=registered.byte_size,
                official_id="wt2-raw-test" if request.dataset == "wikitext2" else None,
                text_protocol="mfq-wt2-text-v1" if request.dataset == "wikitext2" else None)
        comparison = {
            "kind": "perplexity",
            "dataset_sha256": manifest["sha256"],
            "source_sha256": manifest.get("source_sha256"),
            "context_size": request.context_size,
            "chunks": request.chunks,
            "parallel": request.parallel,
            "ubatch_size": request.ubatch_size,
            "score_count": request.score_count,
            "kl_reference": self._input_manifest(request.kl_reference),
            "kl_manifest": self._input_manifest(request.kl_manifest),
        }
        evaluation = await context.evaluation(
            kind="perplexity",
            model_id=artifact.resource.name,
            metrics=metrics,
            parameters=request.model_dump(mode="json"),
            comparison_parameters=comparison,
            dataset_id=request.dataset_id,
            dataset_manifest=manifest,
            runtime_identity=self._runtime_identity(
                executable,
                artifact.resource.name,
                artifact.resource.id,
            ),
        )
        result["evaluation_id"] = str(evaluation.id)
        return result

    async def kernel_benchmark(
        self, context: JobContext, payload: dict[str, Any]
    ) -> dict[str, Any]:
        request = KernelBenchmarkPayload.model_validate(payload)
        artifact = await self._model(request.model)
        runtime = self._required_executable(self.paths.runtime, "MFQ runtime")
        argv = [
            str(runtime),
            "--model",
            str(artifact.path),
            "--tensor",
            request.tensor,
            "--benchmark-reps",
            str(request.repetitions),
        ]
        if request.experts:
            argv.extend(["--benchmark-experts", ",".join(map(str, request.experts))])
        if request.swiglu:
            argv.append("--benchmark-swiglu")
        lines = await self._run(context, argv)
        summary = lines[-1] if lines else ""
        parsed: dict[str, Any] = {
            "summary": summary,
            "model_id": artifact.resource.name,
            "artifact_id": artifact.resource.id,
        }
        for key, value in re.findall(r"([a-z_]+)=([^ ]+)", summary):
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
        evaluation = await context.evaluation(
            kind="kernel_benchmark",
            model_id=artifact.resource.name,
            metrics={
                key: value
                for key, value in parsed.items()
                if key not in {"model_id", "artifact_id"}
            },
            parameters=request.model_dump(mode="json"),
            comparison_parameters={
                "kind": "kernel_benchmark",
                "tensor": request.tensor,
                "repetitions": request.repetitions,
                "experts": request.experts,
                "swiglu": request.swiglu,
            },
            runtime_identity=self._runtime_identity(
                runtime,
                artifact.resource.name,
                artifact.resource.id,
            ),
        )
        parsed["evaluation_id"] = str(evaluation.id)
        return parsed

    async def _run(
        self,
        context: JobContext,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        progress_pattern: re.Pattern[str] | None = None,
        progress_total: int | None = None,
        progress_parser: Callable[[str], tuple[float, str] | tuple[float, str, dict[str, Any]] | None] | None = None,
        final_progress: float = 0.99,
        final_message: str = "Finalizing output",
    ) -> list[str]:
        inherited = (int(env['MFQ_DOWNLOAD_RESERVATION_FD']),) if env and os.name != 'nt' and 'MFQ_DOWNLOAD_RESERVATION_FD' in env else ()
        process = await asyncio.create_subprocess_exec(
            *argv,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            cwd=self.root,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            **({'pass_fds': inherited} if inherited else {}),
        )

        async def stop() -> None:
            if process.returncode is not None:
                return
            with suppress(ProcessLookupError):
                if os.name == 'nt':
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=_TOOL_TERMINATE_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                with suppress(ProcessLookupError):
                    if os.name == 'nt':
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        process.wait(), timeout=_TOOL_KILL_TIMEOUT_SECONDS
                    )

        context.add_cleanup(stop)
        lines: deque[str] = deque(maxlen=_TOOL_OUTPUT_TAIL_LINES)
        assert process.stdout is not None
        try:
            while True:
                raw = await process.stdout.readline()
                if not raw:
                    break
                context.raise_if_cancelled()
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                lines.append(line)
                await context.log(line[:4096])
                if progress_parser is not None:
                    parsed = progress_parser(line)
                    if parsed is not None:
                        progress, message = parsed[:2]
                        await context.progress(min(0.99, max(0.01, progress)), message=message[:200], data=parsed[2] if len(parsed) == 3 else None)
                if progress_pattern is not None:
                    match = progress_pattern.search(line)
                    if match:
                        amount = int(match.group(1))
                        progress = amount / progress_total if progress_total else amount / 100
                        await context.progress(min(0.99, max(0.01, progress)), message=line[:200])
            status = await process.wait()
            if status != 0:
                raise self._failure(
                    "tool_process_failed",
                    f"tool process exited with status {status}",
                    retryable=status in {75, 130, 143},
                )
            await context.progress(final_progress, message=final_message)
            return list(lines)
        except BaseException:
            # Download reservations close before JobContext cleanup. Reap the
            # child here so it cannot hold disk-budget locks during that close.
            await stop()
            raise


    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return document if isinstance(document, dict) else {}

    @staticmethod
    def _last_json_event(lines: list[str], event: str) -> dict[str, Any] | None:
        for line in reversed(lines):
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(document, dict) and document.get("event") == event:
                return document
        return None


    async def _download_result(
        self,
        context: JobContext,
        destination: Path,
        repo_id: str,
        *,
        provider: str,
        revision: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        partials = [
            path
            for path in destination.rglob("*")
            if path.is_file() and path.suffix in {".partial", ".incomplete"}
        ]
        if partials:
            raise self._failure(
                "download_incomplete", "download left incomplete files", retryable=True
            )
        files = [path for path in destination.rglob("*") if path.is_file()]
        total = sum(path.stat().st_size for path in files)
        expected = parameters.get("expected_bytes")
        if isinstance(expected, int) and expected > 0 and total < expected:
            raise self._failure(
                "download_size_mismatch",
                f"downloaded {total} bytes, expected at least {expected}",
                retryable=True,
            )
        if not destination.is_relative_to(self.model_root) and any(destination.is_relative_to(root) for root in self.download_directories.roots.values()):
            from mfq.server.state.catalog import ModelRegistrationError
            try:
                await self.catalog.register_directory(path=destination)
            except ModelRegistrationError:
                pass
        uri = self._artifact_uri(destination)
        await context.artifact(
            name=repo_id,
            uri=uri,
            metadata={
                "source_uris": [f"{provider}://{repo_id}@{revision}"],
                "parameters": parameters,
                "files": len(files),
                "total_bytes": total,
            },
        )
        return {"repo_id": repo_id, "files": len(files), "total_bytes": total, "artifact": uri}

    async def _model(self, identifier: str):
        try:
            return await self.catalog.resolve(identifier)
        except ModelArtifactNotFoundError as error:
            raise self._failure(
                "model_artifact_not_found", f"model was not found: {identifier}"
            ) from error

    def _input(self, value: str) -> Path:
        path = self._resolve(value)
        if not path.exists():
            raise self._failure("input_not_found", f"input does not exist: {value}")
        return path

    async def _download_environment(self, provider: str, request) -> tuple[dict[str, str], bool]:
        endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co') if provider == 'huggingface' else os.environ.get('MODELSCOPE_ENDPOINT', 'https://modelscope.cn')
        filename = next((name for name in request.include if not any(char in name for char in '*?[')
            and name.lower().endswith((*MFQ_SUFFIXES, '.safetensors', '.gguf', '.bin'))), None)
        options = {'direct': True} if request.direct else {}
        if filename and not request.direct:
            options['sample_url'] = hub_download_url(provider, request.repo_id, request.revision, filename)
        return await download_environment(endpoint, **options)

    def download_request(self, kind: str, payload: dict[str, Any]):
        model = HuggingFaceDownloadPayload if kind == 'download.huggingface' else ModelScopeDownloadPayload
        request = model.model_validate(payload)
        destination = self._download_output(request.destination, request.download_directory, create=False)
        return request, destination

    def download_destinations(self, kind: str, request) -> list[Path]:
        destinations = [self._download_output(request.destination, request.download_directory, create=False)]
        provider = kind.split('.')[1]
        parts = Path(request.destination).parts
        prefix = ('models', provider, *request.repo_id.split('/'))
        if parts[:len(prefix)] == prefix:
            mirror = 'modelscope' if provider == 'huggingface' else 'huggingface'
            relative = str(Path('models', mirror, *parts[2:]))
            destinations.append(self._download_output(relative, request.download_directory, create=False))
        return destinations

    @staticmethod
    def selected_download_files(request, files):
        selected = [item for item in files if (not request.include or any(fnmatch.fnmatchcase(item.name, pattern) for pattern in request.include))
            and not any(fnmatch.fnmatchcase(item.name, pattern) for pattern in request.exclude)]
        by_name = {item.name: item for item in files}
        names = with_engram_shards([item.name for item in selected], list(by_name))
        if any(any(fnmatch.fnmatchcase(name, pattern) for pattern in request.exclude) for name in names):
            raise ValueError('Required Engram tables cannot be excluded from the model download.')
        return [by_name[name] for name in names]

    @staticmethod
    def reusable_download_files(destination: Path, files) -> dict[str, int]:
        result = {}
        for item in files:
            if not item.name.lower().endswith(MFQ_SUFFIXES) or item.byte_size <= 0:
                continue
            try:
                path = (destination / item.name).resolve()
                if path.is_relative_to(destination) and path.is_file() and path.stat().st_size == item.byte_size:
                    result[item.name] = item.byte_size
            except OSError:
                continue
        return result

    def download_reuse(self, destination: str, directory: str | None, files) -> dict[str, int]:
        path = self._download_output(destination, directory, create=False)
        return self.reusable_download_files(path, files)

    @staticmethod
    def download_files_present(destination: Path, request, files, *, mfq_only: bool = False) -> bool:
        selected = ToolJobHandlers.selected_download_files(request, files)
        if mfq_only:
            weights = [item for item in selected if item.name.lower().endswith((*MFQ_SUFFIXES, '.gguf'))]
            if weights:
                selected = weights
        if not selected:
            return False
        for item in selected:
            path = (destination / item.name).resolve()
            try:
                complete = path.is_relative_to(destination) and path.is_file() and path.stat().st_size == item.byte_size
            except OSError:
                complete = False
            if not complete:
                return False
        return True

    def _download_output(self, value: str, directory: str | None, *, create: bool = True) -> Path:
        if directory is None:
            path = self._resolve(value)
            if create:
                path.mkdir(parents=True, exist_ok=True)
            return path
        try:
            root = self.download_directories.directory(directory, create=create)
        except ValueError as error:
            raise self._failure("invalid_download_directory", str(error)) from error
        relative = Path(value)
        if relative.is_absolute():
            raise self._failure("absolute_path_not_allowed", "download destinations must be relative")
        if relative.parts[:1] == ("models",):
            relative = Path(*relative.parts[1:])
        destination = (root / relative).resolve()
        if destination == root or not destination.is_relative_to(root):
            raise self._failure("path_outside_workspace", "download destination must stay inside its folder")
        if create:
            destination.mkdir(parents=True, exist_ok=True)
        return destination

    def _output(self, value: str, *, directory: bool = False) -> Path:
        path = self._resolve(value)
        if directory:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            raise self._failure(
                "absolute_path_not_allowed", "tool job paths must be workspace-relative"
            )
        model_path = path.parts[:1] == ("models",)
        if path.parts[:1] == ("download-roots",):
            root = self.download_directories.roots.get(path.parts[1] if len(path.parts) > 1 else "")
            if root is None:
                raise self._failure("path_outside_workspace", "download folder is not registered")
            resolved = (root / Path(*path.parts[2:])).resolve()
            if not resolved.is_relative_to(root):
                raise self._failure("path_outside_workspace", "path is outside the download folder")
            return resolved
        root = self.model_root if model_path else self.root
        resolved = (root / Path(*path.parts[1:]) if model_path else root / path).resolve()
        if not resolved.is_relative_to(root):
            raise self._failure(
                "path_outside_workspace", "path is outside the configured workspace"
            )
        return resolved

    def _artifact_uri(self, path: Path) -> str:
        if not path.is_relative_to(self.model_root) and not path.is_relative_to(self.root):
            for identifier, root in self.download_directories.roots.items():
                if path.is_relative_to(root):
                    return f"workspace://download-roots/{identifier}/{path.relative_to(root).as_posix()}"
        relative = (
            Path("models") / path.relative_to(self.model_root)
            if path.is_relative_to(self.model_root)
            else path.relative_to(self.root)
        )
        return f"workspace://{relative.as_posix()}"

    def remove_workspace_artifact(self, artifact_uri: str) -> dict[str, int]:
        if not artifact_uri.startswith("workspace://"):
            raise self._failure("invalid_artifact_uri", "artifact must use workspace://")
        relative = artifact_uri.removeprefix("workspace://")
        target = self._resolve(relative)
        if target in {self.root, self.model_root, *self.download_directories.roots.values()}:
            raise self._failure("unsafe_artifact_target", "workspace root cannot be removed")
        if not target.exists():
            raise self._failure("artifact_not_found", "workspace artifact does not exist")
        if target.is_dir():
            files = [path for path in target.rglob("*") if path.is_file()]
            total = sum(path.stat().st_size for path in files)
            shutil.rmtree(target)
            return {"files": len(files), "total_bytes": total}
        total = target.stat().st_size
        target.unlink()
        return {"files": 1, "total_bytes": total}

    def workspace_file_manifest(self, artifact_uri: str) -> dict[str, Any]:
        if not artifact_uri.startswith("workspace://"):
            raise self._failure("invalid_artifact_uri", "artifact must use workspace://")
        path = self._input(artifact_uri.removeprefix("workspace://"))
        if not path.is_file():
            raise self._failure("dataset_not_file", "dataset artifact must be a file")
        return self._file_manifest(path)

    @staticmethod
    def _file_manifest(path: Path) -> dict[str, Any]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "sha256": digest.hexdigest(),
            "byte_size": path.stat().st_size,
        }

    def _input_manifest(self, value: str | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return self._file_manifest(self._input(value))

    @staticmethod
    def _runtime_identity(
        executable: Path,
        model_id: str,
        artifact_id: str,
    ) -> dict[str, Any]:
        stat = executable.stat()
        return {
            "executable": executable.name,
            "executable_bytes": stat.st_size,
            "executable_modified_ns": stat.st_mtime_ns,
            "model_id": model_id,
            "artifact_id": artifact_id,
        }

    @staticmethod
    def _required_executable(value: Path | None, name: str) -> Path:
        if value is None or not value.is_file():
            raise ToolJobHandlers._failure("tool_unavailable", f"{name} executable is unavailable")
        return value

    @staticmethod
    def _failure(code: str, message: str, *, retryable: bool = False) -> JobExecutionError:
        return JobExecutionError(ErrorDetail(code=code, message=message, retryable=retryable))
