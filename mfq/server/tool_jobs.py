"""Strict MFQ tool jobs backed by argv-only subprocesses."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mfq.server.catalog import ModelArtifactNotFoundError, ModelCatalog
from mfq.server.jobs import JobContext, JobExecutionError, TypedJobHandler
from mfq.server.models import ErrorDetail
from mfq.server.storage import StorageError


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelScopeDownloadPayload(_Payload):
    repo_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    destination: str = Field(min_length=1, max_length=1024)
    revision: str = Field(default="master", min_length=1, max_length=255)
    repo_type: Literal["model", "dataset"] = "model"
    include: list[str] = Field(default_factory=list, max_length=64)
    exclude: list[str] = Field(default_factory=list, max_length=64)
    max_workers: int = Field(default=8, ge=1, le=16)
    direct: bool = False
    expected_bytes: int | None = Field(default=None, ge=0)


class HuggingFaceDownloadPayload(_Payload):
    repo_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    destination: str = Field(min_length=1, max_length=1024)
    revision: str = Field(default="main", min_length=1, max_length=255)
    repo_type: Literal["model", "dataset", "space"] = "model"
    include: list[str] = Field(default_factory=list, max_length=64)
    exclude: list[str] = Field(default_factory=list, max_length=64)
    max_workers: int = Field(default=8, ge=1, le=16)
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
    window_length: int = Field(default=16_384, ge=2)
    batch_size: int = Field(default=1, ge=1)
    train_tokens: int = Field(default=1_572_864, ge=2)
    seed: int = Field(default=20260810, ge=0)
    accumulation_dtype: Literal["auto", "float32", "float64"] = "auto"
    work_dir: str | None = Field(default=None, max_length=1024)
    keep_hidden: bool = False


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
    standalone_cli: bool = False


class ToolJobHandlers:
    def __init__(self, catalog: ModelCatalog, paths: ToolJobPaths) -> None:
        self.catalog = catalog
        self.paths = paths
        self.root = paths.work_root.expanduser().resolve()

    def _mfq_command(self, *arguments: str) -> list[str]:
        command = [str(self.paths.python)]
        if not self.paths.standalone_cli:
            command.extend(["-m", "mfq.cli"])
        command.extend(arguments)
        return command

    def handlers(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "artifact.import": TypedJobHandler(self.import_artifact, ImportArtifactPayload),
            "calibrate.imatrix": TypedJobHandler(self.calibrate_imatrix, ImatrixCalibrationPayload),
            "model.validate": TypedJobHandler(self.validate_container, ContainerValidationPayload),
            "model.quantize": TypedJobHandler(self.quantize, QuantizePayload),
        }
        if self.paths.modelscope is not None:
            result["download.modelscope"] = TypedJobHandler(
                self.download_modelscope, ModelScopeDownloadPayload
            )
        if self.paths.huggingface is not None:
            result["download.huggingface"] = TypedJobHandler(
                self.download_huggingface, HuggingFaceDownloadPayload
            )
        if self.paths.perplexity is not None:
            result["evaluate.perplexity"] = TypedJobHandler(self.perplexity, PerplexityPayload)
        if self.paths.runtime is not None:
            result["benchmark.kernel"] = TypedJobHandler(
                self.kernel_benchmark, KernelBenchmarkPayload
            )
        return result

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
        window_length: int,
        batch_size: int,
        train_tokens: int,
        seed: int,
        accumulation_dtype: str,
        work_dir: Path | None = None,
        keep_hidden: bool = False,
        progress_start: float = 0.01,
        progress_end: float = 0.99,
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
            "--output",
            str(output),
            "--backend",
            resolved_backend,
            "--device",
            resolved_device,
            "--attention",
            attention,
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

        def progress(line: str) -> tuple[float, str] | None:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return None
            if not isinstance(event, dict):
                return None
            if event.get("event") == "imatrix_layer":
                layer = int(event.get("layer", 0)) + 1
                layers = max(1, int(event.get("layers", 1)))
                fraction = layer / layers
                return (
                    progress_start + (progress_end - progress_start) * fraction,
                    f"Imatrix layer {layer}/{layers}",
                )
            if event.get("event") == "imatrix_saved":
                return progress_end, "Imatrix saved"
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
        return {
            "backend": resolved_backend,
            "device": resolved_device,
            "entries": int(saved.get("entries", 0)),
            "tokens": int(saved.get("tokens", train_tokens)),
            "total_bytes": output.stat().st_size,
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
            window_length=request.window_length,
            batch_size=request.batch_size,
            train_tokens=request.train_tokens,
            seed=request.seed,
            accumulation_dtype=request.accumulation_dtype,
            work_dir=work_dir,
            keep_hidden=request.keep_hidden,
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
        destination = self._output(request.destination, directory=True)
        self._preflight_download(destination, request.expected_bytes)
        executable = self._required_executable(self.paths.modelscope, "ModelScope")
        argv = [
            str(executable),
            "download",
            request.repo_id,
            "--repo-type",
            request.repo_type,
            "--revision",
            request.revision,
            "--local-dir",
            str(destination),
            "--max-workers",
            str(request.max_workers),
        ]
        if request.include:
            argv.extend(["--include", *request.include])
        if request.exclude:
            argv.extend(["--exclude", *request.exclude])
        env = self._environment(direct=request.direct)
        await context.progress(0.01, message="Starting ModelScope download")
        await self._run(context, argv, env=env)
        return await self._download_result(
            context,
            destination,
            request.repo_id,
            provider="modelscope",
            revision=request.revision,
            parameters=request.model_dump(mode="json"),
        )

    async def download_huggingface(
        self, context: JobContext, payload: dict[str, Any]
    ) -> dict[str, Any]:
        request = HuggingFaceDownloadPayload.model_validate(payload)
        destination = self._output(request.destination, directory=True)
        self._preflight_download(destination, request.expected_bytes)
        executable = self._required_executable(self.paths.huggingface, "Hugging Face")
        argv = [
            str(executable),
            "download",
            request.repo_id,
            "--repo-type",
            request.repo_type,
            "--revision",
            request.revision,
            "--local-dir",
            str(destination),
            "--max-workers",
            str(request.max_workers),
        ]
        for pattern in request.include:
            argv.extend(["--include", pattern])
        for pattern in request.exclude:
            argv.extend(["--exclude", pattern])
        await context.progress(0.01, message="Starting Hugging Face download")
        await self._run(context, argv, env=self._environment())
        return await self._download_result(
            context,
            destination,
            request.repo_id,
            provider="huggingface",
            revision=request.revision,
            parameters=request.model_dump(mode="json"),
        )

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
        runtime = self._required_executable(self.paths.runtime, "MFQ runtime")
        output = await self._run(
            context,
            [str(runtime), "--mfq", str(artifact.path), "--check-mfq-container"],
        )
        with suppress(ValueError, StorageError):
            await context.validate_artifact(self._artifact_uri(artifact.path))
        return {
            "model_id": artifact.resource.name,
            "artifact_id": artifact.resource.id,
            "summary": output[-1] if output else "ok",
        }

    async def perplexity(self, context: JobContext, payload: dict[str, Any]) -> dict[str, Any]:
        request = PerplexityPayload.model_validate(payload)
        artifact = await self._model(request.model)
        executable = self._required_executable(self.paths.perplexity, "MFQ perplexity")
        dataset = self._input(request.dataset_file)
        argv = [
            str(executable),
            "--mfq",
            str(artifact.path),
            "--file",
            str(dataset),
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
        comparison = {
            "kind": "perplexity",
            "dataset_sha256": manifest["sha256"],
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
            "--mfq",
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
        progress_parser: Callable[[str], tuple[float, str] | None] | None = None,
        final_progress: float = 0.99,
        final_message: str = "Finalizing output",
    ) -> list[str]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.root,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )

        async def stop() -> None:
            if process.returncode is not None:
                return
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

        context.add_cleanup(stop)
        lines: list[str] = []
        assert process.stdout is not None
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
                    progress, message = parsed
                    await context.progress(min(0.99, max(0.01, progress)), message=message[:200])
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
        return lines


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

    def _preflight_download(self, destination: Path, expected_bytes: int | None) -> None:
        if expected_bytes is None or expected_bytes <= 0:
            return
        existing = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
        required = max(0, expected_bytes - existing)
        free = shutil.disk_usage(destination).free
        reserve = max(512 * 1024 * 1024, expected_bytes // 100)
        if free < required + reserve:
            raise self._failure(
                "insufficient_disk_space",
                f"download requires {required} bytes plus {reserve} bytes reserve, "
                f"but only {free} bytes are free",
            )

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
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise self._failure(
                "path_outside_workspace", "path is outside the configured workspace"
            )
        return resolved

    def _artifact_uri(self, path: Path) -> str:
        return f"workspace://{path.relative_to(self.root).as_posix()}"

    def remove_workspace_artifact(self, artifact_uri: str) -> dict[str, int]:
        if not artifact_uri.startswith("workspace://"):
            raise self._failure("invalid_artifact_uri", "artifact must use workspace://")
        relative = artifact_uri.removeprefix("workspace://")
        target = self._resolve(relative)
        if target == self.root:
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
    def _environment(*, direct: bool = False) -> dict[str, str]:
        env = dict(os.environ)
        if direct:
            for name in (
                "http_proxy",
                "https_proxy",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "all_proxy",
            ):
                env.pop(name, None)
            env["NO_PROXY"] = "*"
            env["no_proxy"] = "*"
        return env

    @staticmethod
    def _failure(code: str, message: str, *, retryable: bool = False) -> JobExecutionError:
        return JobExecutionError(ErrorDetail(code=code, message=message, retryable=retryable))
