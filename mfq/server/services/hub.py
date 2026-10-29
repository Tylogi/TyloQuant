"""Model-hub discovery and the official Tylogi MFQ catalog."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import ssl
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, unquote, urlparse
from urllib.request import proxy_bypass_environment

import httpx

from mfq.architectures.tensor_schema import tensor_schema_for_config
from mfq.formats.header import MFQ_SUFFIXES
from mfq.formats.companions import is_engram_shard, with_engram_shards
from mfq.server.protocol.models import (
    HubMemoryPool,
    HubModelFile,
    HubModelInfo,
    HubModelSearchResult,
    HubModelSummary,
    HubModelVariant,
    HubSystemProfile,
    ModelConfigurationStatus,
    OfficialModelInfo,
    OfficialModelList,
    OfficialModelSource,
)
from mfq.server.runtime.host_memory import (
    host_memory_snapshot,
    metal_recommended_working_set_size,
    total_physical_memory,
)
from mfq.server.services.hardware import hardware_identity, memory_topology
from mfq.server.services.hub_metadata import estimated_resident_weight_bytes, read_mfq_metadata
from mfq.server.services.model_parameters import parameter_breakdown
from mfq.server.services.model_memory import cache_profile
from mfq.server.services.model_attention import attention_profile

HubProvider = Literal["huggingface", "modelscope"]

_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MFQ_SHARD_PATTERN = re.compile(
    r"^(?P<name>.+?)(?:-\d{5}-of-\d{5})?\.(?:mfq|tq)$", re.IGNORECASE
)
_PRECISION_PATTERN = re.compile(
    r"(?:^|[-_.])((?:S|V|Q|NINT|NVQ)\d+(?:\.\d+)?(?:[-_][A-Za-z0-9]+)?)",
    re.IGNORECASE,
)
_METADATA_TIMEOUT = 8.0
_METADATA_DEADLINE = 12.0
_OFFICIAL_CACHE_TTL = 900
_MFQ_DETAIL_RETRY_TTL = 30


class HubError(RuntimeError):
    pass


class _RepositoryNotFound(HubError):
    pass


class _DirectFirstTransport(httpx.AsyncBaseTransport):
    def __init__(self, proxy: str, verify: ssl.SSLContext | bool) -> None:
        self._proxy_url = proxy
        self._proxy: httpx.AsyncHTTPTransport | None = None
        self._direct = httpx.AsyncHTTPTransport(verify=verify, trust_env=False)
        self._proxy_hosts: set[str] = set()
        self._verify = verify

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host not in self._proxy_hosts:
            try:
                return await self._direct.handle_async_request(request)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError):
                if request.method not in {"GET", "HEAD"}:
                    raise
                self._proxy_hosts.add(request.url.host)
        if self._proxy is None:
            self._proxy = httpx.AsyncHTTPTransport(proxy=self._proxy_url, verify=self._verify, trust_env=False)
        return await self._proxy.handle_async_request(request)

    async def aclose(self) -> None:
        try:
            await self._direct.aclose()
        finally:
            if self._proxy is not None:
                await self._proxy.aclose()


def _metadata_client(endpoint: str, **kwargs: Any) -> httpx.AsyncClient:
    from mfq.server.api.network import system_proxy_environment

    environment = system_proxy_environment()
    parsed = urlparse(endpoint)
    proxy = None
    if not proxy_bypass_environment(parsed.hostname or "", {"no": environment["NO_PROXY"]}):
        proxy = (
            environment.get(f"{parsed.scheme}_proxy")
            or environment.get(f"{parsed.scheme.upper()}_PROXY")
            or environment.get("all_proxy") or environment.get("ALL_PROXY")
        )
    verify: ssl.SSLContext | bool = True
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        verify = ssl.create_default_context(
            cafile=os.environ.get("SSL_CERT_FILE"), capath=os.environ.get("SSL_CERT_DIR")
        )
    return httpx.AsyncClient(
        timeout=httpx.Timeout(_METADATA_TIMEOUT, connect=4.0),
        follow_redirects=True, trust_env=False, verify=verify,
        proxy=proxy, **kwargs,
    )


@dataclass(frozen=True)
class _OfficialSourceSpec:
    provider: HubProvider
    repo_id: str

    @property
    def url(self) -> str:
        if self.provider == "huggingface":
            return f"https://huggingface.co/{self.repo_id}"
        return f"https://modelscope.cn/models/{self.repo_id}"


@dataclass(frozen=True)
class _OfficialModelSpec:
    id: str
    name: str
    family: str
    architecture: str
    description: str
    description_zh: str
    parameter_label: str | None
    active_parameter_label: str | None
    modalities: tuple[str, ...]
    capabilities: tuple[str, ...]
    precision_options: tuple[str, ...]
    license: str | None
    sources: tuple[_OfficialSourceSpec, ...]
    supports_ssd_streaming: bool = False


def resolve_hub_reference(
    reference: str, fallback_provider: HubProvider = "huggingface"
) -> tuple[HubProvider, str, str | None]:
    """Normalize a repository id or trusted hub URL without following it."""

    value = reference.strip()
    if _REPOSITORY_PATTERN.fullmatch(value):
        return fallback_provider, value, None
    try:
        parsed = urlparse(value)
    except ValueError as error:
        raise HubError(f"invalid model repository reference: {error}") from error
    host = (parsed.hostname or "").lower()
    if parsed.username or parsed.password:
        raise HubError("model repository links must not contain credentials")
    if parsed.scheme not in {"http", "https"}:
        raise HubError("model repository links must use HTTP or HTTPS")
    if host in {"huggingface.co", "www.huggingface.co"}:
        provider: HubProvider = "huggingface"
    elif host in {"modelscope.cn", "www.modelscope.cn"}:
        provider = "modelscope"
    else:
        raise HubError("only Hugging Face and ModelScope repository links are supported")
    try:
        parts = [unquote(part) for part in parsed.path.split("/") if part]
    except UnicodeDecodeError as error:
        raise HubError("model repository link contains invalid path encoding") from error
    if provider == "modelscope" and parts[:1] == ["models"]:
        parts = parts[1:]
    if len(parts) < 2:
        raise HubError("model repository link is missing an owner or model name")
    repo_id = "/".join(parts[:2])
    if not _REPOSITORY_PATTERN.fullmatch(repo_id):
        raise HubError("model repository id contains unsupported characters")
    revision = None
    if len(parts) > 3 and parts[2] in {"tree", "blob"}:
        revision = "/".join(parts[3:])
    return provider, repo_id, revision


def system_profile(
    *, backend: str = "unknown", runtime_memory_budget_bytes: int | None = None
) -> HubSystemProfile:
    hardware = hardware_identity()
    snapshot = host_memory_snapshot()
    physical = snapshot.total if snapshot is not None else total_physical_memory() or hardware.physical_memory_bytes
    if physical is None:
        physical = memory_topology()['host_memory_total_bytes']
    available = snapshot.reclaimable(active_ratio=0.35) if snapshot is not None else None
    if backend == "metal":
        recommended = metal_recommended_working_set_size()
        if runtime_memory_budget_bytes is None:
            runtime_memory_budget_bytes = recommended
    normalized_backend = backend if backend in {"metal", "cuda", "rocm", "cpu"} else "unknown"
    unified = hardware.unified_memory or any(item.unified for item in hardware.gpu_memory)
    shared_gpu = next((item for item in hardware.gpu_memory if item.unified), None)
    bandwidth = hardware.memory_bandwidth_bytes_per_second
    if bandwidth is None and shared_gpu is not None:
        bandwidth = shared_gpu.bandwidth_bytes_per_second
    memory_pools = [HubMemoryPool(
        kind="vram", device=item.name, capacity_bytes=item.capacity_bytes,
        bandwidth_bytes_per_second=item.bandwidth_bytes_per_second,
    ) for item in hardware.gpu_memory if not item.unified]
    if not memory_pools and not unified and backend in {"cuda", "rocm", "metal"}:
        memory_pools = [HubMemoryPool(kind="vram", device=name) for name in hardware.gpu_names]
    memory_pools.append(HubMemoryPool(
        kind="uma" if unified else "ram", capacity_bytes=physical,
        bandwidth_bytes_per_second=bandwidth,
    ))
    return HubSystemProfile(
        platform=platform.system() or "unknown",
        machine=platform.machine() or "unknown",
        backend=normalized_backend,
        cpu_name=hardware.cpu_name,
        cpu_cores=hardware.cpu_cores,
        gpu_names=list(hardware.gpu_names),
        gpu_cores=hardware.gpu_cores,
        physical_memory_bytes=physical,
        available_memory_bytes=available,
        runtime_memory_budget_bytes=runtime_memory_budget_bytes,
        memory_pools=memory_pools,
    )


def _configuration_status(
    byte_size: int,
    profile: HubSystemProfile,
    *,
    supported: bool | None = True,
    unsupported_reason: str | None = None,
) -> ModelConfigurationStatus:
    required = byte_size if byte_size > 0 else None
    capacity = profile.runtime_memory_budget_bytes or profile.physical_memory_bytes
    reasons: list[str] = []
    if supported is False:
        reasons.append(unsupported_reason or "This format is not directly loadable by MFQ.")
        status: Literal["recommended", "warning", "unknown"] = "warning"
        recommendation = "not_recommended"
    elif supported is None:
        reasons.append("Runtime compatibility could not be verified from repository metadata.")
        status = "unknown"
        recommendation = "unknown"
    elif required is None or capacity is None:
        reasons.append("Configuration requirements could not be determined.")
        status = "unknown"
        recommendation = "unknown"
    elif required > capacity:
        status = "warning"
        if capacity * 10 >= required * 7:
            recommendation = "caution"
            reasons.append(
                "Estimated memory requirement exceeds the detected runtime budget, "
                "but at least 70% is available."
            )
        else:
            recommendation = "not_recommended"
            reasons.append(
                "The detected runtime budget is below 70% of the estimated memory requirement."
            )
    else:
        reasons.append("Fits within the detected runtime memory budget.")
        status = "recommended"
        recommendation = "three_stars"
    return ModelConfigurationStatus(
        status=status,
        recommendation=recommendation,
        required_memory_bytes=required,
        available_memory_bytes=capacity,
        reasons=reasons,
    )


def _model_configuration_status(
    variants: list[HubModelVariant],
    profile: HubSystemProfile,
) -> ModelConfigurationStatus:
    """Rate full residency from the share of loadable precision tiers that fit."""

    candidates = [
        variant
        for variant in variants
        if variant.format in {"mfq", "hf"}
        and variant.configuration.status != "unknown"
        and variant.configuration.required_memory_bytes is not None
    ]
    capacity = profile.runtime_memory_budget_bytes or profile.physical_memory_bytes
    if not candidates:
        return ModelConfigurationStatus(
            status="unknown",
            recommendation="unknown",
            available_memory_bytes=capacity,
            reasons=["Configuration requirements could not be determined."],
        )

    requirements = [
        variant.configuration.required_memory_bytes
        for variant in candidates
        if variant.configuration.required_memory_bytes is not None
    ]
    smallest = min(requirements)
    if capacity is None:
        return ModelConfigurationStatus(
            status="unknown",
            recommendation="unknown",
            required_memory_bytes=smallest,
            reasons=["Configuration requirements could not be determined."],
        )
    fit_count = sum(requirement <= capacity for requirement in requirements)
    total = len(requirements)
    if fit_count == total:
        status: Literal["recommended", "warning", "unknown"] = "recommended"
        recommendation = "three_stars"
        reason = (
            "All published precision tiers' estimated resident weights fit within the detected runtime memory "
            "budget."
        )
    elif fit_count * 2 > total:
        status = "recommended"
        recommendation = "two_stars"
        reason = (
            "More than half of the published precision tiers' estimated resident weights fit within the "
            "detected runtime memory budget."
        )
    elif fit_count > 0:
        status = "recommended"
        recommendation = "one_star"
        reason = (
            "At most half of the published precision tiers' estimated resident weights fit within the detected "
            "runtime memory budget."
        )
    elif capacity * 10 >= smallest * 7:
        status = "warning"
        recommendation = "caution"
        reason = (
            "The detected runtime memory budget covers at least 70% of the estimated resident weight "
            "requirement for the smallest published precision tier."
        )
    else:
        status = "warning"
        recommendation = "not_recommended"
        reason = (
            "The detected runtime memory budget is below 70% of the estimated resident weight "
            "requirement for the smallest published precision tier."
        )
    return ModelConfigurationStatus(
        status=status,
        recommendation=recommendation,
        required_memory_bytes=smallest,
        available_memory_bytes=capacity,
        reasons=[reason],
    )


def _precision_from_name(name: str) -> str | None:
    matches = list(_PRECISION_PATTERN.finditer(name))
    return matches[-1].group(1).replace("_", "-").upper() if matches else None


def _model_variants(
    files: list[HubModelFile],
    profile: HubSystemProfile,
    *,
    runtime_compatible: bool | None = True,
) -> list[HubModelVariant]:
    variants: list[HubModelVariant] = []
    mfq_groups: dict[tuple[str, str], list[HubModelFile]] = {}
    hf_files: list[HubModelFile] = []
    for item in files:
        parent, separator, basename = item.name.rpartition("/")
        match = _MFQ_SHARD_PATTERN.match(basename)
        if match:
            stem = match.group("name")
            if not separator:
                label = stem
            elif parent.rsplit("/", 1)[-1] == stem:
                label = parent
            else:
                label = f"{parent}/{stem}"
            suffix = '.tq' if basename.lower().endswith('.tq') else '.mfq'
            mfq_groups.setdefault((label, suffix), []).append(item)
        elif basename.lower().endswith(".gguf"):
            configuration = _configuration_status(
                item.byte_size,
                profile,
                supported=False,
                unsupported_reason="GGUF must be converted before the MFQ runtime can load it.",
            )
            variants.append(
                HubModelVariant(
                    id=item.name,
                    label=basename.removesuffix(".gguf"),
                    format="gguf",
                    precision=_precision_from_name(basename),
                    files=[item.name],
                    byte_size=item.byte_size,
                    configuration=configuration,
                )
            )
        elif basename.lower().endswith((".safetensors", ".bin")):
            hf_files.append(item)
    by_name = {item.name: item for item in files}
    for (label, suffix), group in mfq_groups.items():
        if all(is_engram_shard(item.name) for item in group):
            continue
        group = [by_name[name] for name in with_engram_shards(
            [item.name for item in group], list(by_name))]
        size = sum(item.byte_size for item in group)
        shared = [item for item in group if is_engram_shard(item.name)]
        primary = [item for item in group if not is_engram_shard(item.name)]
        inspected = all(item.weight_bytes is not None and item.ssd_ple_bytes is not None for item in group)
        inspected = inspected and all(item.weight_bytes == 0 for item in shared)
        weights = sum(item.weight_bytes or 0 for item in group) if inspected else None
        # Before header inspection finishes, shared table files already have
        # an unambiguous streamed-only role. Do not count them as RAM weights.
        ple = sum(item.ssd_ple_bytes or 0 for item in group) if inspected else (
            sum(item.ssd_ple_bytes if item.weight_bytes == 0 and item.ssd_ple_bytes is not None
                else item.byte_size for item in shared) if shared else None)
        estimated = sum(estimated_resident_weight_bytes(
            item.weight_bytes or 0, item.weight_bytes_by_dtype,
        ) for item in group) if inspected else estimated_resident_weight_bytes(sum(item.byte_size for item in primary), {})
        roles = {role: sum(item.estimated_weight_bytes_by_role[role] for item in primary)
                 for role in ('dense', 'experts', 'embedding')} if all(
            all(role in item.estimated_weight_bytes_by_role for role in ('dense', 'experts', 'embedding'))
            for item in primary) else {}
        configuration = _configuration_status(
            estimated, profile, supported=runtime_compatible,
            unsupported_reason="This model architecture is not registered in MFQ.",
        )
        configuration.reasons.append('Estimated resident weights include a 10% allowance on packed or unclassified weights; exclude streamed PLE, KV, prefix caches and temporary workspaces.' if inspected else 'File-size estimate with a 10% allowance; tensor metadata is unavailable.')
        variants.append(
            HubModelVariant(
                id=f"{suffix[1:]}:{label}",
                    label=label,
                    format="mfq",
                    precision=_precision_from_name(label),
                    files=sorted(item.name for item in group)[:256],
                    byte_size=size,
                    resident_weight_bytes=weights,
                    estimated_resident_weight_bytes=estimated,
                    estimated_weight_bytes_by_role=roles,
                    ssd_ple_bytes=ple,
                    configuration=configuration,
                )
        )
    if hf_files:
        size = sum(item.byte_size for item in hf_files)
        variants.append(
            HubModelVariant(
                id="hf:original",
                label="Original weights",
                format="hf",
                precision=None,
                files=sorted(item.name for item in hf_files)[:256],
                byte_size=size,
                configuration=_configuration_status(
                    size,
                    profile,
                    supported=runtime_compatible,
                    unsupported_reason="This model architecture is not registered in MFQ.",
                ),
            )
        )
    if not variants and files:
        size = sum(item.byte_size for item in files)
        variants.append(
            HubModelVariant(
                id="repository:all",
                label="Complete repository",
                format="unknown",
                precision=None,
                files=[],
                byte_size=size,
                configuration=_configuration_status(
                    size,
                    profile,
                    supported=False,
                    unsupported_reason=(
                        "The repository can be downloaded, but direct runtime compatibility "
                        "has not been verified."
                    ),
                ),
            )
        )
    return sorted(variants, key=lambda item: (item.byte_size <= 0, item.byte_size, item.label))


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        result = converter()
        return result if isinstance(result, dict) else {}
    return {}


def _optional_text(value: Any, *, limit: int = 2048) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        text = ", ".join(str(item) for item in value if item is not None)
    elif isinstance(value, dict):
        text = ", ".join(f"{key}: {item}" for key, item in value.items())
    else:
        text = str(value)
    text = " ".join(text.split()).strip()
    return text[:limit] or None


class HubCatalog:
    def __init__(self, *, cache_path: Path | None = None) -> None:
        self._official_cache: dict[tuple[HubProvider, str], HubModelInfo | None] = {}
        self._discovered_sources: set[tuple[HubProvider, str]] = set()
        self._official_cache_time = 0.0
        self._source_cache_times: dict[tuple[HubProvider, str], float] = {}
        self._detail_attempt_times: dict[tuple[HubProvider, str], float] = {}
        self._cache_path = cache_path
        self._refresh_task: asyncio.Task[None] | None = None
        self._closed = False
        self._load_cache()

    def _load_cache(self) -> None:
        if self._cache_path is None:
            return
        try:
            if self._cache_path.stat().st_size > 1 << 20:
                return
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if payload["version"] != 1:
                return
            now = time.time()
            updated_at = float(payload["updated_at"])
            if not math.isfinite(updated_at) or updated_at > now:
                return
            age = now - updated_at
            cache = {}
            times = {}
            for item in payload["models"]:
                model = HubModelInfo.model_validate(item["model"])
                cached_at = float(item["cached_at"])
                if not math.isfinite(cached_at) or cached_at > now:
                    return
                key = (model.provider, model.repo_id)
                if self._is_official_repo(model.repo_id):
                    cache[key] = model
                    times[key] = time.monotonic() - (now - cached_at)
                    self._discovered_sources.add(key)
            self._official_cache = cache
            self._source_cache_times = times
            self._official_cache_time = time.monotonic() - age
        except (OSError, ValueError, KeyError, TypeError):
            # A missing or damaged cache must not prevent browsing.
            return

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        temporary = None
        try:
            payload = json.dumps({
                "version": 1,
                "updated_at": time.time(),
                "models": [
                    {
                        "model": model.model_dump(mode="json"),
                        "cached_at": time.time() - max(
                            0.0, time.monotonic() - self._source_cache_times.get(key, 0.0)
                        ),
                    }
                    for key, model in self._official_cache.items()
                    if model is not None
                ],
            })
            if len(payload.encode("utf-8")) > 1 << 20:
                return
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._cache_path.parent,
                prefix=f".{self._cache_path.name}.", delete=False,
            ) as output:
                temporary = Path(output.name)
                output.write(payload)
            temporary.replace(self._cache_path)
        except OSError:
            pass
        finally:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    temporary.unlink(missing_ok=True)

    async def aclose(self) -> None:
        self._closed = True
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            await asyncio.gather(self._refresh_task, return_exceptions=True)

    async def search(
        self, provider: HubProvider, query: str, *, limit: int
    ) -> HubModelSearchResult:
        if provider == "huggingface":
            return await asyncio.to_thread(self._search_huggingface, query, limit)
        return await self._search_modelscope(query, limit)

    async def info(
        self,
        provider: HubProvider,
        repo_id: str,
        revision: str | None,
        *,
        profile: HubSystemProfile | None = None,
    ) -> HubModelInfo:
        profile = profile or system_profile()
        cached = self._official_cache.get((provider, repo_id))
        if (
            cached is not None
            and revision in {None, cached.revision}
            and time.monotonic() - self._source_cache_times.get(
                (provider, repo_id), self._official_cache_time
            ) < _OFFICIAL_CACHE_TTL
        ):
            self._start_detail_refresh(profile)
            return cached.model_copy(update={
                "variants": _model_variants(
                    cached.files, profile, runtime_compatible=cached.runtime_compatible
                ),
            })
        return await self._fetch_info(provider, repo_id, revision, profile)

    async def download_files(self, provider: HubProvider, repo_id: str, revision: str | None) -> HubModelInfo:
        cached = self._official_cache.get((provider, repo_id))
        if cached is not None and revision in {None, cached.revision} and time.monotonic() - self._source_cache_times.get(
                (provider, repo_id), self._official_cache_time) < _OFFICIAL_CACHE_TTL:
            return cached
        fetch = self._info_huggingface if provider == 'huggingface' else self._info_modelscope
        return await asyncio.wait_for(fetch(repo_id, revision, system_profile()), timeout=_METADATA_DEADLINE)

    async def probe_download_sources(self, sources) -> dict[str, Any]:
        from mfq.server.api.network import hub_download_url, probe_download_route
        async def probe(source) -> dict[str, Any]:
            result = {'provider': source.provider, 'repo_id': source.repo_id, 'bytes_per_second': 0.0}
            try:
                info = await self.download_files(source.provider, source.repo_id, source.revision)
                files = [item for item in info.files if item.byte_size >= 262144 and
                    item.name.lower().endswith((*MFQ_SUFFIXES, '.safetensors', '.gguf', '.bin'))]
                if files:
                    use_proxy, speed = await probe_download_route(hub_download_url(
                        source.provider, source.repo_id, info.revision, files[0].name))
                    result.update(bytes_per_second=speed, route='proxy' if use_proxy else 'direct')
            except (HubError, asyncio.TimeoutError):
                pass
            return result
        return {'data': await asyncio.gather(*(probe(source) for source in sources))}

    async def _fetch_info(
        self, provider: HubProvider, repo_id: str, revision: str | None,
        profile: HubSystemProfile,
    ) -> HubModelInfo:
        async def fetch() -> HubModelInfo:
            if provider == "huggingface":
                return await self._info_huggingface(repo_id, revision, profile)
            return await self._info_modelscope(repo_id, revision, profile)
        try:
            info = await asyncio.wait_for(fetch(), timeout=_METADATA_DEADLINE)
        except asyncio.TimeoutError as error:
            raise HubError("Model repository metadata request timed out.") from error
        info = self._reuse_mfq_files(info)
        key = (provider, repo_id)
        publish = revision is None and self._is_official_repo(repo_id)
        if publish:
            self._official_cache[key] = info
        try:
            return await self._inspect_mfq_files(info, profile, publish=publish)
        finally:
            if publish:
                self._detail_attempt_times[key] = time.monotonic()

    @staticmethod
    def _mfq_file_identity(info: HubModelInfo, item: HubModelFile) -> tuple | None:
        if item.sha256:
            return ('sha256', item.sha256, item.byte_size)
        if re.fullmatch(r'[0-9a-fA-F]{40,64}', info.revision):
            return (info.provider, info.repo_id, info.revision, item.name, item.byte_size)
        return None

    def _reuse_mfq_files(self, info: HubModelInfo) -> HubModelInfo:
        known = {}
        identities = {self._mfq_file_identity(info, item) for item in info.files if item.name.lower().endswith(MFQ_SUFFIXES)}
        update = {}
        for cached in self._official_cache.values():
            if cached is None:
                continue
            cached_identities = {self._mfq_file_identity(cached, item) for item in cached.files if item.name.lower().endswith(MFQ_SUFFIXES)}
            if identities and None not in identities and identities == cached_identities:
                for field in ('architectures', 'modalities', 'parameter_count', 'parameter_breakdown',
                              'mtp_supported', 'cache_profile', 'ple_parameter_count', 'runtime_compatible',
                              'attention_profile', 'config_inspected'):
                    value = getattr(cached, field)
                    missing = getattr(info, field) is None or getattr(info, field) == [] or field == 'config_inspected' and not info.config_inspected
                    if field not in update and missing and value is not None and value != []:
                        if field == 'config_inspected' and not value:
                            continue
                        update[field] = getattr(cached, field)
            for item in cached.files:
                identity = self._mfq_file_identity(cached, item)
                if identity is not None and item.weight_bytes is not None and item.ssd_ple_bytes is not None:
                    if identity not in known or item.estimated_weight_bytes_by_role:
                        known[identity] = item
        files = []
        for item in info.files:
            cached = known.get(self._mfq_file_identity(info, item))
            files.append(item.model_copy(update={field: getattr(cached, field) for field in (
                'weight_bytes', 'weight_bytes_by_dtype', 'ssd_ple_bytes', 'estimated_weight_bytes_by_role',
            )}) if cached is not None else item)
        return info.model_copy(update={**update, 'files': files})

    @staticmethod
    def _missing_mfq_files(info: HubModelInfo) -> list[tuple[int, HubModelFile]]:
        return [(index, item) for index, item in enumerate(info.files)
                if item.name.lower().endswith(MFQ_SUFFIXES) and item.byte_size >= 64
                and (item.weight_bytes is None or item.ssd_ple_bytes is None
                     or is_engram_shard(item.name) and item.weight_bytes != 0)]

    @staticmethod
    def _missing_model_metadata(info: HubModelInfo) -> bool:
        return any(item.name.lower().endswith(MFQ_SUFFIXES) and item.byte_size >= 64 for item in info.files) and (
            not info.config_inspected or not info.architectures or info.mtp_supported is None)

    def _start_detail_refresh(self, profile: HubSystemProfile) -> bool:
        if self._closed or self._refresh_task is not None and not self._refresh_task.done():
            return False
        now = time.monotonic()
        keys = [key for key, info in self._official_cache.items() if info is not None and (self._missing_mfq_files(info) or self._missing_model_metadata(info))
                and now - self._detail_attempt_times.get(key, float('-inf')) >= _MFQ_DETAIL_RETRY_TTL]
        if not keys:
            return False
        self._refresh_task = asyncio.create_task(self._refresh_mfq_details(keys, profile))
        return True

    async def _refresh_mfq_details(self, keys: list[tuple[HubProvider, str]], profile: HubSystemProfile) -> None:
        semaphore = asyncio.Semaphore(4)
        async def inspect(key: tuple[HubProvider, str]) -> None:
            async with semaphore:
                info = self._official_cache.get(key)
                if info is None:
                    return
                try:
                    await self._inspect_mfq_files(self._reuse_mfq_files(info), profile, publish=True)
                finally:
                    self._detail_attempt_times[key] = time.monotonic()
        try:
            await asyncio.gather(*(inspect(key) for key in keys))
        finally:
            self._save_cache()

    async def _inspect_mfq_files(self, info: HubModelInfo, profile: HubSystemProfile, *, publish: bool = False) -> HubModelInfo:
        candidates = self._missing_mfq_files(info)
        if not candidates and not self._missing_model_metadata(info):
            info = info.model_copy(update={'variants': _model_variants(info.files, profile, runtime_compatible=info.runtime_compatible)})
            if publish:
                self._official_cache[(info.provider, info.repo_id)] = info
            return info
        files = list(info.files)
        configs = []
        architectures = []
        predictors = []
        legacy_blocks = []
        file_count = sum(item.name.lower().endswith(MFQ_SUFFIXES) and item.byte_size >= 64 for item in files)
        def result() -> HubModelInfo:
            config = configs[0] if configs else {}
            text = config.get('text_config', config)
            ple_parameters = info.ple_parameter_count
            if isinstance(text, dict) and any(item.ssd_ple_bytes for item in files):
                vocab, width, layers = text.get('ngram_vocab_size_base'), text.get('ple_embed_dim'), text.get('ple_layer_ids')
                if isinstance(vocab, int) and vocab > 0 and isinstance(width, int) and width > 0 and isinstance(layers, list):
                    ple_parameters = vocab * width * len(layers)
            declared = config.get('architectures') or architectures or info.architectures
            if isinstance(declared, str):
                declared = [declared]
            parameters = parameter_breakdown(config) if config else info.parameter_breakdown
            layer_count = text.get('num_hidden_layers') if isinstance(text, dict) else None
            legacy_predictor = isinstance(layer_count, int) and layer_count > 0 and any(block >= layer_count for block in legacy_blocks)
            has_predictor = any(predictors) or legacy_predictor
            mtp_supported = True if has_predictor else info.mtp_supported
            if mtp_supported is None and len(predictors) == file_count:
                mtp_supported = False
            modalities = list(info.modalities) or ['text']
            if config.get('vision_config') and not config.get('language_model_only', False):
                modalities = list(dict.fromkeys([*modalities, 'image', 'video']))
            return info.model_copy(update={
                'files': list(files), 'architectures': list(dict.fromkeys(declared)),
                'runtime_compatible': tensor_schema_for_config(config) is not None if config else info.runtime_compatible,
                'ple_parameter_count': ple_parameters,
                'parameter_breakdown': parameters,
                'cache_profile': cache_profile(config) if config else info.cache_profile,
                'attention_profile': attention_profile(config) if config else info.attention_profile,
                'config_inspected': bool(config) or info.config_inspected,
                'parameter_count': parameters.total if parameters else info.parameter_count,
                'modalities': modalities,
                'mtp_supported': mtp_supported,
                'variants': _model_variants(files, profile, runtime_compatible=tensor_schema_for_config(config) is not None if config else info.runtime_compatible),
            })
        semaphore = asyncio.Semaphore(8)
        endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/') if info.provider == 'huggingface' else 'https://modelscope.cn'
        async with _metadata_client(endpoint) as client:
            async def inspect(index: int, item: HubModelFile) -> None:
                url = f'{endpoint}/{info.repo_id}/resolve/{quote(info.revision, safe="")}/{quote(item.name, safe="/")}' if info.provider == 'huggingface' else f'{endpoint}/api/v1/models/{info.repo_id}/repo'
                params = None if info.provider == 'huggingface' else {'Revision': info.revision, 'FilePath': item.name}
                try:
                    async with semaphore:
                        metadata = await asyncio.wait_for(read_mfq_metadata(client, url, params, item.byte_size), timeout=_METADATA_DEADLINE)
                    files[index] = item.model_copy(update={'weight_bytes': metadata.weight_bytes,
                        'weight_bytes_by_dtype': metadata.weight_bytes_by_dtype, 'ssd_ple_bytes': metadata.ssd_ple_bytes,
                        'estimated_weight_bytes_by_role': metadata.estimated_weight_bytes_by_role})
                    if metadata.config:
                        configs.append(metadata.config)
                    if metadata.architecture:
                        architectures.append(metadata.architecture)
                    predictors.append(metadata.has_mtp_weights)
                    legacy_blocks.append(metadata.last_legacy_block)
                    if publish:
                        self._official_cache[(info.provider, info.repo_id)] = result()
                except (httpx.HTTPError, ValueError, UnicodeError, asyncio.TimeoutError):
                    pass
            await asyncio.gather(*(inspect(index, item) for index, item in candidates))
            if self._missing_model_metadata(result()):
                remaining = [(index, item) for index, item in enumerate(files)
                    if item.name.lower().endswith(MFQ_SUFFIXES) and item.byte_size >= 64 and index not in {i for i, _ in candidates}]
                remaining = remaining[:1] + remaining[:0:-1]
                for index, item in remaining:
                    if not self._missing_model_metadata(result()):
                        break
                    await inspect(index, item)
        return result()

    async def resolve(
        self,
        reference: str,
        fallback_provider: HubProvider,
        *,
        profile: HubSystemProfile | None = None,
    ) -> HubModelInfo:
        provider, repo_id, revision = resolve_hub_reference(reference, fallback_provider)
        return await self.info(provider, repo_id, revision, profile=profile)

    async def official(
        self, *, profile: HubSystemProfile | None = None, refresh: bool = False
    ) -> OfficialModelList:
        profile = profile or system_profile()
        refreshing = self._refresh_task is not None and not self._refresh_task.done()
        stale = (
            self._official_cache_time == 0.0
            or time.monotonic() - self._official_cache_time >= _OFFICIAL_CACHE_TTL
        )
        if not self._closed and not refreshing and (refresh or stale):
            self._refresh_task = asyncio.create_task(self._refresh_official_cache(profile))
            refreshing = True
        elif not refreshing:
            refreshing = self._start_detail_refresh(profile)
        models = [self._official_model(spec, profile) for spec in self._official_specs()]
        models.sort(key=lambda model: model.published_at or model.updated_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return OfficialModelList(system=profile, data=models, refreshing=refreshing)

    async def _refresh_official_cache(
        self, profile: HubSystemProfile
    ) -> None:
        sources = {
            (source.provider, source.repo_id)
            for model in self._official_specs()
            for source in model.sources
        }
        discovered: dict[HubProvider, set[tuple[HubProvider, str]]] = {}
        missing: set[tuple[HubProvider, str]] = set()

        semaphore = asyncio.Semaphore(4)

        async def inspect(provider: HubProvider, repo_id: str) -> None:
            key = (provider, repo_id)
            try:
                async with semaphore:
                    self._official_cache[key] = await self._fetch_info(provider, repo_id, None, profile)
                self._source_cache_times[key] = time.monotonic()
            except _RepositoryNotFound:
                missing.add(key)
            except HubError:
                # Keep the last usable snapshot when a hub is temporarily offline.
                self._official_cache.setdefault(key, None)

        async def discover(provider: HubProvider) -> None:
            try:
                repos = await asyncio.wait_for(self._discover_official(provider), timeout=_METADATA_DEADLINE)
            except (HubError, asyncio.TimeoutError):
                return
            discovered[provider] = {(provider, repo) for repo in repos if self._is_official_repo(repo)}
            additions = discovered[provider] - sources
            self._discovered_sources.update(additions)
            await asyncio.gather(*(inspect(*key) for key in sorted(additions)))

        try:
            await asyncio.gather(*(inspect(*key) for key in sorted(sources)), discover("huggingface"), discover("modelscope"))
            removed = missing | {key for key in self._discovered_sources | self._official_cache.keys()
                if key[0] in discovered and key not in discovered[key[0]]}
            self._discovered_sources.difference_update(removed)
            for key in removed:
                self._official_cache.pop(key, None)
                self._source_cache_times.pop(key, None)
                self._detail_attempt_times.pop(key, None)
            self._official_cache_time = time.monotonic()
        finally:
            self._save_cache()

    @staticmethod
    def _is_official_repo(repo_id: str) -> bool:
        if not _REPOSITORY_PATTERN.fullmatch(repo_id):
            return False
        owner, name = repo_id.casefold().split('/')
        return owner == 'tylogi' and ('mfq' in name or 'tq' in re.split('[-_]', name))

    @staticmethod
    async def _discover_official(provider: HubProvider) -> list[str]:
        endpoint = 'https://huggingface.co/api/models' if provider == 'huggingface' else 'https://modelscope.cn/openapi/v1/models'
        repos = []
        try:
            async with _metadata_client(endpoint) as client:
                page, url = 1, endpoint
                while page <= 150:
                    params = {'author': 'Tylogi', 'limit': 100} if provider == 'huggingface' and page == 1 else {'owner': 'Tylogi', 'page_number': page, 'page_size': 20} if provider == 'modelscope' else None
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    payload = response.json()
                    if provider == 'huggingface':
                        values = payload
                    else:
                        if not isinstance(payload, dict):
                            raise HubError('Invalid ModelScope author discovery response')
                        if payload.get('success') is False:
                            raise HubError('ModelScope author discovery failed')
                        data = payload.get('data', {})
                        if not isinstance(data, dict) or not isinstance(data.get('models'), list):
                            raise HubError('Invalid ModelScope author discovery response')
                        values = data['models']
                    if not isinstance(values, list) or any(not isinstance(item, dict) or not isinstance(item.get('id'), str) for item in values):
                        raise HubError('Invalid official model discovery response')
                    repos.extend(item['id'] for item in values if isinstance(item.get('id'), str) and not item.get('private', False) and HubCatalog._is_official_repo(item['id']))
                    if provider == 'modelscope':
                        if len(values) < 20 or page * 20 >= int(data.get('total_count', page * 20 + 1)):
                            break
                    else:
                        next_url = response.links.get('next', {}).get('url')
                        if not next_url:
                            break
                        parsed = urlparse(next_url)
                        if parsed.scheme != 'https' or parsed.netloc != 'huggingface.co' or parsed.path != '/api/models':
                            raise HubError('Invalid Hugging Face discovery pagination URL')
                        url = next_url
                    page += 1
                else:
                    raise HubError('Official model discovery pagination did not complete')
            return repos
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            raise HubError('Official model discovery failed') from error

    def _official_specs(self) -> list[_OfficialModelSpec]:
        specs: dict[str, _OfficialModelSpec] = {}
        for provider, repo_id in sorted(self._discovered_sources):
            source = _OfficialSourceSpec(provider, repo_id)
            name = repo_id.split('/')[1]
            key = name.casefold()
            if key in specs:
                model = specs[key]
                if source not in model.sources:
                    specs[key] = replace(model, sources=(*model.sources, source))
                continue
            info = self._official_cache.get((provider, repo_id)) or next((value for (_, repo), value in self._official_cache.items() if repo.casefold() == repo_id.casefold() and value is not None), None)
            display_name = re.split(r'-(?:EWQ?|MFQ)(?:-|$)', name, maxsplit=1, flags=re.IGNORECASE)[0]
            ple_label = f'约 {info.ple_parameter_count / 1e9:.1f}B 参数' if info and info.ple_parameter_count else '大规模'
            ple_label_en = f'approximately {info.ple_parameter_count / 1e9:.1f}B parameters' if info and info.ple_parameter_count else 'large PLE tables'
            flash_next = 'flash-next' in display_name.casefold()
            specs[key] = _OfficialModelSpec(
                id='tylogi-' + hashlib.sha256(key.encode()).hexdigest()[:16], name=display_name,
                family=display_name, architecture=info.architectures[0] if info and info.architectures else 'unknown',
                description=(f'High-performance, fast compact MoE model with {ple_label_en}. PLE tables stream row-wise from SSD without full-table memory residency.' if flash_next else info.description if info and info.description else f'{display_name} model published by Tylogi in MFQ format.'),
                description_zh=f'高性能、快速的中小型 MoE 模型，带有{ple_label}的 PLE 表，可高效卸载至 SSD，按行读取且无需整表常驻内存。' if flash_next else f'{display_name} 模型，提供 MFQ 格式的精度版本。', parameter_label=None, active_parameter_label=None,
                modalities=tuple(info.modalities) if info and info.modalities else ('text',), capabilities=('MoE', 'PLE') if flash_next else (), precision_options=(),
                license=info.license if info else None, sources=(source,),
            )
        return list(specs.values())

    def _official_model(
        self, spec: _OfficialModelSpec, profile: HubSystemProfile
    ) -> OfficialModelInfo:
        source_models = [
            self._official_cache.get((source.provider, source.repo_id))
            for source in spec.sources
        ]
        def source_quality(index: int) -> tuple[int, float, int]:
            info = source_models[index]
            if info is None:
                return (-1, 0, -index)
            weights = [item for item in info.files if item.name.lower().endswith(MFQ_SUFFIXES)]
            coverage = sum(item.weight_bytes is not None for item in weights) / len(weights) if weights else 0
            return (int(info.parameter_breakdown is not None) + int(info.cache_profile is not None), coverage, -index)
        selected_index = max(range(len(source_models)), key=source_quality)
        source_info = source_models[selected_index]
        parameters = source_info.parameter_breakdown if source_info else None
        modalities = source_info.modalities if source_info and source_info.modalities else list(spec.modalities)
        capabilities = [name for name, enabled in (
            ('MoE', parameters is not None and parameters.routed_experts > 0),
            ('PLE', source_info is not None and bool(source_info.ple_parameter_count or parameters and parameters.ple)),
            ('MTP', source_info is not None and source_info.mtp_supported is True),
            ('Vision', 'image' in modalities or 'video' in modalities),
            ('Audio', 'audio' in modalities),
        ) if enabled]
        sources = [
            OfficialModelSource(
                provider=source.provider,
                repo_id=source.repo_id,
                revision=(source_models[index].revision if source_models[index] else None),
                url=source.url,
                available=source_models[index] is not None,
            )
            for index, source in enumerate(spec.sources)
        ]
        selected_source = sources[selected_index]
        variants = []
        if source_info is not None:
            variants = _model_variants(
                source_info.files, profile, runtime_compatible=True
            )
        configuration = _model_configuration_status(variants, profile)
        if source_info is None:
            configuration.reasons.append(
                "The catalog is available offline; repository metadata could not be refreshed."
            )
        return OfficialModelInfo(
            id=spec.id,
            name=spec.name,
            family=spec.family,
            architecture=source_info.architectures[0] if source_info and source_info.architectures else spec.architecture,
            attention_profile=source_info.attention_profile if source_info else None,
            description=spec.description,
            description_zh=spec.description_zh,
            parameter_label=spec.parameter_label,
            active_parameter_label=spec.active_parameter_label,
            parameter_breakdown=source_info.parameter_breakdown if source_info else None,
            cache_profile=source_info.cache_profile if source_info else None,
            mtp_supported=source_info.mtp_supported if source_info and source_info.mtp_supported is not None else True if 'MTP' in spec.capabilities else None,
            modalities=modalities,
            capabilities=capabilities,
            precision_options=list(spec.precision_options),
            license=source_info.license if source_info and source_info.license else spec.license,
            supports_ssd_streaming='MoE' in capabilities,
            sources=sources,
            selected_source=selected_source,
            revision=source_info.revision if source_info else "main",
            downloads=source_info.downloads if source_info else 0,
            likes=source_info.likes if source_info else 0,
            updated_at=source_info.updated_at if source_info else None,
            published_at=source_info.published_at if source_info else None,
            variants=variants,
            configuration=configuration,
        )

    @staticmethod
    def _search_huggingface(query: str, limit: int) -> HubModelSearchResult:
        try:
            from huggingface_hub import HfApi

            models = HfApi().list_models(
                search=query,
                sort="downloads",
                limit=limit,
                expand=["downloads", "likes", "lastModified"],
                token=os.environ.get("HF_TOKEN") or None,
            )
            return HubModelSearchResult(
                data=[
                    HubModelSummary(
                        provider="huggingface",
                        repo_id=item.id,
                        source_url=f"https://huggingface.co/{item.id}",
                        author=item.id.split("/", 1)[0] if "/" in item.id else None,
                        downloads=item.downloads or 0,
                        likes=item.likes or 0,
                        updated_at=item.last_modified,
                    )
                    for item in models
                ]
            )
        except Exception as error:
            raise HubError(str(error)) from error

    @staticmethod
    async def _info_huggingface(
        repo_id: str, revision: str | None, profile: HubSystemProfile
    ) -> HubModelInfo:
        try:
            from huggingface_hub import ModelInfo, constants, get_token

            endpoint = constants.ENDPOINT
            path = f"{endpoint}/api/models/{repo_id}"
            if revision is not None:
                path += f"/revision/{quote(revision, safe='')}"
            token = get_token()
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            async with _metadata_client(endpoint, headers=headers) as client:
                response = await client.get(path, params={"blobs": "true"})
                response.raise_for_status()
                info = ModelInfo(**response.json())
            files = []
            for item in info.siblings or []:
                lfs = _mapping(getattr(item, "lfs", None))
                sha256 = lfs.get("sha256")
                files.append(
                    HubModelFile(
                        name=item.rfilename,
                        byte_size=item.size or int(lfs.get("size") or 0),
                        sha256=sha256 if isinstance(sha256, str) else None,
                    )
                )
            card = _mapping(getattr(info, "card_data", None))
            config = _mapping(getattr(info, "config", None))
            runtime_compatible = (
                tensor_schema_for_config(config) is not None if config else None
            )
            architectures = config.get("architectures") or []
            if isinstance(architectures, str):
                architectures = [architectures]
            safetensors = getattr(info, "safetensors", None)
            parameter_count = getattr(safetensors, "total", None)
            if not isinstance(parameter_count, int):
                parameter_count = None
            tags = list(info.tags or [])
            modalities = [
                modality
                for modality in ("text", "image", "video", "audio")
                if modality in " ".join(tags).lower()
            ]
            return HubModelInfo(
                provider="huggingface",
                repo_id=info.id,
                source_url=f"https://huggingface.co/{info.id}",
                author=getattr(info, "author", None)
                or (info.id.split("/", 1)[0] if "/" in info.id else None),
                description=_optional_text(
                    card.get("model_description") or card.get("model_name")
                ),
                revision=revision or info.sha or "main",
                downloads=info.downloads or 0,
                likes=info.likes or 0,
                total_bytes=sum(item.byte_size for item in files),
                updated_at=info.last_modified,
                published_at=info.created_at,
                files=files,
                tags=tags,
                license=_optional_text(card.get("license"), limit=255),
                library=_optional_text(getattr(info, "library_name", None), limit=255),
                pipeline_tag=_optional_text(
                    getattr(info, "pipeline_tag", None), limit=255
                ),
                architectures=[str(value) for value in architectures],
                attention_profile=attention_profile(config),
                modalities=modalities,
                parameter_count=parameter_count,
                gated=bool(getattr(info, "gated", False)),
                runtime_compatible=runtime_compatible,
                variants=_model_variants(
                    files, profile, runtime_compatible=runtime_compatible
                ),
            )
        except Exception as error:
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in {404, 410}:
                raise _RepositoryNotFound("Model repository no longer exists.") from error
            raise HubError(str(error)) from error

    @staticmethod
    async def _search_modelscope(query: str, limit: int) -> HubModelSearchResult:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.put(
                    "https://modelscope.cn/api/v1/models",
                    json={"PageSize": limit, "Name": query},
                )
                response.raise_for_status()
                payload = response.json().get("Data", {})
            values = payload.get("Models", payload.get("models", []))
            result = []
            for item in values:
                owner = item.get("Path") or ""
                name = item.get("Name") or ""
                repo_id = f"{owner}/{name}" if owner and name else name or owner
                if not repo_id:
                    continue
                result.append(
                    HubModelSummary(
                        provider="modelscope",
                        repo_id=repo_id,
                        source_url=f"https://modelscope.cn/models/{repo_id}",
                        author=owner or None,
                        description=_optional_text(
                            item.get("Description") or item.get("ChineseName")
                        ),
                        downloads=int(item.get("Downloads") or 0),
                        likes=int(item.get("Likes") or item.get("Stars") or 0),
                        total_bytes=int(item.get("StorageSize") or 0),
                    )
                )
            return HubModelSearchResult(data=result[:limit])
        except Exception as error:
            raise HubError(str(error)) from error

    @staticmethod
    async def _info_modelscope(
        repo_id: str, revision: str | None, profile: HubSystemProfile
    ) -> HubModelInfo:
        try:
            try:
                from modelscope_hub.config import get_default_config
            except ImportError:
                endpoint = os.environ.get('MODELSCOPE_ENDPOINT', 'https://modelscope.cn').rstrip('/')
                token = os.environ.get('MODELSCOPE_API_TOKEN') or os.environ.get('MODELSCOPE_TOKEN')
            else:
                config = get_default_config()
                endpoint, token = str(config.endpoint).rstrip('/'), config.token
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            cookies = httpx.Cookies()
            if token:
                cookies.set("m_session_id", token, domain=urlparse(endpoint).hostname)
            async with _metadata_client(endpoint, headers=headers, cookies=cookies) as client:
                async def metadata() -> dict[str, Any]:
                    response = await client.get(f"{endpoint}/openapi/v1/models/{repo_id}")
                    if response.status_code == 404:
                        response = await client.get(f"{endpoint}/api/v1/models/{repo_id}")
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("success") is False or payload.get("Code", 200) != 200:
                        raise HubError("ModelScope repository metadata request failed.")
                    return _mapping(payload.get("data", payload.get("Data", payload)))

                async def listing() -> list[dict[str, Any]]:
                    response = await client.get(
                        f"{endpoint}/api/v1/models/{repo_id}/repo/files",
                        params={"Revision": revision or "master", "Recursive": "True"},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("success") is False or payload.get("Code", 200) != 200:
                        raise HubError("ModelScope repository file listing failed.")
                    data = payload.get("data", payload.get("Data", payload))
                    if isinstance(data, dict):
                        data = data.get("Files", data.get("files", []))
                    return data

                tasks = [asyncio.create_task(metadata()), asyncio.create_task(listing())]
                try:
                    model, entries = await asyncio.gather(*tasks, return_exceptions=True)
                    if isinstance(model, Exception):
                        raise model
                    if isinstance(entries, Exception):
                        raise entries
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
            files = [
                HubModelFile(
                    name=item.get("Path") or item.get("path") or item.get("Name") or "",
                    byte_size=int(item.get("Size") or item.get("size") or 0),
                    sha256=(
                        str(item.get("Sha256") or item.get("sha256"))
                        if re.fullmatch(
                            r"[0-9a-f]{64}", str(item.get("Sha256") or item.get("sha256"))
                        )
                        else None
                    ),
                )
                for item in entries
                if item.get("Type", item.get("type", "blob")) != "tree"
            ]
            tags = list(model.get("tags") or model.get("Tags") or [])
            license_name = model.get("license") or model.get("License")
            return HubModelInfo(
                provider="modelscope",
                repo_id=repo_id,
                source_url=f"https://modelscope.cn/models/{repo_id}",
                author=repo_id.split("/", 1)[0],
                description=_optional_text(model.get("description") or model.get("Description")),
                revision=revision or "master",
                downloads=int(model.get("downloads") or model.get("Downloads") or 0),
                likes=int(model.get("likes") or model.get("Likes") or 0),
                total_bytes=sum(item.byte_size for item in files),
                updated_at=(
                    model.get("last_modified") or model.get("updated_at")
                    or model.get("LastModified") or model.get("UpdatedAt")
                    or model.get("LastUpdatedTime")
                ),
                published_at=model.get("CreatedTime") or model.get("created_at"),
                files=files,
                tags=tags,
                license=_optional_text(license_name, limit=255),
                gated=bool(model.get("gated", False)),
                modalities=[
                    modality
                    for modality in ("text", "image", "video", "audio")
                    if modality in " ".join(tags).lower()
                ],
                runtime_compatible=None,
                variants=_model_variants(
                    files, profile, runtime_compatible=None
                ),
            )
        except Exception as error:
            if (isinstance(error, httpx.HTTPStatusError) and error.response.status_code in {404, 410}
                    and error.request.url.path.rstrip('/').endswith('/models/' + repo_id)):
                raise _RepositoryNotFound("Model repository no longer exists.") from error
            raise HubError(str(error)) from error


__all__ = [
    "HubCatalog",
    "HubError",
    "HubProvider",
    "resolve_hub_reference",
    "system_profile",
]
