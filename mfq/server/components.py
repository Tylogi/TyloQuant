"""Optional runtime components downloaded independently from model containers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from mfq.server.jobs import JobContext, JobExecutionError
from mfq.server.models import ErrorDetail

VOICE_OUTPUT_COMPONENT_ID = "minicpmo45-token2wav"
VOICE_OUTPUT_REPOSITORY = "openbmb/MiniCPM-o-4_5"
VOICE_OUTPUT_REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"


@dataclass(frozen=True)
class ComponentFile:
    path: str
    size: int
    sha256: str


VOICE_OUTPUT_FILES = (
    ComponentFile(
        "assets/system_ref_audio.wav",
        539_032,
        "2c4109b2d685e1923ed66433eb08c92047a1f67510629a27edf49af4e5c606dd",
    ),
    ComponentFile(
        "assets/token2wav/campplus.onnx",
        28_303_423,
        "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73",
    ),
    ComponentFile(
        "assets/token2wav/flow.pt",
        623_466_603,
        "15ccff24256ff61537c7f8b51e025116b83405f3fb017b54b008fc97da115446",
    ),
    ComponentFile(
        "assets/token2wav/flow.yaml",
        1_099,
        "723295d37bf11f5f1b896ca4f2f4c81ebc2fbb3e51b753c2507ef8461c751486",
    ),
    ComponentFile(
        "assets/token2wav/hift.pt",
        83_390_254,
        "3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879",
    ),
    ComponentFile(
        "assets/token2wav/speech_tokenizer_v2_25hz.onnx",
        496_082_973,
        "d43342aa12163a80bf07bffb94c9de2e120a8df2f9917cd2f642e7f4219c6f71",
    ),
)
VOICE_OUTPUT_TOTAL_BYTES = sum(item.size for item in VOICE_OUTPUT_FILES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class VoiceOutputComponent:
    """Install and verify the official MiniCPM-o Token2Wav assets."""

    def __init__(self, work_root: str | Path) -> None:
        self.root = Path(work_root).expanduser().resolve() / "components" / VOICE_OUTPUT_COMPONENT_ID
        self._manifest = self.root / "manifest.json"
        self._install_lock = asyncio.Lock()
        self._installing = False
        self._last_error: str | None = None

    def ready(self) -> bool:
        try:
            manifest = json.loads(self._manifest.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return False
        if (
            manifest.get("component") != VOICE_OUTPUT_COMPONENT_ID
            or manifest.get("revision") != VOICE_OUTPUT_REVISION
        ):
            return False
        return all(
            (self.root / item.path).is_file()
            and (self.root / item.path).stat().st_size == item.size
            for item in VOICE_OUTPUT_FILES
        )

    def downloaded_bytes(self) -> int:
        total = 0
        for item in VOICE_OUTPUT_FILES:
            target = self.root / item.path
            partial = target.with_name(target.name + ".part")
            candidate = target if target.is_file() else partial
            if candidate.is_file():
                total += min(candidate.stat().st_size, item.size)
        return total

    def status(self) -> dict[str, Any]:
        ready = self.ready()
        return {
            "id": VOICE_OUTPUT_COMPONENT_ID,
            "state": "ready" if ready else "installing" if self._installing else "missing",
            "ready": ready,
            "installed_bytes": VOICE_OUTPUT_TOTAL_BYTES if ready else self.downloaded_bytes(),
            "total_bytes": VOICE_OUTPUT_TOTAL_BYTES,
            "repository": VOICE_OUTPUT_REPOSITORY,
            "revision": VOICE_OUTPUT_REVISION,
            "error": self._last_error,
        }

    async def install(self, context: JobContext) -> dict[str, Any]:
        async with self._install_lock:
            self._installing = True
            self._last_error = None
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                await context.progress(0.001, message="Preparing voice output component")
                completed = 0
                timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=timeout,
                    trust_env=True,
                    headers={"User-Agent": "MFQ-Studio/voice-component"},
                ) as client:
                    for item in VOICE_OUTPUT_FILES:
                        context.raise_if_cancelled()
                        target = self.root / item.path
                        if await self._valid_file(target, item):
                            completed += item.size
                            await context.progress(
                                completed / VOICE_OUTPUT_TOTAL_BYTES,
                                message=f"Verified {Path(item.path).name}",
                            )
                            continue
                        if target.exists():
                            target.unlink()
                        await self._download_file(context, client, item, completed)
                        completed += item.size

                manifest = {
                    "component": VOICE_OUTPUT_COMPONENT_ID,
                    "repository": VOICE_OUTPUT_REPOSITORY,
                    "revision": VOICE_OUTPUT_REVISION,
                    "total_bytes": VOICE_OUTPUT_TOTAL_BYTES,
                    "files": [item.__dict__ for item in VOICE_OUTPUT_FILES],
                }
                temporary = self._manifest.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, self._manifest)
                await context.progress(1.0, message="Voice output component ready")
                return {
                    **self.status(),
                    "state": "ready",
                    "ready": True,
                    "installed_bytes": VOICE_OUTPUT_TOTAL_BYTES,
                }
            except JobExecutionError:
                raise
            except Exception as error:
                self._last_error = str(error)
                raise JobExecutionError(
                    ErrorDetail(
                        code="voice_component_download_failed",
                        message=f"voice output component download failed: {error}",
                        retryable=True,
                    )
                ) from error
            finally:
                self._installing = False

    async def _valid_file(self, target: Path, item: ComponentFile) -> bool:
        if not target.is_file() or target.stat().st_size != item.size:
            return False
        return await asyncio.to_thread(_sha256, target) == item.sha256

    async def _download_file(
        self,
        context: JobContext,
        client: httpx.AsyncClient,
        item: ComponentFile,
        completed: int,
    ) -> None:
        target = self.root / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        if partial.is_file() and partial.stat().st_size > item.size:
            partial.unlink()
        url_path = quote(item.path, safe="/")
        url = (
            f"https://huggingface.co/{VOICE_OUTPUT_REPOSITORY}/resolve/"
            f"{VOICE_OUTPUT_REVISION}/{url_path}?download=true"
        )
        last_reported = -1
        last_report_time = 0.0
        for verification_attempt in range(2):
            for transfer_attempt in range(8):
                context.raise_if_cancelled()
                offset = partial.stat().st_size if partial.is_file() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                try:
                    async with client.stream("GET", url, headers=headers) as response:
                        if response.status_code == 416:
                            if offset == item.size:
                                os.replace(partial, target)
                                break
                            partial.unlink(missing_ok=True)
                            raise OSError(
                                f"server rejected the partial range for {item.path}"
                            )
                        response.raise_for_status()
                        append = response.status_code == 206 and offset > 0
                        if not append:
                            offset = 0
                        with partial.open("ab" if append else "wb") as handle:
                            downloaded = offset
                            async for chunk in response.aiter_bytes(4 * 1024 * 1024):
                                context.raise_if_cancelled()
                                handle.write(chunk)
                                downloaded += len(chunk)
                                now = time.monotonic()
                                overall = completed + min(downloaded, item.size)
                                if (
                                    overall - last_reported >= 16 * 1024 * 1024
                                    or now - last_report_time >= 1
                                ):
                                    last_reported = overall
                                    last_report_time = now
                                    await context.progress(
                                        min(0.999, overall / VOICE_OUTPUT_TOTAL_BYTES),
                                        message=f"Downloading {Path(item.path).name}",
                                        data={
                                            "downloaded_bytes": overall,
                                            "total_bytes": VOICE_OUTPUT_TOTAL_BYTES,
                                        },
                                    )
                        actual_size = partial.stat().st_size
                        if actual_size != item.size:
                            raise OSError(
                                f"{item.path} has {actual_size} bytes; expected {item.size}"
                            )
                        os.replace(partial, target)
                        break
                except (httpx.HTTPError, OSError) as error:
                    if partial.is_file() and partial.stat().st_size > item.size:
                        partial.unlink()
                    if transfer_attempt == 7:
                        raise
                    resumed = partial.stat().st_size if partial.is_file() else 0
                    await context.progress(
                        min(
                            0.999,
                            (completed + min(resumed, item.size))
                            / VOICE_OUTPUT_TOTAL_BYTES,
                        ),
                        message=f"Resuming {Path(item.path).name}",
                        data={
                            "downloaded_bytes": completed + min(resumed, item.size),
                            "total_bytes": VOICE_OUTPUT_TOTAL_BYTES,
                            "retry": transfer_attempt + 1,
                            "error": str(error),
                        },
                    )
                    await asyncio.sleep(min(0.5 * (2**transfer_attempt), 5.0))
            if await self._valid_file(target, item):
                return
            target.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
            if verification_attempt == 1:
                raise OSError(f"SHA-256 verification failed for {item.path}")
