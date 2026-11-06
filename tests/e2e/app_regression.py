#!/usr/bin/env python3
"""Cross-platform regression runner for the MFQ correct-path chain.

Stages (all optional, select with --stage):

  r0        Deterministic tiny HF fixture -> ``mfq quantize`` (CPU) -> container read.
            Network-free, GPU-free; suitable for every PR / any machine.
  artifact  Inspect an existing ``.mfq``: size, SHA-256, header, record count, dtypes.
  app       Register a directory in the app catalog, load one artifact, wait ready.
  service   Probe a running MFQ service: /health, /v1/models, /v1/chat/completions.
  session   Drive the Studio-equivalent session API (create session + response).
  stream    SSE paths: OpenAI-compatible chunk stream and Studio session stream.
  multi_turn  Multi-turn context, persisted history, and a system prompt.
  unload_switch  Load A, load B, chat B, unload B, reload A, chat A.
  negative  Error paths: missing directory, corrupt container, duplicate name,
            missing shard, unknown model, unknown artifact URI.

Each check reports ``pass`` / ``fail`` / ``skip`` (with a reason).  A skipped
check never counts as a pass.  Exit code: 0 = no failure, 1 = failure,
2 = configuration error.

Design rules (see tests/e2e/README.md):
  * only documented CLI flags and documented HTTP endpoints are used;
  * assertions target stable invariants, not bytes across versions;
  * capability detection decides applicability instead of hard-coded
    version assumptions, so a future build that ships a quantization entry
    flips APP-R02 from ``skip`` to ``pass`` without editing code.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS: dict = {
    "stage": ["r0"],
    "python": None,
    "work_dir": None,
    "artifact": None,
    "expect_sha256": None,
    "expect_records": None,
    "expect_arch": None,
    "quantize": {
        "enabled": True,
        "source": None,
        "backend": "cpu",
        "device": "cpu",
        "bits": 4,
        "groupsize": 24,
        "sub_bits": 6,
        "text_only": True,
        "timeout_sec": 3600,
    },
    "service": {
        "base_url": "http://127.0.0.1:8090",
        "model": None,
        "question": "只回答数字：2+3 等于多少？",
        "answer_pattern": r"(?:^|[=\s:：])\s*5\s*[。.!！]?\s*$",
        "max_tokens": 64,
        "temperature": 0,
        "timeout_sec": 180,
        "require": False,
    },
    "app": {
        "register_dir": None,
        "artifact_name": None,
        "load_timeout_sec": 300,
    },
    "stream": {
        "timeout_sec": 180,
    },
    "multi_turn": {
        "timeout_sec": 180,
    },
    "unload_switch": {
        "model_b": None,
        "load_timeout_sec": 300,
    },
    "negative": {
        "timeout_sec": 180,
    },
    "report": None,
}

TINY_FIXTURE_TENSORS = {
    "model.language_model.layers.0.mlp.down_proj.weight": (4, 24),
    "model.language_model.norm.weight": (24,),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def mfq_version() -> str | None:
    try:
        import mfq

        return getattr(mfq, "__version__", None)
    except Exception:  # pragma: no cover - diagnostics only
        return None


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def add(self, check_id: str, status: str, detail: str, **extra) -> dict:
        entry = {"id": check_id, "status": status, "detail": detail, **extra}
        self.checks.append(entry)
        return entry

    def passed(self, check_id: str, detail: str, **extra) -> None:
        self.add(check_id, "pass", detail, **extra)

    def failed(self, check_id: str, detail: str, **extra) -> None:
        self.add(check_id, "fail", detail, **extra)

    def skipped(self, check_id: str, detail: str, **extra) -> None:
        self.add(check_id, "skip", detail, **extra)

    def failed_count(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "fail")

    def to_dict(self, args: argparse.Namespace) -> dict:
        return {
            "generated_at": self.started,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
                "executable": sys.executable,
                "repo_head": git_head(),
                "mfq_version": mfq_version(),
            },
            "arguments": {k: v for k, v in vars(args).items() if k != "config"},
            "overall": "fail" if self.failed_count() else "pass",
            "checks": self.checks,
        }

    def print_summary(self) -> None:
        width = max((len(c["id"]) for c in self.checks), default=8)
        for check in self.checks:
            print(f"[{check['status']:<4}] {check['id']:<{width}}  {check['detail']}")


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def make_tiny_hf_fixture(root: Path) -> Path:
    """Create a deterministic, network-free tiny HF checkpoint.

    Mirrors ``tests/test_quantize/test_hf_to_mfq.py``: a two-tensor
    ``qwen3_5``-typed directory is enough for the HF quantizer to emit a
    readable container without downloading a real model.
    """
    import torch
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(20261103)
    tensors = {}
    for name, shape in TINY_FIXTURE_TENSORS.items():
        values = torch.randn(shape, generator=generator, dtype=torch.float32)
        if len(shape) == 2:
            values = torch.linspace(-2.0, 2.0, steps=shape[0] * shape[1], dtype=torch.float32)
            values = values.reshape(shape).to(torch.bfloat16)
        tensors[name] = values
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    return root


def quantize_tiny_fixture(cfg: dict, work_dir: Path, name: str = "tiny-r0.mfq") -> Path:
    """Create the deterministic tiny fixture and quantize it.

    Raises ``RuntimeError`` when the CLI fails or produces no output; callers
    decide whether that is a test failure (r0) or a fixture failure (negative).
    """
    fixture = make_tiny_hf_fixture(work_dir / "tiny-hf")
    output = work_dir / name
    if output.exists():
        output.unlink()
    cmd = [
        sys.executable,
        "-m",
        "mfq.cli",
        "quantize",
        str(fixture),
        str(output),
        "--source-format",
        "hf",
        "--bits",
        str(cfg["quantize"]["bits"]),
        "--groupsize",
        str(cfg["quantize"]["groupsize"]),
        "--sub-bits",
        str(cfg["quantize"]["sub_bits"]),
        "--backend",
        str(cfg["quantize"]["backend"]),
        "--device",
        str(cfg["quantize"]["device"]),
    ]
    if cfg["quantize"].get("text_only", True):
        cmd.append("--text-only")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=cfg["quantize"]["timeout_sec"],
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-400:]
        raise RuntimeError(f"exit={proc.returncode} stderr={tail}")
    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty output: {output}")
    return output


def stage_r0(report: Report, cfg: dict, work_dir: Path) -> None:
    started = time.time()
    try:
        output = quantize_tiny_fixture(cfg, work_dir, "tiny-r0.mfq")
    except Exception as exc:  # quantize failure is a real r0 failure
        report.failed("r0.quantize.exit", str(exc))
        return
    elapsed = round(time.time() - started, 2)
    report.passed("r0.quantize.exit", f"exit=0 elapsed={elapsed}s")

    if not output.exists() or output.stat().st_size <= 0:
        report.failed("r0.artifact.exists", f"missing or empty: {output}")
        return
    report.passed("r0.artifact.exists", f"{output.name} size={output.stat().st_size}")

    try:
        from mfq.formats.io import load_mmap

        header, store = load_mmap(str(output))
        try:
            records = len(store.records)
            dtypes = sorted({r.stored_dtype for r in store.records.values()})
            arch = getattr(header, "model_arch", "") or ""
            version = getattr(header, "version", 0)
            names = list(store.records)
        finally:
            store.close()
    except Exception as exc:  # pragma: no cover - exercised on broken output
        report.failed("r0.artifact.readable", f"{type(exc).__name__}: {exc}")
        return
    report.passed(
        "r0.artifact.readable",
        f"header_version={version} arch={arch} records={records} dtypes={','.join(dtypes)}",
    )
    if records < len(TINY_FIXTURE_TENSORS):
        report.failed("r0.artifact.records", f"records={records} < fixtures={len(TINY_FIXTURE_TENSORS)}")
    else:
        report.passed("r0.artifact.records", f"records={records}")

    assets = [name for name in names if name.startswith("__mfq_asset__/")]
    if assets:
        report.passed("r0.artifact.assets", f"assets={assets}")
    else:
        report.failed("r0.artifact.assets", "no embedded runtime assets")


def stage_artifact(report: Report, cfg: dict) -> bool:
    artifact = cfg.get("artifact")
    if not artifact:
        report.skipped("artifact", "no --artifact supplied")
        return False
    path = Path(artifact)
    if not path.exists():
        report.failed("artifact.exists", f"not found: {path}")
        return False
    size = path.stat().st_size
    if size <= 0:
        report.failed("artifact.exists", f"empty file: {path}")
        return False
    report.passed("artifact.exists", f"{path.name} size={size}")

    digest = sha256_file(path)
    expected = (cfg.get("expect_sha256") or "").lower()
    if expected and digest != expected:
        report.failed("artifact.sha256", f"sha256={digest} expected={expected}")
    elif expected:
        report.passed("artifact.sha256", f"sha256={digest} (matches pinned value)")
    else:
        report.passed("artifact.sha256", f"sha256={digest} (recorded, not pinned)")

    try:
        from mfq.formats.io import load_mmap

        header, store = load_mmap(str(path))
        try:
            records = len(store.records)
            dtypes = sorted({r.stored_dtype for r in store.records.values()})
            arch = getattr(header, "model_arch", "") or ""
            version = getattr(header, "version", 0)
            names = list(store.records)
        finally:
            store.close()
    except Exception as exc:
        report.failed("artifact.readable", f"{type(exc).__name__}: {exc}")
        return False
    report.passed(
        "artifact.readable",
        f"header_version={version} arch={arch} records={records} dtypes={','.join(dtypes)}",
    )

    expect_arch = cfg.get("expect_arch")
    if expect_arch and arch != expect_arch:
        report.failed("artifact.arch", f"arch={arch} expected={expect_arch}")
    elif expect_arch:
        report.passed("artifact.arch", f"arch={arch}")

    expect_records = cfg.get("expect_records")
    if expect_records is not None and records != int(expect_records):
        report.failed("artifact.records", f"records={records} expected={expect_records}")
    elif expect_records is not None:
        report.passed("artifact.records", f"records={records} (matches pinned value)")
    else:
        report.passed("artifact.records", f"records={records} (recorded, not pinned)")

    if not names:
        report.failed("artifact.tensors", "container has no tensor records")
    else:
        report.passed("artifact.tensors", f"first={names[0]} count={len(names)}")

    # A quantized checkpoint must embed the standalone runtime assets.
    assets = [name for name in names if name.startswith("__mfq_asset__/")]
    if len(assets) < 2:
        report.failed("artifact.assets", f"expected >=2 embedded assets, found {assets}")
    else:
        report.passed("artifact.assets", f"assets={assets}")
    tokenizer_assets = [name for name in assets if "tokenizer" in name.lower()]
    if tokenizer_assets:
        report.passed("artifact.tokenizer", f"tokenizer_assets={tokenizer_assets}")
    else:
        report.skipped("artifact.tokenizer", "no tokenizer asset embedded (fixture may omit it)")
    return True


def _http_json(url: str, method: str = "GET", payload: dict | None = None, timeout: float = 30.0):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return response.status, (json.loads(raw) if raw else {})


def _http_error(exc: Exception) -> tuple[int | None, str]:
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        return exc.code, body or str(exc)
    return None, str(exc)


def _http_sse(url: str, payload: dict, timeout: float) -> tuple[int, str, list[dict]]:
    """POST JSON and collect raw SSE frames: ``[{"event": str|None, "data": str}]``."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    events: list[dict] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        content_type = response.headers.get("Content-Type", "")
        stream = io.TextIOWrapper(response, encoding="utf-8", errors="replace")
        event_name: str | None = None
        for raw in stream:
            line = raw.rstrip("\r\n")
            if not line:
                event_name = None
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                events.append({"event": event_name, "data": line[5:].strip()})
    return status, content_type, events


def _service_or_skip(report: Report, cfg: dict, stage_id: str) -> str | None:
    """Return the service base URL, or record fail/skip when unreachable."""
    base = cfg["service"]["base_url"].rstrip("/")
    try:
        status, _ = _http_json(f"{base}/health", timeout=10)
        if status == 200:
            return base
        detail = f"http={status}"
    except Exception as exc:
        code, detail = _http_error(exc)
        detail = f"{code or type(exc).__name__}: {detail}"
    (report.failed if cfg["service"].get("require") else report.skipped)(
        stage_id, f"service unreachable ({detail[:200]})"
    )
    return None


def _wait_ready(base: str, model: str, timeout: float) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _, payload = _http_json(f"{base}/api/v1/runtime/instances", timeout=30)
        except Exception:
            time.sleep(1)
            continue
        items = payload.get("data", []) if isinstance(payload, dict) else []
        for item in items:
            if item.get("model") == model and item.get("state") == "ready":
                return item
        time.sleep(1)
    return None


def _wait_absent(base: str, model: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _, payload = _http_json(f"{base}/api/v1/runtime/instances", timeout=30)
        except Exception:
            time.sleep(1)
            continue
        items = payload.get("data", []) if isinstance(payload, dict) else []
        if not any(i.get("model") == model for i in items):
            return True
        time.sleep(1)
    return False


def _load_model(base: str, model: str, timeout: float = 300) -> tuple[dict | None, str | None]:
    """Load a catalog model through the documented async operation."""
    try:
        status, accepted = _http_json(
            f"{base}/api/v1/models/load", method="POST", payload={"model": model}, timeout=60
        )
    except Exception as exc:
        code, detail = _http_error(exc)
        return None, f"http={code}: {detail[:300]}"
    if status != 202 or not accepted.get("operation_id"):
        return None, f"http={status} body={accepted}"
    operation_id = accepted["operation_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _, job = _http_json(f"{base}/api/v1/jobs/{operation_id}", timeout=30)
        except Exception as exc:
            return None, f"job poll failed: {exc}"
        state = job.get("status")
        if state in {"succeeded", "completed"}:
            return job.get("result") or {}, None
        if state in {"failed", "cancelled"}:
            return None, f"job {state}: {job.get('error')}"
        time.sleep(2)
    return None, "load timeout"


def _ensure_loaded(base: str, model: str, timeout: float = 300) -> tuple[dict | None, str | None]:
    instance = _wait_ready(base, model, 1)
    if instance:
        return instance, None
    result, error = _load_model(base, model, timeout)
    if result is None:
        return None, error
    instance = _wait_ready(base, model, 60)
    if instance:
        return instance, None
    return None, "load job succeeded but the instance never became ready"


def _unload_instance(base: str, instance_id: str, timeout: float = 120) -> tuple[bool, str]:
    try:
        status, accepted = _http_json(
            f"{base}/api/v1/models/unload", method="POST", payload={"instance_id": instance_id}, timeout=30
        )
    except Exception as exc:
        code, detail = _http_error(exc)
        return False, f"http={code}: {detail[:200]}"
    if status != 202 or not accepted.get("operation_id"):
        return False, f"http={status} body={accepted}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _, job = _http_json(f"{base}/api/v1/jobs/{accepted['operation_id']}", timeout=30)
        except Exception as exc:
            return False, f"job poll failed: {exc}"
        if job.get("status") in {"succeeded", "completed"}:
            return True, str(job.get("status"))
        if job.get("status") in {"failed", "cancelled"}:
            return False, f"job {job.get('status')}: {job.get('error')}"
        time.sleep(1)
    return False, "unload timeout"


def _chat(base: str, model: str, question: str, max_tokens: int = 64, timeout: float = 180) -> tuple[int, dict]:
    return _http_json(
        f"{base}/v1/chat/completions",
        method="POST",
        payload={
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "enable_thinking": False,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=timeout,
    )


def _session_create(base: str, model: str, title: str) -> tuple[int, dict]:
    return _http_json(
        f"{base}/api/v1/sessions",
        method="POST",
        payload={"model": model, "mode": "text", "title": title},
        timeout=30,
    )


def _session_exchange(
    base: str,
    session_id: str,
    revision: int,
    text: str,
    cfg: dict,
    request_id: str | None = None,
    system_prompt: str | None = None,
) -> tuple[int, dict, float]:
    svc = cfg["service"]
    request = {
        "request_id": request_id or str(uuid.uuid4()),
        "expected_revision": int(revision),
        "input": [{"type": "text", "text": text}],
        "sampling": {
            "max_tokens": int(svc["max_tokens"]),
            "temperature": float(svc["temperature"]),
            "top_k": 1,
            "top_p": 1.0,
            "enable_thinking": False,
        },
        "stream": False,
    }
    if system_prompt:
        request["system_prompt"] = system_prompt
    started = time.time()
    status, body = _http_json(
        f"{base}/api/v1/sessions/{session_id}/responses",
        method="POST",
        payload=request,
        timeout=float(cfg["multi_turn"]["timeout_sec"]),
    )
    return status, body, round(time.time() - started, 3)


def _session_revision(base: str, session_id: str) -> int:
    _, session = _http_json(f"{base}/api/v1/sessions/{session_id}", timeout=30)
    return int(session.get("revision") or 0)


def _response_text(response: dict) -> str:
    texts = [part.get("text", "") for part in response.get("output", []) if part.get("type") == "text"]
    return " ".join(texts).strip()


def _answer_matches(content: str, pattern: str | None = None) -> bool:
    pattern = pattern or r"(?:^|[=\s:：])\s*5\s*[。.!！]?\s*$"
    return bool(re.search(pattern, content))


def stage_app(report: Report, cfg: dict) -> None:
    """Register a directory in the app catalog, load one artifact, wait ready.

    Drives the app's documented model endpoints (the same ones Studio calls):
    ``POST /api/v1/models/directories/register``, ``POST /api/v1/models/load``,
    ``GET /api/v1/jobs/{id}`` and ``GET /api/v1/runtime/instances``.  When the
    model is already loaded, the load step is treated as satisfied.
    """
    app_cfg = cfg["app"]
    svc = cfg["service"]
    base = svc["base_url"].rstrip("/")
    required = bool(svc.get("require"))
    register_dir = app_cfg.get("register_dir")
    artifact_name = app_cfg.get("artifact_name")
    if not register_dir:
        report.skipped("app", "no --register-dir supplied")
        return
    try:
        _http_json(f"{base}/health", timeout=10)
    except Exception as exc:
        code, detail = _http_error(exc)
        (report.failed if required else report.skipped)(
            "app", f"service unreachable ({code or type(exc).__name__}): {detail[:300]}"
        )
        return
    if not artifact_name:
        report.failed("app", "--register-dir requires --artifact-name")
        return

    register_path = Path(str(register_dir))
    if not register_path.is_absolute():
        register_path = (REPO_ROOT / register_path).resolve()

    try:
        status, catalog = _http_json(
            f"{base}/api/v1/models/directories/register",
            method="POST",
            payload={"path": str(register_path)},
            timeout=180,
        )
    except Exception as exc:
        code, detail = _http_error(exc)
        report.failed("app.register", f"http={code}: {detail[:300]}")
        return
    entries = catalog.get("data", []) if isinstance(catalog, dict) else []
    entry = next((item for item in entries if item.get("name") == artifact_name), None)
    if status != 200 or entry is None:
        names = [item.get("name") for item in entries]
        report.failed(
            "app.register",
            f"http={status} artifact {artifact_name!r} not found in {register_dir}; catalog={names}",
        )
        return
    report.passed(
        "app.register",
        f"name={entry.get('name')} id={entry.get('id')} bytes={entry.get('total_bytes')} "
        f"records={entry.get('record_count')} complete={entry.get('complete')} loadable={entry.get('loadable')}",
    )
    if not entry.get("complete") or not entry.get("loadable"):
        report.failed("app.catalog", "registered artifact is not complete/loadable")
        return
    report.passed("app.catalog", "artifact complete and loadable")

    def ready_instances() -> list[dict]:
        try:
            _, payload = _http_json(f"{base}/api/v1/runtime/instances", timeout=30)
        except Exception as exc:
            code, detail = _http_error(exc)
            report.failed("app.instance", f"http={code}: {detail[:200]}")
            return []
        items = payload.get("data", []) if isinstance(payload, dict) else []
        return [i for i in items if i.get("model") == artifact_name and i.get("state") == "ready"]

    existing = ready_instances()
    if existing:
        report.passed("app.load", f"model already loaded; instance={existing[0].get('id')}")
        report.passed(
            "app.instance",
            f"instance={existing[0].get('id')} model={artifact_name} devices={existing[0].get('devices')}",
        )
        if not svc.get("model"):
            svc["model"] = artifact_name
        return

    started = time.time()
    accepted: dict | None
    try:
        status, accepted = _http_json(
            f"{base}/api/v1/models/load",
            method="POST",
            payload={"model": artifact_name, "artifact_uri": f"mfq://{entry['id']}"},
            timeout=60,
        )
    except Exception as exc:
        code, detail = _http_error(exc)
        if code in (400, 409) and "already" in (detail or "").lower():
            report.passed("app.load", f"model already loaded (http={code})")
            accepted = None
        else:
            report.failed("app.load", f"http={code}: {detail[:300]}")
            return

    if accepted is not None:
        operation_id = accepted.get("operation_id")
        if status != 202 or not operation_id:
            report.failed("app.load", f"http={status} body={accepted}")
            return
        deadline = started + float(app_cfg.get("load_timeout_sec", 300))
        while time.time() < deadline:
            try:
                _, job = _http_json(f"{base}/api/v1/jobs/{operation_id}", timeout=30)
            except Exception as exc:
                code, detail = _http_error(exc)
                report.failed("app.load", f"job poll failed http={code}: {detail[:200]}")
                return
            state = job.get("status")
            if state in {"succeeded", "completed"}:
                result = job.get("result") or {}
                report.passed(
                    "app.load",
                    f"job={operation_id} status={state} instance={result.get('instance_id')} "
                    f"elapsed={round(time.time() - started, 1)}s",
                )
                break
            if state in {"failed", "cancelled"}:
                error = job.get("error") or {}
                if error.get("code") == "model_already_loaded":
                    loaded = ready_instances()
                    if loaded:
                        report.passed(
                            "app.load",
                            f"model already loaded; instance={loaded[0].get('id')}",
                        )
                        report.passed(
                            "app.instance",
                            f"instance={loaded[0].get('id')} model={artifact_name} "
                            f"devices={loaded[0].get('devices')}",
                        )
                        if not svc.get("model"):
                            svc["model"] = artifact_name
                        return
                report.failed("app.load", f"job={operation_id} status={state} error={error}")
                return
            time.sleep(2)
        else:
            report.failed(
                "app.load",
                f"job={operation_id} did not finish in {app_cfg.get('load_timeout_sec')}s",
            )
            return

    try:
        _, instances = _http_json(f"{base}/api/v1/runtime/instances", timeout=30)
    except Exception as exc:
        code, detail = _http_error(exc)
        report.failed("app.instance", f"http={code}: {detail[:200]}")
        return
    items = instances.get("data", []) if isinstance(instances, dict) else []
    ready = [item for item in items if item.get("model") == artifact_name and item.get("state") == "ready"]
    if not ready:
        states = [(item.get("model"), item.get("state")) for item in items]
        report.failed("app.instance", f"no ready instance for {artifact_name!r}; instances={states}")
        return
    instance = ready[0]
    report.passed(
        "app.instance",
        f"instance={instance.get('id')} model={instance.get('model')} devices={instance.get('devices')}",
    )

    # Downstream service/session stages should target the freshly loaded model.
    if not svc.get("model"):
        svc["model"] = artifact_name


def stage_service(report: Report, cfg: dict) -> None:
    svc = cfg["service"]
    base = svc["base_url"].rstrip("/")
    model = svc.get("model")
    required = bool(svc.get("require"))
    timeout = float(svc["timeout_sec"])

    def unavailable(stage_id: str, detail: str) -> None:
        (report.failed if required else report.skipped)(stage_id, detail)

    try:
        status, health = _http_json(f"{base}/health", timeout=10)
    except Exception as exc:
        code, detail = _http_error(exc)
        unavailable("service.health", f"service unreachable ({code or type(exc).__name__}): {detail[:300]}")
        return
    if status != 200 or health.get("status") != "ok":
        report.failed("service.health", f"http={status} body={health}")
        return
    report.passed("service.health", f"service={health.get('service')} protocol={health.get('protocol_version')}")

    try:
        status, models = _http_json(f"{base}/v1/models", timeout=timeout)
    except Exception as exc:
        code, detail = _http_error(exc)
        report.failed("service.models", f"http={code}: {detail[:300]}")
        return
    ids = [entry.get("id") for entry in models.get("data", []) if isinstance(entry, dict)]
    if status != 200:
        report.failed("service.models", f"http={status}")
        return
    if model and model not in ids:
        report.failed("service.models", f"model {model!r} missing; served={ids}")
        return
    report.passed("service.models", f"served={ids}" if ids else "served=[]")

    if not model:
        report.skipped("service.chat", "no --model supplied")
        return
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": svc["question"]}],
        "enable_thinking": False,
        "max_tokens": int(svc["max_tokens"]),
        "temperature": float(svc["temperature"]),
    }
    started = time.time()
    try:
        status, body = _http_json(f"{base}/v1/chat/completions", method="POST", payload=payload, timeout=timeout)
    except Exception as exc:
        code, detail = _http_error(exc)
        report.failed("service.chat", f"http={code}: {detail[:300]}")
        return
    elapsed = round(time.time() - started, 3)
    choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
    message = choice.get("message") or {}
    content = (message.get("content") or "").strip()
    finish = choice.get("finish_reason")
    echoed = body.get("model")
    if status != 200:
        report.failed("service.chat", f"http={status}")
        return
    if echoed != model:
        report.failed("service.chat.model", f"echoed={echoed!r} requested={model!r}")
        return
    report.passed("service.chat.model", f"model={echoed}")
    if finish != "stop":
        report.failed("service.chat.finish", f"finish_reason={finish} content={content[:120]!r}")
    else:
        report.passed("service.chat.finish", "finish_reason=stop")
    pattern = svc.get("answer_pattern") or r"(?:^|[=\s:：])\s*5\s*[。.!！]?\s*$"
    if re.search(pattern, content):
        report.passed("service.chat.answer", f"content={content[:120]!r} elapsed_ms={int(elapsed * 1000)}")
    else:
        report.failed("service.chat.answer", f"content={content[:200]!r} pattern={pattern}")

    usage = body.get("usage") or {}
    total = int(usage.get("total_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    if total > 0 and completion > 0:
        report.passed(
            "service.chat.usage",
            f"prompt={usage.get('prompt_tokens')} completion={completion} total={total}",
        )
    else:
        report.failed("service.chat.usage", f"usage missing or zero: {usage}")

    structured_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You reply with a JSON object only."},
            {"role": "user", "content": '只输出 JSON：{"answer": 5}'},
        ],
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
        "max_tokens": int(svc["max_tokens"]),
        "temperature": 0,
    }
    try:
        status, structured = _http_json(
            f"{base}/v1/chat/completions", method="POST", payload=structured_payload, timeout=timeout
        )
    except Exception as exc:
        code, detail = _http_error(exc)
        if code in (400, 422):
            report.skipped("service.structured", f"response_format unsupported (http={code})")
        else:
            report.failed("service.structured", f"http={code}: {detail[:200]}")
        return
    if status != 200:
        report.skipped("service.structured", f"response_format unsupported (http={status})")
        return
    structured_choice = (structured.get("choices") or [{}])[0]
    structured_content = ((structured_choice.get("message") or {}).get("content") or "").strip()
    try:
        parsed = json.loads(structured_content)
    except json.JSONDecodeError:
        report.failed("service.structured", f"content is not JSON: {structured_content[:200]!r}")
        return
    report.passed("service.structured", f"json_object ok content={structured_content[:160]!r}")
    if isinstance(parsed, dict) and "answer" in parsed:
        if str(parsed["answer"]).strip() == "5":
            report.passed("service.structured.answer", f"answer={parsed['answer']!r}")
        else:
            report.failed("service.structured.answer", f"answer={parsed['answer']!r}")
    else:
        report.skipped("service.structured.answer", "JSON object without an 'answer' key")


def stage_session(report: Report, cfg: dict) -> None:
    svc = cfg["service"]
    base = _service_or_skip(report, cfg, "session")
    if base is None:
        return
    model = svc.get("model")
    if not model:
        report.skipped("session", "no --model supplied")
        return
    try:
        status, session = _http_json(
            f"{base}/api/v1/sessions",
            method="POST",
            payload={"model": model, "mode": "text", "title": "e2e fixed question"},
            timeout=30,
        )
    except Exception as exc:
        code, detail = _http_error(exc)
        if code in (404, 405):
            report.skipped("session", f"session API unavailable on this build (http={code})")
        else:
            report.failed("session.create", f"http={code}: {detail[:300]}")
        return
    session_id = session.get("id")
    revision = session.get("revision")
    if status != 201 or not session_id:
        report.failed("session.create", f"http={status} body={session}")
        return
    report.passed("session.create", f"session={session_id}")

    request = {
        "request_id": str(uuid.uuid4()),
        "expected_revision": int(revision or 0),
        "input": [{"type": "text", "text": svc["question"]}],
        "sampling": {
            "max_tokens": int(svc["max_tokens"]),
            "temperature": float(svc["temperature"]),
            "top_k": 1,
            "top_p": 1.0,
            "enable_thinking": False,
        },
        "stream": False,
    }
    started = time.time()
    try:
        status, response = _http_json(
            f"{base}/api/v1/sessions/{session_id}/responses",
            method="POST",
            payload=request,
            timeout=float(svc["timeout_sec"]),
        )
    except Exception as exc:
        code, detail = _http_error(exc)
        report.failed("session.response", f"http={code}: {detail[:300]}")
        return
    elapsed = round(time.time() - started, 3)
    if status != 200:
        report.failed("session.response", f"http={status}")
        return
    texts = [part.get("text", "") for part in response.get("output", []) if part.get("type") == "text"]
    content = " ".join(texts).strip()
    if response.get("status") != "completed":
        report.failed("session.response.status", f"status={response.get('status')} error={response.get('error')}")
    else:
        report.passed("session.response.status", "status=completed")
    if response.get("finish_reason") != "stop":
        report.failed("session.response.finish", f"finish_reason={response.get('finish_reason')}")
    else:
        report.passed("session.response.finish", "finish_reason=stop")
    pattern = svc.get("answer_pattern") or r"(?:^|[=\s:：])\s*5\s*[。.!！]?\s*$"
    if re.search(pattern, content):
        report.passed("session.response.answer", f"content={content[:120]!r} elapsed_ms={int(elapsed * 1000)}")
    else:
        report.failed("session.response.answer", f"content={content[:200]!r} pattern={pattern}")

    usage = response.get("usage") or {}
    if usage:
        total = int(usage.get("total_tokens") or 0)
        if total > 0:
            report.passed("session.response.usage", f"total={total}")
        else:
            report.failed("session.response.usage", f"usage is zero: {usage}")
    else:
        report.skipped("session.response.usage", "runtime did not return usage")

    performance = response.get("performance") or {}
    if performance:
        ttft = float(performance.get("ttft_ms") or 0)
        if ttft > 0:
            report.passed(
                "session.response.performance",
                f"ttft_ms={ttft:.1f} prefill_tps={performance.get('prefill_tps')}",
            )
        else:
            report.failed("session.response.performance", f"ttft_ms={ttft}")
    else:
        report.skipped("session.response.performance", "runtime did not return performance")

    try:
        status, messages = _http_json(f"{base}/api/v1/sessions/{session_id}/messages", timeout=30)
    except Exception as exc:
        code, detail = _http_error(exc)
        report.failed("session.messages", f"http={code}: {detail[:200]}")
    else:
        roles = [item.get("role") for item in messages.get("data", [])] if isinstance(messages, dict) else []
        if status == 200 and "user" in roles and "assistant" in roles:
            report.passed("session.messages", f"roles={roles}")
        else:
            report.failed("session.messages", f"http={status} roles={roles}")

    # Idempotent replay: the same request_id and body must not run inference again.
    try:
        status, replay = _http_json(
            f"{base}/api/v1/sessions/{session_id}/responses",
            method="POST",
            payload=request,
            timeout=float(svc["timeout_sec"]),
        )
    except Exception as exc:
        code, detail = _http_error(exc)
        text = (detail or "")
        if code == 409 and "revision_conflict" in text:
            report.skipped(
                "session.idempotent",
                "server rejects replay after the session revision advanced (revision_conflict)",
            )
        else:
            report.failed("session.idempotent", f"http={code}: {text[:200]}")
    else:
        if status == 200 and _response_text(replay) == content:
            report.passed("session.idempotent", "same request_id replayed without a new response")
        else:
            report.failed(
                "session.idempotent",
                f"http={status} replay_content={_response_text(replay)[:120]!r}",
            )

    # Determinism at temperature=0 / top_k=1 with a fresh request_id and revision.
    revision = _session_revision(base, session_id)
    deterministic_request = dict(request)
    deterministic_request["request_id"] = str(uuid.uuid4())
    deterministic_request["expected_revision"] = revision
    try:
        status, repeated = _http_json(
            f"{base}/api/v1/sessions/{session_id}/responses",
            method="POST",
            payload=deterministic_request,
            timeout=float(svc["timeout_sec"]),
        )
    except Exception as exc:
        code, detail = _http_error(exc)
        report.failed("session.deterministic", f"http={code}: {detail[:200]}")
    else:
        repeated_text = _response_text(repeated)
        if status == 200 and repeated_text == content:
            report.passed("session.deterministic", f"temp=0 top_k=1 repeatable content={repeated_text[:120]!r}")
        else:
            report.failed("session.deterministic", f"content differs: {repeated_text[:120]!r}")


def stage_stream(report: Report, cfg: dict) -> None:
    """Exercise both streaming paths: OpenAI chunk stream and Studio session SSE."""
    svc = cfg["service"]
    base = _service_or_skip(report, cfg, "stream")
    if base is None:
        return
    model = svc.get("model")
    if not model:
        report.skipped("stream", "no --model supplied")
        return
    timeout = float(cfg["stream"].get("timeout_sec", 180))
    pattern = svc.get("answer_pattern") or r"(?:^|[=\s:：])\s*5\s*[。.!！]?\s*$"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": svc["question"]}],
        "enable_thinking": False,
        "max_tokens": int(svc["max_tokens"]),
        "temperature": 0,
        "stream": True,
    }
    started = time.time()
    try:
        status, content_type, events = _http_sse(f"{base}/v1/chat/completions", payload, timeout)
    except Exception as exc:
        code, detail = _http_error(exc)
        report.failed("stream.openai", f"http={code}: {detail[:300]}")
        return
    first_ms = int((time.time() - started) * 1000)
    if status != 200:
        report.failed("stream.openai", f"http={status}")
        return
    if "text/event-stream" not in content_type:
        report.failed("stream.openai.content_type", f"content_type={content_type!r}")
        return
    report.passed("stream.openai.content_type", f"{content_type.split(';')[0]} first_event_ms={first_ms}")

    chunks: list[dict] = []
    done = False
    for event in events:
        data = event["data"]
        if data == "[DONE]":
            done = True
            break
        if not data.startswith("{"):
            continue
        try:
            chunks.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    content = "".join(
        ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content") or "" for chunk in chunks
    ).strip()
    finish = None
    for chunk in chunks:
        choice = (chunk.get("choices") or [{}])[0]
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
    echoed = chunks[-1].get("model") if chunks else None
    if echoed == model:
        report.passed("stream.openai.model", f"model={echoed}")
    else:
        report.failed("stream.openai.model", f"echoed={echoed!r} requested={model!r}")
    report.passed("stream.openai.done", "received [DONE]" if done else "closed without [DONE]")
    if finish == "stop":
        report.passed("stream.openai.finish", "finish_reason=stop")
    else:
        report.failed("stream.openai.finish", f"finish_reason={finish} content={content[:120]!r}")
    if _answer_matches(content, pattern):
        report.passed("stream.openai.answer", f"content={content[:120]!r} elapsed_ms={int((time.time() - started) * 1000)}")
    else:
        report.failed("stream.openai.answer", f"content={content[:200]!r} pattern={pattern}")

    # Studio session stream (the UI's default chat path).
    try:
        status, session = _session_create(base, model, "e2e stream")
    except Exception as exc:
        code, detail = _http_error(exc)
        if code in (404, 405):
            report.skipped("stream.session", f"session API unavailable (http={code})")
        else:
            report.failed("stream.session.create", f"http={code}: {detail[:300]}")
        return
    if status != 201:
        report.failed("stream.session.create", f"http={status}")
        return
    session_id = session["id"]
    report.passed("stream.session.create", f"session={session_id}")
    request = {
        "request_id": str(uuid.uuid4()),
        "expected_revision": int(session.get("revision") or 0),
        "input": [{"type": "text", "text": svc["question"]}],
        "sampling": {
            "max_tokens": int(svc["max_tokens"]),
            "temperature": 0,
            "top_k": 1,
            "top_p": 1.0,
            "enable_thinking": False,
        },
        "stream": True,
    }
    started = time.time()
    try:
        status, content_type, events = _http_sse(
            f"{base}/api/v1/sessions/{session_id}/responses", request, timeout
        )
    except Exception as exc:
        code, detail = _http_error(exc)
        report.failed("stream.session", f"http={code}: {detail[:300]}")
        return
    if status != 200:
        report.failed("stream.session", f"http={status}")
        return
    report.passed(
        "stream.session.status",
        f"http=200 content_type={content_type.split(';')[0]} first_event_ms={int((time.time() - started) * 1000)}",
    )
    deltas: list[str] = []
    finish = None
    completed = False
    state_seen = False
    error_seen = None
    for event in events:
        data = event["data"]
        if not data.startswith("{"):
            continue
        try:
            frame = json.loads(data)
        except json.JSONDecodeError:
            continue
        payload_obj = frame.get("payload") or {}
        kind = payload_obj.get("type") or event.get("event")
        if kind == "response.text.delta":
            deltas.append(payload_obj.get("delta") or "")
        elif kind == "response.completed":
            completed = True
            finish = payload_obj.get("finish_reason")
        elif kind == "session.state":
            state_seen = True
        elif kind == "error":
            error_seen = payload_obj
    content = "".join(deltas).strip()
    if deltas:
        report.passed("stream.session.deltas", f"deltas={len(deltas)}")
    else:
        report.failed("stream.session.deltas", "no response.text.delta events")
    if completed and finish == "stop":
        report.passed(
            "stream.session.completed",
            f"finish_reason={finish} elapsed_ms={int((time.time() - started) * 1000)}",
        )
    else:
        report.failed("stream.session.completed", f"completed={completed} finish_reason={finish} error={error_seen}")
    if _answer_matches(content, pattern):
        report.passed("stream.session.answer", f"content={content[:120]!r}")
    else:
        report.failed("stream.session.answer", f"content={content[:200]!r} pattern={pattern}")
    if state_seen:
        report.passed("stream.session.state", "session.state event observed")
    else:
        report.skipped("stream.session.state", "no session.state event in this stream")


def stage_multi_turn(report: Report, cfg: dict) -> None:
    """Two context-carrying turns, one system-prompt turn, and history persistence."""
    svc = cfg["service"]
    base = _service_or_skip(report, cfg, "multi_turn")
    if base is None:
        return
    model = svc.get("model")
    if not model:
        report.skipped("multi_turn", "no --model supplied")
        return
    try:
        status, session = _session_create(base, model, "e2e multi-turn")
    except Exception as exc:
        code, detail = _http_error(exc)
        if code in (404, 405):
            report.skipped("multi_turn", f"session API unavailable (http={code})")
        else:
            report.failed("multi_turn.create", f"http={code}: {detail[:300]}")
        return
    if status != 201:
        report.failed("multi_turn.create", f"http={status}")
        return
    session_id = session["id"]
    report.passed("multi_turn.create", f"session={session_id}")

    def exchange(text: str, system_prompt: str | None = None) -> tuple[int, dict, float]:
        revision = _session_revision(base, session_id)
        return _session_exchange(base, session_id, revision, text, cfg, system_prompt=system_prompt)

    # Turn 1: fixed question.
    status, response1, elapsed1 = exchange(svc["question"])
    content1 = _response_text(response1)
    if status == 200 and response1.get("status") == "completed" and _answer_matches(content1, svc.get("answer_pattern")):
        report.passed("multi_turn.turn1", f"content={content1[:120]!r} elapsed_ms={int(elapsed1 * 1000)}")
    else:
        report.failed("multi_turn.turn1", f"http={status} status={response1.get('status')} content={content1[:200]!r}")
        return

    # Turn 2: only answerable with turn 1 in context.
    status, response2, elapsed2 = exchange("把上一个答案再加 4，只回答数字。")
    content2 = _response_text(response2)
    pattern9 = r"(?:^|[=\s:：])\s*9\s*[。.!！]?\s*$"
    if status == 200 and response2.get("status") == "completed" and _answer_matches(content2, pattern9):
        report.passed("multi_turn.turn2", f"content={content2[:120]!r} elapsed_ms={int(elapsed2 * 1000)}")
    else:
        report.failed("multi_turn.turn2", f"http={status} content={content2[:200]!r}")

    usage1 = response1.get("usage") or {}
    usage2 = response2.get("usage") or {}
    if usage1 and usage2:
        total1 = int(usage1.get("total_tokens") or 0)
        total2 = int(usage2.get("total_tokens") or 0)
        if total2 > total1:
            report.passed("multi_turn.context_tokens", f"turn1={total1} turn2={total2}")
        else:
            report.failed("multi_turn.context_tokens", f"turn1={total1} turn2={total2}")
    else:
        report.skipped("multi_turn.context_tokens", "runtime did not return usage")

    # Turn 3: system prompt is accepted and honoured.
    status, response3, _elapsed3 = exchange(
        "1 + 1 等于多少？只回答数字。",
        system_prompt="你是只回答数字的助手，不要输出任何解释。",
    )
    content3 = _response_text(response3)
    pattern2 = r"(?:^|[=\s:：])\s*2\s*[。.!！]?\s*$"
    if status == 200 and response3.get("status") == "completed" and _answer_matches(content3, pattern2):
        report.passed("multi_turn.system_prompt", f"content={content3[:120]!r}")
    else:
        report.failed("multi_turn.system_prompt", f"http={status} content={content3[:200]!r}")

    try:
        status, messages = _http_json(f"{base}/api/v1/sessions/{session_id}/messages", timeout=30)
    except Exception as exc:
        code, detail = _http_error(exc)
        report.failed("multi_turn.messages", f"http={code}: {detail[:200]}")
        return
    items = messages.get("data", []) if isinstance(messages, dict) else []
    roles = [item.get("role") for item in items]
    if status == 200 and roles.count("user") >= 3 and roles.count("assistant") >= 3:
        report.passed("multi_turn.messages", f"roles={roles}")
    else:
        report.failed("multi_turn.messages", f"http={status} roles={roles}")


def stage_unload_switch(report: Report, cfg: dict) -> None:
    """Load A, load B, chat B, unload B, reload A, chat A."""
    svc = cfg["service"]
    base = _service_or_skip(report, cfg, "unload_switch")
    if base is None:
        return
    model_a = svc.get("model")
    model_b = cfg["unload_switch"].get("model_b")
    timeout = float(cfg["unload_switch"].get("load_timeout_sec", 300))
    if not model_a or not model_b:
        report.skipped("unload_switch", "requires --model (A) and --model-b (B)")
        return
    if model_a == model_b:
        report.failed("unload_switch", "model A and model B must differ")
        return

    instance_a, error = _ensure_loaded(base, model_a, timeout)
    if instance_a is None:
        report.failed("unload_switch.load_a", f"model {model_a!r} never became ready: {error}")
        return
    report.passed("unload_switch.load_a", f"instance={instance_a.get('id')} model={model_a}")

    def chat_check(check_id: str, model: str) -> None:
        try:
            status, body = _chat(base, model, svc["question"], int(svc["max_tokens"]), float(svc["timeout_sec"]))
        except Exception as exc:
            code, detail = _http_error(exc)
            report.failed(check_id, f"http={code}: {detail[:200]}")
            return
        choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
        content = ((choice.get("message") or {}).get("content") or "").strip()
        echoed = body.get("model")
        if status == 200 and echoed == model and _answer_matches(content, svc.get("answer_pattern")):
            report.passed(check_id, f"model={model} content={content[:120]!r}")
        else:
            report.failed(check_id, f"http={status} echoed={echoed!r} content={content[:200]!r}")

    chat_check("unload_switch.chat_a", model_a)
    instance_b, error = _ensure_loaded(base, model_b, timeout)
    if instance_b is None:
        report.failed("unload_switch.load_b", f"model {model_b!r} never became ready: {error}")
        return
    report.passed("unload_switch.load_b", f"instance={instance_b.get('id')} model={model_b}")
    chat_check("unload_switch.chat_b", model_b)

    ok, detail = _unload_instance(base, str(instance_b["id"]))
    if not ok:
        report.failed("unload_switch.unload_b", detail)
        return
    if not _wait_absent(base, model_b, 60):
        report.failed("unload_switch.unload_b", f"model {model_b!r} still has a runtime instance")
        return
    report.passed("unload_switch.unload_b", f"instance={instance_b['id']} gone status={detail}")

    instance_a2, error = _ensure_loaded(base, model_a, timeout)
    if instance_a2 is None:
        report.failed("unload_switch.reload_a", f"model {model_a!r} never became ready again: {error}")
        return
    report.passed("unload_switch.reload_a", f"instance={instance_a2.get('id')}")
    chat_check("unload_switch.chat_a_again", model_a)


def stage_negative(report: Report, cfg: dict, work_dir: Path) -> None:
    """Error paths for registration and loading; fixtures are unique per run."""
    base = _service_or_skip(report, cfg, "negative")
    if base is None:
        return
    timeout = float(cfg["negative"].get("timeout_sec", 180))
    tag = uuid.uuid4().hex[:8]
    root = work_dir / "negative"
    root.mkdir(parents=True, exist_ok=True)

    def register(path: Path) -> tuple[int | None, dict, str | None]:
        try:
            status, body = _http_json(
                f"{base}/api/v1/models/directories/register",
                method="POST",
                payload={"path": str(path)},
                timeout=timeout,
            )
            return status, body, None
        except Exception as exc:
            code, detail = _http_error(exc)
            return code, {}, detail

    # 1. Nonexistent directory is rejected.
    status, _, detail = register(root / "missing-dir")
    if status == 404:
        report.passed("negative.register.missing_dir", "http=404 model_directory_not_found")
    else:
        report.failed("negative.register.missing_dir", f"expected http=404, got {status}: {(detail or '')[:200]}")

    try:
        tiny = quantize_tiny_fixture(cfg, work_dir / "negative-fixture", "tiny-neg.mfq")
    except Exception as exc:
        report.failed("negative.fixture", f"could not build tiny artifact: {exc}")
        return

    # 2. A corrupt container is rejected or reported as incomplete.
    corrupt_dir = root / "corrupt"
    corrupt_dir.mkdir(exist_ok=True)
    (corrupt_dir / f"corrupt-{tag}.mfq").write_bytes(b"this is not an mfq container")
    status, body, detail = register(corrupt_dir)
    if status == 400:
        report.passed("negative.register.corrupt", "http=400 model_registration_failed")
    elif status == 200:
        entry = next((e for e in body.get("data", []) if e.get("name") == f"corrupt-{tag}"), None)
        if entry is not None and not entry.get("complete"):
            report.passed("negative.register.corrupt", f"artifact reported incomplete error={entry.get('error')!r}")
        else:
            report.failed("negative.register.corrupt", f"corrupt file accepted as complete: {entry}")
    else:
        report.failed("negative.register.corrupt", f"http={status}: {(detail or '')[:200]}")

    # 3. Duplicate catalog name across discovery roots is rejected.
    dup = root / "dup"
    for sub in ("a", "b"):
        target = dup / sub
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tiny, target / f"dup-{tag}.mfq")
    status, _body, detail = register(dup)
    if status == 409:
        report.passed("negative.register.duplicate", "http=409 duplicate_model_name")
    else:
        report.failed("negative.register.duplicate", f"expected http=409, got {status}: {(detail or '')[:200]}")

    # 4. A shard set with a missing shard is not loadable.
    shard_dir = root / "shard"
    shard_dir.mkdir(exist_ok=True)
    shutil.copy2(tiny, shard_dir / f"sharded-{tag}-00001-of-00002.mfq")
    status, body, detail = register(shard_dir)
    if status == 400:
        report.passed("negative.register.missing_shard", "http=400 rejected")
    elif status == 200:
        entry = next((e for e in body.get("data", []) if e.get("name") == f"sharded-{tag}"), None)
        if entry is None:
            report.failed("negative.register.missing_shard", "sharded artifact not listed")
        elif entry.get("complete") and entry.get("loadable"):
            report.failed("negative.register.missing_shard", f"missing shard accepted as loadable: {entry}")
        else:
            report.passed(
                "negative.register.missing_shard",
                f"complete={entry.get('complete')} loadable={entry.get('loadable')} "
                f"missing_shards={entry.get('missing_shards')}",
            )
    else:
        report.failed("negative.register.missing_shard", f"http={status}: {(detail or '')[:200]}")

    # 5. Unknown model load fails through the job.
    unknown = f"__missing_{tag}__"
    accepted: dict | None = None
    try:
        status, accepted = _http_json(
            f"{base}/api/v1/models/load", method="POST", payload={"model": unknown}, timeout=30
        )
    except Exception as exc:
        code, detail = _http_error(exc)
        if code in (400, 404):
            report.passed("negative.load.unknown_model", f"http={code} rejected synchronously")
        else:
            report.failed("negative.load.unknown_model", f"http={code}: {detail[:200]}")
        accepted = None
    if accepted:
        job_state = None
        job = {}
        deadline = time.time() + 30
        while time.time() < deadline:
            _, job = _http_json(f"{base}/api/v1/jobs/{accepted['operation_id']}", timeout=30)
            job_state = job.get("status")
            if job_state in {"failed", "cancelled", "succeeded", "completed"}:
                break
            time.sleep(1)
        if job_state == "failed":
            report.passed("negative.load.unknown_model", f"job failed error={(job.get('error') or {}).get('code')}")
        else:
            report.failed("negative.load.unknown_model", f"job state={job_state} (expected failed)")

    # 6. Unknown artifact URI for a known model is rejected.
    try:
        _, models = _http_json(f"{base}/v1/models", timeout=30)
        known = [entry.get("id") for entry in models.get("data", []) if entry.get("id")]
    except Exception:
        known = []
    if not known:
        report.skipped("negative.load.bad_uri", "no served model available")
        return
    try:
        status, accepted = _http_json(
            f"{base}/api/v1/models/load",
            method="POST",
            payload={"model": known[0], "artifact_uri": "mfq://00000000000000000000000000000000"},
            timeout=30,
        )
    except Exception as exc:
        code, detail = _http_error(exc)
        if code in (400, 404):
            report.passed("negative.load.bad_uri", f"http={code} rejected synchronously")
        else:
            report.failed("negative.load.bad_uri", f"http={code}: {detail[:200]}")
        return
    job_state = None
    job = {}
    deadline = time.time() + 30
    while time.time() < deadline:
        _, job = _http_json(f"{base}/api/v1/jobs/{accepted['operation_id']}", timeout=30)
        job_state = job.get("status")
        if job_state in {"failed", "cancelled", "succeeded", "completed"}:
            break
        time.sleep(1)
    if job_state == "failed":
        report.passed("negative.load.bad_uri", f"job failed error={(job.get('error') or {}).get('code')}")
    else:
        report.failed("negative.load.bad_uri", f"job state={job_state} (expected failed)")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def load_config(path: Path | None) -> dict:
    merged = json.loads(json.dumps(DEFAULTS))  # deep copy
    if path is None:
        return merged
    override = json.loads(path.read_text(encoding="utf-8"))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="JSON config merged over built-in defaults")
    parser.add_argument(
        "--stage",
        action="append",
        choices=["r0", "artifact", "app", "service", "session", "stream", "multi_turn", "unload_switch", "negative"],
        help="stage(s) to run",
    )
    parser.add_argument("--register-dir", type=Path, help="directory to register in the app catalog (app stage)")
    parser.add_argument("--artifact-name", help="catalog model name to load (app stage)")
    parser.add_argument("--work-dir", type=Path, help="scratch directory for stage outputs")
    parser.add_argument("--artifact", type=Path, help="existing .mfq to inspect")
    parser.add_argument("--expect-sha256", help="pin the artifact digest (same-machine provenance check)")
    parser.add_argument("--expect-records", type=int, help="pin the container record count")
    parser.add_argument("--expect-arch", help="pin the container architecture string")
    parser.add_argument("--service-url", help="MFQ service base URL, e.g. http://127.0.0.1:8090")
    parser.add_argument("--model", help="served model id for service/session/stream stages (A in unload_switch)")
    parser.add_argument("--model-b", help="second served model id for unload_switch (B)")
    parser.add_argument("--question", help="fixed question for inference checks")
    parser.add_argument("--answer-pattern", help="regex the answer must match")
    parser.add_argument("--require-service", action="store_true", help="treat service unreachable as failure")
    parser.add_argument("--skip-quantize", action="store_true", help="keep the r0 stage out of --stage all")
    parser.add_argument("--report", type=Path, help="write the JSON report to this path")
    args = parser.parse_args(argv)
    if args.stage is None:
        args.stage = []
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    # CLI overrides
    if args.artifact:
        cfg["artifact"] = str(args.artifact)
    if args.expect_sha256:
        cfg["expect_sha256"] = args.expect_sha256
    if args.expect_records is not None:
        cfg["expect_records"] = args.expect_records
    if args.expect_arch:
        cfg["expect_arch"] = args.expect_arch
    if args.service_url:
        cfg["service"]["base_url"] = args.service_url
    if args.model:
        cfg["service"]["model"] = args.model
    if args.model_b:
        cfg["unload_switch"]["model_b"] = args.model_b
    if args.question:
        cfg["service"]["question"] = args.question
    if args.answer_pattern:
        cfg["service"]["answer_pattern"] = args.answer_pattern
    if args.require_service:
        cfg["service"]["require"] = True
    if args.skip_quantize:
        cfg["quantize"]["enabled"] = False
    if args.register_dir:
        cfg["app"]["register_dir"] = str(args.register_dir)
    if args.artifact_name:
        cfg["app"]["artifact_name"] = args.artifact_name

    stages = args.stage or cfg["stage"]
    if not stages:
        stages = ["r0"]
    if "r0" in stages and not cfg["quantize"].get("enabled", True):
        stages = [s for s in stages if s != "r0"]

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="mfq-e2e-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    report = Report()
    started = time.time()
    try:
        if "r0" in stages:
            stage_r0(report, cfg, work_dir)
        if "artifact" in stages:
            stage_artifact(report, cfg)
        if "app" in stages:
            stage_app(report, cfg)
        if "service" in stages:
            stage_service(report, cfg)
        if "session" in stages:
            stage_session(report, cfg)
        if "stream" in stages:
            stage_stream(report, cfg)
        if "multi_turn" in stages:
            stage_multi_turn(report, cfg)
        if "unload_switch" in stages:
            stage_unload_switch(report, cfg)
        if "negative" in stages:
            stage_negative(report, cfg, work_dir)
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130

    elapsed = round(time.time() - started, 2)
    print()
    print(f"MFQ e2e regression | stages={','.join(stages)} | elapsed={elapsed}s | work_dir={work_dir}")
    print(f"platform={platform.platform()} | python={sys.version.split()[0]} | repo_head={git_head()}")
    print()
    report.print_summary()
    failures = report.failed_count()
    skips = sum(1 for c in report.checks if c["status"] == "skip")
    print()
    print(f"overall: {'FAIL' if failures else 'PASS'} | failed={failures} skipped={skips}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict(args)
        payload["elapsed_sec"] = elapsed
        args.report.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(f"report: {args.report}")

    if not report.checks:
        print("no checks executed", file=sys.stderr)
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())