"""Persistent, restartable executor for the production V4F EW conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


def _atomic_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_progress(path: Path) -> dict | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 64 * 1024))
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _run_stage(
    *,
    name: str,
    command: list[str],
    progress_path: Path,
    run_dir: Path,
    state_path: Path,
    minimum_free_bytes: int,
    attempts: int,
    stall_seconds: int,
) -> None:
    for attempt in range(1, attempts + 1):
        started = time.time()
        previous_mtime = (
            progress_path.stat().st_mtime if progress_path.exists() else started
        )
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdin=subprocess.DEVNULL,
        )
        _atomic_json(
            state_path,
            {
                "status": "running",
                "stage": name,
                "attempt": attempt,
                "manager_pid": os.getpid(),
                "worker_pid": process.pid,
                "command": command,
                "started_unix": started,
                "progress_path": str(progress_path),
            },
        )
        print(
            json.dumps(
                {
                    "event": "stage_started",
                    "stage": name,
                    "attempt": attempt,
                    "worker_pid": process.pid,
                }
            ),
            flush=True,
        )
        last_progress = time.time()
        while True:
            result = process.poll()
            now = time.time()
            if progress_path.exists():
                current_mtime = progress_path.stat().st_mtime
                if current_mtime > previous_mtime:
                    previous_mtime = current_mtime
                    last_progress = now
            if _free_bytes(run_dir) < minimum_free_bytes:
                process.terminate()
                process.wait(timeout=60)
                raise RuntimeError(
                    f"{name} stopped: free disk fell below "
                    f"{minimum_free_bytes / 1e9:.1f} GB"
                )
            if result is not None:
                if result == 0:
                    event = _latest_progress(progress_path)
                    _atomic_json(
                        state_path,
                        {
                            "status": "stage_complete",
                            "stage": name,
                            "attempt": attempt,
                            "manager_pid": os.getpid(),
                            "worker_pid": process.pid,
                            "seconds": now - started,
                            "last_progress": event,
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "event": "stage_complete",
                                "stage": name,
                                "seconds": now - started,
                            }
                        ),
                        flush=True,
                    )
                    return
                print(
                    json.dumps(
                        {
                            "event": "stage_failed",
                            "stage": name,
                            "attempt": attempt,
                            "exit_code": result,
                        }
                    ),
                    flush=True,
                )
                break
            if now - last_progress > stall_seconds:
                process.terminate()
                process.wait(timeout=60)
                print(
                    json.dumps(
                        {
                            "event": "stage_stalled",
                            "stage": name,
                            "attempt": attempt,
                            "stall_seconds": now - last_progress,
                        }
                    ),
                    flush=True,
                )
                break
            if int(now - started) % 30 == 0:
                print(
                    json.dumps(
                        {
                            "event": "heartbeat",
                            "stage": name,
                            "attempt": attempt,
                            "worker_pid": process.pid,
                            "free_gb": _free_bytes(run_dir) / 1e9,
                            "last_progress": _latest_progress(progress_path),
                        }
                    ),
                    flush=True,
                )
            time.sleep(1)
        if attempt < attempts:
            time.sleep(10)
    raise RuntimeError(f"{name} failed after {attempts} attempts")


def _preflight(args) -> tuple[Path, Path]:
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    required = (
        run_dir / "recipe.json",
        run_dir / "allocation.json",
        run_dir / "scheme.json",
        Path(args.gate_manifest).resolve(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing production artifacts: {missing}")
    if output.exists():
        raise FileExistsError(f"V4F output already exists: {output}")
    if not Path(args.input).joinpath(
        "model.safetensors.index.json"
    ).is_file():
        raise FileNotFoundError("V4F source index is absent")
    if not Path(args.imatrix).is_file():
        raise FileNotFoundError("V4F imatrix is absent")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if _free_bytes(run_dir) < args.minimum_free_gb * 1_000_000_000:
        raise RuntimeError(
            f"free disk is below {args.minimum_free_gb} GB"
        )
    gate = subprocess.run(
        [
            sys.executable,
            str(Path(args.gate_checker).resolve()),
            str(Path(args.gate_manifest).resolve()),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    print(gate.stdout, end="", flush=True)
    if gate.returncode:
        print(gate.stderr, end="", file=sys.stderr, flush=True)
        raise RuntimeError("long-run production gate rejected the run")
    return run_dir, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reap-csv", required=True)
    parser.add_argument("--imatrix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temp-dir", required=True)
    parser.add_argument("--gate-manifest", required=True)
    parser.add_argument("--gate-checker", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--row-chunk", type=int, default=512)
    parser.add_argument("--minimum-free-gb", type=int, default=90)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--stall-seconds", type=int, default=600)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    run_dir, output = _preflight(args)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "gpu": torch.cuda.get_device_name(),
                    "free_gb": _free_bytes(run_dir) / 1e9,
                }
            ),
            flush=True,
        )
        return

    pid_path = run_dir / "executor.pid"
    state_path = run_dir / "executor_state.json"
    if pid_path.exists():
        raise FileExistsError(f"executor PID file already exists: {pid_path}")
    pid_path.write_text(str(os.getpid()), encoding="ascii")
    minimum_free_bytes = args.minimum_free_gb * 1_000_000_000
    train_command = [
        sys.executable,
        "-m",
        "mfq.tools.quantize_v4f_to_mfq",
        "train-codebooks",
        "--input",
        str(Path(args.input).resolve()),
        "--run-dir",
        str(run_dir),
        "--reap-csv",
        str(Path(args.reap_csv).resolve()),
        "--imatrix",
        str(Path(args.imatrix).resolve()),
        "--device",
        args.device,
        "--train-rows",
        "64",
        "--sample-seed",
        "20260723",
        "--bank-iterations",
        "3",
        "--bank-refine-steps",
        "1",
        "--kmeans-iterations",
        "6",
        "--expert-batch",
        "32",
        "--jsc-iterations",
        "4",
        "--jsc-refine-steps",
        "2",
    ]
    convert_command = [
        sys.executable,
        "-m",
        "mfq.tools.quantize_v4f_to_mfq",
        "convert",
        "--input",
        str(Path(args.input).resolve()),
        "--run-dir",
        str(run_dir),
        "--reap-csv",
        str(Path(args.reap_csv).resolve()),
        "--imatrix",
        str(Path(args.imatrix).resolve()),
        "--output",
        str(output),
        "--temp-dir",
        str(Path(args.temp_dir).resolve()),
        "--device",
        args.device,
        "--row-chunk",
        str(args.row_chunk),
    ]
    try:
        _run_stage(
            name="train-codebooks",
            command=train_command,
            progress_path=run_dir / "codebook_progress.jsonl",
            run_dir=run_dir,
            state_path=state_path,
            minimum_free_bytes=minimum_free_bytes,
            attempts=args.attempts,
            stall_seconds=args.stall_seconds,
        )
        partial_output = output.with_suffix(output.suffix + ".partial")
        if partial_output.exists():
            partial_output.unlink()
        _run_stage(
            name="convert",
            command=convert_command,
            progress_path=run_dir / "convert_progress.jsonl",
            run_dir=run_dir,
            state_path=state_path,
            minimum_free_bytes=minimum_free_bytes,
            attempts=args.attempts,
            stall_seconds=args.stall_seconds,
        )
        if not output.is_file() or output.stat().st_size < 39_900_000_000:
            raise RuntimeError("completed V4F output has an invalid size")
        completed = {
            "status": "complete",
            "manager_pid": os.getpid(),
            "output": str(output),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "completed_unix": time.time(),
        }
        _atomic_json(run_dir / "completed.json", completed)
        _atomic_json(state_path, completed)
        print(json.dumps(completed), flush=True)
    except Exception as exc:
        _atomic_json(
            state_path,
            {
                "status": "failed",
                "manager_pid": os.getpid(),
                "error": repr(exc),
                "failed_unix": time.time(),
            },
        )
        raise
    finally:
        pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
