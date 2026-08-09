"""Shared implementation for the two pre-launcher one-click chat shims.

The public runtime is ``tpq launch``.  ``chat_glm52.py`` and
``chat_dsv4.py`` remain importable/executable for existing deployments, but
their interpreter bootstrap, adjacent-model discovery and ``/stop`` handling
belong here so the compatibility paths cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .presets import load_manifest


@dataclass(frozen=True)
class LegacyChatPreset:
    label: str
    model_names: tuple[str, ...]
    default_spec: int
    missing_model_message: str


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


def model_context_limit(model: str | os.PathLike[str]) -> int:
    """Read the declared context ceiling without allocating model state."""
    try:
        _root, manifest = load_manifest(model)
        config = manifest["config"]
    except (OSError, KeyError, TypeError, ValueError):
        return 1_048_576
    return int(config.get("max_position_embeddings", 1_048_576))


def find_adjacent_model(
    caller_file: str | os.PathLike[str],
    model_names: Sequence[str],
) -> str | None:
    """Search the two historical model locations used by the old shims."""
    package_dir = Path(caller_file).resolve().parent
    for name in model_names:
        for parent in (package_dir.parent, package_dir.parent.parent):
            candidate = parent / name
            if (candidate / "tpq.json").is_file():
                return str(candidate.resolve())
    return None


def build_default_argv(
    model: str | None,
    user_argv: Sequence[str],
    preset: LegacyChatPreset,
) -> list[str]:
    """Add historical defaults while keeping every explicit option authoritative."""
    cleaned = [value for value in user_argv if value != "--no-think"]
    args: list[str] = []
    if not _has_option(cleaned, "--model"):
        if model is None and not any(
            value in ("-h", "--help") for value in cleaned
        ):
            raise ValueError(preset.missing_model_message)
        if model is not None:
            args += ["--model", model]
    if not _has_option(cleaned, "--device"):
        args += ["--device", "cuda"]
    if not _has_option(cleaned, "--spec"):
        args += ["--spec", str(preset.default_spec)]
    if not _has_option(cleaned, "--temp"):
        args += ["--temp", "0"]
    if model is not None and not _has_option(cleaned, "--max-ctx"):
        args += ["--max-ctx", str(model_context_limit(model))]
    if not _has_option(cleaned, "--max-new") and not _has_option(
        cleaned,
        "--no-max-new",
    ):
        args.append("--no-max-new")
    if "--no-think" not in user_argv and not _has_option(cleaned, "--think"):
        args.append("--think")
    return args + cleaned


def _ensure_torch_python(caller_file: str, label: str) -> None:
    """On Windows, restart the original shim with a Python that has torch."""
    try:
        import torch  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    candidates = (
        os.environ.get("TPQ_PYTHON", ""),
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "python.exe"),
        os.path.join(
            os.environ.get("PROGRAMDATA", ""),
            "anaconda3",
            "python.exe",
        ),
        os.path.expanduser(r"~\anaconda3\python.exe"),
        os.path.expanduser(r"~\miniconda3\python.exe"),
    )
    for executable in candidates:
        if (
            executable
            and os.path.exists(executable)
            and os.path.abspath(executable) != os.path.abspath(sys.executable)
        ):
            print(
                f"[{label}] 当前解释器无 torch，改用 {executable} 重启…",
                flush=True,
            )
            os.execv(
                executable,
                [executable, os.path.abspath(caller_file), *sys.argv[1:]],
            )
    raise RuntimeError(
        "当前 Python 没有 torch。请使用安装过 torch 的 Python，"
        "或设置 TPQ_PYTHON。"
    )


class StopCommand:
    """Non-blocking ``/stop`` reader with state isolated per chat process."""

    def __init__(self) -> None:
        self.buffer = ""

    def __call__(self) -> bool:
        if os.name == "nt":
            import msvcrt

            while msvcrt.kbhit():
                char = msvcrt.getwch()
                if char == "\x03":
                    raise KeyboardInterrupt
                if char in ("\x00", "\xe0"):
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                    continue
                if char in ("\b", "\x7f"):
                    self.buffer = self.buffer[:-1]
                    continue
                if char in ("\r", "\n"):
                    line, self.buffer = self.buffer.strip(), ""
                    return line == "/stop"
                self.buffer += char
            return False

        import select

        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
        except (OSError, ValueError):
            return False
        if not readable:
            return False
        line = sys.stdin.readline()
        return line != "" and line.strip() == "/stop"


def run_legacy_chat(
    preset: LegacyChatPreset,
    caller_file: str,
    user_argv: Sequence[str] | None = None,
) -> None:
    """Run one compatibility shim through the shared modern chat command."""
    _ensure_torch_python(caller_file, preset.label)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    argv = list(sys.argv[1:] if user_argv is None else user_argv)
    model = find_adjacent_model(caller_file, preset.model_names)
    try:
        args = build_default_argv(model, argv, preset)
    except ValueError as error:
        print(f"[{preset.label}] {error}", flush=True)
        raise SystemExit(2) from error

    from .chat import main as chat_main

    try:
        chat_main(args, should_stop=StopCommand())
    except KeyboardInterrupt:
        print("\n[已停止]", flush=True)


__all__ = [
    "LegacyChatPreset",
    "StopCommand",
    "build_default_argv",
    "find_adjacent_model",
    "model_context_limit",
    "run_legacy_chat",
]
