"""TPQ package entry point: ``python -m tpq <command>``."""

from __future__ import annotations

from importlib import import_module
import sys

from . import __version__


COMMANDS = ("launch", "check", "benchmark", "chat", "serve")
_COMMAND_DESCRIPTIONS = {
    "launch": "auto-detect model and start chat/API with production presets",
    "check": "validate model files, memory, GPUs and TP capacity",
    "benchmark": "measure reproducible steady single-request decode token/s",
    "chat": "interactive chat or one-shot generation",
    "serve": "OpenAI-compatible HTTP API server",
}


def _help_text() -> str:
    command_lines = "\n".join(
        f"  {command:<8} {_COMMAND_DESCRIPTIONS[command]}" for command in COMMANDS
    )
    return (
        "usage: python -m tpq <command> [options]\n\n"
        f"commands:\n{command_lines}\n\n"
        "Run `python -m tpq <command> --help` for command-specific help."
    )


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in {"-V", "--version"}:
        print(f"tpq-inference {__version__}")
        return
    if not argv or argv[0] in {"-h", "--help"}:
        print(_help_text())
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd}\n\n{_help_text()}")
        sys.exit(1)
    command_module = import_module(f".{cmd}", __package__)
    command_module.main(rest)


if __name__ == "__main__":
    main()
