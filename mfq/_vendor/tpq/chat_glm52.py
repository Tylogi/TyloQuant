"""GLM-5.2 CCCP 量化模型的专用一键聊天入口。

启动参数和交互行为与 ``chat_dsv4.py`` 对应：默认使用 CUDA、贪心解码、
思维链和 EOS 驱动的无人工输出上限；输入完整的 ``/stop`` 行或按 Ctrl+C
可以停止生成。当前 GLM CCCP 产物不含 MTP，因此投机解码默认关闭。
"""

from __future__ import annotations

import json
import os
import sys


def _ensure_torch_python() -> None:
    """在 Windows 双击启动时寻找已经安装 torch 的 Python。"""
    try:
        import torch  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    candidates = [
        os.environ.get("TPQ_PYTHON", ""),
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "python.exe"),
        os.path.expanduser(r"~\anaconda3\python.exe"),
        os.path.expanduser(r"~\miniconda3\python.exe"),
    ]
    for executable in candidates:
        if (
            executable
            and os.path.exists(executable)
            and os.path.abspath(executable) != os.path.abspath(sys.executable)
        ):
            print(
                f"[chat_glm52] 当前解释器无 torch，改用 {executable} 重启…",
                flush=True,
            )
            os.execv(
                executable,
                [executable, os.path.abspath(__file__), *sys.argv[1:]],
            )
    raise RuntimeError(
        "当前 Python 没有 torch。请使用安装过 torch 的 Python，"
        "或设置 TPQ_PYTHON。"
    )


_ensure_torch_python()

_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _has_option(argv: list[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in argv)


def model_context_limit(model: str) -> int:
    """Read the declared context ceiling without allocating a KV cache."""
    try:
        with open(
            os.path.join(model, "cccp.json"),
            "r",
            encoding="utf-8",
        ) as handle:
            config = json.load(handle)["config"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return 1_048_576
    return int(config.get("max_position_embeddings", 1_048_576))


def build_default_argv(
    model: str | None,
    user_argv: list[str],
) -> list[str]:
    """Add GLM-5.2 defaults while keeping explicit options authoritative."""
    cleaned = [value for value in user_argv if value != "--no-think"]
    args: list[str] = []
    if not _has_option(cleaned, "--model"):
        if model is None and not any(
            value in ("-h", "--help") for value in cleaned
        ):
            raise ValueError("未找到 GLM-5.2 模型目录，请用 --model 指定")
        if model is not None:
            args += ["--model", model]
    if not _has_option(cleaned, "--device"):
        args += ["--device", "cuda"]
    if not _has_option(cleaned, "--spec"):
        args += ["--spec", "0"]
    if not _has_option(cleaned, "--temp"):
        args += ["--temp", "0"]
    if model is not None and not _has_option(cleaned, "--max-ctx"):
        args += ["--max-ctx", str(model_context_limit(model))]
    if not _has_option(cleaned, "--max-new") and not _has_option(
        cleaned,
        "--no-max-new",
    ):
        args.append("--no-max-new")
    if "--no-think" not in user_argv and not _has_option(
        cleaned,
        "--think",
    ):
        args.append("--think")
    return args + cleaned


def find_model() -> str | None:
    """按 L、M、S 档位顺序查找相邻的 GLM-5.2 CCCP 模型目录。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for name in (
        "GLM-5.2-CCCP-L",
        "GLM-5.2-CCCP-M",
        "GLM-5.2-CCCP-S",
    ):
        for candidate in (
            os.path.join(here, "..", name),
            os.path.join(here, "..", "..", name),
        ):
            if os.path.exists(os.path.join(candidate, "cccp.json")):
                return os.path.abspath(candidate)
    return None


_stop_buffer = ""


def poll_stop_command() -> bool:
    """Non-blockingly consume a complete ``/stop`` line from the terminal."""
    global _stop_buffer
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
                _stop_buffer = _stop_buffer[:-1]
                continue
            if char in ("\r", "\n"):
                line, _stop_buffer = _stop_buffer.strip(), ""
                return line == "/stop"
            _stop_buffer += char
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


def main() -> None:
    user_argv = sys.argv[1:]
    model = find_model()
    try:
        args = build_default_argv(model, user_argv)
    except ValueError as exc:
        print(f"[chat_glm52] {exc}", flush=True)
        raise SystemExit(2) from exc

    from tpq.chat import main as chat_main

    try:
        chat_main(args, should_stop=poll_stop_command)
    except KeyboardInterrupt:
        print("\n[已停止]", flush=True)


if __name__ == "__main__":
    main()
