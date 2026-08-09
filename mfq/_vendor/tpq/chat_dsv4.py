"""Legacy DeepSeek-V4 one-click chat shim.

New deployments should use ``python -m tpq launch chat``.  This filename is
kept so existing desktop shortcuts continue to work.
"""

from __future__ import annotations

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpq.legacy_chat import LegacyChatPreset, run_legacy_chat


PRESET = LegacyChatPreset(
    label="chat_dsv4",
    model_names=(
        "DeepSeek-V4-Flash-DSpark-tpq-m",
        "DeepSeek-V4-Flash-DSpark-tpq-s",
        "DeepSeek-V4-Flash-DSpark-tpq-l",
    ),
    default_spec=5,
    missing_model_message="未找到模型目录，请用 --model 指定",
)


def main() -> None:
    run_legacy_chat(PRESET, __file__)


if __name__ == "__main__":
    main()
