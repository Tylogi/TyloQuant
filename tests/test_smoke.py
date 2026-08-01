"""冒烟测试：确保包可导入、版本可读、CLI 可解析。"""

from __future__ import annotations

import mfq
from mfq import cli


def test_version_is_string():
    assert isinstance(mfq.__version__, str)
    assert mfq.__version__


def test_cli_help_exits_zero():
    import pytest

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_cli_no_subcommand_returns_zero():
    assert cli.main([]) == 0
