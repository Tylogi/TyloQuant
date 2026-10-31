"""Platform capabilities used by filesystem and runtime test fixtures."""

import errno
import sys
from pathlib import Path

import pytest


def symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except NotImplementedError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    except OSError as error:
        if getattr(error, "winerror", None) == 1314 or error.errno in {
            errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP,
        }:
            pytest.skip(f"symlink creation requires filesystem support and permission: {error}")
        raise


def launch_python_runtime(path: Path, monkeypatch) -> None:
    """Run only this fixture script with the test interpreter on every platform."""
    from mfq.server.runtime.runtime_pool import RuntimePool

    launch = RuntimePool._launch_configuration
    script = path.resolve()

    def launch_python(self, *args, **kwargs):
        command, environment = launch(self, *args, **kwargs)
        if Path(command[0]) == script:
            command = [sys.executable, *command]
        return command, environment

    monkeypatch.setattr(RuntimePool, "_launch_configuration", launch_python)
