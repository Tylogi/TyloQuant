"""Native crash probes require an explicitly enabled diagnostic test build."""

import asyncio
import os
from pathlib import Path

import pytest

from mfq.server.runtime.client import BackendError, StdioRuntimeClient


@pytest.mark.parametrize("kind", ["invalid_parameter", "terminate", "purecall", "access_violation"])
def test_runtime_reports_fatal_stack_and_process_exit(kind, caplog, tmp_path):
    executable = os.environ.get("MFQ_FATAL_TEST_RUNTIME")
    if os.name != "nt" or not executable:
        pytest.skip("requires a Windows MFQ_ENABLE_FATAL_TESTS runtime")

    async def run():
        process = await asyncio.create_subprocess_exec(
            str(Path(executable).resolve()), "--help",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "MFQ_TEST_RUNTIME_FATAL": kind},
        )
        client = StdioRuntimeClient(process)
        try:
            with pytest.raises(BackendError) as caught:
                await asyncio.wait_for(client.health(), timeout=30)
            message = str(caught.value)
            report = caplog.text
            (tmp_path / "runtime-stderr.txt").write_text(report, encoding="utf-8")
            assert caught.value.status_code == 503
            assert "runtime process exited with status" in message
            assert f"runtime_fatal error={kind if kind != 'access_violation' else 'unhandled_exception'}" in message
            assert "stacktrace:" in report
            assert "install_fatal_diagnostics" in report
            assert "fatal_diagnostics.cpp:" in report
            if kind == "terminate":
                assert "MFQ fatal diagnostic test exception" in report
            assert "last stderr" in caplog.text
            assert await asyncio.wait_for(process.wait(), timeout=5) != 0
        finally:
            await client.aclose()
            if process.returncode is None:
                process.terminate()
                await process.wait()

    asyncio.run(run())
