import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench import model_bench
from bench import report_workflow as workflow
from tests.test_benchmark_report import fixture


def test_archive_keeps_private_input_and_publishes_verified_report(tmp_path):
    pytest.importorskip("matplotlib", reason="report rendering requires optional matplotlib")
    raw = fixture()
    output = workflow.archive_report(raw, {"model": "Test model", "hardware": "Test GPU"}, tmp_path)
    workflow.verify_report(output)
    private = tmp_path / "private" / output.name
    assert workflow.report.read_json(private / "raw.json") == raw
    assert workflow.report.read_json(private / "status.json")["state"] == "complete"
    latest = workflow.report.read_json(tmp_path / "public/latest.json")
    assert latest["run_id"] == output.name
    assert latest["report"] == f"{output.name}/index.html"
    assert f"{output.name}/index.html" in (tmp_path / "public/index.html").read_text()
    if os.name == "posix":
        assert (tmp_path / "private").stat().st_mode & 0o777 == 0o700
    for path in (tmp_path / "public").rglob("*"):
        if path.is_file():
            assert b"private-marker" not in path.read_bytes()
            assert b"secret-not-for-report" not in path.read_bytes()


@pytest.mark.parametrize("failure", ["invalid", "renderer", "verification"])
def test_failure_preserves_previous_latest_and_private_diagnostics(tmp_path, monkeypatch, failure):
    public = tmp_path / "public"
    public.mkdir()
    latest = public / "latest.json"
    latest.write_text('{"previous":"unchanged"}')
    raw = fixture()
    if failure == "invalid":
        raw["results"][0]["runs"].pop()
    elif failure == "renderer":

        def fail(*args):
            raise RuntimeError("render failed")

        monkeypatch.setattr(workflow.report, "build_report", fail)
    else:
        monkeypatch.setattr(
            workflow.report, "build_report", lambda bundle, path, theme: path.mkdir()
        )
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        workflow.archive_report(raw, {}, tmp_path)
    assert latest.read_text() == '{"previous":"unchanged"}'
    assert list(public.iterdir()) == [latest]
    private = next((tmp_path / "private").iterdir())
    assert workflow.report.read_json(private / "raw.json") == raw
    assert workflow.report.read_json(private / "status.json")["state"] == "failed"
    assert (private / "error.txt").is_file()


def test_publication_lock_rejects_concurrent_publisher(tmp_path):
    with (
        workflow.publication_lock(tmp_path),
        pytest.raises(ValueError, match="busy"),
        workflow.publication_lock(tmp_path),
    ):
        pytest.fail("lock was not held")
    assert not (tmp_path / ".publish-lock").exists()


@pytest.mark.skipif(os.name != "posix", reason="Executable stdio fixture requires POSIX")
def test_runner_automatically_publishes_after_success(tmp_path):
    pytest.importorskip("matplotlib", reason="report rendering requires optional matplotlib")
    runtime = tmp_path / "fake-runtime"
    runtime.write_text(f"""#!{sys.executable}
import json, sys
print(json.dumps({{"type":"ready"}}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("op") == "shutdown": break
    result = {{"metrics": {{"prefill_tokens":32,"prefill_ms":10,"decode_ms":20,"ttft_ms":12,
              "generation_tps":250,"mtp_used":0,"mtp_drafted_tokens":0,"mtp_accepted_tokens":0}},
              "usage":{{"prompt_tokens":32,"completion_tokens":8}},"finish_reason":"length"}}
    print(json.dumps({{"type":"event","id":request["id"],"data":result}}), flush=True)
    print(json.dumps({{"type":"done","id":request["id"]}}), flush=True)
""")
    runtime.chmod(0o755)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"model": "Fixture model", "hardware": "Fixture GPU"}))
    root = tmp_path / "archive"
    command = [
        sys.executable,
        model_bench.__file__,
        "private-model.mfq",
        "--runtime",
        str(runtime),
        "--prefill-lengths",
        "32",
        "--decode-tokens",
        "8",
        "--repetitions",
        "2",
        "--report-root",
        str(root),
        "--report-metadata",
        str(metadata),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    latest = workflow.report.read_json(root / "public/latest.json")
    bundle = workflow.report.read_json(root / "public" / latest["run_id"] / "bundle.json")
    assert bundle["validation"]["measured"] == 2
    assert bundle["protocol"]["modes"] == ["decode"]
    assert bundle["identity"]["runtime_sha256"]
    assert bundle["protocol"]["seed"] == 1
    workflow.verify_report(root / "public" / latest["run_id"])


@pytest.mark.parametrize("setting", [{"seed": 123}, {"cooldown_seconds": 60}])
def test_report_preflight_conflict_prevents_runtime_start(tmp_path, monkeypatch, setting):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"protocol": setting}))
    runtime = tmp_path / "runtime"
    runtime.write_bytes(b"fixture binary")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_bench.py",
            "model.mfq",
            "--runtime",
            str(runtime),
            "--report-root",
            str(tmp_path / "archive"),
            "--report-metadata",
            str(metadata),
        ],
    )
    monkeypatch.setattr(model_bench, "run_model", lambda *args: pytest.fail("runtime was started"))
    with pytest.raises(SystemExit) as error:
        model_bench.main()
    assert error.value.code == 2


@pytest.mark.parametrize("failure", ["index", "latest", "status"])
def test_publication_failure_restores_both_success_pointers(tmp_path, monkeypatch, failure):
    pytest.importorskip("matplotlib", reason="report rendering requires optional matplotlib")
    public = tmp_path / "public"
    public.mkdir()
    originals = {"index.html": "old history", "latest.json": '{"old":true}'}
    for name, value in originals.items():
        (public / name).write_text(value)
    real_replace, real_write = workflow.os.replace, workflow.write_json
    if failure in ("index", "latest"):

        def fail_replace(source, target):
            if Path(source).name == {"index": "index.html", "latest": "latest.json"}[failure]:
                raise OSError("injected catalog replacement failure")
            return real_replace(source, target)

        monkeypatch.setattr(workflow.os, "replace", fail_replace)
    else:

        def fail_status(path, value):
            if path.name == "status.json" and value["state"] == "complete":
                raise OSError("injected status write failure")
            return real_write(path, value)

        monkeypatch.setattr(workflow, "write_json", fail_status)
    with pytest.raises(OSError):
        workflow.archive_report(fixture(), {}, tmp_path)
    for name, value in originals.items():
        assert (public / name).read_text() == value
    private = next((tmp_path / "private").iterdir())
    assert workflow.report.read_json(private / "status.json")["state"] == "failed"
