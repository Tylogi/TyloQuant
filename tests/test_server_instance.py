from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mfq.server.api import create_app
from mfq.server.instance import ServerInstanceLock, server_identity
from tests.platform_helpers import symlink_or_skip


def test_identity_is_stable_per_app_and_does_not_disclose_paths(tmp_path):
    identity = server_identity(tmp_path)
    with TestClient(create_app(server_identity=identity, api_key="secret")) as client:
        first = client.get("/health/identity").json()
        assert client.get("/health/identity").json() == first == identity
        assert first["pid"] == os.getpid()
        assert first["launch_mode"] == "cli"
        assert first["studio_protocol_version"] == "1.0"
        assert str(tmp_path) not in json.dumps(first)
        assert "secret" not in json.dumps(first)
    assert server_identity(tmp_path)["instance_id"] != identity["instance_id"]
    assert server_identity(tmp_path)["workspace_id"] == identity["workspace_id"]
    assert server_identity(tmp_path / "other")["workspace_id"] != identity["workspace_id"]


def test_old_page_cannot_operate_on_a_replacement_server(tmp_path):
    identity = server_identity(tmp_path)
    with TestClient(create_app(server_identity=identity)) as client:
        denied = client.post("/api/v1/models/register", headers={"X-MFQ-Instance-ID": str(uuid4())}, json={})
        assert denied.status_code == 409
        assert denied.json()["error"]["code"] == "server_instance_changed"
        assert client.get("/health", headers={"X-MFQ-Instance-ID": identity["instance_id"]}).status_code == 200
        assert client.get("/health").status_code == 200
        cors = client.options("/api/v1/runtime/instances", headers={"Origin": "tauri://localhost",
            "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "X-MFQ-Instance-ID"})
        assert cors.status_code == 200


def test_realtime_connections_are_bound_to_the_confirmed_instance(tmp_path):
    identity = server_identity(tmp_path)
    with TestClient(create_app(server_identity=identity)) as client:
        with pytest.raises(WebSocketDisconnect) as error, client.websocket_connect(
            f"/api/v1/realtime?server_instance={uuid4()}"):
            pytest.fail("replacement server must not accept the socket")
        assert error.value.code == 1008
        assert "server instance changed" in error.value.reason
        with pytest.raises(WebSocketDisconnect) as error, client.websocket_connect(
            f"/api/v1/realtime?server_instance={identity['instance_id']}"):
            pytest.fail("contract endpoint has no audio transport")
        assert error.value.code == 1013


def test_directory_lock_survives_data_directory_removal(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    with ServerInstanceLock(data):
        shutil.rmtree(data)
        data.mkdir()
        with pytest.raises(RuntimeError, match="already managed"), ServerInstanceLock(data):
            pytest.fail("deleted data must not bypass the process lock")
        with ServerInstanceLock(tmp_path / "other"):
            pass
    with ServerInstanceLock(data):
        pass


def test_directory_lock_rejects_symlink_aliases(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    alias = tmp_path / "alias"
    symlink_or_skip(alias, data, target_is_directory=True)
    with ServerInstanceLock(data):
        with pytest.raises(RuntimeError, match="already managed"), ServerInstanceLock(alias):
            pytest.fail("duplicate writer")


def test_directory_lock_releases_after_exception(tmp_path):
    with pytest.raises(ValueError), ServerInstanceLock(tmp_path):
        raise ValueError("startup failure")
    with ServerInstanceLock(tmp_path):
        pass


def test_process_exit_releases_lock_without_deleting_the_lockfile(tmp_path):
    code = """import sys
from pathlib import Path
from mfq.server.instance import ServerInstanceLock
with ServerInstanceLock(Path(sys.argv[1])):
    print('ready', flush=True)
    sys.stdin.buffer.read()
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(RuntimeError, match="already managed"), ServerInstanceLock(tmp_path):
            pytest.fail("duplicate process")
        child.kill()
        child.wait(timeout=10)
        with ServerInstanceLock(tmp_path) as instance:
            assert instance.path.is_file()
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


def test_cli_takes_lock_before_building_or_opening_database(tmp_path, monkeypatch, capsys):
    from mfq.cli import _build_parser
    from mfq.commands.serve import _run
    args = _build_parser().parse_args(["serve", "--data-dir", str(tmp_path)])
    called = []
    monkeypatch.setattr("mfq.commands.serve._run_locked", lambda *_: called.append(True) or 0)
    with ServerInstanceLock(tmp_path):
        assert _run(args) == 1
    error = capsys.readouterr().err
    assert "already managed" in error
    assert "Traceback" not in error
    assert called == []
    assert _run(args) == 0
    assert called == [True]
    assert not (tmp_path / "server.sqlite3").exists()


def test_studio_identity_requires_owned_lifetime(tmp_path, monkeypatch):
    from mfq.cli import _build_parser
    from mfq.commands.serve import _run
    instance_id = str(uuid4())
    args = _build_parser().parse_args(["serve", "--data-dir", str(tmp_path), "--studio-instance-id", instance_id])
    with pytest.raises(ValueError, match="requires --exit-on-stdin-close"):
        _run(args)
    identities = []
    monkeypatch.setattr("mfq.commands.serve._run_locked", lambda _, identity: identities.append(identity) or 0)
    args.exit_on_stdin_close = True
    assert _run(args) == 0
    assert identities[0]["instance_id"] == instance_id
    assert identities[0]["launch_mode"] == "studio"
