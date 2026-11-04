from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from mfq.server.api import create_app


def test_authentication_policy_is_hot_persistent_and_shared(tmp_path: Path) -> None:
    async def run() -> None:
        path = tmp_path / "auth.json"
        app = create_app(auth_settings_path=path)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/api/v1/auth/policy")).json() == {"enabled": False, "has_key": False}
            assert (await client.put("/api/v1/auth/policy", json={"enabled": True})).status_code == 400
            assert not path.exists()
            secret = "example-policy-key"
            response = await client.put("/api/v1/auth/policy", json={"enabled": True, "api_key": secret})
            assert response.status_code == 200
            assert response.json() == {"enabled": True, "has_key": True}
            assert secret not in path.read_text()
            assert (await client.get("/v1/models")).status_code == 401
            denied = await client.post("/v1/messages", headers={"x-api-key": "incorrect"}, json={})
            assert denied.status_code == 401
            assert denied.json()["error"]["type"] == "authentication_error"
            assert (await client.get("/api/v1/auth/policy", headers={"Authorization": f"Bearer {secret}"})).status_code == 200
            allowed = await client.post("/v1/messages", headers={"x-api-key": secret}, json={})
            assert allowed.status_code not in (401, 403)
            preflight = await client.options("/api/v1/auth/policy", headers={"Origin": "http://127.0.0.1:5174", "Access-Control-Request-Method": "PUT", "Access-Control-Request-Headers": "authorization,content-type"})
            assert preflight.status_code == 200
            new_secret = "example-replacement-key"
            changed = await client.put("/api/v1/auth/policy", headers={"Authorization": f"Bearer {secret}"}, json={"enabled": True, "api_key": new_secret})
            assert changed.status_code == 200
            assert (await client.get("/api/v1/auth/policy", headers={"Authorization": f"Bearer {secret}"})).status_code == 401

        restarted = create_app(auth_settings_path=path)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restarted), base_url="http://test") as client:
            assert (await client.get("/api/v1/auth/policy")).status_code == 401
            headers = {"Authorization": f"Bearer {new_secret}"}
            assert (await client.get("/api/v1/auth/policy", headers=headers)).status_code == 200
            assert (await client.put("/api/v1/auth/policy", headers=headers, json={"enabled": False})).status_code == 200
            assert (await client.get("/api/v1/auth/policy")).json() == {"enabled": False, "has_key": True}
            assert not json.loads(path.read_text())["enabled"]
            assert secret not in path.read_text() and new_secret not in path.read_text()
        assert not create_app(auth_settings_path=path).state.auth_policy.enabled
    asyncio.run(run())


@pytest.mark.skipif(os.name != "posix", reason="Windows chmod has no owner-only POSIX mode bits; ACLs are separate")
def test_authentication_policy_file_is_owner_only(tmp_path: Path) -> None:
    async def run() -> None:
        path = tmp_path / "auth.json"
        policy = create_app(auth_settings_path=path).state.auth_policy
        await policy.configure(True, "example-policy-key")
        assert path.stat().st_mode & 0o777 == 0o600
        await policy.configure(True, "example-replacement-key")
        assert path.stat().st_mode & 0o777 == 0o600

    asyncio.run(run())


def test_authentication_changes_fail_closed(tmp_path: Path) -> None:
    async def run() -> None:
        app = create_app(api_key="example-initial-key", auth_settings_path=tmp_path / "auth.json")
        app.state.listener = SimpleNamespace(config=SimpleNamespace(host="0.0.0.0"))
        headers = {"Authorization": "Bearer example-initial-key"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.put("/api/v1/auth/policy", headers=headers, json={"enabled": False})
            assert response.status_code == 400
            for key in ("", "has space", "中文", "x" * 513):
                response = await client.put("/api/v1/auth/policy", headers=headers, json={"enabled": True, "api_key": key})
                assert response.status_code == 400
            assert app.state.auth_policy.matches("example-initial-key")
            def fail(*args):
                raise OSError("disk full")
            app.state.auth_policy._save = fail
            response = await client.put("/api/v1/auth/policy", headers=headers, json={"enabled": True, "api_key": "example-new-key"})
            assert response.status_code == 500
            assert (await client.get("/api/v1/auth/policy", headers=headers)).status_code == 200
            assert not app.state.auth_policy.matches("example-new-key")
    asyncio.run(run())


def test_desktop_can_read_authentication_failures_without_relaxing_cors() -> None:
    async def run() -> None:
        app = create_app(api_key="example-desktop-key")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for origin in ("tauri://localhost", "http://tauri.localhost", "https://tauri.localhost", "http://127.0.0.1:5174"):
                headers = {"Origin": origin}
                denied = await client.get("/api/v1/auth/policy", headers=headers)
                assert denied.status_code == 401
                assert denied.headers["access-control-allow-origin"] == origin
                assert denied.json()["error"]["code"] == "unauthorized"
                assert denied.headers["www-authenticate"] == "Bearer"
                invalid = await client.get("/api/v1/auth/policy", headers={**headers, "Authorization": "Bearer incorrect"})
                assert invalid.status_code == 401
                assert invalid.headers["access-control-allow-origin"] == origin
                allowed = await client.get("/api/v1/auth/policy", headers={**headers, "Authorization": "Bearer example-desktop-key"})
                assert allowed.status_code == 200
                assert allowed.headers["access-control-allow-origin"] == origin
            denied = await client.get("/api/v1/auth/policy", headers={"Origin": "https://untrusted.example"})
            assert denied.status_code == 401
            assert "access-control-allow-origin" not in denied.headers
    asyncio.run(run())
