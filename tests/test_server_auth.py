from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from mfq.server.api import create_app
from mfq.server.auth import ApiKeyManager, hash_api_key
from mfq.server.storage import SCHEMA_VERSION, SessionStore


def test_scoped_keys_rotate_revoke_and_never_persist_plaintext(tmp_path: Path) -> None:
    async def run() -> None:
        database = tmp_path / "mfq.server.sqlite3"
        store = SessionStore(database)
        manager = ApiKeyManager(store, "root-secret")
        app = create_app(api_keys=manager)
        transport = httpx.ASGITransport(app=app)
        root = {"Authorization": "Bearer root-secret"}
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/api/v1/auth/keys",
                headers=root,
                json={"name": "jobs", "scopes": ["jobs"]},
            )
            assert created.status_code == 201, created.text
            body = created.json()
            token = body["token"]
            key_id = body["key"]["id"]
            assert token.startswith("mfq_")

            job_key = {"Authorization": f"Bearer {token}"}
            assert (await client.get("/api/v1/jobs", headers=job_key)).status_code == 501
            assert (await client.get("/api/v1/sessions", headers=job_key)).status_code == 403
            assert (await client.get("/api/v1/auth/keys", headers=job_key)).status_code == 403

            operator = await client.post(
                "/api/v1/auth/keys",
                headers=root,
                json={"name": "operator", "role": "operator"},
            )
            assert operator.status_code == 201
            operator_header = {"Authorization": f"Bearer {operator.json()['token']}"}
            assert (await client.get("/api/v1/jobs", headers=operator_header)).status_code == 501
            assert (
                await client.get("/api/v1/models/directories", headers=operator_header)
            ).status_code == 501
            assert (
                await client.get("/api/v1/auth/keys", headers=operator_header)
            ).status_code == 403

            viewer = await client.post(
                "/api/v1/auth/keys",
                headers=root,
                json={"name": "viewer", "role": "viewer"},
            )
            viewer_header = {"Authorization": f"Bearer {viewer.json()['token']}"}
            assert (
                await client.get("/api/v1/models/directories", headers=viewer_header)
            ).status_code == 403

            rotated = await client.post(f"/api/v1/auth/keys/{key_id}/rotate", headers=root)
            assert rotated.status_code == 200
            replacement = rotated.json()["token"]
            assert replacement != token
            assert (await client.get("/api/v1/jobs", headers=job_key)).status_code == 401
            replacement_header = {"Authorization": f"Bearer {replacement}"}
            assert (await client.get("/api/v1/jobs", headers=replacement_header)).status_code == 501

            revoked = await client.post(f"/api/v1/auth/keys/{key_id}/revoke", headers=root)
            assert revoked.status_code == 200
            assert (await client.get("/api/v1/jobs", headers=replacement_header)).status_code == 401

        database_bytes = database.read_bytes()
        assert token.encode() not in database_bytes
        assert replacement.encode() not in database_bytes
        with store._connection() as connection:
            row = connection.execute(
                "SELECT key_hash FROM api_keys WHERE id = ?", (key_id,)
            ).fetchone()
        assert row["key_hash"] == hash_api_key(replacement)

    asyncio.run(run())


def test_api_key_schema_eleven_migrates(tmp_path: Path) -> None:
    database = tmp_path / "mfq.server.sqlite3"
    store = SessionStore(database)
    with store._connection() as connection:
        connection.execute("UPDATE schema_meta SET value = '11' WHERE key = 'schema_version'")
    migrated = SessionStore(database)
    with migrated._connection() as connection:
        version = connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()["value"]
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'api_keys'"
        ).fetchone()
    assert version == str(SCHEMA_VERSION)
    assert table is not None
