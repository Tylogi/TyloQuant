from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from mfq.server.models import (
    CreateGenerationPresetRequest,
    ResponseRequestSettings,
    SamplingParams,
    SessionMode,
    UpdateGenerationPresetRequest,
)
from mfq.server.service import ServerService, ServiceError
from mfq.server.storage import SessionStore
from tests.test_server_service import FakeBackend


def _request(name: str, *, temperature: float = 0.4) -> CreateGenerationPresetRequest:
    return CreateGenerationPresetRequest(
        name=name,
        model="model-a",
        mode=SessionMode.TEXT,
        settings=ResponseRequestSettings(
            sampling=SamplingParams(temperature=temperature),
            system_prompt="Be concise.",
        ),
        context_size=8192,
        metadata={"icon": "P"},
    )


def test_generation_presets_persist_update_and_delete(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = tmp_path / "mfq.server.sqlite3"
        service = ServerService(SessionStore(database), FakeBackend())
        created = await service.create_generation_preset(_request("Precise"))
        assert created.model == "model-a"
        assert created.settings.sampling.temperature == 0.4
        assert created.metadata == {"icon": "P"}

        reopened = ServerService(SessionStore(database), FakeBackend())
        listed = await reopened.list_generation_presets()
        assert [item.id for item in listed.data] == [created.id]

        updated = await reopened.update_generation_preset(
            created.id,
            UpdateGenerationPresetRequest.model_validate(
                _request("Precise", temperature=0.2).model_dump(mode="json")
            ),
        )
        assert updated.settings.sampling.temperature == 0.2
        assert updated.metadata == {"icon": "P"}

        with pytest.raises(ServiceError) as conflict:
            await reopened.create_generation_preset(_request("Precise"))
        assert conflict.value.status_code == 409

        await reopened.delete_generation_preset(created.id)
        assert (await reopened.list_generation_presets()).data == []
        with pytest.raises(ServiceError) as missing:
            await reopened.delete_generation_preset(created.id)
        assert missing.value.status_code == 404

    asyncio.run(scenario())


def test_schema_eight_database_migrates_to_generation_presets(tmp_path: Path) -> None:
    database = tmp_path / "mfq.server.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO schema_meta(key, value) VALUES ('schema_version', '8')")
    reopened = SessionStore(database)
    assert reopened.list_generation_presets() == []
