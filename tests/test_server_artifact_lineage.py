from __future__ import annotations

from pathlib import Path

from mfq.server.storage import SCHEMA_VERSION, SessionStore


def test_artifact_lineage_tracks_validation_and_schema_migration(tmp_path: Path) -> None:
    database = tmp_path / "mfq.server.sqlite3"
    store = SessionStore(database)
    producer = store.create_job("model.quantize", {"scheme": "NINT4", "input": "source"})
    validation = store.create_job("model.validate", {"model": "output"})
    lineage = store.record_artifact_lineage(
        artifact_uri="workspace://output/model.mfq",
        artifact_name="model.mfq",
        producer_job_id=producer.id,
        source_uris=["workspace://source"],
        metadata={"total_bytes": 16},
    )
    store.record_artifact_validation(lineage.artifact_uri, validation.id)
    found = store.list_artifact_lineage(artifact_uri=lineage.artifact_uri)[0]
    assert found.parameters["scheme"] == "NINT4"
    assert found.metadata["total_bytes"] == 16
    assert found.validation_job_ids == [validation.id]

    with store._connection() as connection:
        connection.execute("UPDATE schema_meta SET value = '10' WHERE key = 'schema_version'")
    migrated = SessionStore(database)
    with migrated._connection() as connection:
        version = connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()["value"]
    assert version == str(SCHEMA_VERSION)
