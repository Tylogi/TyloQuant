import asyncio
import hashlib
import struct

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.api.auth import required_scope
from mfq.server.runtime.client import BackendError
from mfq.server.runtime.prefix_disk_budget import inspect_prefix_cache, purge_prefix_cache, read_prefix_cache_tokens, maintain_prefix_disk_budget
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.protocol.models import RuntimeMemoryPolicy
from mfq.server.services.service import ServerService
from mfq.server.state.catalog import ModelCatalog
from mfq.server.state.storage import SessionStore
from tests.test_prefix_disk_budget import block
from tests.test_runtime_memory_policy import cached_instance
from tests.platform_helpers import symlink_or_skip


def text_record(path, namespace, digest, tokens):
    payload = struct.pack('<' + 'q' * len(tokens), *tokens)
    target = path.with_name(path.name + '.tokens')
    target.write_bytes(struct.pack('<8s32s32sI32s', b'MFQTXT1\0', bytes.fromhex(namespace * 64),
        bytes.fromhex(digest * 64), len(tokens), hashlib.sha256(payload).digest()) + payload)
    return target


def test_inventory_decodes_metadata_without_reading_kv_payloads(tmp_path):
    path = block(tmp_path, 'a', '1')
    text_record(path, 'a', '1', list(range(256)))
    (path.parent.parent / 'identity.txt').write_text('mfq-cache-identity-v1\nnamespace=' + 'a' * 64 +
        '\ncodec=qwen35-hybrid-kv-v3\ncontext=8192\narchitecture=qwen3_5\nshard=0:Qwen3.5-Test.mfq:100:1\n')
    value = inspect_prefix_cache(tmp_path, 'a' * 64)
    group = value['data'][0]
    assert group['model_name'] == 'Qwen3.5-Test' and group['context_size'] == 8192
    assert group['max_prefix_tokens'] == 256 and group['text_blocks'] == 1
    assert value['blocks'][0]['text_available']
    assert value['total_bytes'] > path.stat().st_size
    assert read_prefix_cache_tokens(tmp_path, 'a' * 64, '1' * 64)['tokens'] == list(range(256))


def test_purge_removes_only_selected_owned_blocks_and_text(tmp_path):
    first, other = block(tmp_path, 'a', '1'), block(tmp_path, 'b', '2')
    sidecar = text_record(first, 'a', '1', [42] * 256)
    note = first.parent / 'notes.txt'
    note.write_text('keep')
    expected = first.stat().st_size + sidecar.stat().st_size
    result = purge_prefix_cache(tmp_path, 'a' * 64)
    assert result == {'released_bytes': expected, 'removed_blocks': 1, 'failed_blocks': 0}
    assert not first.exists() and not sidecar.exists() and other.exists() and note.read_text() == 'keep'


def test_text_integrity_and_missing_legacy_text_are_explicit(tmp_path):
    path = block(tmp_path, 'a', '1')
    assert read_prefix_cache_tokens(tmp_path, 'a' * 64, '1' * 64)['reason'] == 'text_not_saved'
    sidecar = text_record(path, 'a', '1', [42] * 256)
    content = bytearray(sidecar.read_bytes())
    content[-1] ^= 1
    sidecar.write_bytes(content)
    assert read_prefix_cache_tokens(tmp_path, 'a' * 64, '1' * 64)['reason'] == 'text_invalid'
    assert read_prefix_cache_tokens(tmp_path, '../escape', '1' * 64)['reason'] == 'invalid_cache_id'


def test_low_disk_space_reclaims_below_quota_to_preserve_headroom(tmp_path, monkeypatch):
    path = block(tmp_path, 'a', '1')
    monkeypatch.setattr('mfq.server.runtime.prefix_disk_budget.shutil.disk_usage', lambda _: type('Disk', (), {'free': 1})())
    result = maintain_prefix_disk_budget(tmp_path, 100 << 30, evict=True)
    assert result['prefix_cache_total_disk_max_bytes'] == 0 and not path.exists()


def test_busy_purge_does_not_delete_anything(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        pool.memory_policy = RuntimeMemoryPolicy(prefix_directory=str(tmp_path / 'prefix-cache'))
        path = block(pool._prefix_cache_directory(), 'a', '1')
        instance = cached_instance(tmp_path, 'a', 0, 0, active=1)
        pool._instances = {instance.id: instance}
        with pytest.raises(BackendError) as error:
            await pool.purge_prefix_cache()
        assert error.value.status_code == 409 and path.exists()
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("namespace", [None, 'a' * 64])
def test_idle_purge_waits_for_read_only_monitoring_without_deleting_while_it_is_active(tmp_path, namespace):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        pool.memory_policy = RuntimeMemoryPolicy(prefix_directory=str(tmp_path / 'prefix-cache'))
        path = block(pool._prefix_cache_directory(), 'a', '1')
        instance = cached_instance(tmp_path, 'a', 0, 0)
        pool._instances = {instance.id: instance}
        calls = []
        async def refresh():
            assert instance.read_control_leases == 0
            calls.append('refresh')
        async def clear():
            assert instance.read_control_leases == 0
            calls.append('clear')
        instance.backend.refresh_prefix_cache_index = refresh
        instance.backend.clear_runtime_cache = clear
        async with pool._runtime_control_lease(instance.id, read_only=True):
            assert (await pool.inspect_prefix_cache())['can_clear']
            purging = asyncio.create_task(pool.purge_prefix_cache(namespace))
            await asyncio.sleep(.05)
            assert not purging.done() and path.exists() and not calls
        result = await asyncio.wait_for(purging, 1)
        assert result['removed_blocks'] == 1 and not path.exists()
        assert instance.control_leases == instance.read_control_leases == 0
        assert pool._memory_configuration_job is None
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_inference_starting_during_read_only_drain_cancels_purge_without_deletion(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        pool.memory_policy = RuntimeMemoryPolicy(prefix_directory=str(tmp_path / 'prefix-cache'))
        path = block(pool._prefix_cache_directory(), 'a', '1')
        instance = cached_instance(tmp_path, 'a', 0, 0)
        pool._instances = {instance.id: instance}
        async with pool._runtime_control_lease(instance.id, read_only=True):
            purging = asyncio.create_task(pool.purge_prefix_cache())
            await asyncio.sleep(.05)
            assert not purging.done() and path.exists()
            instance.active_requests = 1
            instance.state = 'busy'
        with pytest.raises(BackendError) as rejected:
            await purging
        assert rejected.value.code == 'runtime_busy' and rejected.value.status_code == 409
        assert path.exists() and pool._memory_configuration_job is None
        assert instance.control_leases == instance.read_control_leases == 0
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_read_only_drain_timeout_keeps_cache_and_does_not_claim_maintenance(tmp_path, monkeypatch):
    import mfq.server.runtime.runtime_pool as runtime_module
    from types import SimpleNamespace
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        pool.memory_policy = RuntimeMemoryPolicy(prefix_directory=str(tmp_path / 'prefix-cache'))
        path = block(pool._prefix_cache_directory(), 'a', '1')
        instance = cached_instance(tmp_path, 'a', 0, 0)
        pool._instances = {instance.id: instance}
        async with pool._runtime_control_lease(instance.id, read_only=True):
            clock = iter([10, 13])
            with monkeypatch.context() as patch:
                patch.setattr(runtime_module, 'time', SimpleNamespace(monotonic=lambda: next(clock)))
                with pytest.raises(BackendError) as rejected:
                    await pool.purge_prefix_cache()
            assert rejected.value.code == 'runtime_busy' and rejected.value.status_code == 409
            assert path.exists() and pool._memory_configuration_job is None
            assert not pool._prefix_budget_lock.locked()
            assert instance.control_leases == instance.read_control_leases == 1
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_cancelling_read_only_drain_releases_lock_without_deleting_cache(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        pool.memory_policy = RuntimeMemoryPolicy(prefix_directory=str(tmp_path / 'prefix-cache'))
        path = block(pool._prefix_cache_directory(), 'a', '1')
        instance = cached_instance(tmp_path, 'a', 0, 0)
        pool._instances = {instance.id: instance}
        async with pool._runtime_control_lease(instance.id, read_only=True):
            purging = asyncio.create_task(pool.purge_prefix_cache())
            await asyncio.sleep(.05)
            assert not purging.done()
            purging.cancel()
            with pytest.raises(asyncio.CancelledError):
                await purging
            assert path.exists() and pool._memory_configuration_job is None
            assert not pool._prefix_budget_lock.locked()
        assert instance.control_leases == instance.read_control_leases == 0
        pool._instances.clear()
        await pool.aclose()
    asyncio.run(run())


def test_management_api_clears_unloaded_models_and_rejects_paths(tmp_path):
    async def run():
        pool = RuntimePool(ModelCatalog([]), tmp_path / 'runtime')
        service = ServerService(SessionStore(tmp_path / 'state.sqlite3'), pool, runtime_manager=pool)
        path = block(pool._prefix_cache_directory(), 'a', '1')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url='http://test') as client:
            listing = await client.get('/api/v1/runtime/cache/entries')
            assert listing.status_code == 200 and listing.json()['total_blocks'] == 1
            assert listing.json()['can_clear']
            invalid = await client.post('/api/v1/runtime/cache/purge', json={'namespace': '../escape'})
            assert invalid.status_code == 422 and path.exists()
            result = await client.post('/api/v1/runtime/cache/purge', json={'namespace': None})
            assert result.status_code == 200 and result.json()['removed_blocks'] == 1 and not path.exists()
        await service.aclose()
    asyncio.run(run())


def test_cache_text_and_purge_require_admin_scope():
    assert required_scope('GET', '/api/v1/runtime/cache/entries') == 'admin'
    assert required_scope('GET', '/api/v1/runtime/cache/entries/a/b/text') == 'admin'
    assert required_scope('POST', '/api/v1/runtime/cache/purge') == 'admin'


def test_text_reassembles_parent_chain_and_detects_reclaimed_blocks(tmp_path):
    first, second = block(tmp_path, 'a', '1'), block(tmp_path, 'a', '2')
    raw = bytearray(second.read_bytes())
    raw[76:108] = bytes.fromhex('1' * 64)
    second.write_bytes(raw)
    text_record(first, 'a', '1', [41] * 256)
    text_record(second, 'a', '2', [42] * 256)
    record = read_prefix_cache_tokens(tmp_path, 'a' * 64, '2' * 64)
    assert record['tokens'] == [41] * 256 + [42] * 256
    assert inspect_prefix_cache(tmp_path)['data'][0]['max_prefix_tokens'] == 512
    first.unlink()
    assert read_prefix_cache_tokens(tmp_path, 'a' * 64, '2' * 64)['reason'] == 'incomplete_chain'


def test_text_sidecar_symlink_is_neither_read_nor_deleted(tmp_path):
    path = block(tmp_path, 'a', '1')
    external = tmp_path / 'private.txt'
    external.write_text('keep')
    sidecar = path.with_name(path.name + '.tokens')
    symlink_or_skip(sidecar, external)
    assert read_prefix_cache_tokens(tmp_path, 'a' * 64, '1' * 64)['reason'] == 'text_not_saved'
    purge_prefix_cache(tmp_path)
    assert external.read_text() == 'keep' and sidecar.is_symlink()


def test_purge_reports_partial_sidecar_failure_without_losing_released_count(tmp_path, monkeypatch):
    from pathlib import Path
    path = block(tmp_path, 'a', '1')
    sidecar = text_record(path, 'a', '1', [42] * 256)
    size = path.stat().st_size
    unlink = Path.unlink
    def fail_text(item, *args, **kwargs):
        if item == sidecar:
            raise PermissionError('locked')
        return unlink(item, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', fail_text)
    result = purge_prefix_cache(tmp_path)
    assert result == {'removed_blocks': 1, 'released_bytes': size, 'failed_blocks': 1}
    assert not path.exists() and sidecar.exists()


def test_invalid_tokenizer_returns_unavailable_instead_of_server_error(tmp_path):
    from mfq.server.runtime.prefix_cache_text import decode_prefix_cache_tokens
    (tmp_path / 'tokenizer.json').write_text('invalid tokenizer')
    with pytest.raises(ValueError):
        decode_prefix_cache_tokens(tmp_path, [42])
