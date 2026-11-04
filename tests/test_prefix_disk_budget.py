import os
import struct
from pathlib import Path

from mfq.server.runtime.prefix_disk_budget import maintain_prefix_disk_budget
from tests.platform_helpers import symlink_or_skip


def block(root, namespace, digest, payload=32, accessed=1):
    path = root / (namespace * 64) / (digest * 2) / (digest * 64 + '.mfqkv')
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack('<8sI32s32s32sIIQ32s', b'MFQKVB1\0', 1, bytes.fromhex(namespace * 64),
        bytes.fromhex(digest * 64), bytes(32), 256, 256, payload, bytes(32))
    path.write_bytes(header + bytes(payload))
    os.utime(path, (accessed, accessed))
    return path


def test_disk_budget_is_shared_across_model_namespaces(tmp_path):
    old = block(tmp_path, 'a', '1')
    recent = block(tmp_path, 'b', '2', accessed=2)
    size = recent.stat().st_size
    before = maintain_prefix_disk_budget(tmp_path, size, evict=False)
    assert before['prefix_cache_total_disk_bytes'] == 2 * size
    assert old.exists() and recent.exists()
    result = maintain_prefix_disk_budget(tmp_path, size, evict=True)
    assert result['prefix_cache_total_disk_bytes'] == size
    assert result['prefix_cache_total_disk_blocks'] == 1
    assert result['evicted_blocks'] == 1
    assert not old.exists() and recent.exists()


def test_auto_budget_uses_available_space_plus_existing_cache(tmp_path, monkeypatch):
    path = block(tmp_path, 'a', '1')
    free = 4 << 30
    monkeypatch.setattr('mfq.server.runtime.prefix_disk_budget.shutil.disk_usage', lambda _: type('Disk', (), {'free': free})())
    result = maintain_prefix_disk_budget(tmp_path, None, evict=True)
    assert result['prefix_cache_total_disk_max_bytes'] == (free + path.stat().st_size) // 2
    assert path.exists()


def test_eviction_never_follows_links_or_deletes_unmanaged_files(tmp_path):
    cache = tmp_path / 'cache'
    external = tmp_path / 'external'
    path = block(external, 'b', '2')
    cache.mkdir()
    symlink_or_skip(cache / ('b' * 64), external / ('b' * 64), target_is_directory=True)
    owned = block(cache, 'a', '1')
    symlink_or_skip(owned.parent / ('2' * 64 + '.mfqkv'), path)
    unrelated = owned.parent / 'notes.txt'
    unrelated.write_text('keep')
    malformed = owned.parent / ('3' * 64 + '.mfqkv')
    malformed.write_bytes(b'not a cache block')
    result = maintain_prefix_disk_budget(cache, 0, evict=True)
    assert result['evicted_blocks'] == 1
    assert path.exists() and unrelated.read_text() == 'keep' and malformed.exists()
    assert (owned.parent / ('2' * 64 + '.mfqkv')).is_symlink()


def test_restart_rediscovers_durable_blocks_without_session_ids(tmp_path):
    path = block(tmp_path, 'a', '1')
    first = maintain_prefix_disk_budget(tmp_path, 4096, evict=False)
    second = maintain_prefix_disk_budget(Path(str(tmp_path)), 4096, evict=False)
    assert first == second
    assert second['prefix_cache_total_disk_bytes'] == path.stat().st_size
