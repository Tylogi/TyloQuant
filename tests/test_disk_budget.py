from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from mfq.server.runtime.disk_budget import DiskBudget
from mfq.server.services.tool_jobs import ToolJobHandlers, ToolJobPaths
from mfq.server.state.catalog import ModelCatalog


@pytest.fixture
def budget(tmp_path, monkeypatch):
    monkeypatch.setenv('MFQ_DISK_BUDGET_DIR', str(tmp_path / 'reservations'))
    return DiskBudget()


def test_concurrent_downloads_cannot_reserve_the_same_space(budget, tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=100))
    def reserve():
        try:
            return budget.reserve(tmp_path, 70)
        except OSError:
            return None
    with ThreadPoolExecutor(2) as workers:
        results = list(workers.map(lambda _: reserve(), range(2)))
    leases = [item for item in results if item is not None]
    assert len(leases) == 1
    assert budget.reserved(tmp_path) == 70
    leases[0].update(20)
    assert budget.reserved(tmp_path) == 20
    leases[0].update(80)
    assert budget.reserved(tmp_path) == 80
    with pytest.raises(OSError):
        leases[0].update(101)
    assert budget.reserved(tmp_path) == 80
    leases[0].close()
    assert budget.reserved(tmp_path) == 0


def test_dead_process_releases_reservation_without_losing_live_ones(budget, tmp_path):
    live = budget.reserve(tmp_path, 123)
    try:
        code = 'from pathlib import Path; from mfq.server.runtime.disk_budget import DiskBudget; import os,sys; x=DiskBudget().reserve(Path(sys.argv[1]),456); os._exit(0)'
        subprocess.run([sys.executable, '-c', code, str(tmp_path)], check=True)
        assert budget.reserved(tmp_path) == 123
        assert len(list(budget.root.glob('*.lease'))) == 1
    finally:
        live.close()


def test_other_volume_does_not_consume_download_space(budget, tmp_path, monkeypatch):
    from mfq.server.runtime import disk_budget
    monkeypatch.setattr(disk_budget, 'volume', lambda path: 1 if path.name == 'one' else 2)
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=100))
    first = budget.reserve(tmp_path / 'one', 90)
    second = budget.reserve(tmp_path / 'two', 90)
    try:
        assert budget.reserved(tmp_path / 'one') == 90
        assert budget.reserved(tmp_path / 'two') == 90
    finally:
        first.close(); second.close()


@pytest.mark.parametrize('outcome', ['success', 'failure', 'cancel'])
def test_download_releases_budget_after_process_stops(budget, tmp_path, monkeypatch, outcome):
    async def run():
        handlers = ToolJobHandlers(ModelCatalog([]), ToolJobPaths(tmp_path, Path(sys.executable), None, None, None, None))
        entered = asyncio.Event()
        stopped = False
        async def transfer(*args, **kwargs):
            nonlocal stopped
            assert budget.reserved(tmp_path) >= 1024
            entered.set()
            try:
                if outcome == 'cancel':
                    await asyncio.Event().wait()
                if outcome == 'failure':
                    raise RuntimeError('transfer failed')
            finally:
                assert budget.reserved(tmp_path) >= 1024
                stopped = True
        monkeypatch.setattr(handlers, '_run', transfer)
        task = asyncio.create_task(handlers._run_download(SimpleNamespace(), ['test'], {}, tmp_path, 1024))
        await entered.wait()
        if outcome == 'cancel':
            task.cancel()
        if outcome == 'success':
            await task
        else:
            with pytest.raises(asyncio.CancelledError if outcome == 'cancel' else RuntimeError):
                await task
        assert stopped and budget.reserved(tmp_path) == 0
    asyncio.run(run())


def test_sparse_partial_file_does_not_release_unwritten_download_space(budget, tmp_path, monkeypatch):
    async def run():
        directory = tmp_path / 'download'
        directory.mkdir()
        size = 10 << 20
        with (directory / 'weights.incomplete').open('wb') as stream:
            if os.name == 'nt':
                import ctypes
                import msvcrt
                from ctypes import wintypes

                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                control = kernel.DeviceIoControl
                control.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                    wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
                control.restype = wintypes.BOOL
                returned = wintypes.DWORD()
                handle = msvcrt.get_osfhandle(stream.fileno())
                # NTFS truncate alone allocates space; FSCTL_SET_SPARSE creates real holes.
                if not control(handle, 0x900C4, None, 0, None, 0, ctypes.byref(returned), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                seek = kernel.SetFilePointerEx
                seek.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD]
                seek.restype = wintypes.BOOL
                extend = kernel.SetEndOfFile
                extend.argtypes = [wintypes.HANDLE]
                extend.restype = wintypes.BOOL
                # The CRT truncate implementation writes zeros when extending a file.
                if not seek(handle, size, None, 0) or not extend(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                stream.truncate(size)
        handlers = ToolJobHandlers(ModelCatalog([]), ToolJobPaths(tmp_path, Path(sys.executable), None, None, None, None))
        async def transfer(*args, **kwargs):
            assert budget.reserved(directory) >= size + (512 << 20) - 4096
        monkeypatch.setattr(handlers, '_run', transfer)
        await handlers._run_download(SimpleNamespace(), ['test'], {}, directory, size)
        assert budget.reserved(directory) == 0
    asyncio.run(run())


def test_duplicate_destination_is_rejected_until_download_ends(budget, tmp_path):
    first = budget.reserve(tmp_path, 123, exclusive=True)
    try:
        with pytest.raises(ValueError, match='already in use'):
            budget.reserve(tmp_path, 456, exclusive=True)
        assert budget.reserved(tmp_path) == 123
    finally:
        first.close()
    retry = budget.reserve(tmp_path, 456, exclusive=True)
    retry.close()


@pytest.mark.parametrize('loaded', [False, True])
def test_download_reclaims_lru_prefix_blocks_and_preserves_hot_budget(budget, tmp_path, monkeypatch, loaded):
    from mfq.server.runtime.runtime_pool import RuntimePool
    from mfq.server.protocol.models import RuntimeInstanceState
    from tests.test_prefix_disk_budget import block
    directory = tmp_path / 'cache'
    old = block(directory, 'a', '1')
    recent = block(directory, 'a', '2', accessed=2)
    size = old.stat().st_size
    model = tmp_path / 'model.mfq'
    (directory / ('a' * 64) / 'identity.txt').write_text('mfq-cache-identity-v1\nnamespace=' + 'a' * 64 + '\nmodel_path=' + str(model))
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=1000 + (0 if old.exists() else size)))
    reservation = budget.reserve(tmp_path, 1000 + size, reclaimable=2 * size)
    async def run():
        calls = []
        async def resize(hot, *, disk_target_bytes):
            calls.append((hot, disk_target_bytes))
            old.unlink()
            return {'status': 'ok'}
        instance = SimpleNamespace(state=RuntimeInstanceState.READY, prefix_concurrent_maintenance=True,
            control_leases=0, read_control_leases=0, artifact=SimpleNamespace(path=model),
            memory=SimpleNamespace(prefix_cache_limit_bytes=1234), backend=SimpleNamespace(set_prefix_cache_budget=resize))
        async def release(item, **kwargs):
            item.control_leases -= 1
            item.read_control_leases -= 1
        pool = SimpleNamespace(_prefix_cache_directory=lambda: directory, _prefix_budget_lock=asyncio.Lock(),
            _lock=asyncio.Lock(), _loading_model_names=set(), _memory_configuration_job=None,
            _instances={'model': instance} if loaded else {}, _release_control_lease=release)
        await RuntimePool.reclaim_prefix_disk_for_download(pool, tmp_path)
        assert not old.exists() and recent.exists()
        assert calls == ([(1234, size)] if loaded else [])
        assert instance.control_leases == instance.read_control_leases == 0
    try:
        asyncio.run(run())
    finally:
        reservation.close()


@pytest.fixture(scope='module')
def native_writer(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler or os.name == 'nt':
        pytest.skip('native C++ compiler required')
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp('disk-native') / 'writer'
    source = r'''
#include "mfq_paged_prefix_cache.h"
#include "mfq/qsa_kv_store.h"
#include <iostream>
int main(int argc, char** argv) {
    mfq::cache::PagedPrefixCacheConfig prefix;
    prefix.cache_dir = std::filesystem::path(argv[1]) / "prefix";
    prefix.compatibility_key = "disk-budget-test";
    prefix.block_size_tokens = 4;
    prefix.min_disk_free_bytes = 0;
    mfq::cache::PagedPrefixCache cache(prefix);
    std::int64_t tokens[] = {1,2,3,4};
    cache.store({}, tokens, 4, std::make_shared<const std::vector<std::uint8_t>>(512, 7));
    cache.flush();
    std::cout << cache.metrics().writes << " ";
    mfq::cache::QsaKvStoreConfig kv;
    kv.directory = argv[1]; kv.budget_bytes = 4096; kv.buffer_bytes = 4096;
    try {
        mfq::cache::QsaKvStore store(kv);
        auto block = store.write(std::vector<std::uint8_t>(2048, 9));
        store.flush();
        std::cout << (store.read(block)->at(0) == 9 ? "ok" : "corrupt");
    } catch (const std::exception& error) { std::cout << error.what(); }
}
'''
    result = subprocess.run([compiler, '-std=c++17', '-pthread', '-Wno-deprecated-declarations',
        '-I' + str(root / 'cpp_runtime/core'), '-I' + str(root / 'cpp_runtime/core/include'),
        '-x', 'c++', '-', str(root / 'cpp_runtime/core/mfq_paged_prefix_cache.cpp'),
        str(root / 'cpp_runtime/core/qsa_kv_store.cpp'), '-o', str(output),
        *(['-lcrypto'] if sys.platform != 'darwin' else [])], input=source, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return output


def test_native_prefix_and_active_kv_respect_python_download_reservation(budget, tmp_path, native_writer):
    free = shutil.disk_usage(tmp_path).free
    reservation = budget.reserve(tmp_path, free + (1 << 30), reclaimable=1 << 30)
    try:
        result = subprocess.run([str(native_writer), str(tmp_path)], capture_output=True, text=True, timeout=10, check=True)
        assert result.stdout.startswith('0 ')
        assert 'reserved for downloads' in result.stdout
    finally:
        reservation.close()
    result = subprocess.run([str(native_writer), str(tmp_path)], capture_output=True, text=True, timeout=10, check=True)
    assert result.stdout == '1 ok'
