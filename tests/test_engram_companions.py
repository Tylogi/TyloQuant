from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from mfq.formats.companions import is_engram_table, with_engram_shards
from mfq.formats.header import FileHeader
from mfq.formats.io import open_mmap, save
from mfq.formats.shards import shard_file_versions
from mfq.server.protocol.models import CreateJobRequest, HubModelFile, HubModelInfo, HubSystemProfile
from mfq.server.services.hub import HubCatalog, _model_variants
from mfq.server.services.service import ServerService
from mfq.server.services.tool_jobs import ToolJobHandlers
from mfq.server.state.catalog import ModelCatalog
from mfq.server.services.model_deletion import model_deletion_plan
from mfq.server.state.storage import SessionStore
from tests.test_download_directory import make_handlers
from tests.test_server_service import FakeBackend


TABLE = 'model.block.1.associative_memory.embedding.weight'
SHARED = 'Engram/model-EW-Engram-MXFP8-00002-of-00003.mfq'


@pytest.mark.parametrize('name, expected', [
    (TABLE, True), (TABLE + '.tq_row_shard.00000', True),
    ('model.block.1.engram.embed.weight.tq_row_shard.2', True),
    (TABLE + '.tq_row_shard.', False), (TABLE + '.tq_row_shard.invalid', False),
    ('model.output.weight.tq_row_shard.00000', False),
])
def test_engram_tq_row_shard_names(name, expected):
    assert is_engram_table(name) is expected


def checkpoint(root, precision='Q3', suffix='.mfq'):
    paths = [root / precision / f'model-EW-{precision}-{index:05d}-of-00003{suffix}'
             for index in (1, 3)]
    table = root / SHARED.replace(".mfq", suffix)
    for index, path, name in ((0, paths[0], 'model.token_embedding.weight'),
                              (1, table, TABLE), (2, paths[1], 'model.output.weight')):
        if index == 1 and table.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        save(path, FileHeader(version=2, model_arch='test', extra={
            'split.no': index, 'split.count': 3,
            'split.tensors.count': 3, 'split.records.count': 3,
        }), {name: np.full((2, 4), index + 1, dtype=np.float16)})
    return paths, table


@pytest.mark.parametrize("suffix", [".mfq", ".tq", ".TQ"])
def test_two_precision_tiers_read_one_shared_table_without_copies(tmp_path, suffix):
    first, table = checkpoint(tmp_path, 'Q3', suffix)
    second, _ = checkpoint(tmp_path, 'Q2.5', suffix)
    for entry in (first[0], first[-1], second[0]):
        with open_mmap(entry) as store:
            assert store.paths[1] == table
            np.testing.assert_array_equal(store[TABLE], np.full((2, 4), 2, dtype=np.float16))
            assert len(store.records) == 3
        artifact = ModelCatalog._inspect(tmp_path, entry)
        assert artifact.resource.complete and artifact.resource.shard_count == 3
        assert artifact.resource.ssd_ple_bytes > 0
        assert str(table) not in model_deletion_plan(artifact).files
    assert len(list(tmp_path.rglob(f'*Engram*{suffix}'))) == 1
    assert len(list(tmp_path.rglob(f'*{suffix}'))) == 5


def test_checkpoint_versions_track_shared_table_changes(tmp_path):
    paths, table = checkpoint(tmp_path)
    before = shard_file_versions(paths[0])
    assert len(before) == 3
    assert before == shard_file_versions(paths[-1])
    assert before[1][0] == str(table.resolve())
    table.write_bytes(table.read_bytes() + b'changed')
    after = shard_file_versions(paths[0])
    assert after[0] == before[0] and after[2] == before[2]
    assert after[1] != before[1]
    table.unlink()
    with pytest.raises(FileNotFoundError):
        shard_file_versions(paths[0])


def test_missing_or_ambiguous_shared_table_is_rejected(tmp_path):
    paths, table = checkpoint(tmp_path)
    duplicate = table.with_name(table.name.replace('MXFP8', 'MXFP4'))
    duplicate.write_bytes(table.read_bytes())
    with pytest.raises(ValueError, match='ambiguous shared Engram'):
        open_mmap(paths[0])
    duplicate.unlink()
    table.unlink()
    with pytest.raises(FileNotFoundError, match='missing MFQ shard'):
        open_mmap(paths[0])


def test_shared_name_does_not_allow_replacing_non_table_weights(tmp_path):
    paths, table = checkpoint(tmp_path)
    save(table, FileHeader(version=2, model_arch='test', extra={
        'split.no': 1, 'split.count': 3, 'split.tensors.count': 3, 'split.records.count': 3,
    }), {'model.evil.weight': np.ones((2, 4), dtype=np.float16)})
    with pytest.raises(ValueError, match='non-table records'):
        open_mmap(paths[0])


def test_legacy_complete_checkpoint_takes_priority_over_shared_tables(tmp_path):
    paths, table = checkpoint(tmp_path)
    local = paths[0].with_name(paths[0].name.replace('00001-of', '00002-of'))
    local.write_bytes(table.read_bytes())
    table.with_name(table.name.replace('MXFP8', 'MXFP4')).write_bytes(table.read_bytes())
    with open_mmap(paths[0]) as store:
        assert store.paths[1] == local


def test_shared_shards_never_cross_models_or_split_counts():
    primary = 'Q3/model-EW-Q3-00001-of-00003.mfq'
    unrelated = ['Engram/other-EW-Engram-MXFP8-00002-of-00003.mfq',
                 'Engram/model-EW-Engram-MXFP8-00002-of-00004.mfq']
    assert with_engram_shards([primary], [primary, *unrelated]) == [primary]


def test_catalog_merges_shared_tables_into_each_precision_and_hides_table_only_tier():
    files = [HubModelFile(name=SHARED, byte_size=500, weight_bytes=0, ssd_ple_bytes=480,
                         estimated_weight_bytes_by_role={'dense': 0, 'experts': 0, 'embedding': 0})]
    for precision in ('Q2.5', 'Q3'):
        files.extend(HubModelFile(name=f'{precision}/model-EW-{precision}-{index:05d}-of-00003.mfq',
            byte_size=100, weight_bytes=80, ssd_ple_bytes=0,
            estimated_weight_bytes_by_role={'dense': 0, 'experts': 88, 'embedding': 0}) for index in (1, 3))
    variants = _model_variants(files, HubSystemProfile(platform='test', machine='test', backend='unknown'))
    assert len(variants) == 2
    assert {variant.precision for variant in variants} == {'Q2.5', 'Q3'}
    for variant in variants:
        assert SHARED in variant.files and len(variant.files) == 3
        assert variant.byte_size == 700 and variant.resident_weight_bytes == 160
        assert variant.ssd_ple_bytes == 480
        assert variant.estimated_weight_bytes_by_role['experts'] == 176


def test_table_size_is_excluded_while_metadata_is_pending_and_old_cache_is_refreshed():
    files = [HubModelFile(name=SHARED, byte_size=500, weight_bytes=480, ssd_ple_bytes=0),
             HubModelFile(name='Q3/model-EW-Q3-00001-of-00003.mfq', byte_size=100)]
    profile = HubSystemProfile(platform='test', machine='test', backend='unknown')
    variant, = _model_variants(files, profile)
    assert variant.byte_size == 600 and variant.ssd_ple_bytes == 500
    assert variant.estimated_resident_weight_bytes == 110
    info = HubModelInfo(provider='modelscope', repo_id='owner/model', revision='master', files=files)
    assert {item.name for _, item in HubCatalog._missing_mfq_files(info)} == {item.name for item in files}


@pytest.mark.parametrize('provider', ['modelscope', 'huggingface'])
@pytest.mark.parametrize('suffix', ['.mfq', '.tq'])
def test_download_preflight_adds_required_tables_and_detects_missing_dependency(tmp_path, provider, suffix):
    shared = SHARED.replace('.mfq', suffix)
    async def run():
        handlers = make_handlers(tmp_path)
        primary = f'Q3/model-EW-Q3-00001-of-00003{suffix}'
        files = [HubModelFile(name=primary, byte_size=5), HubModelFile(name=shared, byte_size=7)]
        hub = SimpleNamespace(download_files=AsyncMock(return_value=SimpleNamespace(files=files)))
        service = ServerService(SessionStore(tmp_path / 'state.db'), FakeBackend(),
                                hub_catalog=hub, tool_handlers=handlers)
        request = CreateJobRequest(kind=f'download.{provider}', payload={
            'repo_id': 'owner/model', 'destination': 'models/owner/model', 'include': [f'Q3/*{suffix}'],
        })
        await service._check_download_job(request)
        assert shared in request.payload['include'] and request.payload['expected_bytes'] == 12
        payload, destination = handlers.download_request(request.kind, request.payload)
        target = destination / primary
        target.parent.mkdir(parents=True)
        target.write_bytes(b'12345')
        assert not ToolJobHandlers.download_files_present(destination, payload, files)
        target = destination / shared
        target.parent.mkdir(parents=True)
        target.write_bytes(b'1234567')
        assert ToolJobHandlers.download_files_present(destination, payload, files)
    asyncio.run(run())


def test_explicit_exclusion_cannot_silently_produce_an_incomplete_model(tmp_path):
    handlers = make_handlers(tmp_path)
    primary = 'Q3/model-EW-Q3-00001-of-00003.mfq'
    payload, _ = handlers.download_request('download.modelscope', {
        'repo_id': 'owner/model', 'destination': 'models/model',
        'include': ['Q3/*.mfq'], 'exclude': ['Engram/*'],
    })
    with pytest.raises(ValueError, match='cannot be excluded'):
        handlers.selected_download_files(payload, [HubModelFile(name=name, byte_size=10)
                                                   for name in (primary, SHARED)])


def test_different_precision_downloads_can_queue_in_the_shared_directory(tmp_path):
    async def run():
        from mfq.server.services.jobs import JobManager
        handlers = make_handlers(tmp_path)
        store = SessionStore(tmp_path / 'queue.db')
        waiting = asyncio.Event()
        async def download(context, payload):
            await waiting.wait()
            return {}
        jobs = JobManager(store, {'download.modelscope': download})
        files = [HubModelFile(name=name, byte_size=10) for name in
                 (SHARED, 'Q3/model-EW-Q3-00001-of-00003.mfq', 'Q2.5/model-EW-Q2.5-00001-of-00003.mfq')]
        hub = SimpleNamespace(download_files=AsyncMock(return_value=SimpleNamespace(files=files)))
        service = ServerService(store, FakeBackend(), jobs=jobs, hub_catalog=hub, tool_handlers=handlers)
        def request(precision):
            return CreateJobRequest(kind='download.modelscope', payload={
                'repo_id': 'owner/model', 'destination': 'models/owner/model',
                'include': [SHARED, f'{precision}/model-EW-{precision}-00001-of-00003.mfq'],
            })
        try:
            first = await service.create_job(request('Q3'))
            second = await service.create_job(request('Q2.5'))
            repeated = await service.create_job(request('Q3'))
            assert first.id != second.id and first.id == repeated.id
            assert len(store.list_jobs()) == 2
        finally:
            await jobs.close()
    asyncio.run(run())


def test_shared_directory_download_and_registration_do_not_interleave(tmp_path):
    async def run():
        handlers = make_handlers(tmp_path)
        request, destination = handlers.download_request('download.modelscope', {
            'repo_id': 'owner/model', 'destination': 'models/owner/model',
        })
        events = []
        async def download(context, argv, env, destination, expected, *, selection=None):
            events.append(('start', context.name))
            await asyncio.sleep(0)
        async def result(context, *args, **kwargs):
            await asyncio.sleep(0)
            events.append(('registered', context.name))
            return {}
        handlers._run_download, handlers._download_result = download, result
        contexts = [SimpleNamespace(name=name, log=AsyncMock()) for name in ('Q3', 'Q2.5')]
        await asyncio.gather(*(handlers._download_and_register(context, [], {}, destination, request, 'modelscope')
                               for context in contexts))
        assert events == [('start', 'Q3'), ('registered', 'Q3'), ('start', 'Q2.5'), ('registered', 'Q2.5')]
    asyncio.run(run())


def test_download_progress_ignores_other_tiers_but_counts_shared_tables_and_partial_files(tmp_path):
    async def run():
        handlers = make_handlers(tmp_path)
        request, destination = handlers.download_request('download.modelscope', {
            'repo_id': 'owner/model', 'destination': 'models/owner/model',
            'include': ['Q3/*.mfq', SHARED], 'expected_bytes': 4096,
        })
        (destination / 'Q2').mkdir(parents=True)
        (destination / 'Q2/other.mfq').write_bytes(b'x' * 65536)
        table = destination / SHARED
        table.parent.mkdir(parents=True)
        table.write_bytes(b'x' * 1024)
        partial = destination / '.cache/huggingface/download/Q3/model.mfq.hash.incomplete'
        partial.parent.mkdir(parents=True)
        captured = []
        class Context:
            async def progress(self, value, **kwargs):
                captured.append(kwargs['data'])
        async def transfer(*args, **kwargs):
            partial.write_bytes(b'x' * 1024)
            await asyncio.sleep(1.1)
        handlers._run = transfer
        await handlers._run_download(Context(), [], {}, destination, 4096, selection=request)
        assert captured[-1]['downloaded_bytes'] == 2048
        assert captured[-1]['files_completed'] == 1
    asyncio.run(run())


def test_resuming_old_download_persists_new_engram_dependency(tmp_path):
    async def run():
        from mfq.server.services.jobs import JobManager
        from mfq.server.protocol.models import ErrorDetail
        handlers = make_handlers(tmp_path)
        store = SessionStore(tmp_path / 'resume.db')
        primary = 'Q3/model-EW-Q3-00001-of-00003.mfq'
        previous = store.create_job('download.modelscope', {
            'repo_id': 'owner/model', 'destination': 'models/owner/model', 'include': [primary]})
        store.fail_job(previous.id, ErrorDetail(code='interrupted', message='fixture'))
        observed = []
        async def download(context, payload):
            observed.append(payload)
            return {}
        jobs = JobManager(store, {'download.modelscope': download})
        files = [HubModelFile(name=name, byte_size=10) for name in (primary, SHARED)]
        hub = SimpleNamespace(download_files=AsyncMock(return_value=SimpleNamespace(files=files)))
        service = ServerService(store, FakeBackend(), jobs=jobs, hub_catalog=hub, tool_handlers=handlers)
        try:
            resumed = await service.retry_job(previous.id)
            assert resumed.id == previous.id
            assert SHARED in resumed.payload['include']
            for _ in range(100):
                if observed:
                    break
                await asyncio.sleep(.001)
            assert SHARED in observed[0]['include']
        finally:
            await jobs.close()
    asyncio.run(run())
