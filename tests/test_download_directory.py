from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from mfq.server.api import create_app
from mfq.server.services.jobs import JobExecutionError
from mfq.server.services.tool_jobs import ToolJobHandlers, ToolJobPaths
from mfq.server.state.catalog import ModelCatalog
from tests.platform_helpers import symlink_or_skip


def make_handlers(workspace: Path) -> ToolJobHandlers:
    return ToolJobHandlers(ModelCatalog([workspace / 'models']),
        ToolJobPaths(workspace, Path(sys.executable), None, None, None, None, internal_modelscope=True, internal_huggingface=True))


def test_download_directory_api_persists_and_uses_server_folder_ids(tmp_path: Path) -> None:
    async def run() -> None:
        workspace = tmp_path / 'workspace'
        chosen = tmp_path / 'downloads'
        chosen.mkdir()
        handlers = make_handlers(workspace)
        service = SimpleNamespace(tool_handlers=handlers, catalog=handlers.catalog)
        app = create_app(service)
        folder = await handlers.catalog.browse_directories(path=chosen)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            initial = (await client.get('/api/v1/hub/download-directory')).json()
            assert initial['actual_directory'] == str(workspace / 'models')
            assert initial['directory'] is None
            assert isinstance(initial['disk_total_bytes'], int)
            assert 0 <= initial['disk_free_bytes'] <= initial['disk_total_bytes']
            changed = await client.put('/api/v1/hub/download-directory', json={'directory_id': folder.current_id})
            assert changed.status_code == 200
            assert changed.json()['actual_directory'] == str(chosen)
            reclaim = await client.put('/api/v1/hub/download-directory', json={'evict_prefix_cache': True})
            assert reclaim.status_code == 200 and reclaim.json()['evict_prefix_cache'] is True
            assert reclaim.json()['actual_directory'] == str(chosen)
            restarted = make_handlers(workspace)
            assert restarted.download_directories.status()['actual_directory'] == str(chosen)
            assert restarted.download_directories.evict_prefix_cache is True
            bad = await client.put('/api/v1/hub/download-directory', json={'directory_id': '0' * 32})
            assert bad.status_code == 400
            assert handlers.download_directories.status()['actual_directory'] == str(chosen)
            reset = await client.put('/api/v1/hub/download-directory', json={'directory_id': None})
            assert reset.status_code == 200 and reset.json()['directory'] is None
        assert not list(chosen.iterdir())
    asyncio.run(run())


@pytest.mark.parametrize('use_symlink', [False, True])
def test_download_space_uses_selected_filesystem_and_updates_free_bytes(tmp_path, monkeypatch, use_symlink):
    from mfq.server.services import download_directory
    default = tmp_path / 'models'
    chosen = tmp_path / 'external'
    chosen.mkdir()
    link = chosen
    if use_symlink:
        link = tmp_path / 'external-link'
        symlink_or_skip(link, chosen, target_is_directory=True)
    available = 200
    calls = []
    def usage(path):
        calls.append(path)
        return SimpleNamespace(total=1024 if path == chosen else 2048, free=available)
    monkeypatch.setattr(download_directory.shutil, 'disk_usage', usage)
    settings = download_directory.DownloadDirectorySettings(tmp_path, default)
    assert settings.status()['disk_total_bytes'] == 2048
    selected = asyncio.run(settings.configure(link))
    assert selected['actual_directory'] == str(chosen)
    assert selected['disk_total_bytes'] == 1024 and selected['disk_free_bytes'] == 200
    available = 0
    assert settings.status()['disk_free_bytes'] == 0
    assert calls == [default, chosen, chosen]


def test_unavailable_download_drive_keeps_path_without_reporting_another_disk(tmp_path, monkeypatch):
    from mfq.server.services import download_directory
    settings = download_directory.DownloadDirectorySettings(tmp_path, tmp_path / 'models')
    selected = tmp_path / 'unmounted' / 'models'
    settings.selected = selected
    def unavailable(path):
        assert path == selected
        raise OSError('volume unavailable')
    monkeypatch.setattr(download_directory.shutil, 'disk_usage', unavailable)
    status = settings.status()
    assert status['actual_directory'] == str(selected)
    assert status['disk_total_bytes'] is None and status['disk_free_bytes'] is None


@pytest.mark.parametrize('provider', ['modelscope', 'huggingface'])
def test_downloads_keep_their_chosen_folder_and_register_external_results(tmp_path: Path, monkeypatch, provider: str) -> None:
    async def run() -> None:
        workspace = tmp_path / 'workspace'
        chosen = tmp_path / 'chosen'
        later = tmp_path / 'later'
        chosen.mkdir(); later.mkdir()
        handlers = make_handlers(workspace)
        await handlers.download_directories.configure(chosen)
        await handlers.download_directories.configure(later)
        handlers = make_handlers(workspace)
        registrations = []
        artifacts = []
        async def register(**values):
            registrations.append(values['path'])
        async def transfer(context, argv, env, destination, expected, *, selection=None):
            (destination / 'model-00001-of-00002.mfq').write_bytes(b'first')
            (destination / 'model-00002-of-00002.mfq').write_bytes(b'second')
        async def environment(*args, **kwargs):
            return {}, False
        async def ignore(*args, **kwargs):
            pass
        async def artifact(**values):
            artifacts.append(values)
        monkeypatch.setattr(handlers.catalog, 'register_directory', register)
        monkeypatch.setattr(handlers, '_run_download', transfer)
        monkeypatch.setattr('mfq.server.services.tool_jobs.download_environment', environment)
        context = SimpleNamespace(log=ignore, progress=ignore, artifact=artifact)
        execute = getattr(handlers, f'download_{provider}')
        result = await execute(context, {'repo_id': 'example/model', 'destination': f'models/{provider}/example/model/S4', 'download_directory': str(chosen)})
        destination = chosen / provider / 'example/model/S4'
        assert result['files'] == 2
        assert registrations == [destination]
        assert (destination / 'model-00002-of-00002.mfq').read_bytes() == b'second'
        assert not list(later.iterdir())
        assert handlers._resolve(result['artifact'].removeprefix('workspace://')) == destination
        assert artifacts[0]['metadata']['parameters']['download_directory'] == str(chosen)
        with pytest.raises(JobExecutionError, match='must stay inside'):
            handlers._download_output('models/../../outside', str(chosen))
        with pytest.raises(JobExecutionError, match='not been selected'):
            handlers._download_output('models/example/model', str(tmp_path))
        identifier = handlers.download_directories.identifier(chosen)
        with pytest.raises(JobExecutionError, match='root cannot be removed'):
            handlers.remove_workspace_artifact(f'workspace://download-roots/{identifier}/')
        assert chosen.is_dir()
    asyncio.run(run())
