"""Live but unhealthy managed runtimes are replaced using fake stdio only."""

import asyncio
from types import SimpleNamespace

import pytest
import httpx

from mfq.server.protocol.models import ModelLoadRequest, RuntimeInstanceState, SamplingParams
from mfq.server.runtime.backend import BackendDelta, OpenAIChatBackend
from mfq.server.runtime.client import BackendError, HttpRuntimeClient
from tests.test_runtime_stdio_client import _client
from tests.test_server_concurrency import _instance, _pool


class FakeProcess:
    """An event-driven process; no OS process is launched or signalled."""

    def __init__(self):
        self.returncode = None
        self.exited = asyncio.Event()
        self.terminations = 0

    def terminate(self):
        self.terminations += 1
        self.returncode = -1
        self.exited.set()

    kill = terminate

    async def wait(self):
        await self.exited.wait()
        return self.returncode


class FakeBackend(OpenAIChatBackend):
    def __init__(self, status=None, error_code=None, health=None):
        self.error_code = error_code
        self.closed = False
        self.started = asyncio.Event()
        self.request_gate = None

        def handler(request):
            if request['op'] == 'generate':
                if self.error_code:
                    return [{'v': 1, 'id': request['id'], 'type': 'error', 'error': {
                        'code': self.error_code, 'message': 'fake failure', 'status_code': 500}}]
                return [{'v': 1, 'id': request['id'], 'type': 'done'}]
            data = (health or {'status': 'ok'}) if request['op'] == 'health' else (status or {})
            return [{'v': 1, 'id': request['id'], 'type': 'result', 'data': data}]

        super().__init__(_client(handler), model_type='qwen35')

    async def stream(self, **_kwargs):
        self.started.set()
        if self.request_gate is not None:
            await self.request_gate.wait()
        async with self._runtime.generate({'model': 'model', 'messages': [], 'stream': True}) as events:
            async for _event in events:
                pass
        yield BackendDelta(content_delta='restored', finish_reason='stop')

    async def aclose(self):
        self.closed = True
        await super().aclose()


def setup_pool(monkeypatch, *, status=None, error_code=None, health=None):
    pool = _pool(monkeypatch, backend='metal')
    instance = _instance(capacity=1, backend=FakeBackend(status, error_code, health))
    instance.process = FakeProcess()
    pool._instances[instance.id] = instance
    pool._last_instance_id = instance.id
    saved = ModelLoadRequest(model='model', context_size=8192, pin=True, idle_ttl_seconds=600,
                             max_requests=8, max_queued_requests=67, queue_timeout_seconds=6.25)
    pool._load_requests['model'] = saved
    loads = []
    loading = asyncio.Event()
    release = asyncio.Event()

    async def load(_context, request):
        assert instance.process.returncode is not None
        loads.append(request)
        loading.set()
        await release.wait()
        replacement = _instance(capacity=1, backend=FakeBackend())
        replacement.artifact = instance.artifact
        replacement.process = FakeProcess()
        replacement.context_size = request['context_size']
        pool._instances[replacement.id] = replacement
        pool._last_instance_id = replacement.id

    async def resolve(_model):
        return instance.artifact

    pool.catalog = SimpleNamespace(resolve=resolve)
    monkeypatch.setattr(pool, 'load', load)
    return pool, instance, saved, loads, loading, release


async def drain(pool):
    return [delta.content_delta async for delta in pool.stream(
        model='model', messages=[{'role': 'user', 'content': 'hello'}], sampling=SamplingParams())]


async def finish_recovery(pool, release):
    tasks = tuple(pool._background_tasks)
    release.set()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)


@pytest.mark.parametrize('status,health', [
    ({'healthy': False}, None), ({'healthy': 0}, None), ({'state': 'unhealthy'}, None),
    ({'state': 'failed'}, None), ({'status': 'unhealthy'}, None),
    ({'runtime_state': 'unhealthy'}, None), ({}, {'status': 'unhealthy'}),
])
def test_unhealthy_alive_runtime_is_restarted_and_recovers(monkeypatch, status, health):
    async def run():
        pool, instance, saved, loads, loading, release = setup_pool(monkeypatch, status=status, health=health)
        try:
            assert instance.process.returncode is None
            await pool._refresh_instance_usage(instance)
            await asyncio.wait_for(loading.wait(), timeout=2)
            assert instance.process.terminations == 1
            assert instance.backend.closed
            for model in ('model', instance.artifact.resource.id):
                with pytest.raises(BackendError) as raised:
                    await pool.preflight(model=model, messages=[], sampling=SamplingParams())
                assert raised.value.status_code == 503
                assert raised.value.code == 'runtime_restarting'
                assert raised.value.retryable
            await finish_recovery(pool, release)
            assert len(loads) == 1
            assert loads[0]['context_size'] == saved.context_size
            assert loads[0]['pin'] and loads[0]['idle_ttl_seconds'] == 600
            assert loads[0] == {**saved.model_dump(mode='python'),
                                'artifact_uri': f'mfq://{instance.artifact.resource.id}'}
            assert await drain(pool) == ['restored']
        finally:
            release.set()
            await pool.aclose()

    asyncio.run(run())


@pytest.mark.parametrize('code', ['backend_failure', 'runtime_unhealthy', 'backend_unhealthy'])
def test_fatal_request_error_restarts_once_and_rejects_waiting_requests(monkeypatch, code):
    async def run():
        pool, instance, _saved, loads, loading, release = setup_pool(monkeypatch, error_code=code)
        queued = None
        active = None
        try:
            # Leave one caller waiting in the admission gate while a second
            # caller reports the sticky error through the real stdio client.
            instance.backend.request_gate = asyncio.Event()
            active = asyncio.create_task(drain(pool))
            await asyncio.wait_for(instance.backend.started.wait(), timeout=2)
            queued = asyncio.create_task(drain(pool))
            for _ in range(20):
                if instance.queued_requests:
                    break
                await asyncio.sleep(0)
            assert instance.queued_requests == 1
            instance.backend.request_gate.set()
            with pytest.raises(BackendError) as raised:
                await asyncio.wait_for(active, timeout=2)
            assert raised.value.status_code == 503
            assert raised.value.code == 'runtime_restarting'
            with pytest.raises(BackendError) as waiting_error:
                await asyncio.wait_for(queued, timeout=2)
            assert waiting_error.value.code == 'runtime_restarting'
            await asyncio.wait_for(loading.wait(), timeout=2)
            await pool._recover_unhealthy_runtime(instance, 'duplicate notification')
            assert instance.process.terminations == 1
            await finish_recovery(pool, release)
            assert len(loads) == 1
            assert await drain(pool) == ['restored']
        finally:
            if queued is not None:
                queued.cancel()
                await asyncio.gather(queued, return_exceptions=True)
            if active is not None:
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
            release.set()
            await pool.aclose()

    asyncio.run(run())


@pytest.mark.parametrize('code', ['invalid_request', 'resource_exhausted', 'backend_timeout', 'backend_http_503'])
def test_ordinary_request_error_does_not_restart(monkeypatch, code):
    async def run():
        pool, instance, _saved, loads, _loading, release = setup_pool(monkeypatch, error_code=code)
        try:
            with pytest.raises(BackendError) as raised:
                await drain(pool)
            assert raised.value.code == code
            assert instance.state == RuntimeInstanceState.READY
            assert instance.process.returncode is None
            assert not loads and not pool._recovering_runtimes
            instance.backend.error_code = None
            assert await drain(pool) == ['restored']
        finally:
            release.set()
            await pool.aclose()

    asyncio.run(run())


def test_health_http_503_is_explicit_without_matching_generic_503():
    error = HttpRuntimeClient._http_error(503, b'{"status":"unhealthy","model":"test"}')
    assert error.code == 'runtime_unhealthy'
    assert HttpRuntimeClient._http_error(503, b'{"error":{"code":"resource_exhausted"}}').code == 'resource_exhausted'


def test_missing_health_fields_are_compatible(monkeypatch):
    async def run():
        pool, instance, _saved, loads, _loading, release = setup_pool(monkeypatch)
        try:
            await pool._refresh_instance_usage(instance)
            assert instance.state == RuntimeInstanceState.READY
            assert not loads
        finally:
            release.set()
            await pool.aclose()

    asyncio.run(run())


def test_periodic_monitor_recovers_without_an_inference_request(monkeypatch):
    async def run():
        pool, instance, _saved, loads, loading, release = setup_pool(monkeypatch, health={'status': 'unhealthy'})

        async def noop(*_args, **_kwargs):
            return None

        monkeypatch.setattr(pool, '_maintain_prefix_disk_budget', noop)
        try:
            await pool.start()
            pool._idle_reaper_wakeup.set()
            await asyncio.wait_for(loading.wait(), timeout=2)
            await finish_recovery(pool, release)
            assert instance.process.terminations == 1
            assert len(loads) == 1
            assert await drain(pool) == ['restored']
        finally:
            release.set()
            await pool.aclose()

    asyncio.run(run())


def test_status_query_starts_recovery_and_returns_503(monkeypatch):
    async def run():
        pool, instance, _saved, _loads, loading, release = setup_pool(monkeypatch, status={'healthy': False})

        async def noop(*_args, **_kwargs):
            return None

        monkeypatch.setattr(pool, '_maintain_prefix_disk_budget', noop)
        try:
            with pytest.raises(BackendError) as raised:
                await pool.runtime_status()
            assert raised.value.code == 'runtime_restarting' and raised.value.status_code == 503
            await asyncio.wait_for(loading.wait(), timeout=2)
            await finish_recovery(pool, release)
            assert await drain(pool) == ['restored']
        finally:
            release.set()
            await pool.aclose()

    asyncio.run(run())


def test_recovery_respects_existing_cooldown_and_pool_close_cancels_it(monkeypatch):
    async def run():
        pool, instance, _saved, loads, _loading, release = setup_pool(monkeypatch, status={'healthy': False})
        pool.load_failure_cooldown_seconds = 60
        assert pool._begin_runtime_revival('model')
        try:
            await pool._refresh_instance_usage(instance)
            await asyncio.wait_for(instance.process.exited.wait(), timeout=2)
            await asyncio.sleep(0)
            assert not loads
            assert 'model' in pool._recovering_runtimes
        finally:
            release.set()
            tasks = tuple(pool._background_tasks)
            await pool.aclose()
        assert all(task.done() for task in tasks)
        assert not pool._recovering_runtimes

    asyncio.run(run())


@pytest.mark.parametrize('stream', [False, True])
def test_http_requests_get_503_during_recovery_then_succeed(monkeypatch, stream):
    from mfq.server.api import create_app
    from tests.test_server_openai_compat import _Service

    async def run():
        pool, instance, _saved, _loads, loading, release = setup_pool(monkeypatch, status={'healthy': False})
        try:
            await pool._refresh_instance_usage(instance)
            await asyncio.wait_for(loading.wait(), timeout=2)
            app = create_app(_Service(pool))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                body = {'model': 'model', 'messages': [{'role': 'user', 'content': 'hello'}], 'stream': stream}
                response = await client.post('/v1/chat/completions', json=body)
                assert response.status_code == 503
                assert response.headers['retry-after'] == '1'
                assert response.json()['error']['type'] == 'runtime_restarting'
                await finish_recovery(pool, release)
                response = await client.post('/v1/chat/completions', json=body)
                assert response.status_code == 200 and 'restored' in response.text
        finally:
            release.set()
            await pool.aclose()

    asyncio.run(run())
