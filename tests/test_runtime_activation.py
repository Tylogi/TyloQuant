import asyncio
from types import SimpleNamespace

import pytest

from mfq.server.protocol.models import RuntimeInstanceState, RuntimeMemoryPolicy, SamplingParams
from mfq.server.runtime.backend import BackendDelta
from mfq.server.runtime.client import BackendError
from mfq.server.runtime.runtime_pool import RuntimePool
from mfq.server.state.catalog import ModelCatalog
from tests.test_server_models import _fake_runtime, _model, _TestJobContext


@pytest.mark.parametrize("checkpoint", ["budget", "voice", "commit"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_model_activation_is_published_only_after_the_load_transaction(tmp_path, monkeypatch, checkpoint, cancelled):
    async def run():
        _model(tmp_path / "activating.mfq", architecture="qwen35")
        executable = tmp_path / "fake-runtime"
        _fake_runtime(executable, monkeypatch)
        pool = RuntimePool(ModelCatalog([tmp_path], cache_seconds=0), executable, startup_timeout_seconds=5)
        pool.memory_policy = RuntimeMemoryPolicy(prefix_directory=str(tmp_path / "prefix-cache"))
        entered, release = asyncio.Event(), asyncio.Event()

        class Context(_TestJobContext):
            async def progress(self, value, **kwargs):
                if checkpoint == "commit" and value == 1.0 and kwargs.get("message") == "Model ready":
                    entered.set()
                    await release.wait()

        context = Context()
        rebalance = pool._rebalance_resident_budget

        async def delayed_rebalance(*args, **kwargs):
            if checkpoint == "budget" and kwargs.get("activating") is not None:
                entered.set()
                await release.wait()
            return await rebalance(*args, **kwargs)

        monkeypatch.setattr(pool, "_rebalance_resident_budget", delayed_rebalance)
        if checkpoint == "voice":
            pool.voice_component = SimpleNamespace(ready=lambda: True)

            async def delayed_voice(instance):
                assert pool._realtime_activation_lock.locked()
                assert instance.state == RuntimeInstanceState.LOADING and instance.control_leases == 1
                entered.set()
                await release.wait()
                return {"active": True}

            monkeypatch.setattr(pool, "_activate_realtime_instance", delayed_voice)
        loading = asyncio.create_task(pool.load(context, {"model": "activating"}))
        waiter = None
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            instance = next(iter(pool._instances.values()))
            assert (await pool.instances()).data[0].state == RuntimeInstanceState.LOADING
            assert (await pool.runtime_models())["data"] == []
            assert instance.control_leases == 1
            async def output(**_kwargs):
                yield BackendDelta(content_delta="activation-ok", finish_reason="stop")

            async def request():
                return [delta.content_delta async for delta in pool.stream(model="activating",
                    messages=[{"role": "user", "content": "activation check"}], sampling=SamplingParams(max_tokens=1))]

            monkeypatch.setattr(instance.backend, "stream", output)
            waiter = asyncio.create_task(request())
            await asyncio.sleep(0.02)
            assert not waiter.done()
            if cancelled:
                loading.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await loading
                await context.cleanup()
                with pytest.raises(BackendError) as failure:
                    await asyncio.wait_for(waiter, timeout=1)
                assert failure.value.code == "runtime_start_failed"
                assert (await pool.instances()).data == []
                assert (await pool.runtime_models())["data"] == []
                assert instance.process.returncode is not None
            else:
                release.set()
                result = await asyncio.wait_for(loading, timeout=5)
                assert result["instance_id"] == str(instance.id)
                assert await asyncio.wait_for(waiter, timeout=1) == ["activation-ok"]
                assert (await pool.instances()).data[0].state == RuntimeInstanceState.READY
                assert instance.control_leases == 0
                assert (await pool.runtime_models())["data"][0]["id"] == "activating"
        finally:
            release.set()
            await asyncio.gather(loading, return_exceptions=True)
            await context.cleanup()
            if waiter is not None:
                await asyncio.gather(waiter, return_exceptions=True)
            await pool.aclose()

    asyncio.run(run())
