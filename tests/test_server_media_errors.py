"""Invalid media is a client error on both public inference protocols."""

import asyncio
import base64
import io
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from mfq.server.api import create_app
from mfq.server.runtime.backend import OpenAIChatBackend
from mfq.server.runtime.client import HttpRuntimeClient


def _wrong_image_mime():
    data = io.BytesIO()
    Image.new("RGB", (2, 2)).save(data, format="PNG")
    return "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode("ascii")


INVALID_URLS = [
    "data:image/png;base64,aGVsbG8=",
    "data:image/png;base64,iVBORw0KGgo=",
    "data:image/png;base64,%%%",
    "data:text/plain;base64,aGVsbG8=",
    "data:image/png;base64,",
    "http://example.invalid/image.png",
    _wrong_image_mime(),
]


@pytest.mark.parametrize("url", INVALID_URLS)
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_image_is_400(url, protocol, stream):
    async def run():
        async def unexpected(request):
            pytest.fail(f"invalid media reached runtime: {request.url}")

        native = httpx.AsyncClient(transport=httpx.MockTransport(unexpected))
        backend = OpenAIChatBackend(HttpRuntimeClient("http://runtime", client=native), model_type="nemotron_h")
        service = SimpleNamespace(backend=backend, resolve_model_alias=lambda name: name)
        app = create_app(service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            content = [{"type": "image_url", "image_url": {"url": url}}]
            endpoint = "/v1/chat/completions"
            if protocol == "anthropic":
                endpoint = "/v1/messages"
                if url.startswith("data:") and "," in url:
                    header, data = url.split(",", 1)
                    source = {"type": "base64", "media_type": header[5:].split(";")[0], "data": data}
                else:
                    source = {"type": "url", "url": url}
                content = [{"type": "image", "source": source}]
            response = await client.post(endpoint, json={
                "model": "tiny", "max_tokens": 1, "stream": stream,
                "messages": [{"role": "user", "content": content}],
            })
            assert response.status_code == 400, response.text
            assert response.json()["error"]["message"]
            if protocol == "anthropic":
                assert response.json()["error"]["type"] == "invalid_request_error"
        await native.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("status", [{"status": "unhealthy"}, {"healthy": False}, {"runtime_state": "failed"}])
def test_health_reflects_failed_runtime(status):
    async def run():
        async def runtime_status():
            return status

        app = create_app(SimpleNamespace(runtime_status=runtime_status))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 503
            assert response.json()["status"] == "unhealthy"

    asyncio.run(run())
