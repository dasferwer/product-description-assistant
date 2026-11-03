import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from ollama import AsyncClient

from app.main import app, get_gateway
from app.models import Audit
from app.prompts import AUDIT_PROMPT
from app.provider import OllamaGateway, ServiceError


@pytest.mark.parametrize("status,expected_status,code,attempts", [
    (401, 502, "provider_authorization", 1),
    (429, 503, "provider_rate_limit", 2),
    (500, 502, "provider_error", 2),
])
@pytest.mark.parametrize("body", [
    b"[]", b"null", b"7", b'"unexpected"', b"true", b"",
    b"<html>private provider details</html>", b"\xff",
    b'{"error": ["private provider details"]}',
    b"9" * 4301, b"[" * 1500 + b"]" * 1500,
])
def test_http_status_survives_malformed_error_body(
        product, monkeypatch, status, expected_status, code, attempts, body):
    monkeypatch.setattr("app.provider.asyncio.sleep", AsyncMock())
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, content=body)

    sdk = AsyncClient(host="http://example.invalid", transport=httpx.MockTransport(handler))
    app.dependency_overrides[get_gateway] = lambda: OllamaGateway(sdk, "test-model")
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/generate", json=product)
        assert response.status_code == expected_status
        assert response.json()["error"]["code"] == code
        assert "private provider details" not in response.text
        assert len(calls) == attempts
    finally:
        app.dependency_overrides.clear()
        asyncio.run(sdk.__aexit__(None, None, None))


def test_malformed_temporary_http_failure_can_recover(monkeypatch):
    monkeypatch.setattr("app.provider.asyncio.sleep", AsyncMock())
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(500, content=b"[]")
        return httpx.Response(200, json={
            "model": "test-model", "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps({
                "supported": True, "problems": [], "missing_data": [], "conflicts": [],
            })},
        })

    async def run():
        async with AsyncClient(host="http://example.invalid", transport=httpx.MockTransport(handler)) as sdk:
            return await OllamaGateway(sdk, "test-model")._parse(
                AUDIT_PROMPT, {"source_fields": {}}, Audit)

    assert asyncio.run(run()).supported
    assert len(calls) == 2


def test_attribute_error_without_http_failure_is_not_hidden():
    client = AsyncMock()
    client.chat.side_effect = AttributeError("application failure")

    with pytest.raises(AttributeError, match="application failure"):
        asyncio.run(OllamaGateway(client, "test-model")._parse(
            AUDIT_PROMPT, {"source_fields": {}}, Audit))
    assert client.chat.await_count == 1


def test_provider_timeout_cancels_each_bounded_attempt(monkeypatch):
    monkeypatch.setattr("app.provider.CALL_TIMEOUT", 0.01)
    monkeypatch.setattr("app.provider.asyncio.sleep", AsyncMock())
    cancelled = []

    async def wait(**kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    client = AsyncMock()
    client.chat.side_effect = wait
    with pytest.raises(ServiceError) as error:
        asyncio.run(OllamaGateway(client, "test-model")._parse(
            AUDIT_PROMPT, {"source_fields": {}}, Audit))
    assert (error.value.status, error.value.code) == (504, "provider_timeout")
    assert client.chat.await_count == len(cancelled) == 2


def test_caller_cancellation_is_not_retried():
    started = asyncio.Event()
    cancelled = []

    async def wait(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    client = AsyncMock()
    client.chat.side_effect = wait

    async def run():
        task = asyncio.create_task(OllamaGateway(client, "test-model")._parse(
            AUDIT_PROMPT, {"source_fields": {}}, Audit))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert client.chat.await_count == len(cancelled) == 1
