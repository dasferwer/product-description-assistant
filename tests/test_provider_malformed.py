import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from ollama import AsyncClient

from app.main import app, get_gateway
from app.models import Audit
from app.prompts import AUDIT_PROMPT
from app.provider import OllamaGateway


def envelope(content: str) -> bytes:
    return json.dumps({
        "model": "test-model", "message": {"role": "assistant", "content": content},
        "done": True, "done_reason": "stop",
    }).encode()


@pytest.mark.parametrize("body", [
    b"[]", b"null", b"7", b"\xff", b"9" * 4301,
    envelope("9" * 4301), envelope("[" * 1500 + "]" * 1500),
])
def test_malformed_external_responses_have_controlled_bounded_error(product, body):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=body, headers={"Content-Type": "application/json"})

    sdk = AsyncClient(host="http://example.invalid", transport=httpx.MockTransport(handler))
    gateway = OllamaGateway(sdk, "test-model")
    app.dependency_overrides[get_gateway] = lambda: gateway
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/generate", json=product)
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "invalid_llm_output"
        assert len(calls) == 2
        assert "Internal Server Error" not in response.text
    finally:
        app.dependency_overrides.clear()
        asyncio.run(sdk.__aexit__(None, None, None))


def test_application_validation_errors_are_not_hidden_as_provider_failures(monkeypatch):
    def broken_validation(value):
        raise TypeError("application validation failure")

    monkeypatch.setattr(Audit, "model_validate", broken_validation)

    async def run():
        async with AsyncClient(host="http://example.invalid", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=envelope("{}")),
        )) as sdk:
            await OllamaGateway(sdk, "test-model")._parse(AUDIT_PROMPT, {"source_fields": {}}, Audit)

    with pytest.raises(TypeError, match="application validation failure"):
        asyncio.run(run())
