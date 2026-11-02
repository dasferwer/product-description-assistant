from fastapi.testclient import TestClient

from app.main import app


def test_missing_configuration_returns_503(monkeypatch, product):
    monkeypatch.setattr("app.main.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with TestClient(app) as client:
        response = client.post("/generate", json=product)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_not_configured"


def test_docs_and_schema_available_without_configuration(monkeypatch):
    monkeypatch.setattr("app.main.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
    assert "/generate" in schema["paths"]


def test_local_provider_does_not_require_api_key(monkeypatch):
    monkeypatch.setattr("app.main.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with TestClient(app):
        assert app.state.gateway is not None
        assert app.state.gateway.model == "test-model"
