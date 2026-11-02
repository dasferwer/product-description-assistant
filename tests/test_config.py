from fastapi.testclient import TestClient

from app.main import app


def test_missing_configuration_returns_503(monkeypatch, product):
    monkeypatch.setattr("app.main.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with TestClient(app) as client:
        response = client.post("/generate", json=product)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_not_configured"


def test_docs_and_schema_available_without_key(monkeypatch):
    monkeypatch.setattr("app.main.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 200
        schema = client.get("/openapi.json").json()
    assert "/generate" in schema["paths"]
