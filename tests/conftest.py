"""Искусственные данные для тестов; это не товары из задания."""

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_gateway
from app.models import Audit, Draft


@pytest.fixture
def product():
    path = Path(__file__).parent / "fixtures" / "synthetic_product.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def draft(product):
    def claim(text, source="description", quote=None):
        return {"text": text, "sources": [source]}

    return {
        "status": "ready",
        "seo_title": claim(product["name"], "name"),
        "seo_description": claim(product["description"]),
        "search_queries": [
            claim(product["name"], "name"),
            claim("измерения избыточного давления воздуха"),
            claim("контроля давления воздуха"),
        ],
        "audience": claim("Оператор стенда", quote="Оператор стенда"),
        "tasks": claim("контроля давления воздуха"),
        "description_characteristics": [],
        "missing_data": [],
        "conflicts": [],
    }


@pytest.fixture
def gateway(draft):
    gateway = AsyncMock()
    gateway.generate.return_value = Draft.model_validate(copy.deepcopy(draft))
    gateway.audit.return_value = Audit(
        supported=True, problems=[], missing_data=[], conflicts=[]
    )
    return gateway


@pytest.fixture
def client(gateway):
    app.dependency_overrides[get_gateway] = lambda: gateway
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
