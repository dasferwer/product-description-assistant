import json
from pathlib import Path

import pytest

from app.description import labelled_characteristics
from app.models import Product
from app.service import explicit_conflicts


def temperature_product(text: str) -> Product:
    data = json.loads(Path("examples/testo-174-t.input.json").read_text())
    data["description"] = text
    return Product.model_validate(data)


@pytest.mark.parametrize("text", [
    "Диапазон измерения температуры −30…+70 °C, а диапазон хранения −40…+85 °C.",
    "Диапазон хранения −40…+85 °C, а диапазон измерения температуры −30…+70 °C.",
    "Диапазон измерения температуры −30…+70 °C и диапазон температуры хранения −40…+85 °C.",
])
def test_distinct_ranges_are_associated_with_their_own_property(text):
    assert explicit_conflicts(temperature_product(text)) == []


def test_real_measurement_conflict_is_not_hidden_by_matching_storage_range():
    product = temperature_product(
        "Диапазон измерения температуры 0…+50 °C, а диапазон хранения −30…+70 °C.")
    conflict = explicit_conflicts(product)
    assert len(conflict) == 1
    assert conflict[0].field == "specs/диапазон измерения температуры"
    assert conflict[0].sources[1].quote == "0…+50 °C"


def test_identical_labelled_values_are_extracted_once(product):
    product["specs"] = {}
    product["description"] += " Диапазон: 0–10 кПа."
    items = labelled_characteristics(Product.model_validate(product))
    assert sum(item.name.casefold() == "диапазон" for item in items) == 1


def test_different_labelled_values_return_conflict_before_llm(client, product, gateway):
    product["specs"] = {}
    product["description"] += " Диапазон: 0–20 кПа."
    response = client.post("/generate", json=product)
    result = response.json()
    assert result["status"] == "needs_clarification"
    assert result["conflicts"][0]["field"] == "description"
    assert len({source["quote"] for source in result["conflicts"][0]["sources"]}) == 2
    gateway.generate.assert_not_awaited()


def test_long_note_is_not_forced_into_characteristics(product):
    product["description"] += "\nПримечание: " + "текст " * 210
    assert labelled_characteristics(Product.model_validate(product)) == []


def test_long_technical_value_is_preserved_without_truncation(product):
    value = "При заданных условиях; " * 60 + "действительно при t99 = 40 мин"
    product["description"] += "\nПогрешность в условиях испытания: " + value
    items = labelled_characteristics(Product.model_validate(product))
    assert len(items) == 1
    assert items[0].value == value
    assert value in items[0].source.quote


@pytest.mark.parametrize("extra", [" Диапазон: 0–10 кПа.", "\nПримечание: " + "дополнительные условия; " * 50])
def test_ready_plan_survives_duplicate_or_unselected_long_note(client, product, draft, extra):
    from unittest.mock import AsyncMock
    from app.main import app, get_gateway
    from app.models import Audit, GenerationPlan, SourceReview
    from app.provider import OllamaGateway

    product["specs"] = {}; product["description"] += extra
    gateway = OllamaGateway(AsyncMock(), "test-model")
    calls = []

    async def parse(prompt, payload, schema):
        calls.append(schema)
        if schema is SourceReview:
            return SourceReview(decisions=[{"fact_id": key, "kind": "product_data"}
                                           for key in payload["candidates"]])
        if schema is GenerationPlan:
            return GenerationPlan(**{key: value for key, value in draft.items()
                if key not in {"seo_description", "search_queries"}}, description_parts=list(payload["facts"]))
        return Audit(supported=True, problems=[], missing_data=[], conflicts=[])

    gateway._parse = parse
    app.dependency_overrides[get_gateway] = lambda: gateway
    result = client.post("/generate", json=product)
    assert result.status_code == 200
    assert result.json()["status"] == "ready"
    rows = result.json()["key_characteristics"]
    assert sum(row["name"].casefold() == "диапазон" for row in rows) == 1
    assert not any(row["name"] == "Примечание" for row in rows)
    assert calls == [SourceReview, GenerationPlan, Audit]
