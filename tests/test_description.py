import json
from pathlib import Path

import pytest

from app.description import compose_description, description_facts
from app.models import Product


def test_short_passport_and_specs_form_description_without_losing_conditions():
    data = json.loads(Path("examples/testo-174-t.input.json").read_text())
    data["description"] = ". ".join(data["description"].split(". ")[:3]).rstrip(".") + "."
    product = Product.model_validate(data)
    facts = description_facts(product)
    priority = ["specs/ресурс батарей", "specs/погрешность температуры", *facts]
    result = compose_description(facts, priority)
    assert 800 <= len(result.text) <= 1200
    assert product.specs["ресурс батарей"] in result.text
    assert product.specs["погрешность температуры"] in result.text
    assert len(result.sources) == len(set(result.sources))


def test_unknown_fact_reference_is_rejected(product):
    with pytest.raises(ValueError):
        compose_description(description_facts(Product.model_validate(product)), ["nonexistent"])


def test_insufficient_facts_are_not_padded():
    product = Product(name="Прибор", manufacturer="Завод", specs={}, description="Измеряет давление.")
    with pytest.raises(ValueError):
        compose_description(description_facts(product), ["name"])


def test_unselected_facts_cannot_fill_description(product):
    product = Product.model_validate(product)
    with pytest.raises(ValueError):
        compose_description(description_facts(product), ["name", "manufacturer"])


def test_multiline_passport_keeps_complete_conditional_facts():
    data = json.loads(Path("examples/testo-174-h.input.json").read_text())
    data["description"] = (
        "Применяется на складах, в офисах и на производстве.\n"
        + "\n".join(f"{key}: {value}" for key, value in data["specs"].items())
    )
    data["specs"] = {}
    facts = description_facts(Product.model_validate(data))
    assert len(facts) > 15
    accuracy = next(text for text, _ in facts.values() if text.startswith("базовая погрешность"))
    assert "t99 = 30 мин" in accuracy and "±0,03 %RH/K" in accuracy
    result = compose_description(facts, list(facts))
    assert 800 <= len(result.text) <= 1200
