import pytest

from app.models import Draft, Product
from app.service import explicit_conflicts


@pytest.mark.parametrize("first,second", [
    ("±0,1 кПа при +25 °C", "±0,2 кПа при −20 °C"),
    ("−30…+70 °C при измерении", "−40…+85 °C при хранении"),
    ("±0,1 кПа в диапазоне 0…+25 °C", "±0,2 кПа в диапазоне −20…0 °C"),
    ("±0,1 кПа (при +25 °C)", "±0,2 кПа (при −20 °C)"),
    ("±0,1 кПа при+25 °C", "±0,2 кПа при−20 °C"),
    ("±0,1 кПа при давлении 10 MPa", "±0,2 кПа при давлении 10 mPa"),
])
def test_explicitly_different_conditions_are_not_a_conflict(product, first, second):
    product["specs"] = {}
    product["description"] += f"\nПараметр: {first}.\nПараметр: {second}."
    assert explicit_conflicts(Product.model_validate(product)) == []


@pytest.mark.parametrize("unknown_spec", [False, True])
def test_conditional_values_are_both_preserved_in_ready_response(client, product, gateway, draft, unknown_spec):
    product["specs"] = {"погрешность": None} if unknown_spec else {}
    values = ["±0,1 кПа при +25 °C", "±0,2 кПа при −20 °C"]
    product["description"] += "".join(f"\nПогрешность: {value}." for value in values)
    draft["description_characteristics"] = [{"name": "погрешность", "value": value,
        "source": {"field": "description", "quote": f"Погрешность: {value}."}} for value in values]
    gateway.generate.return_value = Draft.model_validate(draft)
    response = client.post("/generate", json=product)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["conflicts"] == []
    assert [row["value"] for row in body["key_characteristics"] if row["name"] == "погрешность"] == values
    assert not any(issue["field"] == "specs/погрешность" for issue in body["missing_data"])
    gateway.audit.assert_awaited_once()


@pytest.mark.parametrize("second", ["±0,2 кПа при +25 °C", "±0,2 кПа при 25 °С",
                                     "±0,2 кПа (при +25 °C)", "±0,2 кПа (при 25 °С)"])
def test_same_conditions_with_different_values_still_require_clarification(client, product, gateway, second):
    product["specs"] = {}
    product["description"] += f"\nПогрешность: ±0,1 кПа при +25 °C.\nПогрешность: {second}."
    body = client.post("/generate", json=product).json()
    assert body["status"] == "needs_clarification"
    assert body["conflicts"]
    gateway.generate.assert_not_awaited()


def test_nonadjacent_matching_conditions_do_not_hide_real_conflict(product):
    product["specs"] = {}
    product["description"] += (
        "\nПогрешность: ±0,1 кПа при +25 °C.\nПогрешность: ±0,2 кПа при −20 °C."
        "\nПогрешность: ±0,3 кПа при +25 °C.")
    conflicts = explicit_conflicts(Product.model_validate(product))
    assert len(conflicts) == 1
    assert "±0,1" in conflicts[0].sources[0].quote
    assert "±0,3" in conflicts[0].sources[1].quote


def test_conditional_numeric_aliases_are_deduplicated_without_losing_conditions(client, product, gateway, draft):
    product["specs"] = {}
    values = ["±0,1 кПа при +25 °C", "±0.1 kPa при 25 °С"]
    product["description"] += "".join(f"\nПогрешность: {value}." for value in values)
    draft["description_characteristics"] = [{"name": "погрешность", "value": value,
        "source": {"field": "description", "quote": f"Погрешность: {value}."}} for value in values]
    gateway.generate.return_value = Draft.model_validate(draft)
    body = client.post("/generate", json=product).json()
    assert body["status"] == "ready"
    assert [row["value"] for row in body["key_characteristics"] if row["name"] == "погрешность"] == values[:1]


@pytest.mark.parametrize("values", [
    ["измерение избыточного давления воздуха", "контроль давления на испытательном стенде"],
    ["0–100 кПа", "0–0,1 МПа"],
])
def test_unproven_string_or_unit_disagreement_is_left_for_semantic_audit(client, product, gateway, draft, values):
    product["specs"] = {}
    product["description"] += "".join(f"\nПараметр: {value}." for value in values)
    draft["description_characteristics"] = [{"name": "параметр", "value": value,
        "source": {"field": "description", "quote": f"Параметр: {value}."}} for value in values]
    gateway.generate.return_value = Draft.model_validate(draft)
    body = client.post("/generate", json=product).json()
    assert body["status"] == "ready"
    assert [row["value"] for row in body["key_characteristics"] if row["name"] == "параметр"] == values
    gateway.audit.assert_awaited_once()


def test_other_unit_does_not_hide_a_proven_same_unit_conflict(product):
    product["specs"] = {}
    product["description"] += (
        "\nДополнительный диапазон: 0–0,1 МПа.\nДополнительный диапазон: 0–100 кПа."
        "\nДополнительный диапазон: 0–200 кПа.")
    conflicts = explicit_conflicts(Product.model_validate(product))
    assert len(conflicts) == 1
    assert "0–100 кПа" in conflicts[0].sources[0].quote
    assert "0–200 кПа" in conflicts[0].sources[1].quote
