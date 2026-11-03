import pytest

from app.models import Draft


def test_ambiguous_labelled_field_is_a_controlled_input_error(client, product, gateway):
    product["specs"] = {"корпус": "сталь", "КОРПУС": "сталь"}
    response = client.post("/generate", json=product)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"
    gateway.generate.assert_not_awaited()


@pytest.mark.parametrize("first,second", [
    ("0–10 кПа", "0-10 kPa"),
    ("±0,5 °C", "±0.5 °С"),
])
def test_equivalent_labelled_values_are_preserved_once_before_audit(
    client, product, gateway, draft, first, second,
):
    product["specs"] = {}
    product["description"] += f"\nПараметр: {first}.\nПараметр: {second}."
    draft["description_characteristics"] = [
        {"name": "Параметр", "value": value,
         "source": {"field": "description", "quote": f"Параметр: {value}."}}
        for value in (first, second)
    ]
    gateway.generate.return_value = Draft.model_validate(draft)
    response = client.post("/generate", json=product)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["conflicts"] == []
    rows = [row for row in body["key_characteristics"] if row["name"] == "Параметр"]
    assert rows == [{"name": "Параметр", "value": first, "status": "provided"}]
    audited = gateway.audit.await_args.args[1]
    assert len(audited.description_characteristics) == 1


def test_different_labelled_values_still_require_clarification(client, product, gateway):
    product["specs"] = {}
    product["description"] += "\nДиапазон: 0–20 кПа."
    response = client.post("/generate", json=product)
    assert response.status_code == 200
    assert response.json()["status"] == "needs_clarification"
    assert response.json()["conflicts"]
    gateway.generate.assert_not_awaited()
