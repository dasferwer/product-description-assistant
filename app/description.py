import re

from app.models import DescriptionClaim, Product


def description_facts(product: Product) -> dict[str, tuple[str, str]]:
    """Целые исходные предложения и значения; второй элемент — поле-источник."""
    facts = {
        "name": (product.name, "name"),
        "manufacturer": ("Производитель — " + product.manufacturer, "manufacturer"),
    }
    sentences = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])", product.description.strip())
    for index, sentence in enumerate(sentences):
        if sentence:
            facts[f"description/{index}"] = (sentence, "description")
    for field, value in product.source_fields().items():
        if field.startswith("specs/"):
            facts[field] = (field[6:] + ": " + value, field)
    return facts


def compose_description(facts: dict[str, tuple[str, str]], priority: list[str]) -> DescriptionClaim:
    if not priority or any(key not in facts for key in priority):
        raise ValueError("Для описания нужны известные идентификаторы фактов.")

    def render(keys: list[str]) -> str:
        prose, specs = [], []
        for key in keys:
            text, source = facts[key]
            if source.startswith("specs/"):
                specs.append(text)
            else:
                prose.append(text if text.endswith((".", "!", "?")) else text + ".")
        if specs:
            prose.append("Характеристики: " + "; ".join(specs) + ".")
        return " ".join(prose)

    selected: list[str] = []
    text = ""
    for key in dict.fromkeys([*priority, *facts]):
        sentence, _ = facts[key]
        if sentence.casefold() in text.casefold():
            continue
        candidate = render([*selected, key])
        if len(candidate) > 1200:
            continue
        selected.append(key)
        text = candidate
        if len(text) >= 800:
            break
    if len(text) < 800:
        raise ValueError(
            f"Из целых подтверждённых фактов получилось {len(text)} символов. "
            "Выбери другой порядок фактов либо запроси действительно отсутствующие сведения; не добавляй наполнитель.")
    return DescriptionClaim(text=text, sources=list(dict.fromkeys(facts[key][1] for key in selected)))
