import re

from app.models import DescriptionClaim, ExtractedCharacteristic, Product, Reference, unknown


def value_conditions(value: str) -> tuple[str, str | None]:
    """Значение и явно записанные условия; исходный текст не изменяется."""
    match = re.search(r"(?:\s+|\s*\()(?:при|для|в\s+(?:условиях|диапазоне|режиме)|во?\s+время)\b\s*",
                      value, re.IGNORECASE)
    if match and value[:match.start()].strip() and value[match.end():].strip():
        condition = value[match.end():].strip()
        if "(" in match.group() and condition.endswith(")"):
            condition = condition[:-1].strip()
        return value[:match.start()].strip(" ;"), condition
    return value, None


def description_facts(product: Product) -> dict[str, tuple[str, str]]:
    """Целые исходные предложения и значения; второй элемент — поле-источник."""
    facts = {
        "name": (product.name, "name"),
        "manufacturer": ("Производитель — " + product.manufacturer, "manufacturer"),
    }
    # Перенос строки или новая подписанная характеристика отделяют факт.
    # Точка с запятой внутри условий погрешности/ресурса не отделяет их от значения.
    sentences = re.split(r"\r?\n|(?<=[.!?])\s+(?=[А-ЯЁA-Z])|;\s+(?=[^;:\n]{1,120}:\s)",
                         product.description.strip())
    for index, sentence in enumerate(sentences):
        if sentence:
            facts[f"description/{index}"] = (sentence, "description")
    for field, value in product.source_fields().items():
        if field.startswith("specs/"):
            facts[field] = (field[6:] + ": " + value, field)
    return facts


def labelled_characteristics(product: Product, *, limit: int | None = 20,
                             facts: dict[str, tuple[str, str]] | None = None) -> list[ExtractedCharacteristic]:
    result, seen = [], set()
    for text, source in (description_facts(product) if facts is None else facts).values():
        if source != "description":
            continue
        match = re.fullmatch(r"([^:;.!?\n]{1,120}):\s*(.+)", text)
        if match is None:
            continue
        name, value = match[1].strip(), match[2].rstrip(".").strip()
        field = product.canonical_field("specs/" + name)
        name = field[6:]
        if (name.casefold() in {"производитель", "изготовитель", "название", "manufacturer", "name",
                               "примечание", "примечания", "комментарий", "комментарии", "note", "notes"}
                or (name in product.specs and not unknown(product.specs[name])) or unknown(value)):
            continue
        identity = (name.casefold(), value)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(ExtractedCharacteristic(name=name, value=value,
                      source=Reference(field="description", quote=text)))
    return result if limit is None else result[:limit]


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
    for key in dict.fromkeys(priority):
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
