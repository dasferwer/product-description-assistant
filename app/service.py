import json
import re

from app.models import (
    Characteristic, ClarificationResponse, Commercial, Conflict, Draft,
    GenerationResponse, Lengths, MissingData, Product, ReadyResponse, Reference, Scalar,
)
from app.provider import InvalidOutput, OpenAIGateway, ServiceError


def unknown(value: Scalar) -> bool:
    return value is None or (isinstance(value, str) and value.strip().casefold() in {
        "", "данных нет", "нет данных", "не указано", "n/a", "-",
    })


def source_fields(product: Product) -> dict[str, str]:
    return {
        "name": product.name, "manufacturer": product.manufacturer,
        "description": product.description,
        **{f"specs/{key}": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
           for key, value in product.specs.items() if not unknown(value)},
    }


def check_reference(reference: Reference, sources: dict[str, str]) -> None:
    if reference.field not in sources or reference.quote not in sources[reference.field]:
        raise InvalidOutput("Цитата отсутствует в указанном исходном поле.")


def check_conflicts(conflicts: list[Conflict], sources: dict[str, str]) -> None:
    for conflict in conflicts:
        if conflict.field not in sources:
            raise InvalidOutput("Противоречие ссылается на неизвестное поле.")
        for reference in conflict.sources:
            check_reference(reference, sources)
        if not any(r.field == conflict.field for r in conflict.sources):
            raise InvalidOutput("Нет свидетельства из спорного поля.")
        if len({(r.field, r.quote) for r in conflict.sources}) < 2:
            raise InvalidOutput("Для противоречия нужны два разных свидетельства.")


# Это фильтр числовых фрагментов, а не доказательство технического смысла.
NUMBER = r"[+−-]?\d+(?:[.,]\d+)?(?:/\d+)?(?:\s*[–−-]\s*[+−-]?\d+(?:[.,]\d+)?)?"
UNITS = (
    r"%|°\s*[CСF]|МПа|кПа|Па|MPa|kPa|Pa|бар|bar|psi|мм|см|мкм|mm|cm|µm|"
    r"кг|мг|г|kg|mg|мВ|кВ|В|mV|kV|V|мА|А|mA|A|кВт|Вт|kW|W|МГц|кГц|Гц|MHz|kHz|Hz|"
    r"кОм|Ом|Ω|л/мин|мл/мин|м/с|об/мин|л|мл|м|m|с|s"
)
QUANTITY = re.compile(NUMBER + rf"(?:\s*(?:{UNITS})(?!\w))?")


def check_draft(product: Product, draft: Draft) -> None:
    sources = source_fields(product)
    check_conflicts(draft.conflicts, sources)
    claims = [c for c in [draft.seo_title, *draft.seo_description,
                         *draft.search_queries, draft.audience, draft.tasks] if c is not None]
    for claim in claims:
        for reference in claim.sources:
            check_reference(reference, sources)
        evidence = {
            re.sub(r"\s+", "", match.group())
            for reference in claim.sources for match in QUANTITY.finditer(reference.quote)
        }
        for match in QUANTITY.finditer(claim.text):
            quantity = re.sub(r"\s+", "", match.group())
            if quantity not in evidence:
                raise InvalidOutput("Число или единица не подтверждены цитатой: " + quantity)
    known_names = {key.casefold() for key in product.specs} | {"производитель"}
    for characteristic in draft.description_characteristics:
        check_reference(characteristic.source, sources)
        if (characteristic.source.field != "description"
                or characteristic.value not in characteristic.source.quote
                or characteristic.name.casefold() in known_names):
            raise InvalidOutput("Извлечённая характеристика не подтверждена или дублирует specs.")
        known_names.add(characteristic.name.casefold())
    if draft.status == "needs_clarification":
        if not any(i.blocking for i in draft.missing_data) and not draft.conflicts:
            raise InvalidOutput("Не указана причина необходимости уточнения.")
        if claims or draft.description_characteristics:
            raise InvalidOutput("Неполная карточка не должна содержать готовый текст.")
        return
    if draft.conflicts or any(item.blocking for item in draft.missing_data):
        raise InvalidOutput("Готовая карточка содержит блокирующие вопросы.")
    if draft.seo_title is None or draft.audience is None or draft.tasks is None:
        raise InvalidOutput("Отсутствуют обязательные блоки карточки.")
    title_length = len(draft.seo_title.text)
    description_length = len(" ".join(c.text for c in draft.seo_description))
    if not 1 <= title_length <= 70 or not 800 <= description_length <= 1200:
        raise InvalidOutput(f"Длины: заголовок {title_length} (1–70), описание {description_length} (800–1200).")
    queries = [c.text.strip().casefold() for c in draft.search_queries]
    if not 3 <= len(queries) <= 5 or len(set(queries)) != len(queries):
        raise InvalidOutput("Нужны 3–5 разных поисковых запросов.")


def unique_issues(issues: list[MissingData]) -> list[MissingData]:
    result: dict[str, MissingData] = {}
    for issue in issues:
        previous = result.get(issue.field)
        if previous is None or issue.blocking:
            result[issue.field] = issue
    return list(result.values())


def characteristics(product: Product, missing: list[MissingData], conflicts: list[Conflict],
                    draft: Draft | None = None) -> list[Characteristic]:
    values = [("manufacturer", "производитель", product.manufacturer)]
    values.extend((f"specs/{key}", key, value) for key, value in product.specs.items())
    result = []
    disputed = {c.field for c in conflicts}
    for field, key, value in values:
        state = "conflicting" if field in disputed else ("missing" if unknown(value) else "provided")
        result.append(Characteristic(name=key, value=value if state == "provided" else "данных нет", status=state))
    if draft is not None:
        result.extend(Characteristic(name=c.name, value=c.value, status="provided")
                      for c in draft.description_characteristics)
    for issue in missing:
        if issue.field.startswith("specs/") and issue.field[6:] not in product.specs:
            result.append(Characteristic(name=issue.field[6:], value="данных нет", status="missing"))
    return result


async def generate(product: Product, gateway: OpenAIGateway | None) -> GenerationResponse:
    missing = [MissingData(field=field, message="данных нет", question=f"Укажите {label}.", blocking=True)
               for field, label in (("name", "название товара"), ("manufacturer", "производителя"),
                                    ("description", "описание или фрагмент паспорта"))
               if unknown(getattr(product, field))]
    missing.extend(MissingData(field=f"specs/{key}", message="данных нет",
                               question=f"Уточните характеристику «{key}».", blocking=False)
                   for key, value in product.specs.items() if unknown(value))
    if any(i.blocking for i in missing):
        return ClarificationResponse(key_characteristics=characteristics(product, missing, []),
                                     missing_data=missing, conflicts=[])
    if gateway is None:
        raise ServiceError(503, "provider_not_configured", "Задайте OPENAI_API_KEY и OPENAI_MODEL в окружении или .env.")
    feedback: list[str] = []
    for _ in range(2):
        try:
            draft = await gateway.generate(product, feedback)
            check_draft(product, draft)
            audit = await gateway.audit(product, draft)
            check_conflicts(audit.conflicts, source_fields(product))
            all_missing = unique_issues([*missing, *draft.missing_data, *audit.missing_data])
            all_conflicts = list({c.field: c for c in [*draft.conflicts, *audit.conflicts]}.values())
            if all_conflicts or any(i.blocking for i in all_missing):
                return ClarificationResponse(
                    key_characteristics=characteristics(product, all_missing, all_conflicts),
                    missing_data=all_missing, conflicts=all_conflicts,
                )
            if not audit.supported or audit.problems:
                raise InvalidOutput("; ".join(audit.problems) or "Проверка достоверности не пройдена.")
            if draft.status == "needs_clarification":
                return ClarificationResponse(
                    key_characteristics=characteristics(product, all_missing, all_conflicts),
                    missing_data=all_missing, conflicts=all_conflicts,
                )
            title = draft.seo_title.text
            description = " ".join(c.text for c in draft.seo_description)
            return ReadyResponse(
                seo_title=title, seo_description=description,
                key_characteristics=characteristics(product, all_missing, [], draft),
                search_queries=[c.text for c in draft.search_queries],
                commercial=Commercial(audience=draft.audience.text, tasks=draft.tasks.text),
                missing_data=all_missing, conflicts=[],
                lengths=Lengths(seo_title=len(title), seo_description=len(description)),
            )
        except InvalidOutput as exc:
            feedback = [str(exc)[:1500]]
    raise ServiceError(502, "invalid_llm_output", "LLM дважды вернула неподтверждённый или некорректный результат. Карточка не выдана.")
