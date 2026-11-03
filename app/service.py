import re
from decimal import Decimal
from typing import TypeVar

from app.description import description_facts, labelled_characteristics, value_conditions
from app.models import (
    Audit, Characteristic, ClarificationResponse, Commercial, Conflict, Draft,
    GenerationResponse, Lengths, MissingData, Product, ReadyResponse, Reference, unknown,
)
from app.provider import InvalidOutput, OllamaGateway, ServiceError, SourceClarification
from app.text import has_model_command, is_russian


ResultT = TypeVar("ResultT", Draft, Audit)


def normalize_references(product: Product, result: ResultT) -> ResultT:
    result = result.model_copy(deep=True)
    try:
        for issue in result.missing_data:
            issue.field = product.canonical_field(issue.field)
        for conflict in result.conflicts:
            conflict.field = product.canonical_field(conflict.field)
            for source in conflict.sources:
                source.field = product.canonical_field(source.field)
        if isinstance(result, Draft):
            extracted = []
            for item in result.description_characteristics:
                item.name = product.canonical_field("specs/" + item.name)[6:]
                item.source.field = product.canonical_field(item.source.field)
                if any(previous.name.casefold() == item.name.casefold()
                       and equivalent_values(previous.value, item.value) for previous in extracted):
                    continue
                extracted.append(item)
            result.description_characteristics = extracted
    except ValueError as exc:
        raise InvalidOutput(str(exc)) from exc
    return result


def check_missing(product: Product, issues: list[MissingData]) -> None:
    for issue in issues:
        if has_model_command(issue.message + " " + issue.question):
            raise InvalidOutput("Вопрос содержит команды для модели вместо уточнения фактов.")
        if issue.field in {"name", "manufacturer"} and not unknown(getattr(product, issue.field)):
            raise InvalidOutput(f"Поле {issue.field} уже заполнено: не запрашивай его повторно.")
        if issue.field.startswith("specs/"):
            key = issue.field[6:]
            if key in product.specs and not unknown(product.specs[key]):
                raise InvalidOutput(
                    f"Поле {issue.field} уже заполнено: используй его значение. "
                    "Вопрос об отсутствующих условиях задавай с field=specs, не объявляй известное значение отсутствующим.")


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
INTEGER = r"\d+(?:[ \u00a0\u202f]\d{3}(?!\d))*"
DECIMAL = INTEGER + r"(?:[.,]\d+)?(?:/\d+)?"
NUMBER = rf"[±+−-]?{DECIMAL}(?:\s*(?:[–−-]|…|\.{{2,3}})\s*[±+−-]?{DECIMAL})?"
UNITS = (
    r"%\s*RH(?:/K)?|%|°\s*[CСF]|МПа|кПа|Па|MPa|kPa|Pa|бар|bar|psi|мм|см|мкм|mm|cm|µm|"
    r"кг|мг|г|kg|mg|мВ|кВ|В|mV|kV|V|мА|А|mA|A|кВт|Вт|kW|W|МГц|кГц|Гц|MHz|kHz|Hz|"
    r"кОм|Ом|Ω|л/мин|мл/мин|м/с|об/мин|л|мл|м|m|мин|ч|дней|дня|год|года|лет|min|h|с|s"
)
QUANTITY = re.compile(NUMBER + rf"(?:\s*(?:{UNITS})(?!\w))?")
BOUND = r"[±+−-]?\d+(?:[.,]\d+)?"
PLAIN_QUANTITY = re.compile(
    rf"(?P<low>{BOUND})(?:(?:[–−-]|…|\.{{2,3}})(?P<high>{BOUND}))?(?P<unit>{UNITS})?")
UNIT_ALIASES = {"°С": "°C", "МПа": "MPa", "кПа": "kPa", "Па": "Pa", "бар": "bar",
                "г": "g", "кг": "kg", "мм": "mm", "см": "cm", "мин": "min", "ч": "h"}


NumericSignature = tuple[tuple[tuple[bool, Decimal], ...], str]


def numeric_signature(value: str) -> NumericSignature | None:
    match = PLAIN_QUANTITY.fullmatch(re.sub(r"\s+", "", value))
    if match is None:
        return None
    numbers = tuple((part.startswith("±"), Decimal(part.lstrip("±").replace("−", "-").replace(",", ".")))
                    for part in (match["low"], match["high"]) if part is not None)
    unit = match["unit"] or ""
    return numbers, UNIT_ALIASES.get(unit, unit)


def condition_signature(value: str) -> NumericSignature | str | None:
    _, condition = value_conditions(value)
    if condition is None:
        return None
    # Регистр неизвестных единиц/кодов не меняется: MPa и mPa различаются.
    return numeric_signature(condition) or re.sub(r"\s+", "", condition)


def equivalent_values(first: str, second: str) -> bool:
    if first == second:
        return True
    if condition_signature(first) != condition_signature(second):
        return False
    first_base, _ = value_conditions(first)
    second_base, _ = value_conditions(second)
    first_signature, second_signature = numeric_signature(first_base), numeric_signature(second_base)
    return first_base == second_base or (first_signature is not None and first_signature == second_signature)


def conflicting_numeric_values(first: str, second: str) -> bool:
    first_condition, second_condition = condition_signature(first), condition_signature(second)
    if first_condition is not None and second_condition is not None and first_condition != second_condition:
        return False
    first_signature = numeric_signature(value_conditions(first)[0])
    second_signature = numeric_signature(value_conditions(second)[0])
    # Только доказуемое числовое расхождение в одной единице. Списки назначений,
    # сложные формулировки и разные единицы требуют семантической проверки.
    return (first_signature is not None and second_signature is not None
            and first_signature[1] == second_signature[1] and first_signature[0] != second_signature[0])


def explicit_conflicts(product: Product) -> list[Conflict]:
    conflicts = []
    sources = product.source_fields()
    for key in product.specs:
        field = f"specs/{key}"
        if field not in sources:
            continue
        expected = numeric_signature(sources[field])
        if expected is None:
            continue
        pattern = rf"(?:^|[.!?\n])\s*{re.escape(key)}\s*[:=—]\s*(.+?)(?=[.!?](?:\s|$)|\n|$)"
        quotes = [match[1].strip() for match in re.finditer(pattern, product.description, re.IGNORECASE)]
        # Каждая величина связывается с описанием непосредственно перед ней,
        # а не со всеми параметрами предложения. Иначе диапазон хранения мог
        # ошибочно подменять измерительный диапазон того же прибора.
        anchors = {word[:6] for word in re.findall(r"[а-яёa-z]{4,}", key.casefold())
                   if not word.startswith(("измер", "measure"))}
        if len(expected[0]) == 2 and anchors:
            for sentence in re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])|\n", product.description):
                previous_range_end = 0
                for quantity in QUANTITY.finditer(sentence):
                    signature = numeric_signature(quantity.group())
                    if signature is None or len(signature[0]) != 2:
                        continue
                    context = sentence[previous_range_end:quantity.start()].casefold()
                    previous_range_end = quantity.end()
                    if (not all(anchor in context for anchor in anchors)
                            or re.search(r"\b(?:при проверке|при испытани\w*|для проверки|при услови\w*)", context)):
                        continue
                    other_conditions = ("хран", "транспорт", "эксплуат", "рабоч")
                    if any(stem in context and stem not in key.casefold() for stem in other_conditions):
                        continue
                    quotes.append(quantity.group())
        for quote in quotes:
            actual = numeric_signature(quote)
            # Сравниваются только простые величины в одной единице: без
            # пересчёта и предположений об отличающихся условиях измерения.
            if actual is not None and actual[1] == expected[1] and actual[0] != expected[0]:
                conflicts.append(Conflict(field=field, question=f"Какое значение характеристики «{key}» верно?",
                    sources=[Reference(field=field, quote=sources[field]), Reference(field="description", quote=quote)]))
                break
    # Повтор одной подписи с разными значениями в самом паспорте тоже
    # требует уточнения, даже если specs пуст и сравнивать его не с чем.
    previous = {}
    for item in labelled_characteristics(product, limit=None):
        name = item.name.casefold()
        signature = numeric_signature(value_conditions(item.value)[0])
        if signature is None:
            continue
        scope = condition_signature(item.value)
        contexts = previous.setdefault((name, signature[1]), {})
        if scope in contexts:
            comparable = [contexts[scope]]
        elif scope is None:
            comparable = list(contexts.values())
        else:
            comparable = [contexts[None]] if None in contexts else []
        for first in comparable:
            if conflicting_numeric_values(first.value, item.value):
                conflicts.append(Conflict(field="description",
                    question=f"Уточните значения и условия характеристики «{item.name}» в описании.",
                    sources=[first.source, item.source]))
                break
        contexts.setdefault(scope, item)
    return conflicts


SENSITIVE_WORDS = re.compile(
    r"\b(обеспеч|гарант|повыш|сниж|предотвращ|безопас|надеж|эффектив|оптималь|"
    r"сертиф|совместим|агрессив|взрыв|вибрац|коррози|нержав|энерг|"
    r"пыл|влаг|водонепрониц|термостой|налич)\w*"
)


def check_sensitive_claim(text: str, evidence: str) -> None:
    software = r"\bпрограмм\w+\s+обеспеч\w+\b"
    normalized = re.sub(software, "", evidence.casefold().replace("ё", "е"))
    checked = re.sub(software, "", text.casefold().replace("ё", "е"))
    for match in SENSITIVE_WORDS.finditer(checked):
        # Сверяются основы слов; склонение не меняет наличие свойства во входе.
        root = match.group(1)
        if root not in normalized:
            raise InvalidOutput(f"Удали неподтверждённое утверждение «{match.group()}»: во входных данных этого свойства или результата нет.")


NEGATIONS = (
    re.compile(r"\bне\s+((?:подходит|предназначен\w*|рекомендуется)\s+(?:для|в)\s+[^.!?;\n]+)"),
    re.compile(r"\bнельзя\s+((?:применять|использовать)\s+(?:для|в)\s+[^.!?;\n]+)"),
)


def check_identity_and_negation(product: Product, text: str) -> None:
    known = re.sub(r"[\W_]+", "", product.manufacturer.casefold())
    for match in re.finditer(r"\b(?:производитель|изготовитель)\s*[:—]\s*([^.;\n]+)", text, re.IGNORECASE):
        declared = re.sub(r"[\W_]+", "", match.group(1).casefold())
        if declared and declared not in known:
            raise InvalidOutput("Производитель в тексте не соответствует входному manufacturer.")
    source = re.sub(r"\s+", " ", product.description.casefold())
    positive = re.sub(r"\s+", " ", text.casefold())
    for pattern in NEGATIONS:
        positive = pattern.sub("", positive)
    for pattern in NEGATIONS:
        for match in pattern.finditer(source):
            if match.group(1) in positive:
                raise InvalidOutput("Утверждение меняет смысл явного запрета из описания.")


def check_draft(product: Product, draft: Draft) -> None:
    sources = product.source_fields()
    check_missing(product, draft.missing_data)
    if draft.status == "ready":
        description_length = len(draft.seo_description.text) if draft.seo_description else 0
        if not 800 <= description_length <= 1200:
            raise InvalidOutput(f"SEO-описание содержит {description_length} символов. Нужны 800–1200: напиши полный связный текст по исходным фактам.")
    check_conflicts(draft.conflicts, sources)
    claims = [c for c in [draft.seo_title, draft.seo_description,
                         *draft.search_queries, draft.audience, draft.tasks] if c is not None]
    evidence_text = " ".join(f"{field}: {value}" for field, value in sources.items())
    evidence = {re.sub(r"\s+", "", match.group())
                for value in [*sources.values(), *product.specs]
                for match in QUANTITY.finditer(value)}
    for claim in claims:
        if has_model_command(claim.text):
            raise InvalidOutput("В карточку попала команда для модели. Используй только факты о приборе.")
        if not is_russian(claim.text):
            raise InvalidOutput("Текст карточки должен быть на русском языке; обозначения и единицы сохраняй.")
        if any(field not in sources for field in claim.sources):
            raise InvalidOutput("Утверждение ссылается на неизвестное исходное поле.")
        check_sensitive_claim(claim.text, evidence_text)
        check_identity_and_negation(product, claim.text)
        for word in re.findall(r"[^\W\d_]+", claim.text):
            if (re.search(r"[а-яё]", word, re.IGNORECASE)
                    and re.search(r"[a-z]", word, re.IGNORECASE)
                    and word.casefold() not in evidence_text.casefold()):
                raise InvalidOutput(f"Слово «{word}» смешивает алфавиты. Напиши русское слово целиком кириллицей; обозначения товара сохраняй как во входе.")
        for match in QUANTITY.finditer(claim.text):
            quantity = re.sub(r"\s+", "", match.group())
            if quantity not in evidence:
                raise InvalidOutput("Число или единица не подтверждены цитатой: " + quantity)
    known_names = {key.casefold() for key, value in product.specs.items() if not unknown(value)} | {"производитель"}
    seen_values = {}
    for characteristic in draft.description_characteristics:
        check_reference(characteristic.source, sources)
        if (characteristic.source.field != "description"
                or characteristic.value not in characteristic.source.quote
                or characteristic.name.casefold() in known_names):
            raise InvalidOutput("Извлечённая характеристика не подтверждена или дублирует specs.")
        previous_values = seen_values.setdefault(characteristic.name.casefold(), [])
        if any(conflicting_numeric_values(value, characteristic.value) for value in previous_values):
            raise InvalidOutput("Разные значения одной характеристики должны сохранять явно различные условия.")
        previous_values.append(characteristic.value)
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
    description_length = len(draft.seo_description.text) if draft.seo_description else 0
    if not 1 <= title_length <= 70 or not 800 <= description_length <= 1200:
        raise InvalidOutput(f"Длины: заголовок {title_length} (1–70), описание {description_length} (800–1200).")
    queries = [c.text.strip().casefold() for c in draft.search_queries]
    if not 3 <= len(queries) <= 5 or len(set(queries)) != len(queries):
        raise InvalidOutput("Нужны 3–5 разных поисковых запросов.")


def unique_issues(issues: list[MissingData]) -> list[MissingData]:
    result: dict[tuple[str, str], MissingData] = {}
    for issue in issues:
        key = (issue.field, issue.question.strip().casefold() if issue.field == "specs" else "")
        previous = result.get(key)
        if previous is None or issue.blocking:
            result[key] = issue
    return list(result.values())


def characteristics(product: Product, missing: list[MissingData], conflicts: list[Conflict],
                    draft: Draft | None = None) -> list[Characteristic]:
    extracted = {}
    if draft:
        for item in draft.description_characteristics:
            extracted.setdefault(item.name.casefold(), []).append(item.value)
    values = [("manufacturer", "производитель", product.manufacturer)]
    values.extend((f"specs/{key}", key, value) for key, value in product.specs.items())
    result = []
    disputed = {field for conflict in conflicts
                for field in [conflict.field, *(ref.field for ref in conflict.sources)]}
    for field, key, value in values:
        if unknown(value) and key.casefold() in extracted and field not in disputed:
            result.extend(Characteristic(name=key, value=entry, status="provided")
                          for entry in extracted[key.casefold()])
            continue
        state = "conflicting" if field in disputed else ("missing" if unknown(value) else "provided")
        result.append(Characteristic(name=key, value=value if state == "provided" else "данных нет", status=state))
    if draft is not None:
        existing = {c.name.casefold() for c in result}
        result.extend(Characteristic(name=c.name, value=c.value, status="provided")
                      for c in draft.description_characteristics if c.name.casefold() not in existing)
    for issue in missing:
        if issue.field.startswith("specs/") and issue.field[6:] not in product.specs:
            result.append(Characteristic(name=issue.field[6:], value="данных нет", status="missing"))
    return result


async def generate(product: Product, gateway: OllamaGateway | None) -> GenerationResponse:
    untrusted = [field for field, value in product.source_fields().items()
                 if has_model_command(field + " " + value)]
    untrusted.extend("specs/" + key for key in product.specs if has_model_command(key))
    if untrusted:
        roots = sorted({field.split("/", 1)[0] for field in untrusted})
        return ClarificationResponse(key_characteristics=[], missing_data=[MissingData(
            field=field, message="Нет проверенных данных: в поле обнаружены команды для LLM.",
            question="Передайте фактические сведения о приборе без команд для модели.", blocking=True,
        ) for field in roots], conflicts=[])
    missing = [MissingData(field=field, message="данных нет", question=f"Укажите {label}.", blocking=True)
               for field, label in (("name", "название товара"), ("manufacturer", "производителя"),
                                    ("description", "описание или фрагмент паспорта"))
               if unknown(getattr(product, field))]
    missing.extend(MissingData(field=f"specs/{key}", message="данных нет",
                               question=f"Уточните характеристику «{key}».", blocking=False)
                   for key, value in product.specs.items() if unknown(value))
    try:
        conflicts = explicit_conflicts(product)
    except ValueError as exc:
        raise ServiceError(422, "invalid_input",
                           "Проверьте названия характеристик: исходные поля нельзя однозначно сопоставить.") from exc
    if any(i.blocking for i in missing) or conflicts:
        return ClarificationResponse(key_characteristics=characteristics(product, missing, conflicts),
                                     missing_data=missing, conflicts=conflicts)
    # Верхняя оценка включает даже повторяющиеся факты и всю пунктуацию.
    # Если и она меньше 800, никакой отбор исходных фрагментов не даст нужный объём.
    if sum(len(text) + 2 for text, _ in description_facts(product).values()) + 16 < 800:
        missing.append(MissingData(field="description", blocking=True,
            message="Недостаточно подтверждённых фактов для описания на 800 символов: дополнительных данных нет.",
            question="Передайте более полный фрагмент паспорта с дополнительными характеристиками или условиями применения."))
        return ClarificationResponse(key_characteristics=characteristics(product, missing, []),
                                     missing_data=missing, conflicts=[])
    if gateway is None:
        raise ServiceError(503, "provider_not_configured", "Задайте OLLAMA_HOST и OLLAMA_MODEL в окружении или .env.")
    feedback: list[str] = []
    last_error: InvalidOutput | None = None
    for _ in range(2):
        try:
            draft = normalize_references(product, await gateway.generate(product, feedback))
            check_draft(product, draft)
            audit = normalize_references(product, await gateway.audit(product, draft))
            check_conflicts(audit.conflicts, product.source_fields())
            check_missing(product, audit.missing_data)
            all_missing = unique_issues([*missing, *draft.missing_data, *audit.missing_data])
            extracted_fields = {f"specs/{c.name}" for c in draft.description_characteristics}
            all_missing = [i for i in all_missing
                           if i.field not in extracted_fields]
            all_conflicts = list({c.field: c for c in [*draft.conflicts, *audit.conflicts]}.values())
            # Сначала учитывается отказ проверяющего. Подтверждённые им новые
            # противоречия/пропуски допустимы как причина отказа, остальные
            # замечания требуют исправления, в том числе для пустой карточки.
            if audit.problems or (not audit.supported and not (audit.conflicts or audit.missing_data)):
                raise InvalidOutput("; ".join(audit.problems) or "Проверка достоверности не пройдена.")
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
            description = draft.seo_description.text
            return ReadyResponse(
                seo_title=title, seo_description=description,
                key_characteristics=characteristics(product, all_missing, [], draft),
                search_queries=[c.text for c in draft.search_queries],
                commercial=Commercial(audience=draft.audience.text, tasks=draft.tasks.text),
                missing_data=all_missing, conflicts=[],
                lengths=Lengths(seo_title=len(title), seo_description=len(description)),
            )
        except SourceClarification as exc:
            insufficient = exc.reason == "insufficient"
            return ClarificationResponse(key_characteristics=[], missing_data=[MissingData(
                field=field, blocking=True,
                message=("Недостаточно проверенных фактов для полного описания; дополнительных данных нет."
                         if insufficient else "Нужно отделить фактические данные о приборе от редакционных указаний."),
                question=("Передайте более полный фрагмент паспорта с дополнительными характеристиками или условиями применения."
                          if insufficient else "Передайте фактические сведения о приборе без указаний автору карточки или модели."),
            ) for field in exc.fields], conflicts=[])
        except InvalidOutput as exc:
            last_error = exc
            feedback = [str(exc)[:1500]]
    raise ServiceError(502, "invalid_llm_output", "LLM дважды вернула неподтверждённый или некорректный результат. Карточка не выдана.") from last_error
