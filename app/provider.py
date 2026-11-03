import asyncio
import copy
import json
from typing import Any, TypeVar

import httpx
from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, ValidationError

from app.description import compose_description, description_facts, labelled_characteristics
from app.models import Audit, Claim, Draft, GenerationPlan, Product, SourceReview, unknown
from app.prompts import AUDIT_PROMPT, GENERATION_PROMPT, SOURCE_REVIEW_PROMPT
from app.text import has_model_command


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class InvalidOutput(Exception):
    pass


class SourceClarification(Exception):
    def __init__(self, fields: list[str], reason: str = "uncertain"):
        super().__init__(reason)
        self.fields = fields
        self.reason = reason


T = TypeVar("T", bound=BaseModel)
CALL_TIMEOUT = 120
CONTEXT_SIZE = 32768
OUTPUT_LIMIT = 4500


def grammar_schema(node: Any) -> Any:
    # Ollama 0.24.0 не компилирует большие maxLength в грамматику.
    # Исходная схема с ограничениями остаётся в Pydantic.
    if isinstance(node, dict):
        return {key: grammar_schema(value) for key, value in node.items() if key != "maxLength"}
    if isinstance(node, list):
        return [grammar_schema(value) for value in node]
    return node


class OllamaGateway:
    def __init__(self, client: AsyncClient, model: str):
        self.client = client
        self.model = model

    async def _parse(self, prompt: str, payload: dict, schema: type[T]) -> T:
        schema_json = schema.model_json_schema()
        fields = payload["source_fields"]
        if schema is SourceReview:
            schema_json["$defs"]["SourceDecision"]["properties"]["fact_id"]["enum"] = list(payload["candidates"])
        for name in ("Reference", "Conflict"):
            definition = schema_json.get("$defs", {}).get(name)
            if definition:
                definition["properties"]["field"]["enum"] = list(fields)
        missing = schema_json.get("$defs", {}).get("MissingData")
        if missing:
            missing["properties"]["field"]["enum"] = list(dict.fromkeys([
                *fields, "specs", *(f"specs/{name}" for name in payload.get("unknown_specs", [])),
            ]))
        for name in ("Claim", "DescriptionClaim"):
            definition = schema_json.get("$defs", {}).get(name)
            if definition:
                definition["properties"]["sources"]["items"]["enum"] = list(payload["source_fields"])
        extracted = schema_json.get("$defs", {}).get("ExtractedCharacteristic")
        if extracted:
            extracted["properties"]["source"] = {
                "type": "object", "properties": {
                    "field": {"const": "description"},
                    "quote": {"type": "string", "minLength": 1},
                }, "required": ["field", "quote"], "additionalProperties": False,
            }
        if schema is GenerationPlan:
            if any(field.startswith("specs/") for field in fields) and not payload.get("unknown_specs"):
                schema_json["properties"]["description_characteristics"]["maxItems"] = 0
            definitions = schema_json.pop("$defs")
            ready, clarification = copy.deepcopy(schema_json), copy.deepcopy(schema_json)
            ready["properties"]["status"] = {"const": "ready"}
            for name in ("seo_title", "audience", "tasks"):
                ready["properties"][name] = next(
                    item for item in ready["properties"][name]["anyOf"] if "$ref" in item)
                clarification["properties"][name] = {"type": "null"}
            ready["properties"]["conflicts"]["maxItems"] = 0
            clarification["properties"]["status"] = {"const": "needs_clarification"}
            ready["properties"]["description_parts"]["minItems"] = 1
            ready["properties"]["description_parts"]["items"]["enum"] = list(payload["facts"])
            clarification["properties"]["description_parts"]["maxItems"] = 0
            clarification["properties"]["description_characteristics"]["maxItems"] = 0
            schema_json = {"$defs": definitions, "anyOf": [ready, clarification]}
        system = prompt
        user = json.dumps(payload, ensure_ascii=False)
        # Байтовая оценка заведомо консервативна для BPE-токенизации Qwen.
        if len((system + user).encode("utf-8")) + OUTPUT_LIMIT + 512 > CONTEXT_SIZE:
            raise ServiceError(422, "input_context_limit", "Сократите паспорт и характеристики: превышен безопасный объём контекста локальной модели.")
        for attempt in range(2):
            try:
                async with asyncio.timeout(CALL_TIMEOUT):
                    try:
                        response = await self.client.chat(
                            model=self.model,
                            messages=[{"role": "system", "content": system},
                                      {"role": "user", "content": user}],
                            format=grammar_schema(schema_json),
                            options={"temperature": 0, "num_ctx": CONTEXT_SIZE, "num_predict": OUTPUT_LIMIT},
                            stream=False,
                            think=False,
                        )
                    except (ValueError, TypeError, RecursionError, AttributeError) as exc:
                        # ResponseError SDK может сам упасть при разборе тела ошибки.
                        # HTTP-статус сохраняется в контексте исходного исключения.
                        if isinstance(exc.__context__, httpx.HTTPStatusError):
                            raise exc.__context__ from exc
                        if isinstance(exc, AttributeError):
                            raise
                        # SDK декодирует HTTP JSON и строит ChatResponse внутри chat.
                        # Ошибки формы внешней оболочки не являются сетевыми сбоями.
                        raise InvalidOutput("Нарушен формат ответа LLM-сервера.") from exc
                if not response.done or response.done_reason == "length" or not response.message.content:
                    raise InvalidOutput("empty_or_incomplete_output")
                try:
                    raw = json.loads(response.message.content)
                except (ValueError, RecursionError) as exc:
                    raise InvalidOutput("Ответ модели не является допустимым JSON.") from exc
                return schema.model_validate(raw)
            except ValidationError as exc:
                details = "; ".join(
                    "/".join(map(str, error["loc"])) + ": " + error["type"]
                    for error in exc.errors()[:8]
                )
                raise InvalidOutput("Нарушена схема ответа: " + details) from exc
            except (ResponseError, httpx.HTTPStatusError, httpx.RequestError, ConnectionError, TimeoutError) as exc:
                status = (exc.response.status_code if isinstance(exc, httpx.HTTPStatusError)
                          else getattr(exc, "status_code", None))
                transient = (
                    isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError))
                    or status in {408, 409, 429}
                    or (isinstance(status, int) and status >= 500)
                )
                if transient and attempt == 0:
                    await asyncio.sleep(0.5)
                    continue
                if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
                    error = ServiceError(504, "provider_timeout", "Превышено время ожидания LLM.")
                elif status in {401, 403}:
                    error = ServiceError(502, "provider_authorization", "LLM-сервер отклонил доступ.")
                elif status == 429:
                    error = ServiceError(503, "provider_rate_limit", "Превышен лимит запросов LLM-сервера.")
                else:
                    error = ServiceError(502, "provider_error", "Не удалось получить ответ LLM. Проверьте Ollama и наличие модели.")
                raise error from exc
        raise RuntimeError("unreachable")

    async def generate(self, product: Product, feedback: list[str]) -> Draft:
        candidates = description_facts(product)
        unknown_ids = set()
        for index, (key, value) in enumerate(product.specs.items()):
            if unknown(value):
                fact_id = f"unknown_spec/{index}"
                candidates[fact_id] = ("Название характеристики без значения: " + key, "specs/" + key)
                unknown_ids.add(fact_id)
        review = await self._parse(SOURCE_REVIEW_PROMPT, {
            "source_fields": {},
            "candidates": {key: {"text": text, "source_field": source}
                           for key, (text, source) in candidates.items()},
            "validation_feedback": feedback,
        }, SourceReview)
        if ({item.fact_id for item in review.decisions} != set(candidates)
                or len(review.decisions) != len(candidates)):
            raise InvalidOutput("Проверь каждый исходный фрагмент ровно один раз; непроверенные данные не разрешены.")
        facts, rejected = {}, []
        for item in review.decisions:
            text, source = candidates[item.fact_id]
            if (has_model_command(text) or item.kind in {"editorial_instruction", "uncertain"}
                    or (item.kind == "unrelated" and source != "description")):
                rejected.append(source.split("/", 1)[0])
            elif item.kind in {"product_data", "operating_instruction"} and item.fact_id not in unknown_ids:
                facts[item.fact_id] = (text, source)
        if rejected:
            raise SourceClarification(sorted(set(rejected)))
        if sum(len(text) + 2 for text, _ in facts.values()) + 16 < 800:
            raise SourceClarification(["description"], "insufficient")
        source_fields = product.source_fields()
        original_description = product.description
        excluded, cursor = [], 0
        for key, (text, source) in candidates.items():
            if source != "description":
                continue
            start = original_description.find(text, cursor)
            if start < 0:
                raise InvalidOutput("Исходный фрагмент отсутствует в описании.")
            cursor = start + len(text)
            if key not in facts:
                excluded.append((start, cursor))
        # При удалении посторонних фрагментов сохраняются исходные разделители
        # остальных: цитаты из соседних предложений должны оставаться дословными.
        for start, end in reversed(excluded):
            source_fields["description"] = source_fields["description"][:start] + source_fields["description"][end:]
        plan = await self._parse(
            GENERATION_PROMPT,
            {"source_fields": source_fields,
             "facts": {key: {"text": text, "characters": len(text)} for key, (text, _) in facts.items()},
             "unknown_specs": [key for key, value in product.specs.items() if unknown(value)],
             "validation_feedback": feedback},
            GenerationPlan,
        )
        description, queries = None, []
        if plan.status == "ready":
            if plan.seo_title is None:
                raise InvalidOutput("Отсутствует SEO-заголовок.")
            try:
                # Дополнение допускается только из уже проверенного набора.
                # Непроверенные исходные предложения сюда не попадают.
                description = compose_description(facts, list(dict.fromkeys([*plan.description_parts, *facts])))
            except ValueError as exc:
                raise InvalidOutput(str(exc)) from exc
            title = plan.seo_title.text
            queries = [Claim(text=text, sources=["name", "description"])
                       for text in (title, title + " характеристики", title + " применение")]
            try:
                labelled = labelled_characteristics(product, facts=facts)
            except ValueError as exc:
                raise InvalidOutput(str(exc)) from exc
            names = {item.name.casefold() for item in labelled}
            original_fields = product.source_fields()
            if any(item.source.quote not in source_fields.get(item.source.field, "")
                   or item.source.quote not in original_fields.get(item.source.field, "")
                   for item in plan.description_characteristics):
                raise InvalidOutput("Характеристика ссылается на фрагмент, не разрешённый как данные о приборе.")
            plan.description_characteristics = [*labelled, *(item for item in plan.description_characteristics
                                                           if item.name.casefold() not in names)][:20]
        return Draft(**plan.model_dump(exclude={"description_parts"}),
                     seo_description=description, search_queries=queries)

    async def audit(self, product: Product, draft: Draft) -> Audit:
        return await self._parse(
            AUDIT_PROMPT, {"source_fields": product.source_fields(),
                          "unknown_specs": [key for key, value in product.specs.items() if unknown(value)],
                          "draft": draft.model_dump()}, Audit
        )
