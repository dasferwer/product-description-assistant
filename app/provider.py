import asyncio
import copy
import json
from typing import Any, TypeVar

import httpx
from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, ValidationError

from app.models import Audit, Draft, Product, unknown
from app.prompts import AUDIT_PROMPT, GENERATION_PROMPT


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class InvalidOutput(Exception):
    pass


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
        for name in ("Reference", "Conflict"):
            definition = schema_json.get("$defs", {}).get(name)
            if definition:
                definition["properties"]["field"]["enum"] = list(fields)
        description = fields.get("description", "")
        if schema is Draft and 800 <= len(description) <= 1200:
            # Уже пригодный исходный абзац сохраняется без перефразирования.
            # Модель по-прежнему может выбрать needs_clarification; аудит
            # проверяет и этот абзац, и остальные сгенерированные блоки.
            schema_json["$defs"]["DescriptionClaim"]["properties"]["text"]["enum"] = [description]
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
        if schema is Draft:
            if any(field.startswith("specs/") for field in fields) and not payload.get("unknown_specs"):
                schema_json["properties"]["description_characteristics"]["maxItems"] = 0
            definitions = schema_json.pop("$defs")
            ready, clarification = copy.deepcopy(schema_json), copy.deepcopy(schema_json)
            ready["properties"]["status"] = {"const": "ready"}
            for name in ("seo_title", "seo_description", "audience", "tasks"):
                ready["properties"][name] = next(
                    item for item in ready["properties"][name]["anyOf"] if "$ref" in item)
                clarification["properties"][name] = {"type": "null"}
            ready["properties"]["conflicts"]["maxItems"] = 0
            clarification["properties"]["status"] = {"const": "needs_clarification"}
            for variant in (ready, clarification):
                variant["properties"].pop("search_queries")
                variant["required"].remove("search_queries")
            for name in ("description_characteristics",):
                clarification["properties"][name]["maxItems"] = 0
            schema_json = {"$defs": definitions, "anyOf": [ready, clarification]}
        system = prompt
        user = json.dumps(payload, ensure_ascii=False)
        # Байтовая оценка заведомо консервативна для BPE-токенизации Qwen.
        if len((system + user).encode("utf-8")) + OUTPUT_LIMIT + 512 > CONTEXT_SIZE:
            raise ServiceError(422, "input_context_limit", "Сократите паспорт и характеристики: превышен безопасный объём контекста локальной модели.")
        for attempt in range(2):
            try:
                async with asyncio.timeout(CALL_TIMEOUT):
                    response = await self.client.chat(
                        model=self.model,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        format=grammar_schema(schema_json),
                        options={"temperature": 0, "num_ctx": CONTEXT_SIZE, "num_predict": OUTPUT_LIMIT},
                        stream=False,
                        think=False,
                    )
                if not response.done or response.done_reason == "length" or not response.message.content:
                    raise InvalidOutput("empty_or_incomplete_output")
                raw = json.loads(response.message.content)
                if schema is Draft and isinstance(raw, dict):
                    title = raw.get("seo_title")
                    if raw.get("status") == "ready" and isinstance(title, dict) and isinstance(title.get("text"), str):
                        text = title["text"]
                        raw["search_queries"] = [
                            {"text": query, "sources": ["name", "description"]}
                            for query in (text, text + " характеристики", text + " применение")
                        ]
                    else:
                        raw["search_queries"] = []
                return schema.model_validate(raw)
            except ValidationError as exc:
                details = "; ".join(
                    "/".join(map(str, error["loc"])) + ": " + error["type"]
                    for error in exc.errors()[:8]
                )
                raise InvalidOutput("Нарушена схема ответа: " + details) from exc
            except json.JSONDecodeError as exc:
                raise InvalidOutput("Ответ не является корректным JSON.") from exc
            except (ResponseError, httpx.RequestError, ConnectionError, TimeoutError) as exc:
                status = getattr(exc, "status_code", None)
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
        return await self._parse(
            GENERATION_PROMPT,
            {"source_fields": product.source_fields(),
             "description_length": len(product.description),
             "unknown_specs": [key for key, value in product.specs.items() if unknown(value)],
             "validation_feedback": feedback},
            Draft,
        )

    async def audit(self, product: Product, draft: Draft) -> Audit:
        return await self._parse(
            AUDIT_PROMPT, {"source_fields": product.source_fields(), "draft": draft.model_dump()}, Audit
        )
