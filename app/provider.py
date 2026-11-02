import asyncio
import json
from typing import TypeVar

import openai
from pydantic import BaseModel, ValidationError

from app.models import Audit, Draft, Product
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
CALL_TIMEOUT = 35


class OpenAIGateway:
    def __init__(self, client: openai.AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def _parse(self, prompt: str, payload: dict, schema: type[T]) -> T:
        for attempt in range(2):
            try:
                async with asyncio.timeout(CALL_TIMEOUT):
                    response = await self.client.responses.parse(
                        model=self.model,
                        input=[{"role": "system", "content": prompt},
                               {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                        text_format=schema,
                        max_output_tokens=8000,
                        reasoning={"effort": "low"},
                        store=False,
                    )
                if response.status != "completed" or response.output_parsed is None:
                    raise InvalidOutput("empty_or_incomplete_output")
                return response.output_parsed
            except (ValidationError, json.JSONDecodeError, openai.APIResponseValidationError) as exc:
                raise InvalidOutput("invalid_schema") from exc
            except (openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError) as exc:
                raise InvalidOutput("incomplete_output") from exc
            except (openai.APIError, TimeoutError) as exc:
                status = getattr(exc, "status_code", None)
                quota = getattr(exc, "code", None) in {"insufficient_quota", "billing_hard_limit_reached"}
                transient = not quota and (
                    isinstance(exc, (openai.APIConnectionError, TimeoutError))
                    or status in {408, 409, 429}
                    or (isinstance(status, int) and status >= 500)
                )
                if transient and attempt == 0:
                    await asyncio.sleep(0.5)
                    continue
                if isinstance(exc, (openai.APITimeoutError, TimeoutError)):
                    error = ServiceError(504, "provider_timeout", "Превышено время ожидания LLM.")
                elif status in {401, 403}:
                    error = ServiceError(502, "provider_authorization", "Провайдер отклонил API-ключ или доступ к модели.")
                elif status == 429:
                    error = ServiceError(503, "provider_rate_limit", "Недоступна квота или превышен лимит запросов LLM.")
                else:
                    error = ServiceError(502, "provider_error", "Не удалось получить ответ LLM. Проверьте доступ и модель.")
                raise error from exc
        raise RuntimeError("unreachable")

    async def generate(self, product: Product, feedback: list[str]) -> Draft:
        return await self._parse(
            GENERATION_PROMPT,
            {"product": product.model_dump(), "validation_feedback": feedback},
            Draft,
        )

    async def audit(self, product: Product, draft: Draft) -> Audit:
        return await self._parse(
            AUDIT_PROMPT, {"product": product.model_dump(), "draft": draft.model_dump()}, Audit
        )
