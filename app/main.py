import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.models import GenerationResponse, Product
from app.provider import OpenAIGateway, ServiceError
from app.service import generate


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(".env", override=False)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    app.state.gateway = None
    if key and model:
        async with AsyncOpenAI(api_key=key, base_url="https://api.openai.com/v1",
                               timeout=30.0, max_retries=0) as client:
            app.state.gateway = OpenAIGateway(client, model)
            yield
    else:
        yield


app = FastAPI(title="Помощник товарных описаний", version="1.0.0", lifespan=lifespan)


def get_gateway(request: Request) -> OpenAIGateway | None:
    return request.app.state.gateway


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": {
        "code": "invalid_input", "message": "Проверьте JSON и типы входных полей.",
        "details": [{"field": "/".join(map(str, item["loc"])), "type": item["type"]}
                    for item in exc.errors()],
    }})


@app.post("/generate", response_model=GenerationResponse)
async def generate_card(product: Product, gateway: OpenAIGateway | None = Depends(get_gateway)) -> GenerationResponse:
    try:
        async with asyncio.timeout(150):
            return await generate(product, gateway)
    except TimeoutError as exc:
        raise ServiceError(504, "request_timeout", "Превышено общее время подготовки карточки.") from exc
