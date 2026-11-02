import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from ollama import AsyncClient

from app.models import GenerationResponse, Product
from app.provider import OllamaGateway, ServiceError
from app.service import generate


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(".env", override=False)
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    model = os.getenv("OLLAMA_MODEL", "").strip()
    app.state.gateway = None
    if host and model:
        async with AsyncClient(host=host, timeout=110.0) as client:
            app.state.gateway = OllamaGateway(client, model)
            yield
    else:
        yield


app = FastAPI(title="Помощник товарных описаний", version="1.0.0", lifespan=lifespan)


def get_gateway(request: Request) -> OllamaGateway | None:
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
async def generate_card(product: Product, gateway: OllamaGateway | None = Depends(get_gateway)) -> GenerationResponse:
    try:
        async with asyncio.timeout(480):
            return await generate(product, gateway)
    except TimeoutError as exc:
        raise ServiceError(504, "request_timeout", "Превышено общее время подготовки карточки.") from exc
