from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import Settings, get_settings
from app.inference.engine import EngineManager
from app.logging_utils import configure_logging
from app.metrics import REQUEST_COUNTER, REQUEST_LATENCY, metrics_app, start_gpu_metrics_poller
from app.models.request import GenerateRequest
from app.models.response import GenerateResponse

import structlog


logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings: Settings = get_settings()
    configure_logging(settings.log_level)
    manager = EngineManager(settings=settings, logger=logger)
    await manager.init_engine()
    app.state.engine_manager = manager
    start_gpu_metrics_poller()
    yield


app = FastAPI(title="vLLM Inference Service (Phi-2)", version="0.1.0", lifespan=lifespan)


def get_manager(request: Request) -> EngineManager:
    return request.app.state.engine_manager  # type: ignore


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=getattr(locals().get("response", None), "status_code", None),
            duration_ms=duration_ms,
            client_ip=getattr(request.client, "host", None),
        )

@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, manager: EngineManager = Depends(get_manager)) -> JSONResponse:
    route = "/v1/generate"
    start = time.perf_counter()
    REQUEST_COUNTER.labels(route=route, status="started").inc()
    try:
        result = await manager.generate_text(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            stop=req.stop,
            repetition_penalty=req.repetition_penalty,
        )
        REQUEST_COUNTER.labels(route=route, status="ok").inc()
        return JSONResponse(result)
    except Exception as e:
        logger.exception("generate_failed", error=str(e))
        REQUEST_COUNTER.labels(route=route, status="error").inc()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        REQUEST_LATENCY.labels(route=route).observe(time.perf_counter() - start)


@app.post("/v1/stream")
async def stream(req: GenerateRequest, manager: EngineManager = Depends(get_manager)) -> StreamingResponse:
    route = "/v1/stream"
    start = time.perf_counter()
    REQUEST_COUNTER.labels(route=route, status="started").inc()

    async def event_generator() -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in manager.stream_text(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                stop=req.stop,
                repetition_penalty=req.repetition_penalty,
            ):
                yield chunk
            REQUEST_COUNTER.labels(route=route, status="ok").inc()
        except Exception as e:
            logger.exception("stream_failed", error=str(e))
            REQUEST_COUNTER.labels(route=route, status="error").inc()
            yield b"data: {\"error\": \"stream failed\"}\n\n"
        finally:
            REQUEST_LATENCY.labels(route=route).observe(time.perf_counter() - start)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


app.mount("/metrics", metrics_app)


