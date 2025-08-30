from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.alerting import get_alert_manager
from app.auth import get_auth_middleware, require_auth
from app.config import Settings, get_settings
from app.errors import ErrorContext, InferenceError, setup_error_handlers
from app.health import get_health_checker
from app.inference.engine import EngineManager
from app.logging_utils import configure_logging
from app.metrics import (
    ACTIVE_CONNECTIONS,
    REQUEST_COUNTER,
    REQUEST_LATENCY,
    metrics_app,
    record_request_metrics,
    set_health_status,
    start_gpu_metrics_poller,
    start_system_metrics_poller,
    start_tokens_per_second_updater,
    tokens_tracker,
    update_service_info,
)
from app.models.request import GenerateRequest
from app.models.response import GenerateResponse
from app.resilience import get_resilience_manager, CircuitState

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings: Settings = get_settings()
    configure_logging(settings.log_level)

    # Initialize service info
    update_service_info(settings.model_name)

    # Initialize health checker
    health_checker = get_health_checker(settings)
    app.state.health_checker = health_checker

    # Initialize auth middleware
    auth_middleware = get_auth_middleware(settings)
    app.state.auth_middleware = auth_middleware

    # Initialize engine manager
    manager = EngineManager(settings=settings, logger=logger)
    try:
        await manager.init_engine()
        app.state.engine_manager = manager
        set_health_status("engine", True)
    except Exception as e:
        logger.error("Failed to initialize engine", error=str(e))
        set_health_status("engine", False)
        raise

    # Start monitoring threads
    start_gpu_metrics_poller()
    start_system_metrics_poller()
    start_tokens_per_second_updater()

    logger.info("Service started successfully")
    yield

    logger.info("Service shutting down")


app = FastAPI(
    title="vLLM Inference Service",
    version="0.2.0",
    description="High-performance LLM inference API powered by vLLM",
    lifespan=lifespan,
)

# Setup error handlers
setup_error_handlers(app)


def get_manager(request: Request) -> EngineManager:
    return request.app.state.engine_manager  # type: ignore


@app.middleware("http")
async def access_log_middleware(request: Request, call_next) -> Response:
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
async def healthz(request: Request) -> dict:
    """Enhanced health check with component status."""
    try:
        health_checker = getattr(request.app.state, "health_checker", None)
        engine_manager = getattr(request.app.state, "engine_manager", None)

        if health_checker:
            health_info = await health_checker.run_health_checks(engine_manager)
        else:
            # Fallback basic health check
            health_info = {
                "status": "ok" if engine_manager and engine_manager._engine else "degraded",
                "timestamp": time.time(),
                "components": {
                    "api": "healthy",
                    "engine": "healthy"
                    if engine_manager and engine_manager._engine
                    else "unhealthy",
                },
            }

        # Update health metrics
        set_health_status("api", True)
        if health_info.get("status") == "healthy":
            set_health_status("engine", True)

        return health_info  # type: ignore
    except Exception as e:
        logger.exception("Health check failed", error=str(e))
        set_health_status("api", False)
        return {"status": "unhealthy", "error": str(e)}


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    manager: EngineManager = Depends(get_manager),
    client_id: str = Depends(require_auth(get_settings())),
) -> JSONResponse:
    route = "/v1/generate"
    start = time.perf_counter()

    # Track active connections and request metrics
    ACTIVE_CONNECTIONS.inc()
    REQUEST_COUNTER.labels(route=route, status="started").inc()

    async with ErrorContext("text_generation") as ctx:
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

            # Record request/response metrics
            record_request_metrics(len(req.prompt), len(result["text"]))

            # Track tokens for TPS calculation
            tokens_tracker.add_tokens(result["num_generated_tokens"])

            REQUEST_COUNTER.labels(route=route, status="ok").inc()
            return JSONResponse(result)

        except Exception as e:
            logger.exception("generate_failed", error=str(e), request_id=ctx.request_id)
            REQUEST_COUNTER.labels(route=route, status="error").inc()

            # Convert to appropriate error type
            if "out of memory" in str(e).lower():
                raise InferenceError("GPU memory exhausted, please try a smaller request") from e
            elif "timeout" in str(e).lower():
                raise InferenceError("Request timed out, please try again") from e
            else:
                raise InferenceError(f"Text generation failed: {str(e)}") from e

        finally:
            ACTIVE_CONNECTIONS.dec()
            REQUEST_LATENCY.labels(route=route).observe(time.perf_counter() - start)


@app.post("/v1/stream")
async def stream(
    req: GenerateRequest,
    manager: EngineManager = Depends(get_manager),
    client_id: str = Depends(require_auth(get_settings())),
) -> StreamingResponse:
    route = "/v1/stream"
    start = time.perf_counter()

    # Track active connections
    ACTIVE_CONNECTIONS.inc()
    REQUEST_COUNTER.labels(route=route, status="started").inc()

    total_tokens = 0

    async def event_generator() -> AsyncGenerator[bytes, None]:
        nonlocal total_tokens
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
                # Estimate tokens from chunk (rough approximation)
                if b'"delta"' in chunk:
                    total_tokens += 1

            # Record final metrics
            record_request_metrics(len(req.prompt), total_tokens * 4)  # Rough char estimate
            tokens_tracker.add_tokens(total_tokens)

            REQUEST_COUNTER.labels(route=route, status="ok").inc()
        except Exception as e:
            logger.exception("stream_failed", error=str(e))
            REQUEST_COUNTER.labels(route=route, status="error").inc()
            yield b'data: {"error": "stream failed"}\n\n'
        finally:
            ACTIVE_CONNECTIONS.dec()
            REQUEST_LATENCY.labels(route=route).observe(time.perf_counter() - start)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


app.mount("/metrics", metrics_app)


@app.get("/alerts")
async def get_alerts() -> dict:
    """Get current active alerts and recent history."""
    alert_manager = get_alert_manager()
    return {
        "active_alerts": alert_manager.get_active_alerts(),
        "recent_history": alert_manager.get_alert_history(limit=50),
        "total_active": len(alert_manager.active_alerts),
    }


@app.post("/alerts/{alert_name}/resolve")
async def resolve_alert(alert_name: str) -> dict:
    """Manually resolve an active alert."""
    alert_manager = get_alert_manager()
    alert_manager.resolve_alert(alert_name)
    return {"message": f"Alert {alert_name} resolved"}


@app.get("/health/detailed")
async def detailed_health(request: Request) -> dict:
    """Get detailed health information including all components."""
    health_checker = getattr(request.app.state, "health_checker", None)
    engine_manager = getattr(request.app.state, "engine_manager", None)

    if not health_checker:
        return {"error": "Health checker not available"}

    return await health_checker.run_health_checks(engine_manager)  # type: ignore


@app.get("/auth/info")
async def auth_info(request: Request) -> dict:
    """Get authentication configuration info."""
    settings = get_settings()

    return {
        "auth_enabled": settings.enable_auth,
        "rate_limit_rpm": settings.rate_limit_rpm,
        "api_keys_configured": len(settings.api_keys or []) if settings.enable_auth else 0,
    }


@app.get("/auth/rate-limit/{client_id}")
async def get_rate_limit_stats(client_id: str, request: Request) -> dict:
    """Get rate limit statistics for a client."""
    auth_middleware = getattr(request.app.state, "auth_middleware", None)
    if not auth_middleware:
        return {"error": "Auth middleware not available"}

    return auth_middleware.rate_limiter.get_stats(client_id)  # type: ignore


@app.get("/resilience/stats")
async def get_resilience_stats() -> dict:
    """Get resilience statistics including circuit breakers and queues."""
    resilience_manager = get_resilience_manager()
    return resilience_manager.get_all_stats()


@app.post("/resilience/circuit-breaker/{name}/reset")
async def reset_circuit_breaker(name: str) -> dict:
    """Reset a circuit breaker to closed state."""
    resilience_manager = get_resilience_manager()
    if name in resilience_manager.circuit_breakers:
        circuit_breaker = resilience_manager.circuit_breakers[name]
        circuit_breaker.state = CircuitState.CLOSED
        circuit_breaker.failure_count = 0
        circuit_breaker.success_count = 0
        logger.info(f"Circuit breaker {name} reset to CLOSED state")
        return {"message": f"Circuit breaker {name} reset successfully"}
    else:
        return {"error": f"Circuit breaker {name} not found"}
