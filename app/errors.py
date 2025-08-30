from __future__ import annotations

import traceback
import uuid
from typing import Any

import structlog
from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

logger = structlog.get_logger()


class InferenceServiceError(Exception):
    """Base exception for inference service errors."""

    def __init__(self, message: str, error_type: str = "service_error", status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code


class ModelLoadError(InferenceServiceError):
    """Error loading or initializing the model."""

    def __init__(self, message: str):
        super().__init__(message, "model_load_error", status.HTTP_503_SERVICE_UNAVAILABLE)


class InferenceError(InferenceServiceError):
    """Error during text generation."""

    def __init__(self, message: str):
        super().__init__(message, "inference_error", status.HTTP_500_INTERNAL_SERVER_ERROR)


class ResourceExhaustedError(InferenceServiceError):
    """Error when resources are exhausted."""

    def __init__(self, message: str):
        super().__init__(message, "resource_exhausted", status.HTTP_503_SERVICE_UNAVAILABLE)


class RateLimitError(InferenceServiceError):
    """Error when rate limit is exceeded."""

    def __init__(self, message: str):
        super().__init__(message, "rate_limit_exceeded", status.HTTP_429_TOO_MANY_REQUESTS)


class ValidationError(InferenceServiceError):
    """Error with request validation."""

    def __init__(self, message: str):
        super().__init__(message, "validation_error", status.HTTP_422_UNPROCESSABLE_ENTITY)


class ErrorHandler:
    """Centralized error handling for the inference service."""

    @staticmethod
    def generate_request_id() -> str:
        """Generate a unique request ID for error tracking."""
        return f"req_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def create_error_response(
        error: Exception, request_id: str | None = None, include_traceback: bool = False
    ) -> dict[str, Any]:
        """Create a standardized error response."""
        if request_id is None:
            request_id = ErrorHandler.generate_request_id()

        # Determine error type and message
        if isinstance(error, InferenceServiceError):
            error_type = error.error_type
            message = error.message
            status_code = error.status_code
        elif isinstance(error, ValidationError | PydanticValidationError | RequestValidationError):
            error_type = "validation_error"
            message = str(error)
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        elif isinstance(error, HTTPException):
            error_type = "http_error"
            message = error.detail
            status_code = error.status_code
        else:
            error_type = "internal_error"
            message = "An internal server error occurred"
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        # Create base response
        error_response = {"error": message, "error_type": error_type, "request_id": request_id}

        # Add traceback in debug mode
        if include_traceback and not isinstance(error, InferenceServiceError):
            error_response["traceback"] = traceback.format_exc()

        # Log the error
        logger.error(
            "request_error",
            error_type=error_type,
            message=message,
            request_id=request_id,
            exception=str(error),
            traceback=traceback.format_exc() if include_traceback else None,
        )

        return error_response, status_code

    @staticmethod
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        request_id = ErrorHandler.generate_request_id()

        # Format validation errors
        errors = []
        for error in exc.errors():
            field = " -> ".join(str(x) for x in error["loc"]) if error["loc"] else "root"
            errors.append(f"{field}: {error['msg']}")

        error_message = "Validation failed: " + "; ".join(errors)

        error_response, status_code = ErrorHandler.create_error_response(
            ValidationError(error_message), request_id=request_id
        )

        return JSONResponse(content=error_response, status_code=status_code)

    @staticmethod
    async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle general exceptions."""
        request_id = ErrorHandler.generate_request_id()

        # Check if this is a development environment
        include_traceback = getattr(request.app.state, "debug", False)

        error_response, status_code = ErrorHandler.create_error_response(
            exc, request_id=request_id, include_traceback=include_traceback
        )

        return JSONResponse(content=error_response, status_code=status_code)

    @staticmethod
    async def inference_service_error_handler(
        request: Request, exc: InferenceServiceError
    ) -> JSONResponse:
        """Handle custom inference service errors."""
        request_id = ErrorHandler.generate_request_id()

        error_response = {
            "error": exc.message,
            "error_type": exc.error_type,
            "request_id": request_id,
        }

        logger.error(
            "service_error", error_type=exc.error_type, message=exc.message, request_id=request_id
        )

        return JSONResponse(content=error_response, status_code=exc.status_code)


def setup_error_handlers(app):
    """Setup error handlers for the FastAPI app."""
    app.add_exception_handler(RequestValidationError, ErrorHandler.validation_error_handler)
    app.add_exception_handler(InferenceServiceError, ErrorHandler.inference_service_error_handler)
    app.add_exception_handler(Exception, ErrorHandler.general_error_handler)


# Context managers for better error handling
class ErrorContext:
    """Context manager for handling errors in specific operations."""

    def __init__(self, operation_name: str, request_id: str | None = None):
        self.operation_name = operation_name
        self.request_id = request_id or ErrorHandler.generate_request_id()

    async def __aenter__(self):
        logger.info(f"Starting {self.operation_name}", request_id=self.request_id)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if isinstance(exc_val, InferenceServiceError):
                logger.error(
                    f"Operation {self.operation_name} failed",
                    error_type=exc_val.error_type,
                    message=exc_val.message,
                    request_id=self.request_id,
                )
            else:
                logger.exception(
                    f"Unexpected error in {self.operation_name}",
                    request_id=self.request_id,
                    exception=str(exc_val),
                )
        else:
            logger.info(f"Completed {self.operation_name}", request_id=self.request_id)

        return False  # Don't suppress exceptions
