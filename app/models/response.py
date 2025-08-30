from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GenerateResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "The ocean waves crash against the shore, bringing salt and foam.",
                "num_prompt_tokens": 8,
                "num_generated_tokens": 12,
                "finish_reason": "stop",
                "latency_ms": 350,
            }
        }
    )

    text: str = Field(..., description="Generated text response")
    num_prompt_tokens: int = Field(..., ge=0, description="Number of tokens in the prompt")
    num_generated_tokens: int = Field(..., ge=0, description="Number of tokens generated")
    finish_reason: str | None = Field(None, description="Reason for finishing generation")
    latency_ms: int = Field(..., ge=0, description="Request latency in milliseconds")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Invalid request parameters",
                "error_type": "validation_error",
                "request_id": "req_123456789",
            }
        }
    )

    error: str = Field(..., description="Error message")
    error_type: str = Field(default="generic", description="Type of error")
    request_id: str | None = Field(None, description="Request identifier")
