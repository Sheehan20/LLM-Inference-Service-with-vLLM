from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Write a short poem about the ocean.",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "stop": ["\n\n"],
                "stream": False,
            }
        }
    )

    prompt: str = Field(..., min_length=1, max_length=10000, description="The input prompt text")
    max_tokens: int = Field(128, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, gt=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: int = Field(-1, description="Top-k sampling; -1 to disable")
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0, description="Repetition penalty")
    stop: list[str] | None = Field(default=None, max_length=10, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming response")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()

    @field_validator("stop")
    @classmethod
    def validate_stop_sequences(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            # Remove empty strings and limit length
            valid_stops = [s for s in v if s and len(s) <= 50]
            if len(valid_stops) != len(v):
                raise ValueError("Stop sequences cannot be empty and must be <= 50 characters")
            return valid_stops
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v != -1 and v <= 0:
            raise ValueError("top_k must be -1 (disabled) or positive integer")
        return v
