from __future__ import annotations

from pydantic import BaseModel


class GenerateResponse(BaseModel):
    text: str
    num_prompt_tokens: int
    num_generated_tokens: int
    finish_reason: str | None = None
    latency_ms: int


