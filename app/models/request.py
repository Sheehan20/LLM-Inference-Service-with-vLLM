from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt text")
    max_tokens: int = Field(64, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(-1, description="Top-k sampling; -1 to disable")
    repetition_penalty: float = Field(1.0, ge=0.0)
    stop: Optional[List[str]] = Field(default=None)
    stream: bool = Field(False)


