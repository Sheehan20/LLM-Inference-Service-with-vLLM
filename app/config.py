from __future__ import annotations

import os
from functools import lru_cache
from pydantic import BaseModel, Field


class Settings(BaseModel):
    model_name: str = Field(default=os.getenv("MODEL_NAME", "microsoft/phi-2"))
    tokenizer: str | None = Field(default=os.getenv("TOKENIZER") or None)

    concurrency_limit: int = Field(default=int(os.getenv("CONCURRENCY_LIMIT", "20")))
    max_num_seqs: int = Field(default=int(os.getenv("MAX_NUM_SEQS", "32")))
    max_model_len: int = Field(default=int(os.getenv("MAX_MODEL_LEN", "2048")))
    gpu_memory_utilization: float = Field(default=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")))
    microbatch_wait_ms: int = Field(default=int(os.getenv("MICROBATCH_WAIT_MS", "8")))

    log_level: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))
    metrics_enabled: bool = Field(default=os.getenv("METRICS_ENABLED", "true").lower() == "true")
    sse_heartbeat_interval_s: float = Field(default=float(os.getenv("SSE_HEARTBEAT_INTERVAL_S", "10.0")))
    max_log_text_chars: int = Field(default=int(os.getenv("MAX_LOG_TEXT_CHARS", "512")))


@lru_cache()
def get_settings() -> Settings:
    return Settings()


