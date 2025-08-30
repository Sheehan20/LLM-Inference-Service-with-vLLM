from __future__ import annotations

import os
from functools import lru_cache

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = structlog.get_logger()


class Settings(BaseModel):
    """Configuration settings for the vLLM inference service."""

    model_config = ConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=(),  # Allow "model_" prefix fields
    )

    # Model configuration
    model_name: str = Field(
        default=os.getenv("MODEL_NAME", "microsoft/phi-2"),
        description="HuggingFace model identifier or local model path",
    )
    tokenizer: str | None = Field(
        default=os.getenv("TOKENIZER") or None, description="Custom tokenizer path (optional)"
    )

    # Performance configuration
    concurrency_limit: int = Field(
        default=int(os.getenv("CONCURRENCY_LIMIT", "20")),
        ge=1,
        le=100,
        description="Maximum concurrent requests",
    )
    max_num_seqs: int = Field(
        default=int(os.getenv("MAX_NUM_SEQS", "32")),
        ge=1,
        le=256,
        description="Maximum number of sequences in a batch",
    )
    max_model_len: int = Field(
        default=int(os.getenv("MAX_MODEL_LEN", "2048")),
        ge=128,
        le=32768,
        description="Maximum model context length",
    )
    gpu_memory_utilization: float = Field(
        default=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")),
        gt=0.0,
        le=1.0,
        description="GPU memory utilization ratio",
    )
    microbatch_wait_ms: int = Field(
        default=int(os.getenv("MICROBATCH_WAIT_MS", "8")),
        ge=0,
        le=1000,
        description="Microbatch wait time in milliseconds",
    )

    # Logging and monitoring
    log_level: str = Field(
        default=os.getenv("LOG_LEVEL", "INFO"),
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    metrics_enabled: bool = Field(
        default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true",
        description="Enable Prometheus metrics",
    )
    sse_heartbeat_interval_s: float = Field(
        default=float(os.getenv("SSE_HEARTBEAT_INTERVAL_S", "10.0")),
        gt=0.0,
        le=300.0,
        description="Server-sent events heartbeat interval",
    )
    max_log_text_chars: int = Field(
        default=int(os.getenv("MAX_LOG_TEXT_CHARS", "512")),
        ge=100,
        le=10000,
        description="Maximum characters to log for text fields",
    )

    # Security and rate limiting
    enable_auth: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_AUTH", "false").lower() == "true",
        description="Enable API authentication",
    )
    api_keys: list[str] | None = Field(
        default=None, description="Valid API keys (comma-separated in env)"
    )
    rate_limit_rpm: int = Field(
        default=int(os.getenv("RATE_LIMIT_RPM", "60")),
        ge=1,
        le=10000,
        description="Rate limit requests per minute",
    )

    # Development and debugging
    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true",
        description="Enable debug mode",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            # Parse comma-separated string from environment
            keys = [key.strip() for key in v.split(",") if key.strip()]
            return keys if keys else None
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_auth_config(self):
        if self.enable_auth and not self.api_keys:
            raise ValueError("API keys must be provided when authentication is enabled")
        return self

    @model_validator(mode="after")
    def validate_performance_config(self):
        if self.max_num_seqs < self.concurrency_limit:
            logger.warning(
                "max_num_seqs is less than concurrency_limit, this may cause performance issues",
                max_num_seqs=self.max_num_seqs,
                concurrency_limit=self.concurrency_limit,
            )
        return self

    def get_env_info(self) -> dict:
        """Get environment information for debugging."""
        return {
            "model_name": self.model_name,
            "concurrency_limit": self.concurrency_limit,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "log_level": self.log_level,
            "debug": self.debug,
            "auth_enabled": self.enable_auth,
            "metrics_enabled": self.metrics_enabled,
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance with validation."""
    try:
        settings = Settings()
        logger.info("Configuration loaded successfully", **settings.get_env_info())
        return settings
    except Exception as e:
        logger.error("Failed to load configuration", error=str(e))
        raise
