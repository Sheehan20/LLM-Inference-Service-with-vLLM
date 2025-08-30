import logging
import sys
from typing import Any

import orjson
import structlog


def _orjson_dumps(obj: Any, *, default: Any) -> str:
    return orjson.dumps(obj, default=default).decode()


def configure_logging(level: str = "INFO") -> structlog.stdlib.BoundLogger:
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            timestamper,
            structlog.processors.add_log_level,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(serializer=_orjson_dumps),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    root_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=root_level, format="%(message)s", stream=sys.stdout)

    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "gunicorn",
        "gunicorn.error",
        "gunicorn.access",
    ):
        logging.getLogger(name).setLevel(root_level)

    return structlog.get_logger()
