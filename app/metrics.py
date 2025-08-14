from __future__ import annotations

import threading
import time
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app


REQUEST_COUNTER = Counter(
    "inference_requests_total",
    "Total number of inference requests processed",
    labelnames=("route", "status"),
)

REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Inference request latency (seconds)",
    labelnames=("route",),
    buckets=(
        0.025,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        10.0,
    ),
)

GENERATED_TOKENS = Counter(
    "inference_generated_tokens_total",
    "Total number of generated tokens",
)

GPU_UTILIZATION = Gauge(
    "gpu_utilization_ratio",
    "GPU compute utilization ratio (0-100)",
    labelnames=("gpu_index",),
)

GPU_MEM_USED = Gauge(
    "gpu_memory_used_bytes",
    "GPU memory used in bytes",
    labelnames=("gpu_index",),
)

GPU_MEM_TOTAL = Gauge(
    "gpu_memory_total_bytes",
    "GPU memory total in bytes",
    labelnames=("gpu_index",),
)


metrics_app = make_asgi_app()


def start_gpu_metrics_poller(poll_interval_seconds: float = 2.0) -> None:
    try:
        import pynvml
    except Exception:
        return

    def _poll() -> None:
        try:
            pynvml.nvmlInit()
        except Exception:
            return

        try:
            device_count = pynvml.nvmlDeviceGetCount()
            while True:
                for i in range(device_count):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        GPU_UTILIZATION.labels(gpu_index=str(i)).set(util.gpu)
                        GPU_MEM_USED.labels(gpu_index=str(i)).set(mem.used)
                        GPU_MEM_TOTAL.labels(gpu_index=str(i)).set(mem.total)
                    except Exception:
                        continue
                time.sleep(poll_interval_seconds)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    thread = threading.Thread(target=_poll, name="gpu-metrics-poller", daemon=True)
    thread.start()


