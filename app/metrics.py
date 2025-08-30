from __future__ import annotations

import threading
import time

import psutil
from prometheus_client import Counter, Gauge, Histogram, Info, make_asgi_app

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

GPU_TEMPERATURE = Gauge(
    "gpu_temperature_celsius",
    "GPU temperature in celsius",
    labelnames=("gpu_index",),
)

CPU_UTILIZATION = Gauge(
    "cpu_utilization_percent",
    "CPU utilization percentage",
)

MEMORY_USAGE = Gauge(
    "memory_usage_bytes",
    "Memory usage in bytes",
    labelnames=("type",),
)

ACTIVE_CONNECTIONS = Gauge(
    "inference_active_connections",
    "Number of active inference connections",
)

QUEUE_SIZE = Gauge(
    "inference_queue_size",
    "Number of requests waiting in queue",
)

TOKENS_PER_SECOND = Gauge(
    "inference_tokens_per_second",
    "Current tokens generated per second",
)

SERVICE_INFO = Info(
    "inference_service",
    "Information about the inference service",
)

HEALTH_STATUS = Gauge(
    "inference_service_health",
    "Service health status (1=healthy, 0=unhealthy)",
    labelnames=("component",),
)

REQUEST_SIZE = Histogram(
    "inference_request_size_characters",
    "Size of inference requests in characters",
    buckets=(50, 100, 200, 500, 1000, 2000, 5000, 10000),
)

RESPONSE_SIZE = Histogram(
    "inference_response_size_characters",
    "Size of inference responses in characters",
    buckets=(50, 100, 200, 500, 1000, 2000, 5000, 10000),
)


metrics_app = make_asgi_app()


def start_system_metrics_poller(poll_interval_seconds: float = 2.0) -> None:
    """Start polling for system metrics including GPU, CPU, and memory."""
    thread = threading.Thread(
        target=_poll_system_metrics,
        args=(poll_interval_seconds,),
        name="system-metrics-poller",
        daemon=True,
    )
    thread.start()


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
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                        GPU_UTILIZATION.labels(gpu_index=str(i)).set(util.gpu)
                        GPU_MEM_USED.labels(gpu_index=str(i)).set(mem.used)
                        GPU_MEM_TOTAL.labels(gpu_index=str(i)).set(mem.total)
                        GPU_TEMPERATURE.labels(gpu_index=str(i)).set(temp)
                    except Exception:
                        # Log GPU metric collection failure but continue with other GPUs
                        # Skip logging to avoid issues with missing logger in thread
                        continue
                time.sleep(poll_interval_seconds)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                # Log NVML shutdown failure, but it's not critical
                # Skip logging to avoid issues with missing logger in thread
                pass

    thread = threading.Thread(target=_poll, name="gpu-metrics-poller", daemon=True)
    thread.start()


def _poll_system_metrics(poll_interval_seconds: float) -> None:
    """Poll system metrics (CPU, memory) continuously."""
    while True:
        try:
            # CPU utilization
            CPU_UTILIZATION.set(psutil.cpu_percent(interval=1))

            # Memory usage
            memory = psutil.virtual_memory()
            MEMORY_USAGE.labels(type="used").set(memory.used)
            MEMORY_USAGE.labels(type="available").set(memory.available)
            MEMORY_USAGE.labels(type="total").set(memory.total)

            time.sleep(poll_interval_seconds)
        except Exception:
            time.sleep(poll_interval_seconds)


def update_service_info(model_name: str, version: str = "0.1.0") -> None:
    """Update service information metrics."""
    SERVICE_INFO.info(
        {"model_name": model_name, "version": version, "framework": "vllm", "api_version": "v1"}
    )


def set_health_status(component: str, healthy: bool) -> None:
    """Set health status for a component."""
    HEALTH_STATUS.labels(component=component).set(1 if healthy else 0)


def record_request_metrics(prompt_length: int, response_length: int) -> None:
    """Record request and response size metrics."""
    REQUEST_SIZE.observe(prompt_length)
    RESPONSE_SIZE.observe(response_length)


class TokensPerSecondTracker:
    """Track tokens per second generation rate."""

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.tokens: list[tuple[float, int]] = []
        self.lock = threading.Lock()

    def add_tokens(self, count: int) -> None:
        """Add generated tokens with timestamp."""
        now = time.time()
        with self.lock:
            self.tokens.append((now, count))
            # Remove old entries
            cutoff = now - self.window_seconds
            self.tokens = [(ts, cnt) for ts, cnt in self.tokens if ts > cutoff]

    def get_tokens_per_second(self) -> float:
        """Get current tokens per second rate."""
        with self.lock:
            if not self.tokens:
                return 0.0

            now = time.time()
            cutoff = now - self.window_seconds
            recent_tokens = [(ts, cnt) for ts, cnt in self.tokens if ts > cutoff]

            if not recent_tokens:
                return 0.0

            total_tokens = sum(cnt for _, cnt in recent_tokens)
            time_span = now - recent_tokens[0][0] if len(recent_tokens) > 1 else self.window_seconds

            return total_tokens / max(time_span, 1.0)


# Global tokens tracker instance
tokens_tracker = TokensPerSecondTracker()


def update_tokens_per_second_metric() -> None:
    """Update the tokens per second gauge."""
    TOKENS_PER_SECOND.set(tokens_tracker.get_tokens_per_second())


def start_tokens_per_second_updater(update_interval_seconds: float = 5.0) -> None:
    """Start background thread to update tokens per second metric."""

    def _update_loop() -> None:
        while True:
            try:
                update_tokens_per_second_metric()
                time.sleep(update_interval_seconds)
            except Exception:
                time.sleep(update_interval_seconds)

    thread = threading.Thread(target=_update_loop, name="tokens-per-second-updater", daemon=True)
    thread.start()
