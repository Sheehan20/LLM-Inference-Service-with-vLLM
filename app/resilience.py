from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit breaker triggered, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5  # Number of failures to trigger circuit
    recovery_timeout: int = 60  # Seconds to wait before trying again
    success_threshold: int = 3  # Successful calls needed to close circuit
    timeout: int = 30  # Request timeout in seconds


class CircuitBreaker:
    """Circuit breaker pattern implementation for resilient API calls."""

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.request_count = 0
        self.lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function call with circuit breaker protection."""
        async with self.lock:
            # Check if circuit should transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")

        self.request_count += 1

        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)

            # Handle success
            await self._handle_success()
            return result

        except Exception:
            # Handle failure
            await self._handle_failure()
            raise

    async def _handle_success(self):
        """Handle successful request."""
        async with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} closed after recovery")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)  # Decay failure count

    async def _handle_failure(self):
        """Handle failed request."""
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker {self.name} opened due to failures",
                        failure_count=self.failure_count,
                        threshold=self.config.failure_threshold,
                    )
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker {self.name} reopened due to failure during recovery"
                )

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "request_count": self.request_count,
            "last_failure_time": self.last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            },
        }


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class RequestQueue:
    """Async request queue with priority and backpressure."""

    def __init__(self, max_size: int = 1000, max_workers: int = None):
        self.max_size = max_size
        self.max_workers = max_workers or asyncio.BoundedSemaphore(
            50
        )  # Limit concurrent processing
        self.queue = asyncio.Queue(maxsize=max_size)
        self.processing = 0
        self.processed = 0
        self.dropped = 0
        self.workers = []
        self._shutdown = False

    async def put(self, item, priority: int = 0):
        """Add item to queue with optional priority."""
        if self.queue.qsize() >= self.max_size:
            self.dropped += 1
            raise QueueFullError("Request queue is full")

        # Wrap item with priority (lower number = higher priority)
        await self.queue.put((priority, time.time(), item))

    async def get(self):
        """Get next item from queue."""
        priority, timestamp, item = await self.queue.get()
        return item

    def qsize(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            "queue_size": self.queue.qsize(),
            "max_size": self.max_size,
            "processing": self.processing,
            "processed": self.processed,
            "dropped": self.dropped,
            "workers": len(self.workers),
        }

    async def start_workers(self, worker_func: Callable, num_workers: int = None):
        """Start worker tasks to process the queue."""
        if num_workers is None:
            num_workers = 10

        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(worker_func, f"worker-{i}"))
            self.workers.append(worker)

        logger.info(f"Started {num_workers} queue workers")

    async def _worker(self, worker_func: Callable, name: str):
        """Worker task to process queue items."""
        while not self._shutdown:
            try:
                async with self.max_workers:  # Limit concurrent processing
                    item = await asyncio.wait_for(self.get(), timeout=1.0)
                    self.processing += 1

                    try:
                        await worker_func(item)
                        self.processed += 1
                    except Exception as e:
                        logger.exception(f"Worker {name} failed to process item", error=str(e))
                    finally:
                        self.processing -= 1
                        self.queue.task_done()

            except asyncio.TimeoutError:
                continue  # Check shutdown flag
            except Exception as e:
                logger.exception(f"Worker {name} error", error=str(e))
                await asyncio.sleep(1)  # Brief pause on error

    async def shutdown(self):
        """Gracefully shutdown the queue."""
        self._shutdown = True

        # Wait for current items to be processed
        await self.queue.join()

        # Cancel worker tasks
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)

        logger.info("Request queue shutdown complete")


class QueueFullError(Exception):
    """Exception raised when request queue is full."""

    pass


class ResilienceManager:
    """Manages circuit breakers and request queues for service resilience."""

    def __init__(self):
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.request_queues: dict[str, RequestQueue] = {}

    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]

    def get_request_queue(
        self, name: str, max_size: int = 1000, max_workers: int = None
    ) -> RequestQueue:
        """Get or create a request queue."""
        if name not in self.request_queues:
            self.request_queues[name] = RequestQueue(max_size, max_workers)
        return self.request_queues[name]

    def get_all_stats(self) -> dict[str, Any]:
        """Get statistics for all circuit breakers and queues."""
        return {
            "circuit_breakers": {
                name: cb.get_stats() for name, cb in self.circuit_breakers.items()
            },
            "request_queues": {
                name: queue.get_stats() for name, queue in self.request_queues.items()
            },
        }

    async def shutdown(self):
        """Shutdown all managed resources."""
        # Shutdown all queues
        for queue in self.request_queues.values():
            await queue.shutdown()

        logger.info("Resilience manager shutdown complete")


# Global resilience manager instance
_resilience_manager: ResilienceManager | None = None


def get_resilience_manager() -> ResilienceManager:
    """Get the global resilience manager instance."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager


# Utility functions for common patterns
async def with_circuit_breaker(name: str, func: Callable, *args, **kwargs):
    """Execute function with circuit breaker protection."""
    manager = get_resilience_manager()
    circuit_breaker = manager.get_circuit_breaker(name)
    return await circuit_breaker.call(func, *args, **kwargs)


async def queue_request(queue_name: str, item, priority: int = 0):
    """Add request to named queue."""
    manager = get_resilience_manager()
    queue = manager.get_request_queue(queue_name)
    await queue.put(item, priority)
