import argparse
import asyncio
import json
import math
import time
from typing import List

import httpx


async def _worker(client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int, latencies: List[float], errors: List[str]) -> None:
    start = time.perf_counter()
    try:
        resp = await client.post(url, json={"prompt": prompt, "max_tokens": max_tokens})
        if resp.status_code != 200:
            errors.append(f"HTTP {resp.status_code}")
        else:
            _ = resp.json()
    except Exception as e:
        errors.append(str(e))
    finally:
        latencies.append(time.perf_counter() - start)


async def run(url: str, concurrency: int, requests: int, prompt: str, max_tokens: int, timeout: float) -> None:
    latencies: List[float] = []
    errors: List[str] = []
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        sem = asyncio.Semaphore(concurrency)

        async def task() -> None:
            async with sem:
                await _worker(client, url, prompt, max_tokens, latencies, errors)

        await asyncio.gather(*[asyncio.create_task(task()) for _ in range(requests)])

    duration = time.perf_counter() - started
    if not latencies:
        print("No results")
        return

    latencies_ms = [x * 1000.0 for x in latencies]
    latencies_ms.sort()
    qps = requests / duration

    def pct(values, p):
        if not values:
            return 0.0
        k = (len(values) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(values) - 1)
        if f == c:
            return values[int(k)]
        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        return d0 + d1

    p50 = pct(latencies_ms, 50)
    p95 = pct(latencies_ms, 95)
    p99 = pct(latencies_ms, 99)
    err_rate = (len(errors) / requests) * 100.0

    print(json.dumps({
        "requests": requests,
        "concurrency": concurrency,
        "duration_s": round(duration, 3),
        "qps": round(qps, 2),
        "latency_ms": {
            "p50": round(p50, 1),
            "p95": round(p95, 1),
            "p99": round(p99, 1)
        },
        "error_rate_percent": round(err_rate, 3),
        "errors": errors[:5],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="Target URL, e.g., http://localhost:8000/v1/generate")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--prompt", type=str, default="Hello, world")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    asyncio.run(run(args.url, args.concurrency, args.requests, args.prompt, args.max_tokens, args.timeout))


if __name__ == "__main__":
    main()


