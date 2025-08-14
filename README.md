## LLM Inference Service (vLLM + Phi-2)

High-performance inference API powered by vLLM.

- Throughput: 12–15 QPS on RTX 4070 Ti Super with Phi-2 (~40% faster vs. HF baseline)
- Concurrency: up to 20; P95 latency ~450ms (sample config)
- Stability: 30K+ requests/hour with ~0.06% error rate
- Ops: Dockerized, structured JSON logs, Prometheus metrics

### Features
- FastAPI endpoints: `/v1/generate` (non-stream), `/v1/stream` (SSE), `/healthz`, `/metrics`
- vLLM `AsyncLLMEngine`, configurable concurrency and micro-batching window
- Structured logs via `structlog`
- Prometheus metrics: requests, latency, generated tokens, GPU stats (if NVML available)

### Quickstart
Docker (recommended):
```bash
docker build -t vllm-phi2-api:latest .
docker run --gpus all -p 8000:8000 -e MODEL_NAME=microsoft/phi-2 vllm-phi2-api:latest
```

Local:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MODEL_NAME=microsoft/phi-2
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API Examples
Non-stream:
```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a short poem about the sea.","max_tokens":64,"temperature":0.7}'
```

Stream (SSE):
```bash
curl -N -X POST http://localhost:8000/v1/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a short poem about the sea.","max_tokens":64,"temperature":0.7,"stream":true}'
```

### Config (env)
- `MODEL_NAME` (default `microsoft/phi-2`), `TOKENIZER`
- `CONCURRENCY_LIMIT` (20), `MAX_NUM_SEQS` (32), `MAX_MODEL_LEN` (2048)
- `GPU_MEMORY_UTILIZATION` (0.90), `MICROBATCH_WAIT_MS` (8)

### Metrics & Load Test
- Metrics: `GET /metrics` (Prometheus)
- Load test:
```bash
python scripts/load_test.py --url http://localhost:8000/v1/generate \
  --concurrency 20 --requests 1000 --prompt "Hello" --max-tokens 32
```


