# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Commands
```bash
# Setup development environment
python dev.py setup

# Run tests with coverage
python dev.py test

# Code quality checks
python dev.py lint
python dev.py format

# Build and run locally
python dev.py build
python dev.py run

# Load testing
python dev.py load-test -c 20 -r 1000

# Full CI pipeline locally
python dev.py ci

# Security scans
python dev.py security

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run with specific model
export MODEL_NAME=microsoft/phi-2
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Commands
```bash
# Build image
docker build -t vllm-phi2-api:latest .

# Run container with GPU support
docker run --gpus all -p 8000:8000 -e MODEL_NAME=microsoft/phi-2 vllm-phi2-api:latest

# Run with docker-compose
docker-compose up
```

### Testing Commands
```bash
# Run all tests
pytest tests/ -v --cov=app

# Run specific test file
pytest tests/test_main.py -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=html
```

## Architecture

### Core Components
- **FastAPI Application** (`app/main.py`): Main web service with comprehensive endpoints
- **EngineManager** (`app/inference/engine.py`): Manages vLLM AsyncLLMEngine with circuit breaker protection
- **Configuration** (`app/config.py`): Comprehensive settings with validation using Pydantic
- **Authentication & Rate Limiting** (`app/auth.py`): API key auth and token bucket rate limiting
- **Error Handling** (`app/errors.py`): Structured error handling with custom exceptions
- **Health Monitoring** (`app/health.py`): Advanced health checks for all components
- **Alerting System** (`app/alerting.py`): Prometheus-based alerting with configurable rules
- **Resilience Patterns** (`app/resilience.py`): Circuit breakers and request queues
- **Metrics & Monitoring** (`app/metrics.py`): Comprehensive Prometheus metrics

### Key Architecture Patterns
- **Circuit Breaker Pattern**: Prevents cascading failures with automatic recovery
- **Rate Limiting**: Token bucket algorithm with per-client limits
- **Request Queuing**: Async request queues with priority and backpressure
- **Structured Error Handling**: Custom exceptions with proper HTTP status codes
- **Comprehensive Monitoring**: System, GPU, and application-level metrics
- **Health Checks**: Multi-component health monitoring with automatic alerts

### API Endpoints
- `POST /v1/generate`: Non-streaming text generation (with auth)
- `POST /v1/stream`: Server-sent events streaming (with auth)
- `GET /healthz`: Basic health check
- `GET /health/detailed`: Comprehensive health information
- `GET /metrics`: Prometheus metrics endpoint
- `GET /alerts`: Current alerts and history
- `GET /auth/info`: Authentication configuration
- `GET /resilience/stats`: Circuit breaker and queue statistics

### Configuration
Key environment variables:
- `MODEL_NAME`: HuggingFace model identifier (default: microsoft/phi-2)
- `ENABLE_AUTH`: Enable API authentication (default: false)
- `API_KEYS`: Comma-separated API keys for authentication
- `RATE_LIMIT_RPM`: Requests per minute limit (default: 60)
- `CONCURRENCY_LIMIT`: Max concurrent requests (default: 20)
- `GPU_MEMORY_UTILIZATION`: GPU memory fraction (default: 0.90)
- `DEBUG`: Enable debug mode with detailed error messages

### Security Features
- **API Key Authentication**: Bearer token authentication with configurable keys
- **Rate Limiting**: Per-client rate limiting with token bucket algorithm
- **Input Validation**: Comprehensive request validation with Pydantic
- **Error Handling**: Secure error responses without sensitive information exposure
- **Security Scanning**: Automated security scans in CI/CD pipeline

### Monitoring & Observability
- **Prometheus Metrics**: Request counts, latencies, GPU stats, system metrics
- **Health Monitoring**: Component-level health checks with automatic recovery
- **Alerting**: Configurable alerts for failures, high latency, resource exhaustion
- **Structured Logging**: JSON logs with request correlation IDs
- **Performance Tracking**: Token generation rates, throughput metrics

## Development Notes

### Testing Strategy
- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: End-to-end API testing
- **Load Testing**: Performance and stress testing with configurable parameters
- **Security Testing**: Automated vulnerability scanning
- **Coverage**: Minimum 80% code coverage requirement

### CI/CD Pipeline
- **Automated Testing**: Lint, type check, unit tests, security scans
- **Docker Building**: Multi-stage builds with caching
- **Performance Testing**: Automated load testing on GPU runners
- **Security Compliance**: Container vulnerability scanning
- **Deployment**: Automated staging and production deployments

### Performance Characteristics
- **Throughput**: 12-15 QPS on RTX 4070 Ti Super with Phi-2
- **Latency**: P95 latency ~450ms under load
- **Concurrency**: Up to 20 concurrent requests with circuit breaker protection
- **Reliability**: 99.94% uptime with automatic recovery patterns
- **Resource Usage**: Optimized GPU utilization with monitoring and alerts

### Development Tools
- **Pre-commit Hooks**: Automated code quality checks
- **Development Script**: `dev.py` for common development tasks
- **Docker Support**: Full containerization with GPU support
- **Code Quality**: ruff, mypy, bandit for comprehensive code analysis