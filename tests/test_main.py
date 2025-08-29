import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.config import Settings
from app.inference.engine import EngineManager


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_engine_manager():
    manager = MagicMock(spec=EngineManager)
    manager.generate_text.return_value = {
        "text": "Test response",
        "num_prompt_tokens": 5,
        "num_generated_tokens": 3,
        "finish_reason": "stop",
        "latency_ms": 100
    }
    
    async def mock_stream():
        yield b'data: {"delta": "Test"}\n\n'
        yield b'data: {"delta": " response"}\n\n'
        yield b'data: {"event": "end", "generated_chars": 13}\n\n'
    
    manager.stream_text.return_value = mock_stream()
    return manager


class TestMainEndpoints:
    def test_healthz(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_generate_success(self, client, mock_engine_manager):
        with patch.object(app.state, 'engine_manager', mock_engine_manager):
            response = client.post(
                "/v1/generate",
                json={
                    "prompt": "Test prompt",
                    "max_tokens": 50,
                    "temperature": 0.7
                }
            )
            
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Test response"
        assert data["num_prompt_tokens"] == 5
        assert data["num_generated_tokens"] == 3
        assert data["finish_reason"] == "stop"
        assert data["latency_ms"] == 100

    def test_generate_invalid_request(self, client):
        response = client.post(
            "/v1/generate",
            json={
                "prompt": "Test prompt",
                "max_tokens": -1,  # Invalid
            }
        )
        assert response.status_code == 422  # Validation error

    def test_generate_missing_prompt(self, client):
        response = client.post(
            "/v1/generate",
            json={
                "max_tokens": 50,
            }
        )
        assert response.status_code == 422  # Validation error

    def test_generate_engine_error(self, client, mock_engine_manager):
        mock_engine_manager.generate_text.side_effect = RuntimeError("Engine error")
        
        with patch.object(app.state, 'engine_manager', mock_engine_manager):
            response = client.post(
                "/v1/generate",
                json={
                    "prompt": "Test prompt",
                    "max_tokens": 50
                }
            )
            
        assert response.status_code == 500
        assert "error" in response.json()

    def test_stream_success(self, client, mock_engine_manager):
        with patch.object(app.state, 'engine_manager', mock_engine_manager):
            response = client.post(
                "/v1/stream",
                json={
                    "prompt": "Test prompt",
                    "max_tokens": 50,
                    "stream": True
                }
            )
            
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        
        # Check that we get streaming response
        content = response.content.decode()
        assert "Test" in content
        assert "response" in content
        assert "end" in content

    def test_stream_engine_error(self, client, mock_engine_manager):
        async def mock_stream_error():
            yield b'data: {"delta": "Test"}\n\n'
            raise RuntimeError("Stream error")
        
        mock_engine_manager.stream_text.return_value = mock_stream_error()
        
        with patch.object(app.state, 'engine_manager', mock_engine_manager):
            response = client.post(
                "/v1/stream",
                json={
                    "prompt": "Test prompt",
                    "max_tokens": 50,
                    "stream": True
                }
            )
            
        assert response.status_code == 200
        content = response.content.decode()
        assert "stream failed" in content

    def test_metrics_endpoint_exists(self, client):
        # The /metrics endpoint is mounted, verify it's accessible
        response = client.get("/metrics")
        # Should not be 404, either 200 or some other status
        assert response.status_code != 404


class TestRequestModels:
    def test_generate_request_defaults(self, client, mock_engine_manager):
        with patch.object(app.state, 'engine_manager', mock_engine_manager):
            response = client.post(
                "/v1/generate",
                json={
                    "prompt": "Test prompt"
                }
            )
            
        assert response.status_code == 200
        # Verify defaults were applied by checking the call
        mock_engine_manager.generate_text.assert_called_once()
        call_kwargs = mock_engine_manager.generate_text.call_args.kwargs
        assert call_kwargs["max_tokens"] == 128  # Default
        assert call_kwargs["temperature"] == 1.0  # Default
        assert call_kwargs["top_p"] == 1.0  # Default

    def test_generate_request_custom_params(self, client, mock_engine_manager):
        with patch.object(app.state, 'engine_manager', mock_engine_manager):
            response = client.post(
                "/v1/generate",
                json={
                    "prompt": "Test prompt",
                    "max_tokens": 200,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 40,
                    "stop": ["<|end|>"],
                    "repetition_penalty": 1.1
                }
            )
            
        assert response.status_code == 200
        call_kwargs = mock_engine_manager.generate_text.call_args.kwargs
        assert call_kwargs["max_tokens"] == 200
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 40
        assert call_kwargs["stop"] == ["<|end|>"]
        assert call_kwargs["repetition_penalty"] == 1.1