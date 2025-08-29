import pytest
import asyncio
from unittest.mock import patch, MagicMock
from scripts.load_test import _worker, run, pct


class TestLoadTest:
    @pytest.mark.asyncio
    async def test_worker_success(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "response"}
        mock_client.post.return_value = mock_response
        
        latencies = []
        errors = []
        
        await _worker(
            client=mock_client,
            url="http://localhost:8000/v1/generate",
            prompt="test",
            max_tokens=50,
            latencies=latencies,
            errors=errors
        )
        
        assert len(latencies) == 1
        assert latencies[0] > 0  # Should record some latency
        assert len(errors) == 0
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_worker_http_error(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client.post.return_value = mock_response
        
        latencies = []
        errors = []
        
        await _worker(
            client=mock_client,
            url="http://localhost:8000/v1/generate",
            prompt="test",
            max_tokens=50,
            latencies=latencies,
            errors=errors
        )
        
        assert len(latencies) == 1
        assert len(errors) == 1
        assert "HTTP 500" in errors[0]

    @pytest.mark.asyncio
    async def test_worker_exception(self):
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Connection failed")
        
        latencies = []
        errors = []
        
        await _worker(
            client=mock_client,
            url="http://localhost:8000/v1/generate",
            prompt="test",
            max_tokens=50,
            latencies=latencies,
            errors=errors
        )
        
        assert len(latencies) == 1
        assert len(errors) == 1
        assert "Connection failed" in errors[0]

    @pytest.mark.asyncio
    async def test_run_integration(self):
        # Mock the entire HTTP client behavior
        with patch("scripts.load_test.httpx.AsyncClient") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"text": "response"}
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            with patch("builtins.print") as mock_print:
                await run(
                    url="http://localhost:8000/v1/generate",
                    concurrency=2,
                    requests=5,
                    prompt="test",
                    max_tokens=10,
                    timeout=30.0
                )
                
                # Should have printed results
                mock_print.assert_called_once()
                printed_output = mock_print.call_args[0][0]
                
                # Verify JSON structure in output
                import json
                result = json.loads(printed_output)
                assert result["requests"] == 5
                assert result["concurrency"] == 2
                assert "qps" in result
                assert "latency_ms" in result
                assert "p50" in result["latency_ms"]
                assert "error_rate_percent" in result

    def test_pct_function(self):
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        assert pct(values, 0) == 10
        assert pct(values, 50) == 55  # Median
        assert pct(values, 100) == 100
        
        # Test with single value
        assert pct([42], 50) == 42
        
        # Test with empty list
        assert pct([], 50) == 0.0

    def test_pct_edge_cases(self):
        # Test with two values
        values = [10, 20]
        assert pct(values, 0) == 10
        assert pct(values, 50) == 15  # Average
        assert pct(values, 100) == 20
        
        # Test exact percentile matches
        values = [1, 2, 3, 4, 5]
        assert pct(values, 0) == 1
        assert pct(values, 25) == 2
        assert pct(values, 50) == 3
        assert pct(values, 75) == 4
        assert pct(values, 100) == 5