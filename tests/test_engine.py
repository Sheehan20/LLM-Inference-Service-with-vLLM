from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from app.config import Settings
from app.inference.engine import EngineManager


@pytest.fixture
def settings():
    return Settings(
        model_name="microsoft/phi-2",
        concurrency_limit=2,
        max_num_seqs=4,
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        microbatch_wait_ms=0,  # Disable for tests
    )


@pytest.fixture
def logger():
    return structlog.get_logger()


@pytest.fixture
def engine_manager(settings, logger):
    return EngineManager(settings=settings, logger=logger)


class TestEngineManager:
    @pytest.mark.asyncio
    async def test_init_engine_success(self, engine_manager):
        mock_engine = AsyncMock()
        mock_engine_args = MagicMock()

        with patch("app.inference.engine.AsyncLLMEngine") as mock_async_engine:
            with patch("app.inference.engine.EngineArgs") as mock_args:
                mock_args.return_value = mock_engine_args
                mock_async_engine.from_engine_args.return_value = mock_engine

                await engine_manager.init_engine()

                assert engine_manager._engine is mock_engine
                mock_args.assert_called_once()
                mock_async_engine.from_engine_args.assert_called_once_with(mock_engine_args)

    @pytest.mark.asyncio
    async def test_init_engine_failure(self, engine_manager):
        with patch("app.inference.engine.AsyncLLMEngine") as mock_async_engine:
            with patch("app.inference.engine.EngineArgs"):
                mock_async_engine.from_engine_args.side_effect = RuntimeError("GPU not found")

                with pytest.raises(RuntimeError, match="GPU not found"):
                    await engine_manager.init_engine()

    @pytest.mark.asyncio
    async def test_generate_text_success(self, engine_manager):
        # Mock engine and request output
        mock_engine = AsyncMock()
        mock_output = MagicMock()
        mock_output.text = "Generated text"
        mock_output.token_ids = [1, 2, 3, 4]
        mock_output.finish_reason = "stop"

        mock_request_output = MagicMock()
        mock_request_output.outputs = [mock_output]
        mock_request_output.prompt_token_ids = [1, 2]

        async def mock_generate(*args, **kwargs):
            yield mock_request_output

        mock_engine.generate = mock_generate
        engine_manager._engine = mock_engine

        with patch("app.inference.engine.SamplingParams") as mock_sampling:
            with patch("app.inference.engine.GENERATED_TOKENS") as mock_counter:
                mock_sampling.return_value = MagicMock()

                result = await engine_manager.generate_text(
                    prompt="Test prompt",
                    max_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    stop=None,
                    repetition_penalty=1.0,
                )

                assert result["text"] == "Generated text"
                assert result["num_prompt_tokens"] == 2
                assert result["num_generated_tokens"] == 4
                assert result["finish_reason"] == "stop"
                assert "latency_ms" in result
                mock_counter.inc.assert_called_once_with(4)

    @pytest.mark.asyncio
    async def test_generate_text_empty_output(self, engine_manager):
        mock_engine = AsyncMock()

        async def mock_generate(*args, **kwargs):
            # Return empty generator
            return
            yield  # unreachable

        mock_engine.generate = mock_generate
        engine_manager._engine = mock_engine

        with patch("app.inference.engine.SamplingParams"):
            with pytest.raises(RuntimeError, match="Empty generation output"):
                await engine_manager.generate_text(
                    prompt="Test prompt",
                    max_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    stop=None,
                    repetition_penalty=1.0,
                )

    @pytest.mark.asyncio
    async def test_stream_text_success(self, engine_manager):
        mock_engine = AsyncMock()
        mock_output1 = MagicMock()
        mock_output1.text = "Hello"
        mock_output2 = MagicMock()
        mock_output2.text = "Hello world"

        mock_request_output1 = MagicMock()
        mock_request_output1.outputs = [mock_output1]
        mock_request_output2 = MagicMock()
        mock_request_output2.outputs = [mock_output2]

        async def mock_generate(*args, **kwargs):
            yield mock_request_output1
            yield mock_request_output2

        mock_engine.generate = mock_generate
        engine_manager._engine = mock_engine

        with patch("app.inference.engine.SamplingParams"):
            with patch("app.inference.engine.GENERATED_TOKENS") as mock_counter:
                chunks = []
                async for chunk in engine_manager.stream_text(
                    prompt="Test prompt",
                    max_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    stop=None,
                    repetition_penalty=1.0,
                ):
                    chunks.append(chunk)

                # Should get delta chunk and end event
                assert len(chunks) >= 2
                assert b"Hello" in chunks[0]
                assert b"world" in chunks[1]
                assert b"end" in chunks[-1]
                mock_counter.inc.assert_called_once()

    def test_build_sampling_params(self, engine_manager):
        with patch("app.inference.engine.SamplingParams") as mock_sampling:
            mock_params = MagicMock()
            mock_sampling.return_value = mock_params

            result = engine_manager._build_sampling_params(
                max_tokens=100,
                temperature=0.8,
                top_p=0.95,
                top_k=40,
                stop=["<|endoftext|>"],
                repetition_penalty=1.1,
            )

            mock_sampling.assert_called_once_with(
                temperature=0.8,
                top_p=0.95,
                top_k=40,
                max_tokens=100,
                stop=["<|endoftext|>"],
                repetition_penalty=1.1,
            )
            assert result is mock_params

    def test_build_sampling_params_no_stop(self, engine_manager):
        with patch("app.inference.engine.SamplingParams") as mock_sampling:
            engine_manager._build_sampling_params(
                max_tokens=100,
                temperature=0.8,
                top_p=0.95,
                top_k=40,
                stop=None,
                repetition_penalty=1.1,
            )

            # Ensure stop=None is passed as None to SamplingParams
            mock_sampling.assert_called_once()
            call_kwargs = mock_sampling.call_args[1]
            assert call_kwargs["stop"] is None
