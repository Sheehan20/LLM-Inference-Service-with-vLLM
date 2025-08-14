from __future__ import annotations

import asyncio
import time
import uuid
from typing import AsyncGenerator, Optional

import orjson
import structlog

from app.config import Settings
from app.metrics import GENERATED_TOKENS


class EngineManager:
    def __init__(self, settings: Settings, logger: structlog.stdlib.BoundLogger) -> None:
        self.settings = settings
        self.logger = logger
        self._engine = None
        self._semaphore = asyncio.Semaphore(self.settings.concurrency_limit)

    async def init_engine(self) -> None:
        t0 = time.perf_counter()
        try:
            # Import lazily to speed up cold start
            from vllm import AsyncLLMEngine
            try:
                from vllm.engine.arg_utils import AsyncEngineArgs as EngineArgs  # type: ignore
            except Exception:
                from vllm.engine.arg_utils import EngineArgs  # type: ignore

            engine_args = EngineArgs(
                model=self.settings.model_name,
                tokenizer=self.settings.tokenizer or self.settings.model_name,
                max_num_seqs=self.settings.max_num_seqs,
                max_model_len=self.settings.max_model_len,
                gpu_memory_utilization=self.settings.gpu_memory_utilization,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            dt = (time.perf_counter() - t0) * 1000
            self.logger.info(
                "engine_initialized",
                model=self.settings.model_name,
                tokenizer=self.settings.tokenizer,
                max_num_seqs=self.settings.max_num_seqs,
                max_model_len=self.settings.max_model_len,
                gpu_memory_utilization=self.settings.gpu_memory_utilization,
                duration_ms=int(dt),
            )
        except Exception as e:
            self.logger.exception("engine_init_failed", error=str(e))
            raise

    def _build_sampling_params(self, *, max_tokens: int, temperature: float, top_p: float, top_k: int, stop: Optional[list[str]], repetition_penalty: float):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop or None,
            repetition_penalty=repetition_penalty,
        )
        return sampling_params

    async def generate_text(self, *, prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: int, stop: Optional[list[str]], repetition_penalty: float) -> dict:
        if self.settings.microbatch_wait_ms > 0:
            await asyncio.sleep(self.settings.microbatch_wait_ms / 1000.0)

        await self._semaphore.acquire()
        try:
            assert self._engine is not None
            sampling_params = self._build_sampling_params(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                repetition_penalty=repetition_penalty,
            )

            request_id = str(uuid.uuid4())
            start = time.perf_counter()
            gen = self._engine.generate(prompt, sampling_params, request_id)

            last_output = None
            async for request_output in gen:
                last_output = request_output

            if last_output is None:
                raise RuntimeError("Empty generation output")

            output = last_output.outputs[0]
            text = output.text
            num_prompt_tokens = len(last_output.prompt_token_ids)
            num_generated_tokens = len(output.token_ids)
            finish_reason = getattr(output, "finish_reason", None)
            duration_ms = int((time.perf_counter() - start) * 1000)

            GENERATED_TOKENS.inc(num_generated_tokens)

            return {
                "text": text,
                "num_prompt_tokens": num_prompt_tokens,
                "num_generated_tokens": num_generated_tokens,
                "finish_reason": finish_reason,
                "latency_ms": duration_ms,
            }
        finally:
            self._semaphore.release()

    async def stream_text(self, *, prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: int, stop: Optional[list[str]], repetition_penalty: float) -> AsyncGenerator[bytes, None]:
        if self.settings.microbatch_wait_ms > 0:
            await asyncio.sleep(self.settings.microbatch_wait_ms / 1000.0)

        await self._semaphore.acquire()
        try:
            assert self._engine is not None
            from vllm import SamplingParams

            sampling_params = self._build_sampling_params(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                repetition_penalty=repetition_penalty,
            )

            request_id = str(uuid.uuid4())
            gen = self._engine.generate(prompt, sampling_params, request_id)

            prev_len = 0
            async for request_output in gen:
                output = request_output.outputs[0]
                text = output.text or ""
                delta = text[prev_len:]
                prev_len = len(text)
                if delta:
                    payload = {"delta": delta}
                    yield b"data: " + orjson.dumps(payload) + b"\n\n"

            # end event
            final_tokens = prev_len  # approximate
            GENERATED_TOKENS.inc(final_tokens)
            yield b"data: " + orjson.dumps({"event": "end", "generated_chars": final_tokens}) + b"\n\n"
        finally:
            self._semaphore.release()


