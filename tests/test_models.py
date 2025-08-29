import pytest
from pydantic import ValidationError
from app.models.request import GenerateRequest
from app.models.response import GenerateResponse


class TestGenerateRequest:
    def test_valid_request(self):
        request = GenerateRequest(
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            stop=["<|end|>"],
            repetition_penalty=1.1
        )
        
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.top_k == 50
        assert request.stop == ["<|end|>"]
        assert request.repetition_penalty == 1.1

    def test_request_defaults(self):
        request = GenerateRequest(prompt="Test prompt")
        
        assert request.prompt == "Test prompt"
        assert request.max_tokens == 128
        assert request.temperature == 1.0
        assert request.top_p == 1.0
        assert request.top_k == -1
        assert request.stop is None
        assert request.repetition_penalty == 1.0

    def test_empty_prompt(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="")
        
        errors = exc_info.value.errors()
        assert any("ensure this value has at least 1 characters" in str(error) for error in errors)

    def test_negative_max_tokens(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="Test", max_tokens=-1)
        
        errors = exc_info.value.errors()
        assert any("ensure this value is greater than 0" in str(error) for error in errors)

    def test_zero_max_tokens(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="Test", max_tokens=0)
        
        errors = exc_info.value.errors()
        assert any("ensure this value is greater than 0" in str(error) for error in errors)

    def test_negative_temperature(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="Test", temperature=-0.1)
        
        errors = exc_info.value.errors()
        assert any("ensure this value is greater than or equal to 0" in str(error) for error in errors)

    def test_high_temperature(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="Test", temperature=2.1)
        
        errors = exc_info.value.errors()
        assert any("ensure this value is less than or equal to 2" in str(error) for error in errors)

    def test_invalid_top_p(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="Test", top_p=1.5)
        
        errors = exc_info.value.errors()
        assert any("ensure this value is less than or equal to 1" in str(error) for error in errors)

    def test_negative_top_p(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="Test", top_p=-0.1)
        
        errors = exc_info.value.errors()
        assert any("ensure this value is greater than 0" in str(error) for error in errors)

    def test_negative_repetition_penalty(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="Test", repetition_penalty=0.5)
        
        errors = exc_info.value.errors()
        assert any("ensure this value is greater than or equal to 1" in str(error) for error in errors)

    def test_high_repetition_penalty(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateRequest(prompt="Test", repetition_penalty=3.0)
        
        errors = exc_info.value.errors()
        assert any("ensure this value is less than or equal to 2" in str(error) for error in errors)


class TestGenerateResponse:
    def test_valid_response(self):
        response = GenerateResponse(
            text="Generated text",
            num_prompt_tokens=10,
            num_generated_tokens=5,
            finish_reason="stop",
            latency_ms=150
        )
        
        assert response.text == "Generated text"
        assert response.num_prompt_tokens == 10
        assert response.num_generated_tokens == 5
        assert response.finish_reason == "stop"
        assert response.latency_ms == 150

    def test_response_with_none_finish_reason(self):
        response = GenerateResponse(
            text="Generated text",
            num_prompt_tokens=10,
            num_generated_tokens=5,
            finish_reason=None,
            latency_ms=150
        )
        
        assert response.finish_reason is None

    def test_response_negative_tokens(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateResponse(
                text="Generated text",
                num_prompt_tokens=-1,
                num_generated_tokens=5,
                finish_reason="stop",
                latency_ms=150
            )
        
        errors = exc_info.value.errors()
        assert any("ensure this value is greater than or equal to 0" in str(error) for error in errors)

    def test_response_negative_latency(self):
        with pytest.raises(ValidationError) as exc_info:
            GenerateResponse(
                text="Generated text",
                num_prompt_tokens=10,
                num_generated_tokens=5,
                finish_reason="stop",
                latency_ms=-1
            )
        
        errors = exc_info.value.errors()
        assert any("ensure this value is greater than or equal to 0" in str(error) for error in errors)