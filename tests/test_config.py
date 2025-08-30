import os

from app.config import Settings, get_settings


class TestSettings:
    def setup_method(self):
        """Clear environment variables and cache before each test."""
        # Clear the LRU cache
        get_settings.cache_clear()

        # Store original environment
        self.original_env = dict(os.environ)

    def teardown_method(self):
        """Restore environment after each test."""
        # Clear environment
        os.environ.clear()
        os.environ.update(self.original_env)

        # Clear the cache again
        get_settings.cache_clear()

    def test_default_settings(self):
        settings = Settings()

        assert settings.model_name == "microsoft/phi-2"
        assert settings.tokenizer is None
        assert settings.concurrency_limit == 20
        assert settings.max_num_seqs == 32
        assert settings.max_model_len == 2048
        assert settings.gpu_memory_utilization == 0.90
        assert settings.microbatch_wait_ms == 8
        assert settings.log_level == "INFO"
        assert settings.metrics_enabled is True
        assert settings.sse_heartbeat_interval_s == 10.0
        assert settings.max_log_text_chars == 512

    def test_settings_from_env(self):
        env_vars = {
            "MODEL_NAME": "custom/model",
            "TOKENIZER": "custom/tokenizer",
            "CONCURRENCY_LIMIT": "10",
            "MAX_NUM_SEQS": "16",
            "MAX_MODEL_LEN": "1024",
            "GPU_MEMORY_UTILIZATION": "0.8",
            "MICROBATCH_WAIT_MS": "5",
            "LOG_LEVEL": "DEBUG",
            "METRICS_ENABLED": "false",
            "SSE_HEARTBEAT_INTERVAL_S": "5.0",
            "MAX_LOG_TEXT_CHARS": "256",
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        settings = Settings()

        assert settings.model_name == "custom/model"
        assert settings.tokenizer == "custom/tokenizer"
        assert settings.concurrency_limit == 10
        assert settings.max_num_seqs == 16
        assert settings.max_model_len == 1024
        assert settings.gpu_memory_utilization == 0.8
        assert settings.microbatch_wait_ms == 5
        assert settings.log_level == "DEBUG"
        assert settings.metrics_enabled is False
        assert settings.sse_heartbeat_interval_s == 5.0
        assert settings.max_log_text_chars == 256

    def test_get_settings_caching(self):
        # Test that get_settings returns the same instance (LRU cache)
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_metrics_enabled_various_values(self):
        test_cases = [
            ("true", True),
            ("TRUE", True),
            ("True", True),
            ("1", True),
            ("false", False),
            ("FALSE", False),
            ("False", False),
            ("0", False),
            ("", False),
            ("anything_else", False),
        ]

        for env_value, expected in test_cases:
            # Clear environment first
            os.environ.clear()
            os.environ.update(self.original_env)

            os.environ["METRICS_ENABLED"] = env_value
            settings = Settings()
            assert (
                settings.metrics_enabled is expected
            ), f"Failed for '{env_value}' - expected {expected}, got {settings.metrics_enabled}"

    def test_tokenizer_empty_string_becomes_none(self):
        os.environ["TOKENIZER"] = ""
        settings = Settings()
        assert settings.tokenizer is None
