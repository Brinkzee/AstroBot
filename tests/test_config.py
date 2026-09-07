import os
from unittest import mock
import pytest

def test_settings_load_defaults():
    from app.config import Settings
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "mock-key"}, clear=True):
        s = Settings(_env_file=None)
        assert s.openai_api_key == "mock-key"
        assert s.openai_model_name == "gpt-4o-mini"
        assert s.openai_temperature == 0.7
        assert s.max_context_tokens == 2000

def test_settings_custom_env():
    from app.config import Settings
    custom_env = {
        "OPENAI_API_KEY": "custom-key",
        "OPENAI_BASE_URL": "https://api.deepseek.com/v1",
        "OPENAI_MODEL_NAME": "deepseek-chat",
        "OPENAI_TEMPERATURE": "0.2",
        "MAX_CONTEXT_TOKENS": "3000",
    }
    with mock.patch.dict(os.environ, custom_env, clear=True):
        s = Settings(_env_file=None)
        assert s.openai_api_key == "custom-key"
        assert s.effective_base_url == "https://api.deepseek.com/v1"
        assert s.openai_model_name == "deepseek-chat"
        assert s.openai_temperature == 0.2
        assert s.max_context_tokens == 3000

def test_settings_api_base_alias():
    from app.config import Settings
    custom_env = {
        "OPENAI_API_KEY": "custom-key",
        "OPENAI_API_BASE": "https://api.openai-proxy.com/v1",
    }
    with mock.patch.dict(os.environ, custom_env, clear=True):
        s = Settings(_env_file=None)
        assert s.effective_base_url == "https://api.openai-proxy.com/v1"
