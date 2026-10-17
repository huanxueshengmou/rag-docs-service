"""聊天模型配置的读取与校验。"""

import pytest
from rag_service.config import ChatConfigurationError, load_chat_settings

ENV_NAMES = (
    "CHAT_API_KEY",
    "CHAT_MODEL",
    "CHAT_BASE_URL",
    "CHAT_TIMEOUT",
    "CHAT_MAX_TOKENS",
)


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return tmp_path / ".env"


def test_complete_config_is_loaded(env_file):
    env_file.write_text(
        "CHAT_API_KEY=fake-key\n"
        "CHAT_MODEL=fake-model\n"
        "CHAT_BASE_URL=https://example.test/v1\n"
        "CHAT_TIMEOUT=45\n"
        "CHAT_MAX_TOKENS=800\n",
        encoding="utf-8",
    )

    settings = load_chat_settings(env_file)

    assert settings == {
        "api_key": "fake-key",
        "model": "fake-model",
        "base_url": "https://example.test/v1",
        "timeout": 45,
        "max_tokens": 800,
    }


def test_missing_key_raises_configuration_error(env_file):
    env_file.write_text(
        "CHAT_MODEL=fake-model\nCHAT_BASE_URL=https://example.test/v1\n",
        encoding="utf-8",
    )

    with pytest.raises(ChatConfigurationError):
        load_chat_settings(env_file)


def test_overrides_take_precedence(env_file):
    env_file.write_text(
        "CHAT_API_KEY=fake-key\n"
        "CHAT_MODEL=fake-model\n"
        "CHAT_BASE_URL=https://example.test/v1\n",
        encoding="utf-8",
    )

    settings = load_chat_settings(
        env_file,
        model_override="override-model",
        base_url_override="https://override.test/v1",
    )

    assert settings["model"] == "override-model"
    assert settings["base_url"] == "https://override.test/v1"


def test_invalid_timeout_is_rejected(env_file):
    env_file.write_text(
        "CHAT_API_KEY=fake-key\n"
        "CHAT_MODEL=fake-model\n"
        "CHAT_BASE_URL=https://example.test/v1\n"
        "CHAT_TIMEOUT=0\n",
        encoding="utf-8",
    )

    with pytest.raises(ChatConfigurationError):
        load_chat_settings(env_file)
