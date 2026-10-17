"""聊天模型配置。

只认 CHAT_* 这一组变量：要么三项齐全，要么直接报配置错误。避免新旧供应
商变量混用导致请求发到错误的服务。
"""

import os

from dotenv import load_dotenv

from .paths import PROJECT_ROOT

DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class ChatConfigurationError(ValueError):
    """聊天模型配置不完整。"""


class ChatRequestError(RuntimeError):
    """聊天请求失败；消息里不包含密钥或上游原始响应。"""


def load_chat_settings(
    env_file=None,
    model_override: str | None = None,
    base_url_override: str | None = None,
) -> dict:
    load_dotenv(env_file or DEFAULT_ENV_FILE, encoding="utf-8", override=False)

    api_key = os.getenv("CHAT_API_KEY")
    model = model_override or os.getenv("CHAT_MODEL")
    base_url = base_url_override or os.getenv("CHAT_BASE_URL")
    if not all((api_key, model, base_url)):
        raise ChatConfigurationError(
            "请完整配置 CHAT_API_KEY、CHAT_MODEL、CHAT_BASE_URL"
        )

    return {
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "timeout": _positive_int("CHAT_TIMEOUT", 30),
        "max_tokens": _positive_int("CHAT_MAX_TOKENS", 600),
    }


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        raise ChatConfigurationError(f"{name} 必须是正整数") from None
    if value < 1:
        raise ChatConfigurationError(f"{name} 必须是正整数")
    return value
