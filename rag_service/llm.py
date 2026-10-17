"""聊天模型调用。"""

from langchain.chat_models import init_chat_model

from .config import ChatRequestError, load_chat_settings


def call_chat(
    messages: list[dict],
    env_file=None,
    model: str | None = None,
    base_url: str | None = None,
) -> str:
    settings = load_chat_settings(env_file, model, base_url)
    llm = init_chat_model(model_provider="openai", max_retries=0, **settings)
    # SDK 和网络错误统一转成 ChatRequestError，由 API 层映射状态码。
    try:
        response = llm.invoke(messages)
    except Exception as error:  # noqa: BLE001
        raise ChatRequestError(
            f"聊天调用失败：{type(error).__name__}，"
            f"状态码：{getattr(error, 'status_code', None) or '未提供'}"
        ) from None
    return response.content
