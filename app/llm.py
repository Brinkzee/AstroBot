from typing import Optional
from langchain_openai import ChatOpenAI
from app.config import settings

def get_chat_model(
    streaming: bool = False,
    temperature: Optional[float] = None,
    model_name: Optional[str] = None
) -> ChatOpenAI:
    """获取统一配置的 OpenAI 兼容 Chat 模型客户端"""
    kwargs = {
        "model": model_name or settings.openai_model_name,
        "api_key": settings.openai_api_key,
        "temperature": settings.openai_temperature if temperature is None else temperature,
        "streaming": streaming,
    }
    if settings.effective_base_url:
        kwargs["base_url"] = settings.effective_base_url

    return ChatOpenAI(**kwargs)
