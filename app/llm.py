from typing import Optional
import httpx
from langchain_openai import ChatOpenAI
from app.config import settings


def clear_llm_cache() -> None:
    """清理 langchain_openai 内部的客户端 LRU 缓存，避免跨 event loop 复用导致死锁"""
    try:
        from langchain_openai.chat_models._client_utils import (
            _cached_async_httpx_client,
            _cached_sync_httpx_client,
        )
        _cached_async_httpx_client.cache_clear()
        _cached_sync_httpx_client.cache_clear()
    except Exception:
        pass


def get_chat_model(
    streaming: bool = False,
    temperature: Optional[float] = None,
    model_name: Optional[str] = None,
    timeout: Optional[float] = 30.0,
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
    if timeout is not None:
        kwargs["timeout"] = httpx.Timeout(timeout)

    return ChatOpenAI(**kwargs)

