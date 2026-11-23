from contextlib import nullcontext
import logging
import threading
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class LangfuseManager:
    _lock: threading.Lock = threading.Lock()
    _handler: Optional[Any] = None
    _client: Optional[Any] = None
    _client_config_key: Optional[tuple] = None

    @classmethod
    def is_enabled(cls) -> bool:
        return bool(
            settings.langfuse_enabled
            and settings.langfuse_public_key
            and settings.langfuse_secret_key
        )

    @classmethod
    def get_client(cls) -> Optional[Any]:
        if not cls.is_enabled():
            return None

        cfg_key = (
            settings.langfuse_public_key,
            settings.langfuse_secret_key,
            settings.langfuse_host,
        )
        with cls._lock:
            if cls._client is not None and (
                cls._client_config_key is None or cls._client_config_key == cfg_key
            ):
                return cls._client
            try:
                from langfuse import Langfuse

                cls._client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                cls._client_config_key = cfg_key
                return cls._client
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse client: {e}")
                return None

    @classmethod
    def get_callback_handler(cls) -> Optional[Any]:
        """返回一个隔离的 Langfuse LangChain CallbackHandler 实例，绝不因观测异常阻断核心对话链路。"""
        if not cls.is_enabled():
            return None

        # 1. Try langfuse.callback (Langfuse v2 SDK)
        try:
            from langfuse.callback import CallbackHandler

            return CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to initialize CallbackHandler from langfuse.callback: {e}")

        # 2. Try langfuse.langchain (Langfuse v3 / v4 SDK)
        try:
            from langfuse.langchain import CallbackHandler

            try:
                return CallbackHandler(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
            except TypeError:
                # v4 signature: register singleton client with explicit credentials first
                if cls.get_client() is None:
                    return None
                return CallbackHandler(public_key=settings.langfuse_public_key)
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse CallbackHandler: {e}")
            return None

    @classmethod
    def trace_attributes_context(
        cls,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Return a Langfuse v4 propagate_attributes context manager (or nullcontext on fallback)."""
        if not cls.is_enabled():
            return nullcontext()
        try:
            from langfuse import propagate_attributes

            return propagate_attributes(metadata=metadata, tags=tags)
        except Exception as e:
            logger.debug(f"propagate_attributes unavailable: {e}")
            return nullcontext()

    @classmethod
    def update_current_trace(
        cls,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        if not cls.is_enabled():
            return

        # 1. Try v2/v3 langfuse_context
        try:
            from langfuse.decorators import langfuse_context

            if metadata:
                langfuse_context.update_current_trace(metadata=metadata)
            if tags:
                langfuse_context.update_current_trace(tags=tags)
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"langfuse_context.update_current_trace skipped: {e}")

        # 2. Langfuse v4 client API (never discard propagate_attributes context manager as a no-op)
        try:
            client = cls.get_client()
            if client is None:
                return

            if hasattr(client, "update_current_trace"):
                kwargs: Dict[str, Any] = {}
                if metadata is not None:
                    kwargs["metadata"] = metadata
                if tags is not None:
                    kwargs["tags"] = tags
                if kwargs:
                    client.update_current_trace(**kwargs)
                return

            if hasattr(client, "update_current_span"):
                span_meta: Dict[str, Any] = dict(metadata) if metadata else {}
                if tags:
                    span_meta.setdefault("tags", list(tags))
                if span_meta:
                    client.update_current_span(metadata=span_meta)
        except Exception as e:
            logger.warning(f"Failed to update current Langfuse trace/span: {e}")

    @classmethod
    def flush(cls) -> None:
        if cls._handler and hasattr(cls._handler, "flush"):
            try:
                cls._handler.flush()
            except Exception as e:
                logger.warning(f"Failed to flush langfuse handler: {e}")
        if cls._client and hasattr(cls._client, "flush"):
            try:
                cls._client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush langfuse client: {e}")

