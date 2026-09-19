import os
import logging
from typing import Optional, Any, Dict, List
from app.config import settings

logger = logging.getLogger(__name__)

class LangfuseManager:
    _handler: Optional[Any] = None
    _client: Optional[Any] = None

    @classmethod
    def is_enabled(cls) -> bool:
        return bool(
            settings.langfuse_enabled and 
            settings.langfuse_public_key and 
            settings.langfuse_secret_key
        )

    @classmethod
    def get_client(cls) -> Optional[Any]:
        if not cls.is_enabled():
            return None
        if cls._client is not None:
            return cls._client
        try:
            from langfuse import Langfuse
            cls._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            return cls._client
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse client: {e}")
            return None

    @classmethod
    def get_callback_handler(cls) -> Optional[Any]:
        if not cls.is_enabled():
            return None

        # Ensure environment variables are set for SDK and background processes
        if settings.langfuse_public_key:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
        if settings.langfuse_secret_key:
            os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        if settings.langfuse_host:
            os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
            os.environ.setdefault("LANGFUSE_BASEURL", settings.langfuse_host)

        # 1. First try langfuse.callback (Langfuse v2 SDK)
        try:
            from langfuse.callback import CallbackHandler
            cls._handler = CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host
            )
            return cls._handler
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to initialize CallbackHandler from langfuse.callback: {e}")

        # 2. Try langfuse.langchain (Langfuse v3 / v4 SDK)
        try:
            from langfuse.langchain import CallbackHandler
            try:
                # Try v2/v3 signature
                cls._handler = CallbackHandler(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host
                )
                return cls._handler
            except TypeError:
                # v4 signature: initialize client first, then pass public_key
                cls.get_client()
                cls._handler = CallbackHandler(public_key=settings.langfuse_public_key)
                return cls._handler
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse CallbackHandler: {e}")
            return None

    @classmethod
    def update_current_trace(cls, metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> None:
        if not cls.is_enabled():
            return

        # Try v2/v3 langfuse_context
        try:
            from langfuse.decorators import langfuse_context
            if metadata:
                langfuse_context.update_current_trace(metadata=metadata)
            if tags:
                langfuse_context.update_current_trace(tags=tags)
            return
        except (ImportError, Exception):
            pass

        # Try v4 propagate_attributes
        try:
            from langfuse import propagate_attributes
            propagate_attributes(metadata=metadata, tags=tags)
        except Exception as e:
            logger.warning(f"Failed to update current trace: {e}")

    @classmethod
    def flush(cls) -> None:
        if cls._handler and hasattr(cls._handler, 'flush'):
            try:
                cls._handler.flush()
            except Exception as e:
                logger.warning(f"Failed to flush langfuse handler: {e}")
        if cls._client and hasattr(cls._client, 'flush'):
            try:
                cls._client.flush()
            except Exception as e:
                logger.warning(f"Failed to flush langfuse client: {e}")

