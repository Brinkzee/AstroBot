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
    def get_callback_handler(cls) -> Optional[Any]:
        if not cls.is_enabled():
            return None
            
        try:
            from langfuse.callback import CallbackHandler
            cls._handler = CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host
            )
            return cls._handler
        except ImportError:
            try:
                from langfuse.langchain import CallbackHandler
                cls._handler = CallbackHandler(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host
                )
                return cls._handler
            except Exception as e:
                logger.warning(f"Failed to import Langfuse CallbackHandler: {e}")
                return None
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse callback handler: {e}")
            return None

    @classmethod
    def update_current_trace(cls, metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> None:
        if not cls.is_enabled():
            return
            
        try:
            from langfuse.decorators import langfuse_context
            if metadata:
                langfuse_context.update_current_trace(metadata=metadata)
            if tags:
                langfuse_context.update_current_trace(tags=tags)
        except Exception as e:
            logger.warning(f"Failed to update current trace: {e}")

    @classmethod
    def flush(cls) -> None:
        if cls._handler and hasattr(cls._handler, 'flush'):
            try:
                cls._handler.flush()
            except Exception as e:
                logger.warning(f"Failed to flush langfuse handler: {e}")
