import pytest
from unittest.mock import patch, MagicMock

from app.config import settings

def test_langfuse_disabled_graceful_fallback():
    # Setup test where settings indicate Langfuse is disabled or keys missing
    with patch.object(settings, 'langfuse_enabled', False), \
         patch.object(settings, 'langfuse_public_key', None), \
         patch.object(settings, 'langfuse_secret_key', None):
        
        from app.services.observability.langfuse_service import LangfuseManager
        handler = LangfuseManager.get_callback_handler()
        assert handler is None

def test_langfuse_enabled_initialization():
    with patch.object(settings, 'langfuse_enabled', True), \
         patch.object(settings, 'langfuse_public_key', 'pk-123'), \
         patch.object(settings, 'langfuse_secret_key', 'sk-123'), \
         patch.object(settings, 'langfuse_host', 'http://test'):
        
        from app.services.observability.langfuse_service import LangfuseManager
        
        mock_handler_class = MagicMock()
        mock_handler_class.return_value = "mock_handler_instance"
        
        mock_callback_module = MagicMock()
        mock_callback_module.CallbackHandler = mock_handler_class
        
        with patch.dict('sys.modules', {'langfuse.callback': mock_callback_module}):
            handler = LangfuseManager.get_callback_handler()
            assert handler == "mock_handler_instance"
            mock_handler_class.assert_called_once_with(
                public_key='pk-123',
                secret_key='sk-123',
                host='http://test'
            )
def test_workflow_compilation_with_langfuse_disabled():
    with patch.object(settings, 'langfuse_enabled', False):
        from app.services.workflow.engine import build_workflow_graph
        graph = build_workflow_graph()
        assert graph is not None
        has_callbacks = hasattr(graph, 'config') and graph.config and 'callbacks' in graph.config
        assert not has_callbacks

def test_workflow_compilation_with_langfuse_enabled():
    with patch.object(settings, 'langfuse_enabled', True), \
         patch.object(settings, 'langfuse_public_key', 'pk-123'), \
         patch.object(settings, 'langfuse_secret_key', 'sk-123'), \
         patch.object(settings, 'langfuse_host', 'http://test'):
         
        with patch('app.services.observability.langfuse_service.LangfuseManager.get_callback_handler') as mock_handler:
            mock_handler_instance = MagicMock()
            mock_handler.return_value = mock_handler_instance
            
            from app.services.workflow.engine import build_workflow_graph
            graph = build_workflow_graph()
            
            assert graph is not None
            # graph should be a RunnableBinding with callbacks
            # Since CompiledGraph isn't simple, we just assert graph exists and mock_handler was called
            mock_handler.assert_called_once()

def test_update_current_trace_safe_when_disabled():
    from app.services.observability.langfuse_service import LangfuseManager
    # Just calling it should not raise an exception
    LangfuseManager.update_current_trace(metadata={"a": 1}, tags=["tag1"])
    LangfuseManager.flush()
