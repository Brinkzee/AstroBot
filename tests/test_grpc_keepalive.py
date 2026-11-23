import pytest
from unittest.mock import patch, MagicMock
from app.services.rag.milvus_client import MilvusKnowledgeStore, DEFAULT_GRPC_OPTIONS


def test_milvus_knowledge_store_grpc_options():
    """测试 MilvusKnowledgeStore 默认注入安全的 gRPC keepalive 参数，防止 too_many_pings 报错"""
    test_db = "./data/milvus/test_keepalive_db.db"
    store = MilvusKnowledgeStore(uri=test_db)
    try:
        grpc_opts = store.client._handler._grpc_options
        assert grpc_opts.get("grpc.keepalive_permit_without_calls") in (False, 0)
        assert grpc_opts.get("grpc.keepalive_time_ms") >= 60000
        assert grpc_opts.get("grpc.http2.max_pings_without_data") == 0
    finally:
        store.close(release_server=True)


def test_milvus_knowledge_store_custom_grpc_options():
    """测试 MilvusKnowledgeStore 允许透传并覆盖自定义 gRPC 选项"""
    test_db = "./data/milvus/test_keepalive_db.db"
    custom_opts = {
        "grpc.keepalive_time_ms": 120000,
        "grpc.max_metadata_size": 16384,
    }
    store = MilvusKnowledgeStore(uri=test_db, grpc_options=custom_opts)
    try:
        grpc_opts = store.client._handler._grpc_options
        assert grpc_opts.get("grpc.keepalive_time_ms") == 120000
        assert grpc_opts.get("grpc.max_metadata_size") == 16384
        assert grpc_opts.get("grpc.keepalive_permit_without_calls") in (False, 0)
    finally:
        store.close(release_server=True)


def test_milvus_lite_server_keepalive_patch():
    """测试 milvus-lite 进程内 gRPC 服务端已挂载宽容的 keepalive 配置"""
    try:
        import milvus_lite.adapter.grpc.server as ml_server
        assert getattr(ml_server, "_astro_keepalive_patched", False) is True
    except ImportError:
        pytest.skip("milvus-lite 未安装，跳过服务端配置测试")
