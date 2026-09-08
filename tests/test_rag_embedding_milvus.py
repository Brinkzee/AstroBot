import os
import shutil
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.config import settings
from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.milvus_client import MilvusKnowledgeStore


def test_config_milvus_uri_and_token():
    """验证 config.py 中新增的 MILVUS_URI 属性及 token 兼容性"""
    assert hasattr(settings, "MILVUS_URI")
    assert settings.MILVUS_URI == "./data/milvus/astro_bot.db"
    # 同时兼容小写字段
    assert hasattr(settings, "milvus_uri")
    assert settings.milvus_uri == "./data/milvus/astro_bot.db"


def test_rag_package_exports():
    """验证 app.services.rag 包导出了关键类"""
    import app.services.rag as rag_pkg
    assert hasattr(rag_pkg, "BGEEmbeddingClient")
    assert hasattr(rag_pkg, "MilvusKnowledgeStore")


def test_bge_embedding_client_mock_query():
    """测试离线/Mock 模式下的单句 1024 维向量生成"""
    with patch.object(settings, "huggingface_token", None):
        client = BGEEmbeddingClient(token=None)
        assert client.is_mock is True
        
        vec = client.embed_query("什么是退货政策？")
        assert isinstance(vec, list)
        assert len(vec) == 1024
        assert all(isinstance(x, float) for x in vec)
        
        # 确定性检验：同一输入生成相同向量
        vec2 = client.embed_query("什么是退货政策？")
        assert vec == vec2
        
        # 不同输入生成不同向量
        vec3 = client.embed_query("保修政策是什么？")
        assert vec != vec3


def test_bge_embedding_client_mock_documents_batch():
    """测试离线/Mock 模式下的批量文档 1024 维向量生成及批大小分批"""
    client = BGEEmbeddingClient(token="mock")
    assert client.is_mock is True
    
    texts = [f"测试知识点文本_{i}" for i in range(5)]
    vectors = client.embed_documents(texts, batch_size=2)
    assert len(vectors) == 5
    for vec in vectors:
        assert len(vec) == 1024
        assert isinstance(vec[0], float)
        
    # 空输入处理
    assert client.embed_documents([]) == []


@pytest.mark.asyncio
async def test_bge_embedding_client_async_methods():
    """测试异步包装接口 aembed_query 与 aembed_documents"""
    client = BGEEmbeddingClient(token="mock")
    
    vec = await client.aembed_query("异步单句向量化")
    assert len(vec) == 1024
    
    vecs = await client.aembed_documents(["异步文档1", "异步文档2"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 1024
    assert len(vecs[1]) == 1024


def test_bge_embedding_client_real_with_mocked_hf():
    """测试使用 HF InferenceClient 的真实调用逻辑（通过 mock 模拟 API 返回）"""
    mock_hf_client = MagicMock()
    # 模拟 feature_extraction 返回 (2, 1024) 的 numpy 数组
    fake_matrix = np.ones((2, 1024), dtype=np.float32) * 0.5
    mock_hf_client.feature_extraction.return_value = fake_matrix
    
    with patch("huggingface_hub.InferenceClient", return_value=mock_hf_client):
        client = BGEEmbeddingClient(token="hf_test_token_123")
        assert client.is_mock is False
        
        vectors = client.embed_documents(["文本A", "文本B"], batch_size=2)
        assert len(vectors) == 2
        assert len(vectors[0]) == 1024
        mock_hf_client.feature_extraction.assert_called_once()


def test_bge_embedding_client_retry_and_fallback_on_error():
    """测试 HF API 遇网络错误时重试并在最终失败后优雅降级至兜底 Mock 模式"""
    mock_hf_client = MagicMock()
    mock_hf_client.feature_extraction.side_effect = RuntimeError("HF API Connection Timeout")
    
    with patch("huggingface_hub.InferenceClient", return_value=mock_hf_client):
        client = BGEEmbeddingClient(token="hf_test_token_123", max_retries=2, retry_delay=0.01)
        # 不应抛出异常，而是优雅兜底返回 1024 维向量
        vec = client.embed_query("网络抖动测试")
        assert len(vec) == 1024
        assert mock_hf_client.feature_extraction.call_count == 2


@pytest.fixture
def temp_milvus_path(tmp_path):
    """为每个单测生成独立的临时 Milvus-Lite db 路径，并在测试后安全清理"""
    db_file = tmp_path / "milvus_test" / "astro_test.db"
    yield str(db_file)
    # fixture teardown
    from milvus_lite.server_manager import server_manager_instance
    try:
        server_manager_instance.release_server(str(db_file))
    except Exception:
        pass
    parent_dir = db_file.parent
    if parent_dir.exists():
        try:
            shutil.rmtree(parent_dir, ignore_errors=True)
        except Exception:
            pass


def test_milvus_knowledge_store_lifecycle_and_init(temp_milvus_path):
    """测试 Milvus-Lite 集合初始化、幂等性与关闭"""
    store = MilvusKnowledgeStore(uri=temp_milvus_path)
    try:
        # 初始集合创建
        store.init_collection(collection_name="knowledge", drop_existing=False)
        assert store.count("knowledge") == 0
        
        # 重复调用（drop_existing=False）应保持幂等
        store.init_collection(collection_name="knowledge", drop_existing=False)
        assert store.count("knowledge") == 0
        
        # 强制删除并重建
        store.init_collection(collection_name="knowledge", drop_existing=True)
        assert store.count("knowledge") == 0
    finally:
        store.close()


def test_milvus_knowledge_store_upsert_and_overwrite(temp_milvus_path):
    """测试按 ID 的 Upsert、幂等覆写及 count 统计"""
    store = MilvusKnowledgeStore(uri=temp_milvus_path)
    try:
        store.init_collection("knowledge")
        
        records = [
            {
                "id": 1001,
                "vector": [0.01] * 1024,
                "category": "物流规则",
                "questions": "什么时候发货？",
                "answer": "下单后48小时内发货。",
                "chunk_text": "全场商品48小时内顺丰发货。",
                "content_type": "faq",
                "is_key_clause": False,
            },
            {
                "id": 1002,
                "vector": [-0.01] * 1024,
                "category": "售后政策",
                "questions": "支持退货吗？",
                "answer": "支持7天无理由退货。",
                "chunk_text": "签收7天内支持无理由退换货。",
                "content_type": "faq",
                "is_key_clause": True,
            },
        ]
        
        # 批量写入
        inserted = store.upsert(records, collection_name="knowledge")
        assert inserted == 2
        assert store.count("knowledge") == 2
        
        # 幂等覆写同一 ID (1001)
        overwrite_records = [
            {
                "id": 1001,
                "vector": [0.02] * 1024,
                "category": "物流规则",
                "questions": "什么时候发货？",
                "answer": "当日16点前付款当天发货。",
                "chunk_text": "当日16点前付款当天发货，次日达。",
                "content_type": "faq",
                "is_key_clause": True,
            }
        ]
        updated = store.upsert(overwrite_records, collection_name="knowledge")
        assert updated == 1
        # 总数仍应为 2 条（覆写未重复插入）
        assert store.count("knowledge") == 2
    finally:
        store.close()


def test_milvus_knowledge_store_search_cosine_and_filter(temp_milvus_path):
    """测试基于 COSINE 相似度的 Top-K 向量检索、min_score 阈值过滤及表达式过滤"""
    store = MilvusKnowledgeStore(uri=temp_milvus_path)
    try:
        store.init_collection("knowledge")
        
        # 准备两组具有明显方向差异的正交/反向向量
        vec_shipping = [0.0] * 1024
        vec_shipping[0] = 1.0  # [1, 0, 0, ...]
        
        vec_aftersale = [0.0] * 1024
        vec_aftersale[1] = 1.0  # [0, 1, 0, ...]
        
        vec_unrelated = [0.0] * 1024
        vec_unrelated[0] = -1.0 # [-1, 0, 0, ...] 与 shipping 完全反向 (cosine = -1.0)
        
        records = [
            {
                "id": 2001,
                "vector": vec_shipping,
                "category": "物流规则",
                "questions": "发货时间",
                "answer": "48小时发货",
                "chunk_text": "48小时内发货",
                "content_type": "faq",
                "is_key_clause": False,
            },
            {
                "id": 2002,
                "vector": vec_aftersale,
                "category": "售后规则",
                "questions": "退货时间",
                "answer": "7天退货",
                "chunk_text": "7天内退货",
                "content_type": "faq",
                "is_key_clause": True,
            },
            {
                "id": 2003,
                "vector": vec_unrelated,
                "category": "其他规则",
                "questions": "其他事项",
                "answer": "其他说明",
                "chunk_text": "反向向量测试",
                "content_type": "note",
                "is_key_clause": False,
            }
        ]
        store.upsert(records, collection_name="knowledge")
        
        # 1. 检索与 shipping 相同的向量
        query_vec = list(vec_shipping)
        hits = store.search(query_vec, top_k=2, min_score=0.0)
        assert len(hits) >= 1
        assert hits[0]["id"] == 2001
        assert hits[0]["distance"] > 0.99
        assert hits[0]["category"] == "物流规则"
        assert hits[0]["chunk_text"] == "48小时内发货"
        
        # 2. min_score 过滤（vec_shipping 与 vec_unrelated 相似度为 -1.0，与 vec_aftersale 为 0.0）
        # min_score = 0.5 只应该保留 id 2001
        strict_hits = store.search(query_vec, top_k=5, min_score=0.5)
        assert len(strict_hits) == 1
        assert strict_hits[0]["id"] == 2001
        
        # 3. 带 filter 检索 (只查询 is_key_clause == true)
        filtered_hits = store.search(query_vec, top_k=5, min_score=-1.0, filter="is_key_clause == true")
        assert len(filtered_hits) == 1
        assert filtered_hits[0]["id"] == 2002
    finally:
        store.close()


def test_milvus_knowledge_store_context_manager(temp_milvus_path):
    """测试 MilvusKnowledgeStore 支持上下文管理器安全关闭"""
    with MilvusKnowledgeStore(uri=temp_milvus_path) as store:
        store.init_collection("knowledge")
        assert store.count("knowledge") == 0
    # 退出 with 后 client 已关闭
    assert store.client is None
