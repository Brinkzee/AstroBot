import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from main import app
from app.db.session import get_db
from app.models.knowledge import KnowledgeChunk

client = TestClient(app)


@pytest.fixture
def mock_db_session():
    mock_session = AsyncMock()
    return mock_session


def test_get_kb_page_html():
    """测试访问 http://localhost:8000/kb 能成功返回 HTML 页面"""
    response = client.get("/kb")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "知识库" in response.text


def test_get_kb_stats():
    """测试获取知识库统计指标接口 GET /api/kb/stats"""
    mock_session = AsyncMock()
    
    # 模拟返回 count
    mock_total = MagicMock()
    mock_total.scalar.return_value = 10
    mock_done = MagicMock()
    mock_done.scalar.return_value = 9
    mock_pending = MagicMock()
    mock_pending.scalar.return_value = 1
    mock_failed = MagicMock()
    mock_failed.scalar.return_value = 0

    mock_session.execute = AsyncMock(side_effect=[mock_total, mock_done, mock_pending, mock_failed])

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with patch("app.api.routes.MilvusKnowledgeStore") as MockStore:
            instance = MockStore.return_value
            instance.count.return_value = 9
            instance.close = MagicMock()

            response = client.get("/api/kb/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["total_chunks"] == 10
            assert data["done_chunks"] == 9
            assert data["pending_chunks"] == 1
            assert data["failed_chunks"] == 0
            assert data["milvus_count"] == 9
            assert data["is_aligned"] is True
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_get_kb_materials():
    """测试获取材料文档清单接口 GET /api/kb/materials"""
    response = client.get("/api/kb/materials")
    assert response.status_code == 200
    materials = response.json()
    assert isinstance(materials, list)
    assert len(materials) >= 1
    filenames = [m["filename"] for m in materials]
    assert any("配送与运费" in f or "退换货" in f or ".md" in f for f in filenames)


def test_get_kb_chunks():
    """测试分页与条件筛选知识切块接口 GET /api/kb/chunks"""
    mock_session = AsyncMock()
    
    # 模拟总数 count
    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 1

    # 模拟记录列表
    mock_chunk = MagicMock(spec=KnowledgeChunk)
    mock_chunk.id = 1
    mock_chunk.category = "运费"
    mock_chunk.questions = "包邮吗？"
    mock_chunk.answer = "满88包邮"
    mock_chunk.section_path = "运费 > 规范"
    mock_chunk.content_type = "policy"
    mock_chunk.is_key_clause = True
    mock_chunk.vectorize_status = "done"
    mock_chunk.vector_id = "1"
    mock_chunk.created_at = MagicMock()
    mock_chunk.created_at.strftime.return_value = "2026-09-08 12:00:00"

    mock_list_res = MagicMock()
    mock_list_res.scalars.return_value.all.return_value = [mock_chunk]

    mock_session.execute = AsyncMock(side_effect=[mock_count_res, mock_list_res])

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        response = client.get("/api/kb/chunks?page=1&page_size=10&status=done&category=运费")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == 1
        assert data["items"][0]["questions"] == "包邮吗？"
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_manual_chunk_create_with_sync():
    """测试手工录入知识块并同步写入向量库 POST /api/kb/chunks/manual (sync_vector=True)"""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with patch("app.api.routes.KnowledgeDualWriter") as MockWriter:
            writer_instance = MockWriter.return_value
            mock_saved_chunk = MagicMock(spec=KnowledgeChunk)
            mock_saved_chunk.id = 88
            mock_saved_chunk.category = "支付方式"
            mock_saved_chunk.questions = "支持信用卡吗？"
            mock_saved_chunk.answer = "支持信用卡和微信支付"
            mock_saved_chunk.section_path = "支付 > 常见问题"
            mock_saved_chunk.content_type = "faq"
            mock_saved_chunk.is_key_clause = False
            mock_saved_chunk.vectorize_status = "done"
            mock_saved_chunk.vector_id = "88"
            mock_saved_chunk.created_at = MagicMock()
            mock_saved_chunk.created_at.strftime.return_value = "2026-09-08 12:00:00"

            writer_instance.write_chunks = AsyncMock(return_value=[mock_saved_chunk])
            writer_instance.close = MagicMock()

            payload = {
                "category": "支付方式",
                "questions": "支持信用卡吗？\n可以用信用卡结账吗？",
                "answer": "支持信用卡和微信支付",
                "section_path": "支付 > 常见问题",
                "content_type": "faq",
                "is_key_clause": False,
                "sync_vector": True,
            }
            response = client.post("/api/kb/chunks/manual", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == 88
            assert data["vectorize_status"] == "done"
            assert data["vector_id"] == "88"
            writer_instance.write_chunks.assert_awaited_once()
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_manual_chunk_create_without_sync():
    """测试手工录入知识块暂不向量化 POST /api/kb/chunks/manual (sync_vector=False)"""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    async def mock_refresh(instance):
        instance.id = 99
        instance.created_at = MagicMock()
        instance.created_at.strftime.return_value = "2026-09-08 12:00:00"

    mock_session.refresh = AsyncMock(side_effect=mock_refresh)

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        payload = {
            "category": "会员积分",
            "questions": "积分如何抵扣？",
            "answer": "100积分抵扣1元现金",
            "section_path": "会员 > 积分规则",
            "content_type": "policy",
            "is_key_clause": False,
            "sync_vector": False,
        }
        response = client.post("/api/kb/chunks/manual", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 99
        assert data["vectorize_status"] == "pending"
        assert data["vector_id"] is None
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_repair_pending_chunks_api():
    """测试一键修复补偿 Pending 知识块 POST /api/kb/repair-pending"""
    mock_session = AsyncMock()

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with patch("app.api.routes.KnowledgeDualWriter") as MockWriter:
            writer_instance = MockWriter.return_value
            writer_instance.repair_pending_chunks = AsyncMock(return_value=3)
            writer_instance.close = MagicMock()

            response = client.post("/api/kb/repair-pending")
            assert response.status_code == 200
            data = response.json()
            assert data["repaired_count"] == 3
            assert data["success"] is True
            writer_instance.repair_pending_chunks.assert_awaited_once()
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_kb_search_api():
    """测试知识库语义检索自测沙盒接口 POST /api/kb/search"""
    with patch("app.api.routes.KnowledgeRetriever") as MockRetriever:
        retriever_instance = MockRetriever.return_value
        retriever_instance.retrieve = AsyncMock(return_value=[
            {
                "id": 12,
                "distance": 0.88,
                "category": "物流服务",
                "questions": "发什么快递？",
                "answer": "默认顺丰速运或中通快递随机派送。",
                "section_path": "物流 > 合作快递",
                "content_type": "policy",
                "is_key_clause": True,
            }
        ])
        retriever_instance.format_faq_hits.return_value = "1. 问：发什么快递？\n   答：默认顺丰速运或中通快递随机派送。"

        payload = {
            "query": "你们用什么快递送货",
            "top_k": 3,
            "min_score": 0.5,
        }
        response = client.post("/api/kb/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "你们用什么快递送货"
        assert data["total_hits"] == 1
        assert data["hits"][0]["id"] == 12
        assert data["hits"][0]["distance"] == 0.88
        assert "顺丰速运" in data["formatted_preview"]
        assert "latency_ms" in data
