"""Task 6 单测套件：在线密集向量语义检索器 KnowledgeRetriever 与 query_faq 工具升级。

覆盖内容：
1. KnowledgeRetriever 初始化与默认参数 (min_score=0.35)；
2. retrieve 空输入过滤与 query 异步向量化；
3. COSINE Top-K 匹配、打分过滤与排序；
4. retrieve_faq_text 规范化多行/单行问答文本格式化及未命中兜底；
5. 基于真实临时 Milvus-Lite 数据库的端到端检索与相似度测试；
6. query_faq 的 LangChain @tool 契约不变性（函数名、参数名、类型、Docstring）；
7. query_faq 优先密集向量检索；
8. query_faq 在向量未命中/Milvus 未灌库时优雅回退 SQL FAQ 表。
"""

import os
import shutil
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import FAQ
from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.milvus_client import MilvusKnowledgeStore
from app.services.rag.retriever import KnowledgeRetriever
from app.tools.business_tools import query_faq


@pytest.fixture
def temp_milvus_path(tmp_path):
    """为检索器测试提供独立的临时 Milvus-Lite db 路径，测试后释放"""
    db_file = tmp_path / "milvus_retriever_test" / "astro_retriever.db"
    yield str(db_file)
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


def test_knowledge_retriever_init_defaults():
    """测试 KnowledgeRetriever 的默认依赖与参数初始化"""
    retriever = KnowledgeRetriever()
    assert retriever.min_score == 0.35
    assert retriever.collection_name == "knowledge"
    assert retriever.store is not None
    assert retriever.embedding_client is not None


@pytest.mark.asyncio
async def test_retrieve_empty_query():
    """测试空输入或纯空白字符过滤，直接返回空列表且不调用 embedding 或 store"""
    mock_store = MagicMock()
    mock_embedding = MagicMock()
    retriever = KnowledgeRetriever(store=mock_store, embedding_client=mock_embedding)

    assert await retriever.retrieve("") == []
    assert await retriever.retrieve("   ") == []
    assert await retriever.retrieve(None) == []

    mock_embedding.aembed_query.assert_not_called()
    mock_store.search.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_vector_search_and_score_filtering():
    """测试 retrieve 向量化查询与 COSINE 相似度阈值过滤"""
    mock_embedding = MagicMock()
    mock_embedding.aembed_query = AsyncMock(return_value=[0.1] * 1024)

    mock_store = MagicMock()
    # 模拟 search 返回两条记录，一条高于 min_score，一条低于
    mock_store.search.return_value = [
        {
            "id": 1,
            "distance": 0.85,
            "category": "售后",
            "questions": "退货政策说明\n怎么退货？",
            "answer": "支持7天无理由退换货。",
        },
        {
            "id": 2,
            "distance": 0.20,  # 低于默认 0.35
            "category": "物流",
            "questions": "发货时间",
            "answer": "48小时内发货。",
        },
    ]

    retriever = KnowledgeRetriever(
        store=mock_store,
        embedding_client=mock_embedding,
        min_score=0.35,
    )

    hits = await retriever.retrieve("我要退货", top_k=2)

    mock_embedding.aembed_query.assert_awaited_once_with("我要退货")
    mock_store.search.assert_called_once()
    assert len(hits) == 1
    assert hits[0]["id"] == 1
    assert hits[0]["distance"] == 0.85


@pytest.mark.asyncio
async def test_retrieve_faq_text_formatting_multiline():
    """测试 retrieve_faq_text 提取 questions 第一行并规范化输出"""
    mock_retriever = KnowledgeRetriever(
        store=MagicMock(),
        embedding_client=MagicMock(),
    )
    # Mock retrieve 方法直接返回标准命中列表
    mock_hits = [
        {
            "id": 101,
            "distance": 0.92,
            "questions": "如何申请退换货？\n退货运费谁出？\n7天无理由退货规则",
            "answer": "签收7天内联系在线客服，质量问题商家包邮退换。",
        },
        {
            "id": 102,
            "distance": 0.81,
            "questions": "退款多久到账？",
            "answer": "退货验收合格后，款项将在1-3个工作日原路退回。",
        },
    ]
    mock_retriever.retrieve = AsyncMock(return_value=mock_hits)

    formatted = await mock_retriever.retrieve_faq_text("退款规则", top_k=2)
    expected = (
        "1. 问：如何申请退换货？\n"
        "   答：签收7天内联系在线客服，质量问题商家包邮退换。\n\n"
        "2. 问：退款多久到账？\n"
        "   答：退货验收合格后，款项将在1-3个工作日原路退回。"
    )
    assert formatted == expected


@pytest.mark.asyncio
async def test_retrieve_faq_text_not_found():
    """测试未命中任何知识时返回标准的友好提示"""
    mock_retriever = KnowledgeRetriever(
        store=MagicMock(),
        embedding_client=MagicMock(),
    )
    mock_retriever.retrieve = AsyncMock(return_value=[])

    res = await mock_retriever.retrieve_faq_text("火星探索任务")
    assert res == "未找到与【火星探索任务】相关的常见问题解答。"

    # 空关键词测试
    empty_res = await mock_retriever.retrieve_faq_text("   ")
    assert empty_res == "未找到与【   】相关的常见问题解答。"


@pytest.mark.asyncio
async def test_milvus_integration_e2e_with_temp_db(temp_milvus_path):
    """基于真实临时 Milvus-Lite 验证端到端写入、向量化、COSINE 检索与文本格式化"""
    embedding_client = BGEEmbeddingClient(mock=True)
    store = MilvusKnowledgeStore(uri=temp_milvus_path)
    try:
        store.init_collection("knowledge", drop_existing=True)

        # 写入真实测试向量记录
        records = [
            {
                "id": 1,
                "vector": embedding_client.embed_query("退货退款政策说明"),
                "category": "售后",
                "questions": "退货政策说明\n支持几天无理由退货？",
                "answer": "支持7天无理由退货，商品需保持原包装。",
                "chunk_text": "分类：售后\n问题：退货政策说明\n内容：支持7天无理由退货",
                "content_type": "faq",
                "is_key_clause": True,
            },
            {
                "id": 2,
                "vector": embedding_client.embed_query("快递发货与运费标准"),
                "category": "物流",
                "questions": "运费是多少？\n包邮条件是什么？",
                "answer": "全场满88元包邮，偏远地区除外。",
                "chunk_text": "分类：物流\n问题：运费是多少\n内容：全场满88包邮",
                "content_type": "faq",
                "is_key_clause": False,
            },
        ]
        store.upsert(records, "knowledge")
        assert store.count("knowledge") == 2

        retriever = KnowledgeRetriever(
            store=store,
            embedding_client=embedding_client,
            min_score=0.35,
        )

        # 检索 "退货退款政策说明" 应该首条匹配 ID=1
        hits = await retriever.retrieve("退货退款政策说明", top_k=2)
        assert len(hits) >= 1
        assert hits[0]["id"] == 1
        assert hits[0]["distance"] > 0.9  # 相同文本嵌入相似度极高

        # 测试 retrieve_faq_text 输出格式
        faq_text = await retriever.retrieve_faq_text("退货退款政策说明", top_k=1)
        assert "1. 问：退货政策说明" in faq_text
        assert "答：支持7天无理由退货，商品需保持原包装。" in faq_text

    finally:
        store.close()


def test_query_faq_contract_immutability():
    """验证 query_faq 的 LangChain @tool 契约 100% 不变"""
    assert query_faq.name == "query_faq"
    assert "查询常见问题解答 (FAQ) 知识库" in query_faq.description
    assert "keyword" in query_faq.args
    assert query_faq.args["keyword"]["type"] == "string"


@pytest.mark.asyncio
async def test_query_faq_dense_retrieval_matched():
    """验证 query_faq 优先走向量检索并成功返回格式化 FAQ"""
    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.retrieve = AsyncMock(return_value=[
        {
            "id": 1,
            "distance": 0.88,
            "questions": "发票如何开具？",
            "answer": "可在订单详情页自助申请电子发票，1个工作日内发送到邮箱。",
        }
    ])
    mock_retriever.format_faq_hits = MagicMock(
        return_value="1. 问：发票如何开具？\n   答：可在订单详情页自助申请电子发票，1个工作日内发送到邮箱。"
    )

    with patch("app.tools.business_tools.get_retriever", return_value=mock_retriever):
        res = await query_faq.ainvoke({"keyword": "发票"})
        assert "发票如何开具？" in res
        assert "可在订单详情页自助申请电子发票" in res
        mock_retriever.retrieve.assert_awaited_once_with("发票", top_k=3)


@pytest.mark.asyncio
async def test_query_faq_fallback_to_sql_when_retriever_empty():
    """验证向量检索未命中或无数据时，query_faq 优雅回退到 MySQL FAQ 表查询"""
    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.retrieve = AsyncMock(return_value=[])

    mock_faq = FAQ(
        id=88,
        question="SQL回退问答",
        answer="这是通过数据库表查询返回的FAQ。",
        category="兜底",
    )
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_faq]
    mock_session.execute.return_value = mock_result

    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None
    mock_session_local = MagicMock(return_value=mock_ctx)

    with patch("app.tools.business_tools.get_retriever", return_value=mock_retriever), \
         patch("app.tools.business_tools.AsyncSessionLocal", mock_session_local):
        res = await query_faq.ainvoke({"keyword": "SQL回退"})
        assert "SQL回退问答" in res
        assert "这是通过数据库表查询返回的FAQ。" in res
        mock_retriever.retrieve.assert_awaited_once()
        mock_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_knowledge_retriever_retrieve_with_category_parameter():
    """测试 KnowledgeRetriever.retrieve 支持 category 和 category_filter 标量分类过滤"""
    mock_embedding = MagicMock()
    mock_embedding.aembed_query = AsyncMock(return_value=[0.1] * 1024)

    mock_store = MagicMock()
    mock_store.search.return_value = [
        {
            "id": 1,
            "distance": 0.88,
            "category": "售后服务",
            "questions": "如何申请退换货？",
            "answer": "联系客服提交申请即可。",
        }
    ]

    retriever = KnowledgeRetriever(store=mock_store, embedding_client=mock_embedding)

    # 1. 验证传入 category 参数（Web 接口常用）正常执行并不抛出异常
    hits = await retriever.retrieve(
        query="退货流程",
        top_k=5,
        min_score=0.4,
        category="售后服务",
    )
    assert len(hits) == 1
    assert hits[0]["category"] == "售后服务"
    mock_store.search.assert_called_with(
        query_vector=[0.1] * 1024,
        top_k=5,
        min_score=0.4,
        filter=None,
        category_filter="售后服务",
        collection_name="knowledge",
    )

    # 2. 验证传入 category_filter 参数也支持
    mock_store.search.reset_mock()
    hits2 = await retriever.retrieve(
        query="退货流程",
        top_k=3,
        category_filter="售后服务",
        extra_unknown_kwarg=True,
    )
    assert len(hits2) == 1
    mock_store.search.assert_called_with(
        query_vector=[0.1] * 1024,
        top_k=3,
        min_score=0.35,
        filter=None,
        category_filter="售后服务",
        collection_name="knowledge",
    )
