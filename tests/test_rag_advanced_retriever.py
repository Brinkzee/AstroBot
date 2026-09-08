"""进阶检索管道串联与 query_faq 工具适配单元测试。

测试覆盖：
1. retrieve_with_strategy(query, strategy="hybrid_rerank") 完整流水线：
   Query 改写 -> Dense 50 + BM25 50 -> RRF 融合 -> BGE 重排 Top-10 -> 首尾重排 -> citations 快照生成；
2. 四种检索策略切流：vector_only, bm25_only, hybrid, hybrid_rerank，统一返回 AdvancedRetrievalResult 结构；
3. category_filter 品类过滤条件在各策略中的生效验证；
4. retrieve 与 retrieve_faq_text 向后兼容方法以及空值/无命中边界处理；
5. query_faq.ainvoke({"keyword": "..."}) 工具调用契约 100% 保持向后兼容。
"""

from dataclasses import is_dataclass
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models import FAQ
from app.services.rag.advanced_retriever import (
    AdvancedKnowledgeRetriever,
    AdvancedRetrievalResult,
)
from app.services.rag.query_processor import QueryUnderstandingResult
from app.tools.business_tools import get_retriever, query_faq


@pytest.fixture
def sample_candidates() -> List[Dict[str, Any]]:
    """构造 10 条模拟候选知识块。"""
    candidates = []
    for i in range(1, 11):
        candidates.append({
            "id": i,
            "chunk_id": i,
            "category": "智能手表",
            "section_path": f"售后政策 > 章节{i}",
            "question": f"智能手表问题{i}",
            "questions": f"智能手表问题{i}\n智能手表别名{i}",
            "answer": f"智能手表解答内容{i}",
            "distance": round(1.0 - i * 0.05, 3),
        })
    return candidates


@pytest.mark.asyncio
async def test_hybrid_rerank_full_pipeline(sample_candidates):
    """验证 hybrid_rerank 完整流水线：
    Query 改写 -> Dense 50 + BM25 50 -> RRF 融合 -> BGE 重排 Top-10 -> 首尾重排 -> citations 快照。
    """
    mock_query_processor = MagicMock()
    mock_qu = QueryUnderstandingResult(
        original_query="那个pro x99的手表用着不顺心能退吗",
        standard_query="星光PRO-X99智能手表退换货政策与流程",
        expanded_keywords=["退换货", "退货政策", "退款流程"],
        bm25_query="星光PRO-X99智能手表退换货政策与流程 退换货 退货政策 退款流程",
    )
    mock_query_processor.aprocess = AsyncMock(return_value=mock_qu)

    mock_embedding_client = MagicMock()
    fake_vector = [0.1] * 1024
    mock_embedding_client.aembed_query = AsyncMock(return_value=fake_vector)

    mock_store = MagicMock()
    mock_store.hybrid_search = MagicMock(return_value=sample_candidates)

    mock_reranker = MagicMock()
    # 模拟重排输出：按 rerank_score 降序排列的 10 条文档
    reranked = []
    for idx, c in enumerate(sample_candidates, 1):
        doc = dict(c)
        doc["rerank_score"] = round(1.0 - idx * 0.08, 4)
        reranked.append(doc)
    mock_reranker.rerank = AsyncMock(return_value=reranked)

    retriever = AdvancedKnowledgeRetriever(
        store=mock_store,
        embedding_client=mock_embedding_client,
        query_processor=mock_query_processor,
        reranker_client=mock_reranker,
    )

    result = await retriever.retrieve_with_strategy(
        query="那个pro x99的手表用着不顺心能退吗",
        strategy="hybrid_rerank",
        category_filter="智能手表",
        top_k=10,
        top_k_per_route=50,
    )

    # 1. 验证返回值类型及基本结构
    assert isinstance(result, AdvancedRetrievalResult)
    assert is_dataclass(result)
    assert result.strategy == "hybrid_rerank"
    assert result.query_understanding == mock_qu
    assert len(result.docs) == 10
    assert len(result.citations) == 10

    # 2. 验证 Query 理解调用
    mock_query_processor.aprocess.assert_awaited_once_with(
        "那个pro x99的手表用着不顺心能退吗"
    )

    # 3. 验证 Dense 向量生成使用了 standard_query
    mock_embedding_client.aembed_query.assert_awaited_once_with(
        "星光PRO-X99智能手表退换货政策与流程"
    )

    # 4. 验证 Milvus hybrid_search 接收 dense_vector, bm25_query 与品类过滤
    mock_store.hybrid_search.assert_called_once_with(
        dense_vector=fake_vector,
        bm25_text="星光PRO-X99智能手表退换货政策与流程 退换货 退货政策 退款流程",
        category_filter="智能手表",
        top_k_per_route=50,
        limit=50,
        collection_name="knowledge",
    )

    # 5. 验证重排客户端调用入参
    mock_reranker.rerank.assert_awaited_once_with(
        query="星光PRO-X99智能手表退换货政策与流程",
        candidates=sample_candidates,
        top_k=10,
    )

    # 6. 验证 Lost-in-the-Middle 首尾重排规律：
    # 原始降序排列为 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # 首尾放置后应为 [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
    expected_ids = [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
    actual_ids = [d["id"] for d in result.docs]
    assert actual_ids == expected_ids

    # 7. 验证 citations 引用快照元数据与编号 1..10 严格对应
    for i, cite in enumerate(result.citations):
        assert cite["n"] == i + 1
        assert cite["chunk_id"] == expected_ids[i]
        assert "section_path" in cite
        assert "question" in cite
        assert "answer" in cite


@pytest.mark.asyncio
async def test_strategy_switching_unified_structure(sample_candidates):
    """验证四种策略切流：vector_only, bm25_only, hybrid, hybrid_rerank 返回结构统一。"""
    mock_qu = QueryUnderstandingResult(
        original_query="手表退换",
        standard_query="智能手表退货规则",
        expanded_keywords=["退换政策"],
        bm25_query="智能手表退货规则 退换政策",
    )
    mock_query_processor = MagicMock()
    mock_query_processor.aprocess = AsyncMock(return_value=mock_qu)

    mock_embedding = MagicMock()
    mock_embedding.aembed_query = AsyncMock(return_value=[0.2] * 1024)

    mock_store = MagicMock()
    mock_store.search_dense = MagicMock(return_value=sample_candidates[:5])
    mock_store.search_bm25 = MagicMock(return_value=sample_candidates[:5])
    mock_store.hybrid_search = MagicMock(return_value=sample_candidates[:5])

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=sample_candidates[:5])

    retriever = AdvancedKnowledgeRetriever(
        store=mock_store,
        embedding_client=mock_embedding,
        query_processor=mock_query_processor,
        reranker_client=mock_reranker,
    )

    strategies = ["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]
    for strat in strategies:
        res = await retriever.retrieve_with_strategy(
            query="手表退换", strategy=strat, top_k=5
        )
        assert isinstance(res, AdvancedRetrievalResult)
        assert res.strategy == strat
        assert res.query_understanding == mock_qu
        assert len(res.docs) <= 5
        assert len(res.citations) == len(res.docs)
        for idx, cite in enumerate(res.citations, 1):
            assert cite["n"] == idx

    # 验证非法策略抛出 ValueError
    with pytest.raises(ValueError, match="不支持的检索策略"):
        await retriever.retrieve_with_strategy("手表退换", strategy="unsupported_strat")


@pytest.mark.asyncio
async def test_category_filter_applied_to_all_strategies(sample_candidates):
    """验证 category_filter 品类标量过滤在各检索策略中正确下发。"""
    mock_qu = QueryUnderstandingResult(
        original_query="充电说明",
        standard_query="智能手表充电规格",
        expanded_keywords=[],
        bm25_query="智能手表充电规格",
    )
    mock_qp = MagicMock()
    mock_qp.aprocess = AsyncMock(return_value=mock_qu)

    mock_emb = MagicMock()
    mock_emb.aembed_query = AsyncMock(return_value=[0.3] * 1024)

    mock_store = MagicMock()
    mock_store.search_dense = MagicMock(return_value=sample_candidates[:3])
    mock_store.search_bm25 = MagicMock(return_value=sample_candidates[:3])
    mock_store.hybrid_search = MagicMock(return_value=sample_candidates[:3])

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=sample_candidates[:3])

    retriever = AdvancedKnowledgeRetriever(
        store=mock_store,
        embedding_client=mock_emb,
        query_processor=mock_qp,
        reranker_client=mock_reranker,
    )

    # 1. vector_only
    await retriever.retrieve_with_strategy(
        "充电说明", strategy="vector_only", category_filter="数码配件"
    )
    mock_store.search_dense.assert_called_with(
        query_vector=[0.3] * 1024,
        top_k=10,
        category_filter="数码配件",
        collection_name="knowledge",
    )

    # 2. bm25_only
    await retriever.retrieve_with_strategy(
        "充电说明", strategy="bm25_only", category_filter="数码配件"
    )
    mock_store.search_bm25.assert_called_with(
        query_text="智能手表充电规格",
        top_k=10,
        category_filter="数码配件",
        collection_name="knowledge",
    )

    # 3. hybrid
    await retriever.retrieve_with_strategy(
        "充电说明", strategy="hybrid", category_filter="数码配件", top_k=5
    )
    mock_store.hybrid_search.assert_called_with(
        dense_vector=[0.3] * 1024,
        bm25_text="智能手表充电规格",
        category_filter="数码配件",
        top_k_per_route=50,
        limit=5,
        collection_name="knowledge",
    )

    # 4. hybrid_rerank
    await retriever.retrieve_with_strategy(
        "充电说明", strategy="hybrid_rerank", category_filter="数码配件", top_k=5, top_k_per_route=20
    )
    mock_store.hybrid_search.assert_called_with(
        dense_vector=[0.3] * 1024,
        bm25_text="智能手表充电规格",
        category_filter="数码配件",
        top_k_per_route=20,
        limit=20,
        collection_name="knowledge",
    )


@pytest.mark.asyncio
async def test_empty_query_and_empty_candidates():
    """验证空查询与底层检索无候选时的优雅边界处理。"""
    mock_store = MagicMock()
    mock_store.hybrid_search = MagicMock(return_value=[])

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=[])

    mock_qp = MagicMock()
    mock_qp.aprocess = AsyncMock(
        return_value=QueryUnderstandingResult(
            original_query="未知内容",
            standard_query="未知内容",
            expanded_keywords=[],
            bm25_query="未知内容",
        )
    )
    mock_emb = MagicMock()
    mock_emb.aembed_query = AsyncMock(return_value=[0.1] * 1024)

    retriever = AdvancedKnowledgeRetriever(
        store=mock_store,
        embedding_client=mock_emb,
        query_processor=mock_qp,
        reranker_client=mock_reranker,
    )

    # 1. 空查询
    empty_res = await retriever.retrieve_with_strategy("")
    assert empty_res.docs == []
    assert empty_res.citations == []

    # 2. 空格查询
    space_res = await retriever.retrieve_with_strategy("   ")
    assert space_res.docs == []
    assert space_res.citations == []

    # 3. 候选库为空
    no_cand_res = await retriever.retrieve_with_strategy("未知内容")
    assert no_cand_res.docs == []
    assert no_cand_res.citations == []
    # 验证候选为空时不浪费计算资源去调用重排
    mock_reranker.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_and_retrieve_faq_text_compatibility(sample_candidates):
    """验证 retrieve 与 retrieve_faq_text 向后兼容方法及格式化输出。"""
    mock_store = MagicMock()
    mock_store.hybrid_search = MagicMock(return_value=sample_candidates[:3])

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=sample_candidates[:3])

    mock_qp = MagicMock()
    mock_qp.aprocess = AsyncMock(
        return_value=QueryUnderstandingResult(
            original_query="退货",
            standard_query="退货政策",
            expanded_keywords=["退货", "退款"],
            bm25_query="退货政策 退货 退款",
        )
    )
    mock_emb = MagicMock()
    mock_emb.aembed_query = AsyncMock(return_value=[0.1] * 1024)

    retriever = AdvancedKnowledgeRetriever(
        store=mock_store,
        embedding_client=mock_emb,
        query_processor=mock_qp,
        reranker_client=mock_reranker,
    )

    # 1. 验证 retrieve() 返回 List[Dict] 且长度为 top_k
    hits = await retriever.retrieve("退货", top_k=2)
    assert isinstance(hits, list)
    assert len(hits) == 2
    assert "answer" in hits[0]

    # 2. 验证 retrieve_faq_text() 返回规范问答文本
    faq_text = await retriever.retrieve_faq_text("退货", top_k=2)
    assert "1. 问：" in faq_text
    assert "答：" in faq_text
    assert "智能手表解答内容" in faq_text

    # 3. 验证 format_faq_hits 空输入兜底
    empty_faq = retriever.format_faq_hits([], keyword="神秘商品")
    assert "未找到与【神秘商品】相关的常见问题解答。" in empty_faq


@pytest.mark.asyncio
async def test_query_faq_tool_contract_with_advanced_retriever():
    """验证 query_faq 工具升级为 AdvancedKnowledgeRetriever 后的端到端兼容性。"""
    mock_retriever = MagicMock(spec=AdvancedKnowledgeRetriever)
    mock_retriever.retrieve = AsyncMock(
        return_value=[
            {
                "id": 101,
                "questions": "如何申请保修？",
                "answer": "可在订单中心申请售后，专人跟进。",
                "section_path": "售后服务 > 保修条例",
            }
        ]
    )
    mock_retriever.format_faq_hits = MagicMock(
        return_value="1. 问：如何申请保修？\n   答：可在订单中心申请售后，专人跟进。"
    )

    with patch("app.tools.business_tools.get_retriever", return_value=mock_retriever):
        # 1. 验证 ainvoke 契约与输出
        res = await query_faq.ainvoke({"keyword": "保修"})
        assert "如何申请保修？" in res
        assert "可在订单中心申请售后" in res
        mock_retriever.retrieve.assert_awaited_once_with("保修", top_k=3)

    # 2. 验证 query_faq 工具元数据与 Docstring 完整无误
    assert query_faq.name == "query_faq"
    assert "常见问题解答" in query_faq.description
    assert "keyword" in str(query_faq.args_schema.model_fields.keys())


@pytest.mark.asyncio
async def test_query_faq_fallback_to_sql_when_retriever_empty():
    """验证底层 AdvancedKnowledgeRetriever 无命中时，query_faq 优雅回退到 MySQL FAQ 表。"""
    mock_retriever = MagicMock(spec=AdvancedKnowledgeRetriever)
    mock_retriever.retrieve = AsyncMock(return_value=[])

    mock_faq = FAQ(
        id=77,
        question="发票抬头支持哪些？",
        answer="支持个人与企业增值税普通发票及专用发票。",
        category="发票",
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
        res = await query_faq.ainvoke({"keyword": "发票抬头"})
        assert "发票抬头支持哪些？" in res
        assert "支持个人与企业增值税普通发票" in res
        mock_retriever.retrieve.assert_awaited_once_with("发票抬头", top_k=3)
        mock_session.execute.assert_awaited_once()
