import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.knowledge_node import knowledge_retrieval_node, knowledge_fallback_node
from app.services.workflow.nodes.gate import confidence_gate
from app.services.rag.advanced_retriever import AdvancedRetrievalResult

@pytest.mark.asyncio
async def test_knowledge_retrieval_high_score_passed():
    mock_retriever = MagicMock()
    mock_hit = MagicMock()
    mock_hit.score = 0.88
    mock_hit.to_dict.return_value = {"chunk_id": 1, "text": "7天无理由退货", "score": 0.88}
    mock_res = MagicMock()
    mock_res.hits = [mock_hit]
    mock_res.citations = [{"index": 1, "source": "退款政策"}]
    mock_retriever.retrieve_with_strategy = AsyncMock(return_value=mock_res)

    state = create_initial_state(1, "怎么退货")
    updated = await knowledge_retrieval_node(state, retriever=mock_retriever)
    
    assert len(updated["retrieved_docs"]) == 1
    assert updated["retrieved_docs"][0]["score"] == 0.88
    # 闸门判定应该放行
    assert confidence_gate({**state, **updated}) == "pass"

@pytest.mark.asyncio
async def test_knowledge_retrieval_low_score_triggers_fallback():
    mock_retriever = MagicMock()
    mock_hit = MagicMock()
    mock_hit.score = 0.20  # 低于 0.35 阈值
    mock_hit.to_dict.return_value = {"chunk_id": 2, "text": "不相关内容", "score": 0.20}
    mock_res = MagicMock()
    mock_res.hits = [mock_hit]
    mock_res.citations = []
    mock_retriever.retrieve_with_strategy = AsyncMock(return_value=mock_res)

    state = create_initial_state(1, "太空飞船如何退货")
    updated = await knowledge_retrieval_node(state, retriever=mock_retriever)
    
    # 闸门判定应走兜底
    assert confidence_gate({**state, **updated}) == "fallback"

    fallback_updated = await knowledge_fallback_node({**state, **updated}, db=None)
    assert "暂未收录" in fallback_updated["response_text"]
    assert fallback_updated["confidence_passed"] is False
    assert fallback_updated["suggested_actions"] == ["transfer_agent", "create_ticket"]
    assert fallback_updated["status"] == "fallback"

@pytest.mark.asyncio
async def test_knowledge_gate_empty_docs_and_edge_threshold():
    # Empty docs -> fallback
    assert confidence_gate({"retrieved_docs": []}) == "fallback"
    assert confidence_gate({}) == "fallback"

    # High confidence -> pass
    assert confidence_gate({"retrieved_docs": [{"score": 0.8}]}) == "pass"
    # Low confidence -> fallback
    assert confidence_gate({"retrieved_docs": [{"score": 0.1}]}) == "fallback"

@pytest.mark.asyncio
async def test_knowledge_fallback_with_db():
    mock_db = MagicMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    state = create_initial_state(42, "冷门问题")
    res = await knowledge_fallback_node(state, db=mock_db)

    assert res["status"] == "fallback"
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_knowledge_retrieval_reads_advanced_retrieval_result_docs():
    """真实 AdvancedRetrievalResult 只有 docs 与 citations，且分数在 rerank_score 中"""
    mock_retriever = MagicMock()
    real_res = AdvancedRetrievalResult(
        docs=[
            {
                "id": 101,
                "chunk_text": "7天无理由退货政策：不影响二次销售可申请退货",
                "section_path": "售后政策 > 退货政策",
                "rerank_score": 0.85,
                "distance": 0.02,
                "entity": {"vector": [0.1, 0.2, 0.3]},
            }
        ],
        citations=[{"n": 1, "section_path": "售后政策 > 退货政策"}],
        strategy="hybrid_rerank",
    )
    mock_retriever.retrieve_with_strategy = AsyncMock(return_value=real_res)

    state = create_initial_state(1, "怎么退货")
    updated = await knowledge_retrieval_node(state, retriever=mock_retriever)

    assert len(updated["retrieved_docs"]) == 1
    doc = updated["retrieved_docs"][0]
    assert doc["score"] == 0.85
    assert "7天无理由退货政策" in doc["text"]
    assert "7天无理由退货政策" in doc["content"]
    assert "entity" not in doc
    assert confidence_gate({**state, **updated}) == "pass"


@pytest.mark.asyncio
async def test_knowledge_fallback_extracts_rerank_score_and_chunk_text():
    """兜底落池应兼容 chunk_text 与 rerank_score 字段"""
    mock_db = MagicMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    state = create_initial_state(43, "偏远地区运费")
    state["retrieved_docs"] = [
        {
            "chunk_text": "新疆西藏等偏远地区运费另计",
            "rerank_score": 0.18,
            "section_path": "配送范围",
        }
    ]
    res = await knowledge_fallback_node(state, db=mock_db)

    assert res["status"] == "fallback"
    entry = mock_db.add.call_args[0][0]
    assert entry.retrieved_chunks is not None
    assert entry.retrieved_chunks[0]["score"] == 0.18
    assert "新疆西藏" in entry.retrieved_chunks[0]["text"]
