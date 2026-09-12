import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.knowledge_node import knowledge_retrieval_node, knowledge_fallback_node
from app.services.workflow.nodes.gate import confidence_gate

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

    # Exactly 0.35 -> pass
    assert confidence_gate({"retrieved_docs": [{"score": 0.35}]}) == "pass"
    # Just below 0.35 -> fallback
    assert confidence_gate({"retrieved_docs": [{"score": 0.349}]}) == "fallback"

@pytest.mark.asyncio
async def test_knowledge_fallback_with_db():
    mock_db = MagicMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()

    state = create_initial_state(42, "冷门问题")
    res = await knowledge_fallback_node(state, db=mock_db)

    assert res["status"] == "fallback"
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()
