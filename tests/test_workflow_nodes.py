import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.pre_nodes import (
    coreference_resolution_node,
    intent_recognition_node,
    chitchat_node,
    complaint_node,
)
from app.services.workflow.nodes.router import route_by_intent

def test_coreference_resolution_passthrough():
    state = create_initial_state(1, "我的订单到哪了")
    updated = coreference_resolution_node(state)
    assert updated["resolved_query"] == "我的订单到哪了"

@pytest.mark.asyncio
async def test_intent_recognition_json_parsing():
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "物流", "reason": "用户询问订单物流轨迹"}'
    ))
    state = create_initial_state(1, "订单1001发到哪里了")
    updated = await intent_recognition_node(state, model=mock_model)
    assert updated["intent"] == "物流"
    assert updated["intent_reason"] == "用户询问订单物流轨迹"

def test_route_by_intent_rules():
    assert route_by_intent({"intent": "闲聊"}) == "chitchat"
    assert route_by_intent({"intent": "投诉"}) == "complaint"
    assert route_by_intent({"intent": "商品咨询"}) == "knowledge"
    assert route_by_intent({"intent": "退款退货"}) == "knowledge"
    assert route_by_intent({"intent": "物流"}) == "business_data"
    assert route_by_intent({"intent": "订单"}) == "business_data"
    assert route_by_intent({"intent": "售后"}) == "business_data"
    # 缺省兜底
    assert route_by_intent({"intent": "未知"}) == "business_data"

def test_chitchat_node_fixed_text():
    state = create_initial_state(1, "你好")
    updated = chitchat_node(state)
    assert "智能客服助手" in updated["response_text"]
    assert updated["status"] == "chitchat"
    assert updated["suggested_actions"] == []

def test_complaint_node_soothing_and_actions():
    state = create_initial_state(1, "我要投诉你们")
    updated = complaint_node(state)
    assert "非常抱歉" in updated["response_text"]
    assert updated["suggested_actions"] == ["transfer_agent", "create_ticket"]
    assert updated["status"] == "complaint"
