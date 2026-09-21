import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage

from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.pre_nodes import (
    VALID_INTENTS,
    ALL_INTENTS,
    INTENT_FOUR_ELEMENTS_PROMPT,
    _parse_intent_payload,
    intent_recognition_node,
)
from app.services.workflow.nodes.router import route_by_intent


def test_valid_intents_and_all_intents_contain_human():
    """验证 9 分类中包含「人工」意图"""
    assert "人工" in VALID_INTENTS
    assert "人工" in ALL_INTENTS
    expected_valid = {"物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "人工"}
    assert VALID_INTENTS == expected_valid
    assert ALL_INTENTS == expected_valid | {"其他"}


def test_route_by_intent_human_normal_and_legacy():
    """验证「人工」意图在正式分支与 legacy 分支均直通 business_data"""
    # 正常（Ch06+）分支
    assert route_by_intent({"intent": "人工"}, legacy=False) == "business_data"
    assert route_by_intent({"intent": "人工"}) == "business_data"

    # legacy (Ch05) 分支
    assert route_by_intent({"intent": "人工"}, legacy=True) == "business_data"


def test_route_by_intent_all_existing_intents_normal():
    """验证正式路由分支下既有各意图的路由行为保持正常"""
    assert route_by_intent({"intent": "闲聊"}, legacy=False) == "chitchat"
    assert route_by_intent({"intent": "投诉"}, legacy=False) == "complaint"
    assert route_by_intent({"intent": "其他"}, legacy=False) == "other_fallback"
    assert route_by_intent({"intent": "商品咨询"}, legacy=False) == "knowledge"
    assert route_by_intent({"intent": "退款退货"}, legacy=False) == "refund_subflow"
    assert route_by_intent({"intent": "售后"}, legacy=False) == "refund_subflow"
    assert route_by_intent({"intent": "物流"}, legacy=False) == "business_data"
    assert route_by_intent({"intent": "订单"}, legacy=False) == "business_data"
    assert route_by_intent({"intent": "未知分类"}, legacy=False) == "other_fallback"
    assert route_by_intent({}, legacy=False) == "other_fallback"


def test_route_by_intent_all_existing_intents_legacy():
    """验证 legacy 路由分支下既有各意图的路由行为保持正常"""
    assert route_by_intent({"intent": "闲聊"}, legacy=True) == "chitchat"
    assert route_by_intent({"intent": "投诉"}, legacy=True) == "complaint"
    assert route_by_intent({"intent": "商品咨询"}, legacy=True) == "knowledge"
    assert route_by_intent({"intent": "退款退货"}, legacy=True) == "knowledge"
    assert route_by_intent({"intent": "物流"}, legacy=True) == "business_data"
    assert route_by_intent({"intent": "订单"}, legacy=True) == "business_data"
    assert route_by_intent({"intent": "售后"}, legacy=True) == "business_data"
    assert route_by_intent({"intent": "未知分类"}, legacy=True) == "business_data"


def test_parse_intent_payload_human():
    """验证 _parse_intent_payload 正确解析人工意图 payload"""
    payload_str = '{"intent": "人工", "confidence": 0.95, "reason": "转人工"}'
    parsed = _parse_intent_payload(payload_str)
    assert parsed["intent"] == "人工"
    assert parsed["confidence"] == 0.95
    assert parsed["intent_reason"] == "转人工"


def test_parse_intent_payload_existing_intents():
    """验证 _parse_intent_payload 对所有既有分类的解析行为"""
    for item in ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"]:
        res = _parse_intent_payload(f'{{"intent": "{item}", "confidence": 0.9, "reason": "test"}}')
        assert res["intent"] == item
        assert res["confidence"] == 0.9
        assert res["intent_reason"] == "test"

    # 非法分类降级为「其他」
    invalid_res = _parse_intent_payload('{"intent": "非法分类", "confidence": 0.9, "reason": "test"}')
    assert invalid_res["intent"] == "其他"


@pytest.mark.asyncio
async def test_intent_recognition_node_with_human_payload():
    """验证 intent_recognition_node 接入 mock 模型返回「人工」payload 时正确提取意图"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "人工", "confidence": 0.98, "reason": "明确诉求转人工客服"}'
    ))
    state = create_initial_state(1, "帮我转人工客服")
    updated = await intent_recognition_node(state, model=mock_model)
    assert updated["intent"] == "人工"
    assert updated["confidence"] == 0.98
    assert updated["intent_reason"] == "明确诉求转人工客服"


@pytest.mark.asyncio
async def test_dual_model_cascade_human_intent_high_confidence():
    """验证双层级联模型下，小模型识别为「人工」且置信度高时，直接采纳不触发大模型"""
    small_model = MagicMock()
    small_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "人工", "confidence": 0.95, "reason": "明确要求转人工"}'
    ))
    large_model = MagicMock()
    large_model.ainvoke = AsyncMock()

    state = create_initial_state(1, "帮我找客服专员")
    result = await intent_recognition_node(state, model=large_model, small_model=small_model)

    assert result["intent"] == "人工"
    assert result["confidence"] == 0.95
    small_model.ainvoke.assert_awaited_once()
    large_model.ainvoke.assert_not_awaited()


def test_intent_prompt_structure_and_few_shots():
    """验证 Prompt 包含 9 分类说明、人工定义、Few-shot 样例与 8 类兜底保护"""
    prompt = INTENT_FOUR_ELEMENTS_PROMPT
    assert "9 种意图" in prompt or "9 分类" in prompt
    assert "人工" in prompt
    assert "明确要求转人工、建工单或者找客服专员跟进" in prompt
    assert "帮我转人工客服" in prompt
    assert "建个工单" in prompt
