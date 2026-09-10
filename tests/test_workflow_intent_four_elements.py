import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.pre_nodes import (
    INTENT_FOUR_ELEMENTS_PROMPT,
    intent_recognition_node,
    other_fallback_node,
    VALID_INTENTS,
)


def test_prompt_four_elements_contains_8_classes_and_few_shots():
    """验证 Prompt 四件套：包含 8 分类枚举、强制 JSON 说明、边界 few-shot 及「其他」兜底保护"""
    prompt = INTENT_FOUR_ELEMENTS_PROMPT
    # 要素 1: 8 分类枚举
    for intent_name in ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"]:
        assert intent_name in prompt

    # 要素 2: 严格 JSON 输出与 confidence 字段
    assert "confidence" in prompt
    assert "intent" in prompt

    # 要素 3: 边界 Few-shot 样例
    assert "羽绒服拉链坏了怎么修" in prompt
    assert "衣服收到了码数小了想退掉" in prompt
    assert "我的快递到哪了赶紧催催" in prompt
    assert "退款怎么还没到账" in prompt
    assert "今天北京天气怎么样" in prompt

    # 要素 4: 兜底保护
    assert "其他" in prompt


@pytest.mark.asyncio
async def test_intent_recognition_strict_json_parsing():
    """测试意图识别标准输出解析（包含 intent, confidence 浮点数, intent_reason）"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='```json\n{"intent": "物流", "confidence": 0.98, "reason": "查询快递派送进度"}\n```'
    ))
    state = create_initial_state(1, "我的包裹到哪了")
    result = await intent_recognition_node(state, model=mock_model)

    assert result["intent"] == "物流"
    assert isinstance(result["confidence"], float)
    assert result["confidence"] == 0.98
    assert result["intent_reason"] == "查询快递派送进度"


@pytest.mark.asyncio
async def test_boundary_few_shot_handling_aftersales_vs_refund():
    """测试边界案例：售后（维修/破损）vs 退款退货（不合适/退钱退货）"""
    mock_model = MagicMock()
    # 案例 1: 售后质保维修
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "售后", "confidence": 0.95, "reason": "商品拉链损坏维修质保属于售后"}'
    ))
    state1 = create_initial_state(1, "羽绒服拉链坏了怎么修")
    res1 = await intent_recognition_node(state1, model=mock_model)
    assert res1["intent"] == "售后"
    assert res1["confidence"] == 0.95

    # 案例 2: 码数小了退货退款
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "退款退货", "confidence": 0.98, "reason": "尺码不符申请退掉属于退款退货"}'
    ))
    state2 = create_initial_state(1, "衣服收到了码数小了想退掉")
    res2 = await intent_recognition_node(state2, model=mock_model)
    assert res2["intent"] == "退款退货"
    assert res2["confidence"] == 0.98


@pytest.mark.asyncio
async def test_boundary_few_shot_handling_logistics_vs_refund():
    """测试边界案例：催发货物流 vs 催退款到账"""
    mock_model = MagicMock()
    # 催发货 -> 物流
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "物流", "confidence": 0.95, "reason": "催促快递属于物流跟踪"}'
    ))
    res_logistics = await intent_recognition_node(create_initial_state(1, "我的快递到哪了赶紧催催"), model=mock_model)
    assert res_logistics["intent"] == "物流"
    assert res_logistics["confidence"] == 0.95

    # 催退款 -> 退款退货
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "退款退货", "confidence": 0.96, "reason": "催退款到账属于退款流程"}'
    ))
    res_refund = await intent_recognition_node(create_initial_state(1, "退款怎么还没到账"), model=mock_model)
    assert res_refund["intent"] == "退款退货"
    assert res_refund["confidence"] == 0.96


@pytest.mark.asyncio
async def test_out_of_domain_query_falls_into_other():
    """测试超出电商业务领域的怪问题稳定落入「其他」"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "其他", "confidence": 0.99, "reason": "提问与电商商城业务无关"}'
    ))
    state = create_initial_state(1, "今天北京天气怎么样？给我写首现代诗")
    result = await intent_recognition_node(state, model=mock_model)

    assert result["intent"] == "其他"
    assert result["confidence"] == 0.99
    assert "电商" in result["intent_reason"] or result["intent_reason"] != ""


@pytest.mark.asyncio
async def test_dual_model_cascade_high_confidence_small_model_not_triggering_large():
    """测试双层模型级联：小模型置信度高（>=0.85）且为 7 类有效业务意图时，直接采纳，不调用大模型"""
    small_model = MagicMock()
    small_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "商品咨询", "confidence": 0.96, "reason": "咨询羽绒服含绒量"}'
    ))
    large_model = MagicMock()
    large_model.ainvoke = AsyncMock()

    state = create_initial_state(1, "这款羽绒服含绒量是多少")
    result = await intent_recognition_node(state, model=large_model, small_model=small_model)

    assert result["intent"] == "商品咨询"
    assert result["confidence"] == 0.96
    small_model.ainvoke.assert_awaited_once()
    large_model.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_dual_model_cascade_low_confidence_triggers_large_model():
    """测试双层模型级联：小模型置信度低（<0.85），唤醒大模型重新判定并采纳大模型结果"""
    small_model = MagicMock()
    small_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "物流", "confidence": 0.72, "reason": "置信度不足"}'
    ))
    large_model = MagicMock()
    large_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "售后", "confidence": 0.93, "reason": "大模型复判为破损件售后处理"}'
    ))

    state = create_initial_state(1, "外包装烂了里面衣服脏了怎么办")
    result = await intent_recognition_node(state, model=large_model, small_model=small_model)

    assert result["intent"] == "售后"
    assert result["confidence"] == 0.93
    assert result["intent_reason"] == "大模型复判为破损件售后处理"
    small_model.ainvoke.assert_awaited_once()
    large_model.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_dual_model_cascade_other_intent_triggers_large_model():
    """测试双层模型级联：小模型判断为「其他」（即便置信度>=0.85），必须经大模型二次确认"""
    small_model = MagicMock()
    small_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "其他", "confidence": 0.88, "reason": "初判为其他"}'
    ))
    large_model = MagicMock()
    large_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "商品咨询", "confidence": 0.91, "reason": "大模型纠正为商品面料功能咨询"}'
    ))

    state = create_initial_state(1, "这件冲锋衣防水透湿指数是啥意思")
    result = await intent_recognition_node(state, model=large_model, small_model=small_model)

    assert result["intent"] == "商品咨询"
    assert result["confidence"] == 0.91
    small_model.ainvoke.assert_awaited_once()
    large_model.ainvoke.assert_awaited_once()


def test_other_fallback_node_response_and_action():
    """测试 other_fallback_node 兜底节点返回友好话术与人工客服转接建议动作"""
    state = create_initial_state(1, "你能帮我写个 Python 爬虫吗")
    res = other_fallback_node(state)

    assert res["status"] == "other_fallback"
    assert res["suggested_actions"] == ["transfer_agent"]
    assert "超出我的业务范围" in res["response_text"]
    assert "转接人工客服" in res["response_text"]


@pytest.mark.asyncio
async def test_json_parsing_error_fallback_to_other():
    """测试模型输出畸形文本或无法解析为合法 JSON 时，优雅降级为「其他」与 0.0 置信度"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content="这是一段毫无结构的纯文本回复，不包含合法 JSON。"
    ))
    state = create_initial_state(1, "随便说点啥")
    result = await intent_recognition_node(state, model=mock_model)

    assert result["intent"] == "其他"
    assert result["confidence"] == 0.0
    assert result["intent_reason"] == "解析异常降级"


@pytest.mark.asyncio
async def test_unrecognized_intent_fallback_to_other():
    """测试模型输出不存在的非法分类标签时，归一化降级为「其他」"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "未知神秘分类", "confidence": 0.95, "reason": "瞎猜的"}'
    ))
    state = create_initial_state(1, "测试未知分类")
    result = await intent_recognition_node(state, model=mock_model)

    assert result["intent"] == "其他"
    assert result["confidence"] == 0.95
