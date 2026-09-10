import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.pre_nodes import (
    coreference_resolution_node,
    coreference_rewrite_node,
    COREFERENCE_REWRITE_SYSTEM_PROMPT,
)


@pytest.mark.asyncio
async def test_coref_rewrite_pronoun_resolution_with_history():
    """测试用例 1: 带指代补全（结合历史上下文补全代词与省略主语）"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content="订单1001极简保暖羽绒服支持退货吗"
    ))

    state = create_initial_state(1, "它能退吗")
    state["messages"] = [
        HumanMessage(content="我想看下订单1001的羽绒服"),
        AIMessage(content="为您找到订单1001极简保暖羽绒服，售价299元，状态为已签收。"),
        HumanMessage(content="它能退吗"),
    ]

    result = await coreference_resolution_node(state, model=mock_model)

    assert result["resolved_query"] == "订单1001极简保暖羽绒服支持退货吗"
    mock_model.ainvoke.assert_called_once()
    called_messages = mock_model.ainvoke.call_args[0][0]
    # 验证 Prompt 中包含了系统提示词与历史中涉及的实体
    assert any("订单1001" in str(m.content) for m in called_messages)
    assert any("它能退吗" in str(m.content) for m in called_messages)


@pytest.mark.asyncio
async def test_coref_rewrite_colloquial_normalization():
    """测试用例 2: 口语归一化（将含混方言口语规范化为标准业务提问）"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content="查询订单物流最新轨迹"
    ))

    state = create_initial_state(1, "货到哪了老铁")

    result = await coreference_resolution_node(state, model=mock_model)

    assert result["resolved_query"] == "查询订单物流最新轨迹"
    mock_model.ainvoke.assert_called_once()
    called_messages = mock_model.ainvoke.call_args[0][0]
    assert any("货到哪了老铁" in str(m.content) for m in called_messages)


@pytest.mark.asyncio
async def test_coref_rewrite_passthrough_protection_greeting():
    """测试用例 3a: 寒暄打招呼原样透传保护 (Hard Gate，无需触发模型调用)"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock()

    state = create_initial_state(1, "你好")

    result = await coreference_resolution_node(state, model=mock_model)

    assert result["resolved_query"] == "你好"
    # 纯问候且无历史，Hard Gate 拦截直接透传，不消耗模型调用
    mock_model.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_coref_rewrite_passthrough_protection_self_contained():
    """测试用例 3b: 完整自包含句子原样透传保护，并验证 coreference_rewrite_node 别名"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content="查询订单1001"
    ))

    state = create_initial_state(1, "查询订单1001")

    # 验证原函数与别名一致性
    result1 = await coreference_resolution_node(state, model=mock_model)
    assert result1["resolved_query"] == "查询订单1001"

    result2 = await coreference_rewrite_node(state, model=mock_model)
    assert result2["resolved_query"] == "查询订单1001"


@pytest.mark.asyncio
async def test_coref_rewrite_exception_fallback():
    """测试用例 4: 模型调用发生异常时，安全降级为原样透传 input_query"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(side_effect=RuntimeError("LLM connection timed out"))

    state = create_initial_state(1, "这单子咋样了")

    result = await coreference_resolution_node(state, model=mock_model)

    assert result["resolved_query"] == "这单子咋样了"


def test_coref_rewrite_sync_compatibility_and_empty():
    """测试向后兼容性：同步调用与空输入直接返回"""
    empty_state = create_initial_state(1, "")
    res_empty = coreference_resolution_node(empty_state)
    assert res_empty["resolved_query"] == ""

    normal_state = create_initial_state(1, "我的订单到哪了")
    res_sync = coreference_resolution_node(normal_state)
    assert res_sync["resolved_query"] == "我的订单到哪了"
