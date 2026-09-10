import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.agent_node import main_agent_node


@pytest.mark.asyncio
async def test_agent_system_message_order_and_refund_instructions_injection():
    """测试用例 1: 传入 order_data 与退货政策，验证注入到 SystemMessage 中的内容包含订单详情与专项裁决指令"""
    mock_model = MagicMock()
    captured_messages = []

    async def mock_ainvoke(messages):
        captured_messages.append(list(messages))
        return AIMessage(content="经核查，订单1001支持退款。您可以点击下方“申请退款”按钮提交退款申请单。")

    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    mock_model.bind_tools.return_value = bound_model
    mock_model.ainvoke = bound_model.ainvoke

    state = create_initial_state(1, "订单1001可以退款吗")
    state["order_id"] = "1001"
    state["intent"] = "退款退货"
    state["order_data"] = {
        "order_id": "1001",
        "订单状态": "已发货",
        "支付金额": "299.00元",
        "商品明细": [
            {"商品名称": "极简保暖羽绒服", "数量": 1, "单价": "299.00元"}
        ],
        "下单时间": "2026-09-05 14:20:00",
    }
    state["retrieved_docs"] = [
        {"text": "7天无理由退货：签收后7天内支持退货退款，运费由买家或运费险承担。"}
    ]

    updated = await main_agent_node(state, model=mock_model, tools=[])

    assert len(captured_messages) > 0
    sys_msg = captured_messages[0][0]
    assert isinstance(sys_msg, SystemMessage)
    content = sys_msg.content

    # 验证订单真实状态注入
    assert "【当前订单真实状态】" in content
    assert "1001" in content
    assert "已发货" in content
    assert "299.00元" in content
    assert "极简保暖羽绒服" in content
    assert "2026-09-05 14:20:00" in content

    # 验证专项裁决指令注入
    assert "【退款退货/售后专注裁决专项指令】" in content
    assert "判定" in content and "支持退款" in content
    assert "严禁" in content and "query_order" in content
    assert "追问" in content
    assert "申请退款" in content

    # 验证政策知识注入
    assert "7天无理由退货" in content


@pytest.mark.asyncio
async def test_agent_avoids_redundant_query_order_with_preprovided_order_data():
    """测试用例 2: Agent 在拥有预取数据时不发起 query_order 工具调用，单步直接收敛输出最终裁决"""
    mock_model = MagicMock()
    # 模拟大模型直接给出最终回答，无需调用 query_order
    final_resp = AIMessage(
        content="经核实，订单1001当前为已发货状态，符合退款条件，支持退款。运费由运费险承担，您可点击下方“申请退款”提交申请。"
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(return_value=final_resp)
    mock_model.bind_tools.return_value = bound_model

    mock_query_order = MagicMock()
    mock_query_order.name = "query_order"
    mock_query_order.ainvoke = AsyncMock(return_value='{"order_id": "1001"}')

    state = create_initial_state(1, "退款订单: 1001")
    state["order_id"] = "1001"
    state["intent"] = "退款退货"
    state["order_data"] = {
        "order_id": "1001",
        "订单状态": "已发货",
        "支付金额": "299.00元",
        "商品明细": [{"商品名称": "极简保暖羽绒服", "数量": 1, "单价": "299.00元"}],
        "下单时间": "2026-09-05 14:20:00",
    }

    updated = await main_agent_node(state, model=mock_model, tools=[mock_query_order])

    # 验证单步直接完成
    assert updated["steps_taken"] == 1
    assert "支持退款" in updated["response_text"]
    assert updated["status"] == "success"
    # 验证未调用 query_order
    mock_query_order.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_agent_refund_approved_appends_suggested_action():
    """测试用例 3: 判定支持退款时，返回的 suggested_actions 包含 apply_refund"""
    mock_model = MagicMock()
    resp = AIMessage(content="您的订单1001支持退换货，运费由运费险覆盖，可点击下方“申请退款”按钮办理。")
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(return_value=resp)
    mock_model.bind_tools.return_value = bound_model
    mock_model.ainvoke = bound_model.ainvoke

    state = create_initial_state(1, "1001能退吗")
    state["order_id"] = "1001"
    state["intent"] = "退款退货"
    state["order_data"] = {"order_id": "1001", "订单状态": "已发货"}
    state["suggested_actions"] = ["view_policy"]

    updated = await main_agent_node(state, model=mock_model, tools=[])

    assert "suggested_actions" in updated
    assert "apply_refund" in updated["suggested_actions"]
    assert "view_policy" in updated["suggested_actions"]
    # 确保没有重复项
    assert updated["suggested_actions"].count("apply_refund") == 1


@pytest.mark.asyncio
async def test_agent_refund_rejected_no_apply_refund_action():
    """测试用例 4: 判定完全不支持退款（如已超过售后时效）时，验证建议动作行为"""
    mock_model = MagicMock()
    resp = AIMessage(content="非常抱歉，您的订单1001下单已超过15天，已超出退换货时效，不支持退款。")
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(return_value=resp)
    mock_model.bind_tools.return_value = bound_model
    mock_model.ainvoke = bound_model.ainvoke

    state = create_initial_state(1, "1001能退吗")
    state["order_id"] = "1001"
    state["intent"] = "退款退货"
    state["order_data"] = {"order_id": "1001", "订单状态": "已完成"}
    state["suggested_actions"] = ["view_policy"]

    updated = await main_agent_node(state, model=mock_model, tools=[])

    assert "suggested_actions" in updated
    assert "apply_refund" not in updated["suggested_actions"]
    assert "view_policy" in updated["suggested_actions"]
    assert "不支持退款" in updated["response_text"]


@pytest.mark.asyncio
async def test_agent_unrelated_query_retains_normal_flow():
    """测试用例 5: 无 order_data 的普通场景（如查物流）保持原工具调用行为正常"""
    mock_model = MagicMock()
    resp1 = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}]
    )
    resp2 = AIMessage(content="订单1001的顺丰快递正在派送中。")
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[resp1, resp2])
    mock_model.bind_tools.return_value = bound_model

    mock_tool = MagicMock()
    mock_tool.name = "query_logistics"
    mock_tool.ainvoke = AsyncMock(return_value='{"status": "派送中"}')

    state = create_initial_state(1, "查一下1001物流")
    # 无 order_data
    updated = await main_agent_node(state, model=mock_model, tools=[mock_tool])

    assert updated["steps_taken"] == 2
    assert "顺丰快递正在派送中" in updated["response_text"]
    assert "suggested_actions" in updated
    assert "apply_refund" not in updated.get("suggested_actions", [])
    mock_tool.ainvoke.assert_awaited_once()

    # 验证 SystemMessage 中无退款专职指令
    sys_msgs = [m for m in updated["messages"] if isinstance(m, SystemMessage)]
    assert len(sys_msgs) > 0
    assert "【当前订单真实状态】" not in sys_msgs[0].content
    assert "【退款退货/售后专注裁决专项指令】" not in sys_msgs[0].content
