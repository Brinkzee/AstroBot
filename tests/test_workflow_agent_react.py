import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.agent_node import main_agent_node

@pytest.mark.asyncio
async def test_main_agent_single_step_tool_call():
    """测试物流查询单步调用工具收敛"""
    mock_model = MagicMock()
    resp1 = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}]
    )
    resp2 = AIMessage(content="订单1001的物流状态为：派送中，顺丰速运承运。")
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[resp1, resp2])
    mock_model.bind_tools.return_value = bound_model

    mock_tool = MagicMock()
    mock_tool.name = "query_logistics"
    mock_tool.ainvoke = AsyncMock(return_value='{"status": "派送中"}')

    state = create_initial_state(1, "订单1001的物流到哪了")
    updated = await main_agent_node(state, model=mock_model, tools=[mock_tool])

    assert "派送中" in updated["response_text"]
    assert updated["steps_taken"] == 2
    assert updated["status"] == "success"

@pytest.mark.asyncio
async def test_main_agent_multistep_react_call():
    """测试复合问题多步 ReAct（先查订单再查物流）"""
    mock_model = MagicMock()
    # 步1: 查订单
    resp1 = AIMessage(
        content="",
        tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}]
    )
    # 步2: 查物流
    resp2 = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c2"}]
    )
    # 步3: 最终总结
    resp3 = AIMessage(content="您购买的羽绒服已发货，目前顺丰正在派送中。")
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[resp1, resp2, resp3])
    mock_model.bind_tools.return_value = bound_model

    tool_order = MagicMock()
    tool_order.name = "query_order"
    tool_order.ainvoke = AsyncMock(return_value='{"order_id": "1001", "name": "羽绒服"}')

    tool_logistics = MagicMock()
    tool_logistics.name = "query_logistics"
    tool_logistics.ainvoke = AsyncMock(return_value='{"status": "派件中"}')

    state = create_initial_state(1, "查一下我买的羽绒服到哪了")
    updated = await main_agent_node(state, model=mock_model, tools=[tool_order, tool_logistics])

    assert updated["steps_taken"] == 3
    assert "顺丰正在派送中" in updated["response_text"]
    assert updated["status"] == "success"

@pytest.mark.asyncio
async def test_main_agent_knowledge_injection():
    """测试知识库证据上下文注入（前3条注入SystemMessage）"""
    mock_model = MagicMock()
    captured_messages = []

    async def mock_ainvoke(messages):
        captured_messages.append(list(messages))
        return AIMessage(content="退货政策是7天无理由退货。")

    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    mock_model.bind_tools.return_value = bound_model
    mock_model.ainvoke = bound_model.ainvoke

    state = create_initial_state(1, "退货规则是什么")
    state["retrieved_docs"] = [
        {"text": "7天无理由退货"},
        {"content": "退货运费险由卖家承担"},
        {"text": "超出7天需联系售后专员"},
        {"text": "多余文档不应注入"},
    ]

    updated = await main_agent_node(state, model=mock_model, tools=[])

    assert "7天无理由退货" in updated["response_text"]
    assert len(captured_messages) > 0
    sys_msg = captured_messages[0][0]
    assert isinstance(sys_msg, SystemMessage)
    assert "【参考知识 1】7天无理由退货" in sys_msg.content
    assert "【参考知识 2】退货运费险由卖家承担" in sys_msg.content
    assert "【参考知识 3】超出7天需联系售后专员" in sys_msg.content
    assert "多余文档不应注入" not in sys_msg.content

@pytest.mark.asyncio
async def test_main_agent_missing_param_asking():
    """测试信息不足时友好追问缺失参数（无工具调用直接收敛）"""
    mock_model = MagicMock()
    resp = AIMessage(content="请问您的订单号是多少呢？我来帮您查询物流进度。")
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(return_value=resp)
    mock_model.bind_tools.return_value = bound_model

    mock_tool = MagicMock()
    mock_tool.name = "query_logistics"

    state = create_initial_state(1, "帮我查下物流")
    updated = await main_agent_node(state, model=mock_model, tools=[mock_tool])

    assert updated["steps_taken"] == 1
    assert "请问您的订单号是多少呢" in updated["response_text"]
    assert updated["status"] == "success"
    mock_tool.ainvoke.assert_not_called()

@pytest.mark.asyncio
async def test_main_agent_max_steps_guard():
    """测试最大 5 步硬截断与最终总结兜底机制"""
    mock_model = MagicMock()
    tool_resp = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}]
    )
    summary_resp = AIMessage(content="查询步数超限，根据目前信息已为您尝试多次查询。")

    bound_model = MagicMock()
    # 循环 5 次 tool_call
    bound_model.ainvoke = AsyncMock(side_effect=[tool_resp, tool_resp, tool_resp, tool_resp, tool_resp])
    mock_model.bind_tools.return_value = bound_model
    # 步数超限后的 final_summary 调用直接使用 llm.ainvoke
    mock_model.ainvoke = AsyncMock(return_value=summary_resp)

    mock_tool = MagicMock()
    mock_tool.name = "query_logistics"
    mock_tool.ainvoke = AsyncMock(return_value='{"status": "查询中"}')

    state = create_initial_state(1, "一直查物流")
    updated = await main_agent_node(state, model=mock_model, tools=[mock_tool])

    assert updated["steps_taken"] == 5
    assert "步数超限" in updated["response_text"]
    assert updated["status"] == "success"
    mock_model.ainvoke.assert_awaited_once()

@pytest.mark.asyncio
async def test_main_agent_token_usage_recording():
    """测试每轮 LLM 推理的 token 消耗统计汇总"""
    mock_model = MagicMock()
    resp1 = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
        response_metadata={"token_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
    )
    resp2 = AIMessage(
        content="物流已送达。",
        response_metadata={"token_usage": {"prompt_tokens": 150, "completion_tokens": 30, "total_tokens": 180}}
    )
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[resp1, resp2])
    mock_model.bind_tools.return_value = bound_model

    mock_tool = MagicMock()
    mock_tool.name = "query_logistics"
    mock_tool.ainvoke = AsyncMock(return_value='{"status": "已送达"}')

    state = create_initial_state(1, "查询订单1001")
    updated = await main_agent_node(state, model=mock_model, tools=[mock_tool])

    assert updated["token_usage"]["prompt_tokens"] == 250
    assert updated["token_usage"]["completion_tokens"] == 50
    assert updated["token_usage"]["total_tokens"] == 300

@pytest.mark.asyncio
async def test_main_agent_tool_exception_handling():
    """测试工具抛出异常时的容错与 ToolMessage 捕获"""
    mock_model = MagicMock()
    resp1 = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}]
    )
    resp2 = AIMessage(content="由于物流服务出现异常，暂无法获取最新状态。")
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[resp1, resp2])
    mock_model.bind_tools.return_value = bound_model

    mock_tool = MagicMock()
    mock_tool.name = "query_logistics"
    mock_tool.ainvoke = AsyncMock(side_effect=RuntimeError("第三方物流接口超时"))

    state = create_initial_state(1, "查一下1001物流")
    updated = await main_agent_node(state, model=mock_model, tools=[mock_tool])

    assert updated["steps_taken"] == 2
    assert "物流服务出现异常" in updated["response_text"]
    # 校验 ToolMessage 记录了异常信息
    tool_msgs = [m for m in updated["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "执行异常: 第三方物流接口超时" in tool_msgs[0].content
