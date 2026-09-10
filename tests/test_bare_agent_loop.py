import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from scripts.bare_agent_loop import run_bare_agent_loop

@pytest.mark.asyncio
async def test_bare_agent_loop_single_step_convergence():
    """测试单步工具调用并自发收敛"""
    mock_llm = MagicMock()
    # 第一轮返回工具调用，第二轮返回文本收敛
    resp1 = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"arg1": "val1"}, "id": "call_123"}]
    )
    resp2 = AIMessage(content="查询结果为：val1已处理完毕")
    
    bound_llm = MagicMock()
    bound_llm.ainvoke = AsyncMock(side_effect=[resp1, resp2])
    mock_llm.bind_tools.return_value = bound_llm

    async def mock_tool(arg1: str):
        return f"tool_result_{arg1}"

    result = await run_bare_agent_loop(
        llm=mock_llm,
        tools={"mock_tool": mock_tool},
        query="帮我查一下",
        max_steps=5,
    )

    assert result["answer"] == "查询结果为：val1已处理完毕"
    assert result["steps"] == 2
    assert len(result["messages"]) == 5  # system + human + ai(tool_call) + tool + ai(final)
    mock_llm.bind_tools.assert_called_once()

@pytest.mark.asyncio
async def test_bare_agent_loop_max_steps_guard():
    """测试超过最大步数强制防御截断"""
    mock_llm = MagicMock()
    # 始终返回工具调用，触发死循环守卫
    resp_loop = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"arg1": "val"}, "id": "call_loop"}]
    )
    bound_llm = MagicMock()
    bound_llm.ainvoke = AsyncMock(return_value=resp_loop)
    mock_llm.bind_tools.return_value = bound_llm

    # 兜底强制总结响应
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="步数超限强制总结"))

    async def mock_tool(arg1: str):
        return "loop"

    result = await run_bare_agent_loop(
        llm=mock_llm,
        tools={"mock_tool": mock_tool},
        query="死循环提问",
        max_steps=3,
    )

    assert result["steps"] == 3
    assert result["answer"] == "步数超限强制总结"

@pytest.mark.asyncio
async def test_bare_agent_loop_sync_tool_and_missing_tool():
    """测试同步工具调用以及找不到工具时的错误处理"""
    mock_llm = MagicMock()
    resp1 = AIMessage(
        content="",
        tool_calls=[
            {"name": "sync_tool", "args": {"val": "123"}, "id": "call_sync"},
            {"name": "unknown_tool", "args": {}, "id": "call_unknown"},
        ],
    )
    resp2 = AIMessage(content="同步工具和缺失工具已处理")

    bound_llm = MagicMock()
    bound_llm.ainvoke = AsyncMock(side_effect=[resp1, resp2])
    mock_llm.bind_tools.return_value = bound_llm

    def sync_tool(val: str):
        return f"sync_{val}"

    result = await run_bare_agent_loop(
        llm=mock_llm,
        tools={"sync_tool": sync_tool},
        query="调用同步与不存在工具",
        max_steps=5,
    )

    assert result["answer"] == "同步工具和缺失工具已处理"
    assert result["steps"] == 2
    assert len(result["messages"]) == 6
    assert result["messages"][3].content == "sync_123"
    assert "Tool unknown_tool not found" in result["messages"][4].content
