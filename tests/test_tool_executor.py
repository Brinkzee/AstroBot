import asyncio
import json
import pytest
from langchain_core.tools import tool

from app.tools.registry import ToolRegistry, default_tool_registry
from app.tools.executor import ToolExecutor


def test_default_tool_registry_contains_five_tools():
    """验证默认工具注册中心已注册全部 5 个业务工具"""
    tools = default_tool_registry.get_all_tools()
    tool_names = [t.name for t in tools]
    assert len(tools) == 5
    assert "query_order" in tool_names
    assert "query_product" in tool_names
    assert "query_logistics" in tool_names
    assert "query_faq" in tool_names
    assert "create_ticket" in tool_names

    # 单个工具获取测试
    order_tool = default_tool_registry.get_tool("query_order")
    assert order_tool is not None
    assert order_tool.name == "query_order"

    assert default_tool_registry.get_tool("non_existent_tool") is None


def test_custom_tool_registry():
    """验证自定义注册中心的注册与检索能力"""
    registry = ToolRegistry()
    assert len(registry.get_all_tools()) == 0

    @tool
    def echo_tool(text: str) -> str:
        """回显工具"""
        return text

    registry.register(echo_tool)
    assert len(registry.get_all_tools()) == 1
    assert registry.get_tool("echo_tool") is not None
    assert registry.get_tool("echo_tool").name == "echo_tool"


@pytest.mark.asyncio
async def test_tool_executor_success():
    """验证工具正常调用的执行结果结构"""
    executor = ToolExecutor(default_tool_registry)
    tool_call = {
        "name": "query_order",
        "args": {"order_id": "1001"},
        "id": "call_order_001",
    }
    result = await executor.execute(tool_call)

    assert result["success"] is True
    assert result["tool_name"] == "query_order"
    assert result["tool_call_id"] == "call_order_001"
    assert result["error"] is None
    assert "1001" in result["output"]
    assert "极简保暖羽绒服" in result["output"]


@pytest.mark.asyncio
async def test_tool_executor_unknown_tool():
    """验证未注册工具执行时优雅返回失败结构，不抛出未捕获异常"""
    executor = ToolExecutor(default_tool_registry)
    tool_call = {
        "name": "non_existent_tool",
        "args": {"param": "value"},
        "id": "call_unknown_002",
    }
    result = await executor.execute(tool_call)

    assert result["success"] is False
    assert result["tool_name"] == "non_existent_tool"
    assert result["tool_call_id"] == "call_unknown_002"
    assert result["error"] is not None
    assert "未找到工具" in result["error"]
    assert "工具 [non_existent_tool] 调用失败" in result["output"]
    assert "请结合此情况向用户做解释并提供帮助" in result["output"]


@pytest.mark.asyncio
async def test_tool_executor_schema_validation_failure():
    """验证工具参数 schema 校验失败时优雅拦截并返回失败结构"""
    executor = ToolExecutor(default_tool_registry)
    # query_order 需要必填的 order_id 字符串参数，传入空字典导致 schema 校验失败
    tool_call = {
        "name": "query_order",
        "args": {},
        "id": "call_validation_003",
    }
    result = await executor.execute(tool_call)

    assert result["success"] is False
    assert result["tool_name"] == "query_order"
    assert result["tool_call_id"] == "call_validation_003"
    assert result["error"] is not None
    assert "参数校验失败" in result["error"]
    assert "工具 [query_order] 调用失败" in result["output"]
    assert "请结合此情况向用户做解释并提供帮助" in result["output"]


@pytest.mark.asyncio
async def test_tool_executor_json_string_args():
    """验证 tool_call 中的 args 为 JSON 字符串时能被正确反序列化并校验执行"""
    executor = ToolExecutor(default_tool_registry)
    tool_call = {
        "name": "query_order",
        "args": json.dumps({"order_id": "1002"}),
        "id": "call_json_str_004",
    }
    result = await executor.execute(tool_call)

    assert result["success"] is True
    assert result["tool_name"] == "query_order"
    assert result["tool_call_id"] == "call_json_str_004"
    assert result["error"] is None
    assert "1002" in result["output"]


@pytest.mark.asyncio
async def test_tool_executor_timeout():
    """验证超时控制 (asyncio.wait_for) 在工具执行超时后能优雅拦截并返回超时错误"""
    registry = ToolRegistry()

    @tool
    async def slow_mock_tool(seconds: float = 0.5) -> str:
        """模拟耗时很长的慢工具"""
        await asyncio.sleep(seconds)
        return "slow_done"

    registry.register(slow_mock_tool)

    # 设置短超时 0.05s，无重试
    executor = ToolExecutor(registry=registry, timeout=0.05, max_retries=0)
    tool_call = {
        "name": "slow_mock_tool",
        "args": {"seconds": 0.2},
        "id": "call_timeout_005",
    }
    result = await executor.execute(tool_call)

    assert result["success"] is False
    assert result["tool_name"] == "slow_mock_tool"
    assert result["tool_call_id"] == "call_timeout_005"
    assert result["error"] is not None
    assert "超时" in result["error"] or "Timeout" in result["error"]
    assert "工具 [slow_mock_tool] 调用失败" in result["output"]
    assert "请结合此情况向用户做解释并提供帮助" in result["output"]


@pytest.mark.asyncio
async def test_tool_executor_retry_success():
    """验证工具首次抛异常后，重试策略 (max_retries=1) 触发重试并成功恢复"""
    registry = ToolRegistry()
    call_count = 0

    @tool
    async def flaky_tool(param: str) -> str:
        """前 1 次调用失败，第 2 次调用成功的抖动工具"""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionResetError("模拟网络瞬时抖动中断")
        return f"recovered: {param}"

    registry.register(flaky_tool)

    executor = ToolExecutor(registry=registry, timeout=2.0, max_retries=1)
    tool_call = {
        "name": "flaky_tool",
        "args": {"param": "hello"},
        "id": "call_retry_006",
    }
    result = await executor.execute(tool_call)

    assert result["success"] is True
    assert result["error"] is None
    assert result["output"] == "recovered: hello"
    assert call_count == 2


@pytest.mark.asyncio
async def test_tool_executor_retry_exhausted_failure():
    """验证工具重试次数耗尽后，返回包含最后异常信息的标准失败结构"""
    registry = ToolRegistry()
    call_count = 0

    @tool
    def always_failing_tool(reason: str) -> str:
        """持续失败的工具"""
        nonlocal call_count
        call_count += 1
        raise RuntimeError(f"底层服务崩溃: {reason}")

    registry.register(always_failing_tool)

    executor = ToolExecutor(registry=registry, timeout=2.0, max_retries=1)
    tool_call = {
        "name": "always_failing_tool",
        "args": {"reason": "数据库连接池耗尽"},
        "id": "call_fail_007",
    }
    result = await executor.execute(tool_call)

    assert result["success"] is False
    assert result["tool_name"] == "always_failing_tool"
    assert result["tool_call_id"] == "call_fail_007"
    assert result["error"] is not None
    assert "数据库连接池耗尽" in result["error"]
    assert "工具 [always_failing_tool] 调用失败" in result["output"]
    assert "请结合此情况向用户做解释并提供帮助" in result["output"]
    assert call_count == 2  # 初始 1 次 + 重试 1 次 = 共 2 次
