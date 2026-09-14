import asyncio
import json
import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.tools.registry import ToolRegistry, default_tool_registry
from app.tools.executor import ToolExecutor


@pytest.mark.asyncio
async def test_schema_validation_missing_required_feedback():
    """测试必填项缺失触发参数校验拦截，返回 status='校验拦下'，error_type='INVALID_ARGS'，并格式化输出回灌"""
    executor = ToolExecutor(default_tool_registry)
    # query_order 需要必填的 order_id 字符串参数，传入空字典
    res = await executor.execute({"name": "query_order", "args": {}, "id": "call_schema_001"})

    assert res["success"] is False
    assert res["status"] == "校验拦下"
    assert res["error_type"] == "INVALID_ARGS"
    assert res["retry_count"] == 0
    assert "order_id" in res["error"] or "必填" in res["error"]
    assert "工具 [query_order] 调用失败: 参数校验未通过:" in res["output"]
    assert "请结合此情况向用户做解释或追问补充必要信息" in res["output"]
    assert isinstance(res["duration_ms"], int)


@pytest.mark.asyncio
async def test_schema_validation_type_error_and_bounds():
    """测试类型错误或越界触发参数校验拦截"""
    registry = ToolRegistry()

    class PriceQueryArgs(BaseModel):
        product_id: str
        discount_rate: float = Field(gt=0.0, le=1.0)

    @tool(args_schema=PriceQueryArgs)
    def query_discount(product_id: str, discount_rate: float) -> str:
        """查询商品折扣率"""
        return f"{product_id}: {discount_rate}"

    registry.register(query_discount)
    executor = ToolExecutor(registry=registry)

    # 1. 传入越界值 discount_rate = 1.5
    res = await executor.execute({
        "name": "query_discount",
        "args": {"product_id": "P101", "discount_rate": 1.5},
        "id": "call_schema_002",
    })
    assert res["success"] is False
    assert res["status"] == "校验拦下"
    assert res["error_type"] == "INVALID_ARGS"
    assert res["retry_count"] == 0
    assert "discount_rate" in res["error"]
    assert "工具 [query_discount] 调用失败: 参数校验未通过:" in res["output"]


@pytest.mark.asyncio
async def test_schema_validation_json_parse_error():
    """测试非法 JSON 字符串参数触发 INVALID_ARGS 且 status='校验拦下'"""
    executor = ToolExecutor(default_tool_registry)
    res = await executor.execute({
        "name": "query_order",
        "args": "not a valid json {{{{",
        "id": "call_schema_003",
    })
    assert res["success"] is False
    assert res["status"] == "校验拦下"
    assert res["error_type"] == "INVALID_ARGS"
    assert res["retry_count"] == 0
    assert "反序列化失败" in res["error"]
    assert "请结合此情况向用户做解释或追问补充必要信息" in res["output"]


@pytest.mark.asyncio
async def test_business_empty_result_no_retry():
    """测试业务空结果（查询落空），标记 status='成功'，error_type='QUERY_MISSED'，严禁重试（retry_count=0）"""
    registry = ToolRegistry()
    call_count = 0

    @tool
    def query_empty_order(order_id: str) -> str:
        """查询空订单"""
        nonlocal call_count
        call_count += 1
        return json.dumps({"order_id": order_id, "message": "未找到相关订单信息"}, ensure_ascii=False)

    registry.register(query_empty_order)
    executor = ToolExecutor(registry=registry, max_retries=1)

    res = await executor.execute({
        "name": "query_empty_order",
        "args": {"order_id": "9999"},
        "id": "call_missed_001",
    })

    assert res["success"] is True
    assert res["status"] == "成功"
    assert res["error_type"] == "QUERY_MISSED"
    assert res["error"] is None
    assert res["retry_count"] == 0
    assert call_count == 1
    assert "未找到相关订单信息" in res["output"]


@pytest.mark.asyncio
async def test_read_operation_transient_network_retry_success():
    """测试读操作遭遇网络瞬时故障触发重试并成功恢复，记录重试次数 retry_count=1"""
    registry = ToolRegistry()
    call_count = 0

    @tool
    def network_glitch_tool(keyword: str) -> str:
        """网络瞬时抖动模拟工具"""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionResetError("模拟网络物理连接重置")
        return f"查询到结果: {keyword}"

    registry.register(network_glitch_tool)
    executor = ToolExecutor(registry=registry, timeout=2.0, max_retries=1)

    res = await executor.execute({
        "name": "network_glitch_tool",
        "args": {"keyword": "羽绒服"},
        "id": "call_retry_001",
    })

    assert res["success"] is True
    assert res["status"] == "成功"
    assert res["error_type"] is None
    assert res["error"] is None
    assert res["retry_count"] == 1
    assert call_count == 2
    assert "查询到结果: 羽绒服" in res["output"]


@pytest.mark.asyncio
async def test_write_operation_no_retry_on_timeout():
    """测试写操作（create_ticket）发生超时绝不自动重试，retry_count 恒为 0"""
    registry = ToolRegistry()
    call_count = 0

    @tool
    async def create_ticket(description: str) -> str:
        """模拟创建工单写操作"""
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.2)
        return "ticket_created"

    registry.register(create_ticket)
    # 超时设为 0.05s，max_retries 设为 1
    executor = ToolExecutor(registry=registry, timeout=0.05, max_retries=1)

    res = await executor.execute(
        {
            "name": "create_ticket",
            "args": {"description": "拉链损坏退货"},
            "id": "call_write_001",
        },
        context={"user_query": "我要投诉建工单", "confirmed": True},
    )

    assert res["success"] is False
    assert res["status"] == "超时"
    assert res["error_type"] == "SYSTEM_FAULT"
    assert res["retry_count"] == 0
    assert call_count == 1


@pytest.mark.asyncio
async def test_write_operation_marked_is_write_no_retry():
    """测试标记为 is_write 的自定义工具发生异常绝不重试"""
    registry = ToolRegistry()
    call_count = 0

    @tool
    def cancel_order(order_id: str) -> str:
        """模拟取消订单写操作"""
        nonlocal call_count
        call_count += 1
        raise ConnectionResetError("网络抖动中断")

    cancel_order.metadata = {"is_write": True}
    registry.register(cancel_order)

    mock_guard = MagicMock()
    mock_guard.check_permission.return_value = (True, None)
    executor = ToolExecutor(registry=registry, timeout=1.0, max_retries=1, permission_guard=mock_guard)
    res = await executor.execute({
        "name": "cancel_order",
        "args": {"order_id": "1001"},
        "id": "call_write_002",
    })

    assert res["success"] is False
    assert res["retry_count"] == 0
    assert call_count == 1
    assert res["error_type"] == "SYSTEM_FAULT"


@pytest.mark.asyncio
async def test_non_whitelisted_exception_not_retried():
    """测试非白名单异常（如业务 ValueError/RuntimeError）不触发重试"""
    registry = ToolRegistry()
    call_count = 0

    @tool
    def business_bug_tool(arg: str) -> str:
        """业务异常模拟"""
        nonlocal call_count
        call_count += 1
        raise ValueError(f"无效参数值业务异常: {arg}")

    registry.register(business_bug_tool)
    executor = ToolExecutor(registry=registry, max_retries=1)

    res = await executor.execute({
        "name": "business_bug_tool",
        "args": {"arg": "bad_val"},
        "id": "call_nowhitelist_001",
    })

    assert res["success"] is False
    assert res["status"] == "失败"
    assert res["error_type"] == "SYSTEM_FAULT"
    assert res["retry_count"] == 0
    assert call_count == 1
    assert "ValueError" in res["error"]


@pytest.mark.asyncio
async def test_enum_translation_and_chinese_formatting():
    """测试内部枚举人话翻译（SHIPPED->已发货, WARRANTY_ACTIVE->在保）以及中文不被 Unicode 转义"""
    registry = ToolRegistry()

    @tool
    def check_status(order_id: str) -> str:
        """检查订单与售后状态"""
        data = {
            "order_id": order_id,
            "order_status": "SHIPPED",
            "warranty_status": "WARRANTY_ACTIVE",
            "sign_status": "SIGNED",
            "chinese_text": "极简保暖羽绒服",
        }
        return json.dumps(data, ensure_ascii=False)

    registry.register(check_status)
    executor = ToolExecutor(registry=registry)

    res = await executor.execute({
        "name": "check_status",
        "args": {"order_id": "1001"},
        "id": "call_enum_001",
    })

    assert res["success"] is True
    assert res["status"] == "成功"
    output = res["output"]
    assert "已发货" in output
    assert "SHIPPED" not in output
    assert "在保" in output
    assert "WARRANTY_ACTIVE" not in output
    assert "已签收" in output
    assert "极简保暖羽绒服" in output
    assert "\\u" not in output


@pytest.mark.asyncio
async def test_return_structure_specification():
    """测试返回值字段完全符合规范，包含所有 9 个字段"""
    executor = ToolExecutor(default_tool_registry)
    res = await executor.execute({
        "name": "query_order",
        "args": {"order_id": "1001"},
        "id": "call_spec_001",
    })

    required_keys = {
        "success",
        "output",
        "tool_name",
        "tool_call_id",
        "status",
        "error",
        "error_type",
        "retry_count",
        "duration_ms",
    }
    assert required_keys.issubset(res.keys())
    assert res["success"] is True
    assert res["status"] == "成功"
    assert res["error"] is None
    assert res["error_type"] is None
    assert res["retry_count"] == 0
    assert isinstance(res["duration_ms"], int)
    assert res["duration_ms"] >= 0
