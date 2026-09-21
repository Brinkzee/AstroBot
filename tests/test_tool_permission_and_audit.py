import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.tools import tool

from app.tools.registry import ToolRegistry, default_tool_registry
from app.tools.executor import ToolExecutor
from app.tools.permission import ToolPermissionGuard
from app.tools.audit import AuditLogger
from app.models.tool_audit_log import ToolAuditLog


# ==========================================
# 1. ToolPermissionGuard 单元测试
# ==========================================

def test_permission_guard_mcp_write_rejected():
    """验证外部 MCP 工具尝试执行写操作被严密拦截"""
    guard = ToolPermissionGuard()
    allowed, reason = guard.check_permission(
        tool_name="update_tracking",
        tool_source="mcp",
        is_write=True,
    )
    assert allowed is False
    assert reason == "外部 MCP 工具一律禁止执行写操作 (权限拒绝)"


def test_permission_guard_mcp_read_allowed():
    """验证外部 MCP 工具执行只读操作正常放行"""
    guard = ToolPermissionGuard()
    allowed, reason = guard.check_permission(
        tool_name="query_logistics",
        tool_source="mcp",
        is_write=False,
    )
    assert allowed is True
    assert reason is None


def test_permission_guard_unauthorized_write_tool_rejected():
    """验证非 create_ticket 的写操作工具被拦截"""
    guard = ToolPermissionGuard()
    allowed, reason = guard.check_permission(
        tool_name="delete_order",
        tool_source="builtin",
        is_write=True,
    )
    assert allowed is False
    assert reason == "未授权的写操作工具 (权限拒绝)"


def test_permission_guard_create_ticket_without_user_request_rejected():
    """验证用户未明确表达建工单诉求时调用 create_ticket 被拦截"""
    guard = ToolPermissionGuard()
    # 1. context 为空
    allowed, reason = guard.check_permission(
        tool_name="create_ticket",
        tool_source="builtin",
        is_write=True,
        context=None,
    )
    assert allowed is False
    assert reason == "客户未明确要求建工单，禁止擅自发起 (权限拒绝)"

    # 2. context 提问中不含白名单关键词
    allowed, reason = guard.check_permission(
        tool_name="create_ticket",
        tool_source="builtin",
        is_write=True,
        context={"user_query": "查询一下羽绒服多少钱", "confirmed": True},
    )
    assert allowed is False
    assert reason == "客户未明确要求建工单，禁止擅自发起 (权限拒绝)"


def test_permission_guard_create_ticket_without_confirmation_rejected():
    """验证用户有诉求但前端未授权确认（confirmed!=True）时调用 create_ticket 被拦截"""
    guard = ToolPermissionGuard()
    # 1. confirmed 为 False
    allowed, reason = guard.check_permission(
        tool_name="create_ticket",
        tool_source="builtin",
        is_write=True,
        context={"user_query": "请帮我转人工客服处理", "confirmed": False},
    )
    assert allowed is False
    assert reason == "建工单操作未获前端用户确认授权 (权限拒绝)"

    # 2. 缺失 confirmed 字段
    allowed, reason = guard.check_permission(
        tool_name="create_ticket",
        tool_source="builtin",
        is_write=True,
        context={"user_query": "我要投诉这个商品质量"},
    )
    assert allowed is False
    assert reason == "建工单操作未获前端用户确认授权 (权限拒绝)"


def test_permission_guard_create_ticket_allowed_when_requested_and_confirmed():
    """验证用户明确诉求（包含工单/建单/人工/专员/投诉）且确认凭据齐全时放行"""
    guard = ToolPermissionGuard()
    for kw in ["我要建工单", "帮我建单", "转人工处理", "找专员跟进", "我想投诉"]:
        allowed, reason = guard.check_permission(
            tool_name="create_ticket",
            tool_source="builtin",
            is_write=True,
            context={"user_query": kw, "confirmed": True},
        )
        assert allowed is True, f"Keyword '{kw}' should be allowed"
        assert reason is None


# ==========================================
# 2. AuditLogger 单元测试
# ==========================================

@pytest.mark.asyncio
async def test_audit_logger_log_call_success():
    """验证 AuditLogger 正确向数据库会话添加 ToolAuditLog 并 commit"""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None
    mock_session_factory = MagicMock(return_value=mock_ctx)

    logger = AuditLogger(session_factory=mock_session_factory)
    log_entry = await logger.log_call(
        tool_name="query_order",
        tool_source="builtin",
        status="成功",
        conversation_id=101,
        tool_call_id="call_audit_01",
        mcp_server=None,
        arguments={"order_id": "1001"},
        result_summary="已查询到订单",
        error_message=None,
        retry_count=0,
        duration_ms=45,
    )

    assert log_entry is not None
    mock_session.add.assert_called_once()
    added = mock_session.add.call_args[0][0]
    assert isinstance(added, ToolAuditLog)
    assert added.tool_name == "query_order"
    assert added.tool_source == "builtin"
    assert added.status == "成功"
    assert added.conversation_id == 101
    assert added.tool_call_id == "call_audit_01"
    assert added.arguments == {"order_id": "1001"}
    assert added.result_summary == "已查询到订单"
    assert added.duration_ms == 45
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_audit_logger_truncation_for_long_summary_and_error():
    """验证超过 500 字符的 result_summary 及超长 error_message 安全截断"""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None
    mock_session_factory = MagicMock(return_value=mock_ctx)

    logger = AuditLogger(session_factory=mock_session_factory)
    long_summary = "A" * 600
    long_error = "E" * 700

    log_entry = await logger.log_call(
        tool_name="query_faq",
        result_summary=long_summary,
        error_message=long_error,
    )

    assert log_entry is not None
    added = mock_session.add.call_args[0][0]
    assert len(added.result_summary) == 500
    assert len(added.error_message) == 512


@pytest.mark.asyncio
async def test_audit_logger_exception_never_raises():
    """验证数据库异常时 AuditLogger 仅记录日志，绝不抛出异常"""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit.side_effect = RuntimeError("模拟数据库连接中断崩溃")
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None
    mock_session_factory = MagicMock(return_value=mock_ctx)

    logger = AuditLogger(session_factory=mock_session_factory)
    # 不应抛出异常，返回 None
    res = await logger.log_call(
        tool_name="query_order",
        status="失败",
        error_message="执行异常",
    )
    assert res is None


# ==========================================
# 3. ToolExecutor 装配与全流程集成测试
# ==========================================

@pytest.mark.asyncio
async def test_executor_permission_denied_mcp_write():
    """验证执行器拦截 MCP 写操作，返回 status='权限拒绝'，error_type='PERMISSION_DENIED'，并落审计"""
    registry = ToolRegistry(tools=[])

    @tool
    def mcp_update_tool(data: str) -> str:
        """模拟外部 MCP 写操作工具"""
        return f"updated: {data}"

    # 注册为 MCP 工具且声明为写操作
    registry.register(mcp_update_tool, tool_source="mcp", mcp_server="logistics")
    mcp_update_tool.metadata = {"is_write": True}

    mock_audit = MagicMock(spec=AuditLogger)
    mock_audit.log_call = AsyncMock(return_value=None)

    executor = ToolExecutor(
        registry=registry,
        permission_guard=ToolPermissionGuard(),
        audit_logger=mock_audit,
    )

    res = await executor.execute({
        "name": "mcp_update_tool",
        "args": {"data": "test"},
        "id": "call_mcp_write_01",
    })

    assert res["success"] is False
    assert res["status"] == "权限拒绝"
    assert res["error_type"] == "PERMISSION_DENIED"
    assert "外部 MCP 工具一律禁止执行写操作 (权限拒绝)" in res["error"]
    assert "权限校验拒绝" in res["output"] or "外部 MCP 工具一律禁止执行写操作" in res["output"]

    # 验证审计留痕落库一条「权限拒绝」记录
    mock_audit.log_call.assert_awaited_once()
    audit_kws = mock_audit.log_call.call_args[1]
    assert audit_kws["tool_name"] == "mcp_update_tool"
    assert audit_kws["tool_source"] == "mcp"
    assert audit_kws["status"] == "权限拒绝"
    assert "外部 MCP 工具一律禁止执行写操作" in audit_kws["error_message"]


@pytest.mark.asyncio
async def test_executor_permission_denied_create_ticket_unrequested():
    """验证无明确诉求调用 create_ticket 被权限门禁拦截并记录审计"""
    mock_audit = MagicMock(spec=AuditLogger)
    mock_audit.log_call = AsyncMock(return_value=None)

    executor = ToolExecutor(
        registry=default_tool_registry,
        permission_guard=ToolPermissionGuard(),
        audit_logger=mock_audit,
    )

    res = await executor.execute(
        tool_call={
            "name": "create_ticket",
            "args": {"conversation_id": 1, "description": "衣服不合身"},
            "id": "call_ticket_unreq",
        },
        context={"user_query": "衣服不合身怎么办"},
    )

    assert res["success"] is False
    assert res["status"] == "权限拒绝"
    assert res["error_type"] == "PERMISSION_DENIED"
    assert "客户未明确要求建工单，禁止擅自发起 (权限拒绝)" in res["error"]

    mock_audit.log_call.assert_awaited_once()
    audit_kws = mock_audit.log_call.call_args[1]
    assert audit_kws["status"] == "权限拒绝"
    assert "客户未明确要求建工单" in audit_kws["error_message"]


@pytest.mark.asyncio
async def test_executor_permission_denied_create_ticket_unconfirmed():
    """验证有诉求但未确认直接调用 create_ticket 被权限门禁拦截并记录审计"""
    mock_audit = MagicMock(spec=AuditLogger)
    mock_audit.log_call = AsyncMock(return_value=None)

    executor = ToolExecutor(
        registry=default_tool_registry,
        permission_guard=ToolPermissionGuard(),
        audit_logger=mock_audit,
    )

    res = await executor.execute(
        tool_call={
            "name": "create_ticket",
            "args": {"conversation_id": 1, "description": "衣服不合身申请人工处理"},
            "id": "call_ticket_unconf",
        },
        context={"user_query": "帮我转人工客服", "confirmed": False},
    )

    assert res["success"] is False
    assert res["status"] == "权限拒绝"
    assert res["error_type"] == "PERMISSION_DENIED"
    assert "建工单操作未获前端用户确认授权 (权限拒绝)" in res["error"]

    mock_audit.log_call.assert_awaited_once()
    audit_kws = mock_audit.log_call.call_args[1]
    assert audit_kws["status"] == "权限拒绝"
    assert "未获前端用户确认授权" in audit_kws["error_message"]


@pytest.mark.asyncio
async def test_executor_all_statuses_audited():
    """验证全量执行状态（成功、校验拦下、权限拒绝、超时、失败）均触发审计留痕记录"""
    registry = ToolRegistry(tools=[])

    @tool
    def succeed_tool(msg: str) -> str:
        """成功工具"""
        return f"ok: {msg}"

    @tool
    async def timeout_tool(seconds: float) -> str:
        """慢工具"""
        await asyncio.sleep(seconds)
        return "done"

    @tool
    def fail_tool() -> str:
        """失败工具"""
        raise ValueError("内部逻辑崩溃")

    registry.register(succeed_tool)
    registry.register(timeout_tool)
    registry.register(fail_tool)

    logged_statuses = []

    mock_audit = MagicMock(spec=AuditLogger)
    async def mock_log(**kwargs):
        logged_statuses.append(kwargs["status"])
        return None

    mock_audit.log_call = AsyncMock(side_effect=mock_log)

    executor = ToolExecutor(
        registry=registry,
        timeout=0.05,
        max_retries=0,
        permission_guard=ToolPermissionGuard(),
        audit_logger=mock_audit,
    )

    # 1. 成功 (成功)
    await executor.execute({
        "name": "succeed_tool",
        "args": {"msg": "hello"},
        "id": "c1",
    })

    # 2. 校验拦下 (校验拦下) - 缺少必填参数
    await executor.execute({
        "name": "succeed_tool",
        "args": {},
        "id": "c2",
    })

    # 3. 权限拒绝 (权限拒绝) - 未授权写操作
    await executor.execute(
        {
            "name": "succeed_tool",
            "args": {"msg": "write"},
            "id": "c3",
        },
        is_write=True,
    )

    # 4. 超时 (超时)
    await executor.execute({
        "name": "timeout_tool",
        "args": {"seconds": 0.2},
        "id": "c4",
    })

    # 5. 失败 (失败)
    await executor.execute({
        "name": "fail_tool",
        "args": {},
        "id": "c5",
    })

    assert logged_statuses == ["成功", "校验拦下", "权限拒绝", "超时", "失败"]


@pytest.mark.asyncio
async def test_executor_audit_failure_never_breaks_execution():
    """验证当 AuditLogger 发生严重异常时，ToolExecutor 正常完成，不抛出异常"""
    registry = ToolRegistry(tools=[])

    @tool
    def harmless_tool(param: str) -> str:
        """正常工具"""
        return f"result: {param}"

    registry.register(harmless_tool)

    mock_audit = MagicMock(spec=AuditLogger)
    mock_audit.log_call = AsyncMock(side_effect=RuntimeError("数据库宕机，审计写入全挂"))

    executor = ToolExecutor(
        registry=registry,
        permission_guard=ToolPermissionGuard(),
        audit_logger=mock_audit,
    )

    res = await executor.execute({
        "name": "harmless_tool",
        "args": {"param": "safe"},
        "id": "c_safe",
    })

    # 尽管审计抛出异常，执行器必须保持高可用正常返回
    assert res["success"] is True
    assert res["status"] == "成功"
    assert "result: safe" in res["output"]
