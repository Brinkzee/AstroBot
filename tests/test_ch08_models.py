import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine, select, inspect, text
from sqlalchemy.orm import Session

from app.db.session import Base
import app.models
from app.models import ToolAuditLog
from app.models.tool_audit_log import ToolSource, ToolStatus
from scripts.init_ch08_db import split_sql_statements, parse_ddl_statements, init_ch08_db


def test_models_init_export():
    assert hasattr(app.models, "ToolAuditLog")
    assert "ToolAuditLog" in app.models.__all__
    assert app.models.ToolAuditLog is ToolAuditLog


def test_tool_audit_log_attributes_and_defaults():
    # 完整属性赋值测试
    log = ToolAuditLog(
        conversation_id=101,
        tool_call_id="call_test_001",
        tool_name="query_order",
        tool_source="builtin",
        mcp_server=None,
        arguments={"order_id": "SN-2026-001"},
        result_summary="已发货，承运商：顺丰速运",
        status="成功",
        error_message=None,
        retry_count=0,
        duration_ms=85,
    )
    assert log.conversation_id == 101
    assert log.tool_call_id == "call_test_001"
    assert log.tool_name == "query_order"
    assert log.tool_source == "builtin"
    assert log.mcp_server is None
    assert log.arguments == {"order_id": "SN-2026-001"}
    assert log.result_summary == "已发货，承运商：顺丰速运"
    assert log.status == "成功"
    assert log.error_message is None
    assert log.retry_count == 0
    assert log.duration_ms == 85
    assert log.id is None

    # 默认值与空值测试
    log_default = ToolAuditLog(
        tool_name="search_faq",
        tool_source="builtin",
        status="成功",
    )
    assert log_default.tool_name == "search_faq"
    assert log_default.tool_source == "builtin"
    assert log_default.status == "成功"
    assert log_default.conversation_id is None
    assert log_default.tool_call_id is None
    assert log_default.mcp_server is None
    assert log_default.arguments is None
    assert log_default.result_summary is None
    assert log_default.error_message is None
    assert log_default.retry_count == 0
    assert log_default.duration_ms is None


def test_models_table_metadata():
    assert ToolAuditLog.__tablename__ == "tool_audit_logs"

    table = ToolAuditLog.__table__
    cols = set(table.columns.keys())
    expected_cols = {
        "id",
        "conversation_id",
        "tool_call_id",
        "tool_name",
        "tool_source",
        "mcp_server",
        "arguments",
        "result_summary",
        "status",
        "error_message",
        "retry_count",
        "duration_ms",
        "created_at",
    }
    assert expected_cols.issubset(cols)

    # 主键断言
    assert [c.name for c in table.primary_key.columns] == ["id"]

    # 字段可空性与约束断言
    assert table.c.id.nullable is False
    assert table.c.conversation_id.nullable is True
    assert table.c.tool_call_id.nullable is True
    assert table.c.tool_name.nullable is False
    assert table.c.tool_source.nullable is False
    assert table.c.mcp_server.nullable is True
    assert table.c.arguments.nullable is True
    assert table.c.result_summary.nullable is True
    assert table.c.status.nullable is False
    assert table.c.error_message.nullable is True
    assert table.c.retry_count.nullable is False
    assert table.c.duration_ms.nullable is True
    assert table.c.created_at.nullable is False

    # 审计表无外键约束断言（conversation_id 仅普通索引，不能挂外键拦截审计）
    assert len(table.foreign_keys) == 0

    # 索引断言 (idx_conversation_id, idx_tool_name, idx_status)
    index_cols = {idx.name: [c.name for c in idx.columns] for idx in table.indexes}
    assert "idx_conversation_id" in index_cols
    assert index_cols["idx_conversation_id"] == ["conversation_id"]
    assert "idx_tool_name" in index_cols
    assert index_cols["idx_tool_name"] == ["tool_name"]
    assert "idx_status" in index_cols
    assert index_cols["idx_status"] == ["status"]


def test_sqlite_in_memory_crud():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # 写入内置工具审计
        log1 = ToolAuditLog(
            conversation_id=1,
            tool_call_id="call_builtin_01",
            tool_name="order_query",
            tool_source="builtin",
            arguments={"order_id": "ORD-12345"},
            result_summary="已发货",
            status="成功",
            retry_count=0,
            duration_ms=42,
        )
        # 写入 MCP 工具审计（失败与错误信息）
        log2 = ToolAuditLog(
            conversation_id=1,
            tool_call_id="call_mcp_01",
            tool_name="shipping_tracker",
            tool_source="mcp",
            mcp_server="logistics_server",
            arguments={"tracking_no": "SF99999"},
            status="校验拦下",
            error_message="快递单号格式校验未通过",
            retry_count=0,
            duration_ms=12,
        )
        session.add_all([log1, log2])
        session.commit()

        assert log1.id is not None
        assert log2.id is not None

        # 查询断言
        saved_logs = session.scalars(
            select(ToolAuditLog).where(ToolAuditLog.conversation_id == 1).order_by(ToolAuditLog.id)
        ).all()
        assert len(saved_logs) == 2
        assert saved_logs[0].tool_name == "order_query"
        assert saved_logs[0].arguments == {"order_id": "ORD-12345"}
        assert saved_logs[0].tool_source == "builtin"
        assert saved_logs[0].status == "成功"

        assert saved_logs[1].tool_source == "mcp"
        assert saved_logs[1].mcp_server == "logistics_server"
        assert saved_logs[1].status == "校验拦下"
        assert saved_logs[1].error_message == "快递单号格式校验未通过"


def test_parse_ddl_statements():
    ddl_path = Path(__file__).resolve().parent.parent / "sql" / "ch08-ddl.sql"
    assert ddl_path.exists()
    statements = parse_ddl_statements(ddl_path)
    assert len(statements) >= 2

    set_stmts = [s for s in statements if s.upper().startswith("SET ")]
    assert len(set_stmts) >= 1
    assert "utf8mb4" in set_stmts[0].lower()

    create_stmts = [s for s in statements if "CREATE TABLE" in s.upper()]
    assert len(create_stmts) == 1
    assert "IF NOT EXISTS" in create_stmts[0].upper()
    assert "tool_audit_logs" in create_stmts[0]


@pytest.mark.asyncio
async def test_init_ch08_db_with_mock_engine():
    mock_engine = MagicMock()
    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_engine.dispose = AsyncMock()

    executed = await init_ch08_db(engine_override=mock_engine)
    assert len(executed) >= 2
    assert mock_conn.execute.call_count >= 2


@pytest.mark.asyncio
async def test_init_ch08_db_sqlite_migration_idempotent():
    engine = create_engine("sqlite:///:memory:")

    # 首次执行建表
    executed1 = await init_ch08_db(engine_override=engine)
    assert len(executed1) >= 2

    insp = inspect(engine)
    tables = insp.get_table_names()
    assert "tool_audit_logs" in tables

    cols = {c["name"] for c in insp.get_columns("tool_audit_logs")}
    expected = {
        "id", "conversation_id", "tool_call_id", "tool_name",
        "tool_source", "mcp_server", "arguments", "result_summary",
        "status", "error_message", "retry_count", "duration_ms", "created_at"
    }
    assert expected.issubset(cols)

    # 验证能直接往表中写数据
    with engine.begin() as conn:
        conn.execute(text(
            """
            INSERT INTO tool_audit_logs (tool_name, tool_source, status, retry_count)
            VALUES ('test_tool', 'builtin', '成功', 0)
            """
        ))
        res = conn.execute(text("SELECT COUNT(*) FROM tool_audit_logs")).scalar()
        assert res == 1

    # 二次执行幂等性测试（不报错且表数据保持）
    executed2 = await init_ch08_db(engine_override=engine)
    assert len(executed2) >= 2

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM tool_audit_logs")).scalar()
        assert count == 1
