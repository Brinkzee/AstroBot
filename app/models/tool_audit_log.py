import enum
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from sqlalchemy import (
    BigInteger,
    Integer,
    SmallInteger,
    String,
    Text,
    Enum,
    DateTime,
    JSON,
    Index,
    func,
)
from sqlalchemy.dialects.mysql import (
    BIGINT as MYSQL_BIGINT,
    INTEGER as MYSQL_INTEGER,
    TINYINT as MYSQL_TINYINT,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ToolSource(str, enum.Enum):
    """工具来源类型枚举。"""
    BUILTIN = "builtin"
    MCP = "mcp"


class ToolStatus(str, enum.Enum):
    """工具调用执行状态枚举。"""
    SUCCESS = "成功"
    FAILED = "失败"
    TIMEOUT = "超时"
    VALIDATION_FAILED = "校验拦下"
    PERMISSION_DENIED = "权限拒绝"


class ToolAuditLog(Base):
    """工具调用审计留痕 ORM 实体。

    对齐 sql/ch08-ddl.sql: tool_audit_logs 表。
    统一执行引擎每次调用落一条，内置和 MCP 工具都记，被权限拒、被校验拦的调用同样落一条。
    审计写入不能被引用约束拦住，conversation_id 仅建普通索引，不挂外键。
    """
    __tablename__ = "tool_audit_logs"
    __table_args__ = (
        Index("idx_conversation_id", "conversation_id"),
        Index("idx_tool_name", "tool_name"),
        Index("idx_status", "status"),
        {"comment": "工具调用审计留痕"},
    )

    def __init__(self, **kwargs):
        kwargs.setdefault("retry_count", 0)
        super().__init__(**kwargs)

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="审计主键",
    )
    conversation_id: Mapped[Optional[int]] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        nullable=True,
        comment="所属会话,无会话上下文的调用为 NULL",
    )
    tool_call_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="模型申请单 id,可对回 messages 流水",
    )
    tool_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="工具名",
    )
    tool_source: Mapped[str] = mapped_column(
        Enum("builtin", "mcp", name="tool_source"),
        nullable=False,
        comment="工具来源:内置 / MCP 接入",
    )
    mcp_server: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="来源 MCP Server 名,内置工具为 NULL",
    )
    arguments: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="调用参数",
    )
    result_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="返回结果,过长截断存摘要",
    )
    status: Mapped[str] = mapped_column(
        Enum("成功", "失败", "超时", "校验拦下", "权限拒绝", name="tool_audit_status"),
        nullable=False,
        comment="本次调用结局",
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
        comment="失败 / 拦下时的原因说明",
    )
    retry_count: Mapped[int] = mapped_column(
        SmallInteger().with_variant(MYSQL_TINYINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        nullable=False,
        default=0,
        server_default="0",
        comment="实际重试次数,写操作默认不重试恒为 0",
    )
    duration_ms: Mapped[Optional[int]] = mapped_column(
        Integer().with_variant(MYSQL_INTEGER(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        nullable=True,
        comment="耗时毫秒",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        server_default=func.now(),
        nullable=False,
        comment="调用时间",
    )
