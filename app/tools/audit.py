import json
import logging
from typing import Any, Dict, Optional

from app.db.session import AsyncSessionLocal
from app.models.tool_audit_log import ToolAuditLog

logger = logging.getLogger(__name__)


class AuditLogger:
    """独立事务异步审计日志服务：全量记录工具调用流水，高可用容灾保障主干绝不阻断"""

    def __init__(self, session_factory: Optional[Any] = None) -> None:
        self.session_factory = session_factory or AsyncSessionLocal

    async def log_call(
        self,
        tool_name: str,
        tool_source: str = "builtin",
        status: str = "成功",
        conversation_id: Optional[int] = None,
        tool_call_id: Optional[str] = None,
        mcp_server: Optional[str] = None,
        arguments: Optional[Any] = None,
        result_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        retry_count: int = 0,
        duration_ms: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[ToolAuditLog]:
        """独立异步事务将调用记录持久化到 tool_audit_logs 表
        
        Args:
            tool_name: 工具名称
            tool_source: 工具来源（builtin / mcp）
            status: 本次调用结局（成功 / 失败 / 超时 / 校验拦下 / 权限拒绝）
            conversation_id: 所属会话 ID
            tool_call_id: 模型申请单 id
            mcp_server: 来源 MCP Server 名
            arguments: 调用参数（字典或可反序列化对象）
            result_summary: 返回结果（超过 500 字符自动安全截断）
            error_message: 失败或拦下时的原因说明（超过 512 字符安全截断）
            retry_count: 实际重试次数
            duration_ms: 耗时毫秒
        """
        try:
            # 1. 摘要与错误安全截断
            if result_summary is not None and len(result_summary) > 500:
                result_summary = result_summary[:500]

            if error_message is not None and len(error_message) > 512:
                error_message = error_message[:512]

            # 2. 参数 JSON 结构安全规整
            parsed_arguments = arguments
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except Exception:
                    parsed_arguments = {"raw": arguments}

            # 3. 独立事务落盘
            async with self.session_factory() as session:
                log_entry = ToolAuditLog(
                    conversation_id=conversation_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_source=tool_source,
                    mcp_server=mcp_server,
                    arguments=parsed_arguments,
                    result_summary=result_summary,
                    status=status,
                    error_message=error_message,
                    retry_count=retry_count,
                    duration_ms=duration_ms,
                )
                session.add(log_entry)
                await session.commit()
                return log_entry

        except Exception as e:
            # 数据库异常仅记录告警日志，绝不抛出异常阻塞主干工具执行
            logger.error(
                f"AuditLogger failed to persist tool audit log for [{tool_name}]: {e}",
                exc_info=True,
            )
            return None


# 默认审计日志服务单例
default_audit_logger = AuditLogger()
