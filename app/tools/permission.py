import logging
import re
from typing import Any, Dict, FrozenSet, Optional, Tuple

logger = logging.getLogger(__name__)

# 客户明确建单诉求关键词白名单
TICKET_INTENT_KEYWORDS: Tuple[str, ...] = (
    "工单",
    "建单",
    "人工",
    "专员",
    "投诉",
)

# 外部 MCP 只读工具显式白名单与只读前缀（默认拒绝未知/写操作工具）
ALLOWED_MCP_READ_TOOLS: FrozenSet[str] = frozenset({
    "query_logistics",
    "check_warranty",
    "query_return_progress",
})

ALLOWED_MCP_READ_PREFIXES: Tuple[str, ...] = (
    "query_",
    "check_",
    "get_",
    "list_",
    "search_",
    "fetch_",
    "lookup_",
    "read_",
    "find_",
    "describe_",
    "inspect_",
    "view_",
    "verify_",
    "calculate_",
    "estimate_",
    "evaluate_",
    "mock_query_",
    "mock_check_",
)

# 任何包含写/变更语义词元的 MCP 工具一律拒绝
MCP_MUTATING_TOKENS: FrozenSet[str] = frozenset({
    "create",
    "delete",
    "remove",
    "drop",
    "insert",
    "add",
    "update",
    "modify",
    "edit",
    "set",
    "put",
    "post",
    "patch",
    "write",
    "cancel",
    "refund",
    "issue",
    "approve",
    "reject",
    "transfer",
    "pay",
    "charge",
    "close",
    "reset",
    "purge",
    "exec",
    "execute",
    "run",
    "submit",
    "revoke",
})


def _is_allowed_mcp_readonly_tool(tool_name: str) -> Tuple[bool, Optional[str]]:
    """对外部 MCP 工具执行零信任只读白名单核验（默认拒绝未列明或带写语义的工具）。"""
    normalized = str(tool_name or "").strip().lower()
    if not normalized:
        return False, "外部 MCP 工具名非法，禁止执行 (权限拒绝)"

    tokens = {tok for tok in re.split(r"[^a-z0-9]+", normalized) if tok}
    if normalized == "create_ticket" or (tokens & MCP_MUTATING_TOKENS):
        return False, "外部 MCP 工具一律禁止执行写操作 (权限拒绝)"

    if normalized in ALLOWED_MCP_READ_TOOLS or normalized.startswith(ALLOWED_MCP_READ_PREFIXES):
        return True, None

    return False, f"外部 MCP 工具 [{tool_name}] 不在只读白名单中，禁止执行 (权限拒绝)"


class ToolPermissionGuard:
    """工具权限门禁：控制读写权限、零信任外部 MCP、唯一写操作 create_ticket 诉求与确认核验"""

    def check_permission(
        self,
        tool_name: str,
        tool_source: str = "builtin",
        is_write: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """检查工具调用权限
        
        Args:
            tool_name: 工具名称
            tool_source: 工具来源（'builtin' 或 'mcp'）
            is_write: 是否为写操作
            context: 会话上下文（包含 user_query, confirmed 凭据等）

        Returns:
            (allowed, deny_reason): 是否放行以及拒绝原因
        """
        normalized_source = str(tool_source or "builtin").strip().lower()

        # 1. 外部 MCP 零信任只读（不信任 MCP Server 声明的 is_write，强制白名单放行）
        if normalized_source == "mcp":
            if is_write or tool_name == "create_ticket":
                logger.warning(
                    f"Permission denied for MCP tool [{tool_name}]: external MCP tools are strictly read-only"
                )
                return False, "外部 MCP 工具一律禁止执行写操作 (权限拒绝)"
            allowed_mcp, mcp_reason = _is_allowed_mcp_readonly_tool(tool_name)
            if not allowed_mcp:
                logger.warning(
                    f"Permission denied for MCP tool [{tool_name}]: {mcp_reason}"
                )
                return False, mcp_reason
            return True, None

        # 2. 内置工具写操作白名单与门禁判定
        # create_ticket 本身具有物理写副作用，即使未显式标记 is_write 亦视为写操作
        effective_write = is_write or (tool_name == "create_ticket")

        if effective_write:
            # 2.1 唯一写操作白名单：本系统合法写操作只有 create_ticket
            if tool_name != "create_ticket":
                logger.warning(
                    f"Permission denied for tool [{tool_name}]: unauthorized write tool"
                )
                return False, "未授权的写操作工具 (权限拒绝)"

            # 2.2 create_ticket 门禁 1：诉求核验
            # 前置检查 context（包含当前轮 user_query/intent 及同会话用户历史发言 history_user_queries）；
            # 若通过工单预览确认流（preview_confirmed=True 且 confirmed=True），亦视为已获用户明确诉求确认。
            ctx = context or {}
            query_candidates = []
            for k in (
                "user_query",
                "query",
                "intent",
                "user_intent",
                "message",
                "text",
                "content",
                "prompt",
            ):
                val = ctx.get(k)
                if isinstance(val, str) and val.strip():
                    query_candidates.append(val)

            history_queries = ctx.get("history_user_queries")
            if isinstance(history_queries, (list, tuple)):
                for item in history_queries:
                    if isinstance(item, str) and item.strip():
                        query_candidates.append(item)

            # 若特定键未命中，尝试扫描 context 所有文本值
            context_text = " ".join(query_candidates)
            if not context_text:
                context_text = " ".join(
                    str(v) for v in ctx.values() if isinstance(v, str)
                )

            has_explicit_intent = any(kw in context_text for kw in TICKET_INTENT_KEYWORDS)
            preview_verified = (
                ctx.get("preview_confirmed") is True and ctx.get("confirmed") is True
            )

            if not has_explicit_intent and not preview_verified:
                logger.warning(
                    "Permission denied for create_ticket: user did not explicitly request ticket creation"
                )
                return False, "客户未明确要求建工单，禁止擅自发起 (权限拒绝)"

            # 2.3 create_ticket 门禁 2：前端确认凭据
            # 检查 context.get("confirmed")；若直接尝试物理执行而未携带确认放行凭证（confirmed=True），判定拒绝
            if ctx.get("confirmed") is not True:
                logger.warning(
                    "Permission denied for create_ticket: missing user confirmation"
                )
                return False, "建工单操作未获前端用户确认授权 (权限拒绝)"

            return True, None

        # 3. 普通只读操作放行
        return True, None


# 默认工具权限门禁单例
default_permission_guard = ToolPermissionGuard()
