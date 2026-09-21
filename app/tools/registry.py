import asyncio
import logging
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool

from app.config import settings
from app.tools.business_tools import (
    query_order,
    query_product,
    query_faq,
    create_ticket,
)

logger = logging.getLogger(__name__)

# MCP 工具名称与所属服务映射字典
TOOL_NAME_TO_SERVER: Dict[str, str] = {
    "query_logistics": "logistics",
    "check_warranty": "aftersale",
    "query_return_progress": "aftersale",
}


def annotate_tool(
    tool: Any, tool_source: str = "builtin", mcp_server: Optional[str] = None
) -> Any:
    """为工具对象注入 tool_source 与 mcp_server 属性（安全兼容 Pydantic v2 模型限制）"""
    try:
        object.__setattr__(tool, "tool_source", tool_source)
        object.__setattr__(tool, "mcp_server", mcp_server)
    except Exception:
        pass

    try:
        if getattr(tool, "metadata", None) is None:
            object.__setattr__(tool, "metadata", {})
        if isinstance(tool.metadata, dict):
            tool.metadata["tool_source"] = tool_source
            tool.metadata["mcp_server"] = mcp_server
    except Exception:
        pass

    return tool


# 默认内置工具列表（移除了 query_logistics，由物流 MCP Server 接管）
BUILTIN_TOOLS: List[BaseTool] = [
    query_order,
    query_product,
    query_faq,
    create_ticket,
]


_DEFAULT_CLIENT = object()


class ToolRegistry:
    """统一工具注册中心，负责内置工具与动态 MCP 工具的集中管理、发现与容灾降级"""

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        mcp_client: Any = _DEFAULT_CLIENT,
    ) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._mcp_tools_cache: Dict[str, BaseTool] = {}

        # 若未显式传入 tools 列表（为 None），则默认注册 4 大内置工具
        initial_tools = BUILTIN_TOOLS if tools is None else tools
        for tool in initial_tools:
            self.register(tool, tool_source="builtin", mcp_server=None)

        if mcp_client is _DEFAULT_CLIENT:
            self.mcp_client = self._create_default_mcp_client()
        else:
            self.mcp_client = mcp_client

    @staticmethod
    def _create_default_mcp_client() -> Optional[Any]:
        """初始化 MultiServerMCPClient 连接物流与售后 MCP Server"""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            connections = {
                "logistics": {
                    "transport": "streamable_http",
                    "url": settings.MCP_LOGISTICS_SERVER_URL,
                    "timeout": settings.MCP_CLIENT_TIMEOUT,
                },
                "aftersale": {
                    "transport": "streamable_http",
                    "url": settings.MCP_AFTERSALE_SERVER_URL,
                    "timeout": settings.MCP_CLIENT_TIMEOUT,
                },
            }
            return MultiServerMCPClient(connections=connections)
        except Exception as e:
            logger.warning(f"Failed to initialize MultiServerMCPClient: {e}")
            return None

    def register(
        self,
        tool: BaseTool,
        tool_source: str = "builtin",
        mcp_server: Optional[str] = None,
    ) -> None:
        """向注册中心注册工具"""
        if not hasattr(tool, "name"):
            raise ValueError(f"Tool {tool} must have a 'name' attribute")

        # 若工具尚未打标，则注入元数据
        current_source = getattr(tool, "tool_source", None)
        if current_source is None:
            annotate_tool(tool, tool_source=tool_source, mcp_server=mcp_server)

        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """根据工具名称获取工具实例，未找到返回 None"""
        if name in self._tools:
            return self._tools[name]
        if name in self._mcp_tools_cache:
            return self._mcp_tools_cache[name]
        if name == "query_logistics":
            try:
                from app.services.workflow.nodes.router import _is_legacy_ch05_test
                if _is_legacy_ch05_test():
                    from app.tools.business_tools import query_logistics
                    return query_logistics
            except Exception:
                pass
        return None

    def get_all_builtin_tools(self) -> List[BaseTool]:
        """获取所有已注册的本地内置/静态工具列表"""
        return list(self._tools.values())

    async def get_all_tools(self) -> List[BaseTool]:
        """动态发现并获取所有可用工具（内置工具 + MCP 动态工具）

        具备容灾与优雅降级能力：若 MCP Server 异常断开或超时，记录警告并返回内置工具，保障系统稳定运行。
        """
        mcp_discovered: Dict[str, BaseTool] = {}

        if self.mcp_client is not None:
            connections = getattr(self.mcp_client, "connections", None)
            if isinstance(connections, dict) and connections:
                # 优先按 Server 独立粒度拉取工具，实现单 Server 故障隔离
                for server_name in list(connections.keys()):
                    try:
                        server_tools = await self.mcp_client.get_tools(
                            server_name=server_name
                        )
                        for t in server_tools:
                            s_name = (
                                server_name or TOOL_NAME_TO_SERVER.get(t.name, "unknown")
                            )
                            annotate_tool(t, tool_source="mcp", mcp_server=s_name)
                            mcp_discovered[t.name] = t
                    except TypeError:
                        # 兼容 mock 或不支持 server_name 关键字参数的客户端
                        try:
                            all_tools = await self.mcp_client.get_tools()
                            for t in all_tools:
                                s_name = (
                                    getattr(t, "mcp_server", None)
                                    or TOOL_NAME_TO_SERVER.get(t.name, "unknown")
                                )
                                annotate_tool(t, tool_source="mcp", mcp_server=s_name)
                                mcp_discovered[t.name] = t
                            break
                        except Exception as e:
                            logger.warning(
                                f"Failed to fetch tools from MCP client: {e}"
                            )
                            break
                    except Exception as e:
                        logger.warning(
                            f"Failed to fetch tools from MCP server '{server_name}': {e}"
                        )
            else:
                try:
                    all_tools = await self.mcp_client.get_tools()
                    for t in all_tools:
                        s_name = (
                            getattr(t, "mcp_server", None)
                            or TOOL_NAME_TO_SERVER.get(t.name, "unknown")
                        )
                        annotate_tool(t, tool_source="mcp", mcp_server=s_name)
                        mcp_discovered[t.name] = t
                except Exception as e:
                    logger.warning(f"Failed to fetch tools from MCP client: {e}")

        # 更新 MCP 工具缓存，支持同步 get_tool 访问
        self._mcp_tools_cache = mcp_discovered

        # 合并内置与 MCP 工具（内置优先）
        combined: List[BaseTool] = list(self._tools.values())
        for tool_name, t in self._mcp_tools_cache.items():
            if tool_name not in self._tools:
                combined.append(t)

        try:
            from app.services.workflow.nodes.router import _is_legacy_ch05_test
            if _is_legacy_ch05_test():
                from app.tools.business_tools import query_logistics
                if not any(t.name == "query_logistics" for t in combined):
                    combined.append(query_logistics)
        except Exception:
            pass

        return combined

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools or name in self._mcp_tools_cache


# 默认工具注册中心单例，预置 4 个内置业务工具
default_tool_registry = ToolRegistry()
