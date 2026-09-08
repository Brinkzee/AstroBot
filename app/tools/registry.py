from typing import Dict, List, Optional
from langchain_core.tools import BaseTool

from app.tools.business_tools import (
    query_order,
    query_product,
    query_logistics,
    query_faq,
    create_ticket,
)


class ToolRegistry:
    """工具注册中心，负责工具的集中注册、检索与发现"""

    def __init__(self, tools: Optional[List[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """注册一个工具"""
        if not hasattr(tool, "name"):
            raise ValueError(f"Tool {tool} must have a 'name' attribute")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """根据工具名称获取工具实例，未找到返回 None"""
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        """获取所有已注册工具列表"""
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# 默认注册中心，预置全部 5 个业务工具
default_tool_registry = ToolRegistry(
    tools=[
        query_order,
        query_product,
        query_logistics,
        query_faq,
        create_ticket,
    ]
)
