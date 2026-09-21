import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.tools import tool, BaseTool

import app.tools
from app.config import settings
from app.tools.registry import ToolRegistry, default_tool_registry


def test_config_mcp_settings():
    """验证 app/config.py 中新增的 MCP 配置项与大写属性"""
    assert hasattr(settings, "mcp_logistics_server_url")
    assert hasattr(settings, "mcp_aftersale_server_url")
    assert hasattr(settings, "mcp_client_timeout")
    assert settings.MCP_LOGISTICS_SERVER_URL == "http://127.0.0.1:8001/mcp"
    assert settings.MCP_AFTERSALE_SERVER_URL == "http://127.0.0.1:8002/mcp"
    assert settings.MCP_CLIENT_TIMEOUT == 5.0


def test_tools_init_exports():
    """验证 app/tools/__init__.py 的导出符号：包含 4 个内置工具与 Registry，下线 query_logistics"""
    assert hasattr(app.tools, "ToolRegistry")
    assert hasattr(app.tools, "default_tool_registry")
    assert hasattr(app.tools, "query_order")
    assert hasattr(app.tools, "query_product")
    assert hasattr(app.tools, "query_faq")
    assert hasattr(app.tools, "create_ticket")
    assert "query_logistics" not in app.tools.__all__


def test_builtin_tools_metadata_and_count():
    """验证默认注册中心仅保留 4 个内置工具，且打上 builtin 元数据标签"""
    builtin_tools = default_tool_registry.get_all_builtin_tools()
    tool_names = [t.name for t in builtin_tools]
    assert len(builtin_tools) == 4
    assert "query_order" in tool_names
    assert "query_product" in tool_names
    assert "query_faq" in tool_names
    assert "create_ticket" in tool_names
    assert "query_logistics" not in tool_names

    # 验证元数据标注
    for t in builtin_tools:
        assert getattr(t, "tool_source", None) == "builtin"
        assert getattr(t, "mcp_server", None) is None

    # 验证同步查询与向后兼容接口
    assert default_tool_registry.get_tool("query_order") is not None
    assert default_tool_registry.get_tool("query_logistics") is None
    assert "query_order" in default_tool_registry
    assert "query_logistics" not in default_tool_registry
    assert len(default_tool_registry) == 4


@pytest.mark.asyncio
async def test_dynamic_tool_registration():
    """验证通过 register 动态挂载新工具后可在 get_all_tools 中查见"""
    registry = ToolRegistry(tools=[])
    tools = await registry.get_all_tools()
    assert len(tools) == 0

    @tool
    def calculate_tax(amount: float) -> float:
        """计算税率"""
        return amount * 0.1

    registry.register(calculate_tax)
    updated_tools = await registry.get_all_tools()
    assert len(updated_tools) == 1
    assert updated_tools[0].name == "calculate_tax"
    assert getattr(updated_tools[0], "tool_source", None) == "builtin"
    assert registry.get_tool("calculate_tax") is not None


@pytest.mark.asyncio
async def test_mcp_client_dynamic_discovery_and_metadata_tagging():
    """验证 Mock MCP 客户端动态发现工具并正确打上 tool_source 与 mcp_server 标签"""
    @tool
    def mock_query_logistics(order_id: str) -> str:
        """查物流"""
        return "logistics_info"

    @tool
    def mock_check_warranty(order_id: str) -> str:
        """查质保"""
        return "warranty_info"

    @tool
    def mock_query_return(order_id: str) -> str:
        """查退货"""
        return "return_info"

    mock_client = MagicMock()
    mock_client.connections = {
        "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:8001/mcp"},
        "aftersale": {"transport": "streamable_http", "url": "http://127.0.0.1:8002/mcp"},
    }

    async def fake_get_tools(server_name=None):
        if server_name == "logistics":
            return [mock_query_logistics]
        elif server_name == "aftersale":
            return [mock_check_warranty, mock_query_return]
        return [mock_query_logistics, mock_check_warranty, mock_query_return]

    mock_client.get_tools = AsyncMock(side_effect=fake_get_tools)

    registry = ToolRegistry(mcp_client=mock_client)
    tools = await registry.get_all_tools()

    tool_names = [t.name for t in tools]
    # 4 个内置 + 3 个 MCP
    assert len(tools) == 7
    assert "query_order" in tool_names
    assert "mock_query_logistics" in tool_names
    assert "mock_check_warranty" in tool_names
    assert "mock_query_return" in tool_names

    tool_map = {t.name: t for t in tools}
    assert getattr(tool_map["query_order"], "tool_source") == "builtin"
    assert getattr(tool_map["query_order"], "mcp_server") is None

    assert getattr(tool_map["mock_query_logistics"], "tool_source") == "mcp"
    assert getattr(tool_map["mock_query_logistics"], "mcp_server") == "logistics"

    assert getattr(tool_map["mock_check_warranty"], "tool_source") == "mcp"
    assert getattr(tool_map["mock_check_warranty"], "mcp_server") == "aftersale"

    assert getattr(tool_map["mock_query_return"], "tool_source") == "mcp"
    assert getattr(tool_map["mock_query_return"], "mcp_server") == "aftersale"

    # 验证 get_tool 也能获取到动态 MCP 工具
    assert registry.get_tool("mock_query_logistics") is not None


@pytest.mark.asyncio
async def test_mcp_client_all_servers_offline_graceful_degradation(caplog):
    """验证所有 MCP Server 离线/异常时优雅降级返回可用内置工具，不抛异常"""
    mock_client = MagicMock()
    mock_client.connections = {
        "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:8001/mcp"},
        "aftersale": {"transport": "streamable_http", "url": "http://127.0.0.1:8002/mcp"},
    }
    mock_client.get_tools = AsyncMock(side_effect=ConnectionRefusedError("MCP Server 无法连接"))

    registry = ToolRegistry(mcp_client=mock_client)
    with caplog.at_level(logging.WARNING):
        tools = await registry.get_all_tools()

    assert len(tools) == 4
    tool_names = [t.name for t in tools]
    assert "query_order" in tool_names
    assert "create_ticket" in tool_names
    assert any("MCP" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_mcp_client_partial_server_failure_graceful_degradation(caplog):
    """验证单 MCP Server (如物流) 崩溃时，另一个 Server (售后) 正常返回，内置工具不受影响"""
    @tool
    def mock_check_warranty(order_id: str) -> str:
        """查质保"""
        return "warranty_info"

    mock_client = MagicMock()
    mock_client.connections = {
        "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:8001/mcp"},
        "aftersale": {"transport": "streamable_http", "url": "http://127.0.0.1:8002/mcp"},
    }

    async def fake_get_tools(server_name=None):
        if server_name == "logistics":
            raise TimeoutError("物流 MCP Server 连接超时")
        elif server_name == "aftersale":
            return [mock_check_warranty]
        return [mock_check_warranty]

    mock_client.get_tools = AsyncMock(side_effect=fake_get_tools)

    registry = ToolRegistry(mcp_client=mock_client)
    with caplog.at_level(logging.WARNING):
        tools = await registry.get_all_tools()

    tool_names = [t.name for t in tools]
    # 4 个内置 + 1 个售后的 mock_check_warranty = 5 个工具
    assert len(tools) == 5
    assert "query_order" in tool_names
    assert "mock_check_warranty" in tool_names
    assert getattr(tools[-1], "mcp_server") == "aftersale"
    assert any("logistics" in record.message for record in caplog.records)
