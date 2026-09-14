import os
from unittest import mock
import pytest
from langchain_core.tools import tool

import run
from app.tools.registry import annotate_tool


class TestRunCh08StorageReadiness:
    """测试 run.py 中 check_storage_readiness 流程与 ch08 DDL 迁移集成"""

    @pytest.mark.asyncio
    async def test_storage_readiness_calls_init_ch08_db(self, capsys):
        """验证 check_storage_readiness 能够自动调用 init_ch08_db() 并输出成功提示"""
        mock_init_ch07 = mock.AsyncMock(return_value=["ALTER TABLE conversations ..."])
        mock_init_ch08 = mock.AsyncMock(return_value=["CREATE TABLE tool_audit_logs ..."])
        mock_session = mock.AsyncMock()
        mock_res_pending = mock.MagicMock()
        mock_res_pending.scalar.return_value = 0
        mock_res_done = mock.MagicMock()
        mock_res_done.scalar.return_value = 10
        mock_session.execute.side_effect = [mock_res_pending, mock_res_done]

        mock_session_ctx = mock.AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_session

        mock_store = mock.MagicMock()
        mock_store.count.return_value = 10

        mock_engine = mock.MagicMock()
        mock_conn = mock.AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        mock_engine.dispose = mock.AsyncMock()

        with (
            mock.patch("scripts.wsl_helper.ensure_mysql_ready"),
            mock.patch("app.db.session.engine", mock_engine),
            mock.patch("scripts.seed_data.seed_all_data", new_callable=mock.AsyncMock, return_value=0),
            mock.patch("scripts.init_ch07_db.init_ch07_db", mock_init_ch07),
            mock.patch("scripts.init_ch08_db.init_ch08_db", mock_init_ch08),
            mock.patch("app.db.session.AsyncSessionLocal", return_value=mock_session_ctx),
            mock.patch("app.services.rag.milvus_client.MilvusKnowledgeStore", return_value=mock_store),
            mock.patch("scripts.wsl_helper.is_port_open", return_value=False),
        ):
            ok = await run.check_storage_readiness(clean_kb=False)
            assert ok is True
            mock_init_ch08.assert_awaited_once()
            out = capsys.readouterr().out
            assert "第八章工具调用审计数据表 (tool_audit_logs) 迁移已就绪 ✅" in out

    @pytest.mark.asyncio
    async def test_storage_readiness_init_ch08_db_failure_intercepts(self, capsys):
        """验证当 init_ch08_db 异常时，check_storage_readiness() 拦截并返回 False"""
        mock_init_ch07 = mock.AsyncMock(return_value=["ALTER TABLE conversations ..."])
        mock_init_ch08 = mock.AsyncMock(side_effect=RuntimeError("CH08 DDL table creation error"))
        mock_engine = mock.MagicMock()
        mock_conn = mock.AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        with (
            mock.patch("scripts.wsl_helper.ensure_mysql_ready"),
            mock.patch("app.db.session.engine", mock_engine),
            mock.patch("scripts.init_ch07_db.init_ch07_db", mock_init_ch07),
            mock.patch("scripts.init_ch08_db.init_ch08_db", mock_init_ch08),
        ):
            ok = await run.check_storage_readiness(clean_kb=False)
            assert ok is False
            mock_init_ch08.assert_awaited_once()
            out = capsys.readouterr().out
            assert "第八章数据表与迁移失败" in out or "CH08 DDL table creation error" in out


class TestRunCh08MCPAndToolRegistry:
    """测试 MCP 服务连通性探测与动态工具注册中心自检"""

    @pytest.mark.asyncio
    async def test_mcp_probe_and_tool_discovery_all_online(self, capsys):
        """验证当 MCP 8001/8002 在线时，探测输出正常且正确打印内置工具与动态 MCP 工具"""
        mock_init_ch07 = mock.AsyncMock(return_value=[])
        mock_init_ch08 = mock.AsyncMock(return_value=[])
        mock_session = mock.AsyncMock()
        mock_res_pending = mock.MagicMock()
        mock_res_pending.scalar.return_value = 0
        mock_res_done = mock.MagicMock()
        mock_res_done.scalar.return_value = 10
        mock_session.execute.side_effect = [mock_res_pending, mock_res_done]
        mock_session_ctx = mock.AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_session

        mock_store = mock.MagicMock()
        mock_store.count.return_value = 10

        mock_engine = mock.MagicMock()
        mock_conn = mock.AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        mock_engine.dispose = mock.AsyncMock()

        @tool
        def fake_mcp_tool(query: str) -> str:
            """fake mcp tool"""
            return "ok"

        annotate_tool(fake_mcp_tool, tool_source="mcp", mcp_server="logistics")

        @tool
        def fake_builtin_tool(query: str) -> str:
            """fake builtin tool"""
            return "ok"

        annotate_tool(fake_builtin_tool, tool_source="builtin", mcp_server=None)

        mock_get_all_tools = mock.AsyncMock(return_value=[fake_builtin_tool, fake_mcp_tool])

        with (
            mock.patch("scripts.wsl_helper.ensure_mysql_ready"),
            mock.patch("app.db.session.engine", mock_engine),
            mock.patch("scripts.seed_data.seed_all_data", new_callable=mock.AsyncMock, return_value=0),
            mock.patch("scripts.init_ch07_db.init_ch07_db", mock_init_ch07),
            mock.patch("scripts.init_ch08_db.init_ch08_db", mock_init_ch08),
            mock.patch("app.db.session.AsyncSessionLocal", return_value=mock_session_ctx),
            mock.patch("app.services.rag.milvus_client.MilvusKnowledgeStore", return_value=mock_store),
            mock.patch("scripts.wsl_helper.is_port_open", return_value=True),
            mock.patch("app.tools.registry.default_tool_registry.get_all_tools", mock_get_all_tools),
        ):
            ok = await run.check_storage_readiness(clean_kb=False)
            assert ok is True
            out = capsys.readouterr().out
            assert "物流 MCP 服务" in out and "在线" in out
            assert "售后 MCP 服务" in out and "在线" in out
            assert "工具注册中心" in out
            assert "fake_mcp_tool" in out or "MCP" in out

    @pytest.mark.asyncio
    async def test_mcp_probe_and_tool_discovery_offline_fallback(self, capsys):
        """验证当 MCP 服务离线时，输出友好降级提示并正确展示纯内置工具"""
        mock_init_ch07 = mock.AsyncMock(return_value=[])
        mock_init_ch08 = mock.AsyncMock(return_value=[])
        mock_session = mock.AsyncMock()
        mock_res_pending = mock.MagicMock()
        mock_res_pending.scalar.return_value = 0
        mock_res_done = mock.MagicMock()
        mock_res_done.scalar.return_value = 10
        mock_session.execute.side_effect = [mock_res_pending, mock_res_done]
        mock_session_ctx = mock.AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_session

        mock_store = mock.MagicMock()
        mock_store.count.return_value = 10

        mock_engine = mock.MagicMock()
        mock_conn = mock.AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        mock_engine.dispose = mock.AsyncMock()

        @tool
        def fake_builtin_tool(query: str) -> str:
            """fake builtin tool"""
            return "ok"

        annotate_tool(fake_builtin_tool, tool_source="builtin", mcp_server=None)

        mock_get_all_tools = mock.AsyncMock(return_value=[fake_builtin_tool])

        with (
            mock.patch("scripts.wsl_helper.ensure_mysql_ready"),
            mock.patch("app.db.session.engine", mock_engine),
            mock.patch("scripts.seed_data.seed_all_data", new_callable=mock.AsyncMock, return_value=0),
            mock.patch("scripts.init_ch07_db.init_ch07_db", mock_init_ch07),
            mock.patch("scripts.init_ch08_db.init_ch08_db", mock_init_ch08),
            mock.patch("app.db.session.AsyncSessionLocal", return_value=mock_session_ctx),
            mock.patch("app.services.rag.milvus_client.MilvusKnowledgeStore", return_value=mock_store),
            mock.patch("scripts.wsl_helper.is_port_open", return_value=False),
            mock.patch("app.tools.registry.default_tool_registry.get_all_tools", mock_get_all_tools),
        ):
            ok = await run.check_storage_readiness(clean_kb=False)
            assert ok is True
            out = capsys.readouterr().out
            assert "未启动" in out or "降级" in out
            assert "工具注册中心" in out


class TestRunCh08PrintBanner:
    """测试启动仪表盘展示 MCP 与工具注册中心信息"""

    def test_print_banner_contains_mcp_and_tool_registry_info(self, capsys):
        """验证 print_banner 包含 MCP 服务拓扑与工具注册中心/审计日志信息"""
        run.print_banner("127.0.0.1", 8000)
        out = capsys.readouterr().out
        assert "MCP" in out
        assert "8001" in out
        assert "8002" in out
        assert "tool_audit_logs" in out or "工具注册中心" in out
