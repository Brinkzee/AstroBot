import os
from unittest import mock
import pytest

import run


class TestRunCheckEnvironment:
    """测试 run.py 中 check_environment 流程与上下文预算集成"""

    def test_check_environment_default_passes(self, capsys):
        """验证在 18000 窗口下预算检查通过并输出详细滑窗预算"""
        with mock.patch.dict(os.environ, {"MODEL_CONTEXT_WINDOW": "18000"}):
            ok = run.check_environment()
            assert ok is True
            out = capsys.readouterr().out
            assert "上下文管理窗口: 18000 Tokens" in out
            assert "总滑窗预算: 5650 Tokens" in out
            assert "层1: 3954" in out
            assert "层2: 1695" in out

    def test_check_environment_ultra_low_window_intercepts(self, capsys):
        """验证当 MODEL_CONTEXT_WINDOW 被 mock/设置为极小值（如 1000）导致 is_usable=False 时，check_environment() 返回 False 拦截启动"""
        with mock.patch.dict(os.environ, {"MODEL_CONTEXT_WINDOW": "1000"}):
            ok = run.check_environment()
            assert ok is False
            out = capsys.readouterr().out
            assert "❌ 上下文预算自检未通过：模型窗口无法容纳单轮稳态交互" in out


class TestRunCheckStorageReadiness:
    """测试 run.py 中 check_storage_readiness 流程与 ch07 DDL 迁移集成"""

    @pytest.mark.asyncio
    async def test_storage_readiness_calls_init_ch07_db(self, capsys):
        """Mock init_ch07_db，验证其在存储检查阶段被正确调用并打印成功提示"""
        mock_init_ch07 = mock.AsyncMock(return_value=["ALTER TABLE conversations ..."])
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
            mock.patch("app.db.session.AsyncSessionLocal", return_value=mock_session_ctx),
            mock.patch("app.services.rag.milvus_client.MilvusKnowledgeStore", return_value=mock_store),
        ):
            ok = await run.check_storage_readiness(clean_kb=False)
            assert ok is True
            mock_init_ch07.assert_awaited_once()
            out = capsys.readouterr().out
            assert "第七章三层会话上下文数据表与字段迁移已就绪 ✅" in out

    @pytest.mark.asyncio
    async def test_storage_readiness_init_ch07_db_failure_intercepts(self, capsys):
        """验证当 init_ch07_db 报错时，check_storage_readiness() 正确捕获并返回 False"""
        mock_init_ch07 = mock.AsyncMock(side_effect=RuntimeError("DDL migration connection timeout"))
        mock_engine = mock.MagicMock()
        mock_conn = mock.AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        with (
            mock.patch("scripts.wsl_helper.ensure_mysql_ready"),
            mock.patch("app.db.session.engine", mock_engine),
            mock.patch("scripts.init_ch07_db.init_ch07_db", mock_init_ch07),
        ):
            ok = await run.check_storage_readiness(clean_kb=False)
            assert ok is False
            mock_init_ch07.assert_awaited_once()
            out = capsys.readouterr().out
            assert "第七章数据表与字段迁移失败" in out or "DDL migration connection timeout" in out


class TestRunPrintBanner:
    """测试仪表盘展示上下文管理状态"""

    def test_print_banner_contains_context_info(self, capsys):
        run.print_banner("127.0.0.1", 8000)
        out = capsys.readouterr().out
        assert "上下文管理" in out
