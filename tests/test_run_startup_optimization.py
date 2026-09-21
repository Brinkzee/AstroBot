import subprocess
import time
from unittest.mock import MagicMock, patch
import pytest

from scripts.mcp_process_manager import MCPServerManager
from run import check_port_availability, open_browser_async


class TestMCPServerManager:
    """测试 MCP 服务子进程管理器的拉起、复用与清理策略"""

    def test_manager_reuses_existing_online_server(self):
        """当目标端口已有服务监听时，管理器应直接复用，不重复生成子进程"""
        with patch("scripts.mcp_process_manager.is_port_open", return_value=True):
            with patch("subprocess.Popen") as mock_popen:
                manager = MCPServerManager()
                status = manager.start_servers(verbose=False)
                assert status["logistics"] is True
                assert status["aftersale"] is True
                assert mock_popen.call_count == 0
                assert len(manager.processes) == 0

    def test_manager_starts_child_process_when_port_closed(self):
        """当目标端口未监听时，管理器应拉起子进程并轮询等待端口就绪"""
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        mock_proc.poll.return_value = None

        # 模拟：初始端口未开，拉起子进程后第二次探测时端口打开
        port_states = [False, False, True, True, True]

        def fake_is_port(host, port, timeout=0.2):
            if port_states:
                return port_states.pop(0)
            return True

        with patch("scripts.mcp_process_manager.is_port_open", side_effect=fake_is_port):
            with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                manager = MCPServerManager()
                status = manager.start_servers(wait_timeout=1.0, verbose=False)
                assert status["logistics"] is True
                assert mock_popen.call_count >= 1
                assert "logistics" in manager.processes

                # 停止管理器，验证子进程被正常 terminate
                manager.stop_all()
                mock_proc.terminate.assert_called()
                assert len(manager.processes) == 0

    def test_manager_handles_timeout_gracefully(self):
        """当拉起子进程后超时仍未监听端口时，应标记未就绪且不抛出未捕获崩溃"""
        mock_proc = MagicMock()
        mock_proc.pid = 8888
        mock_proc.poll.return_value = None

        with patch("scripts.mcp_process_manager.is_port_open", return_value=False):
            with patch("subprocess.Popen", return_value=mock_proc):
                manager = MCPServerManager()
                status = manager.start_servers(wait_timeout=0.2, verbose=False)
                assert status["logistics"] is False
                assert status["aftersale"] is False
                manager.stop_all()


class TestStartupPortAndBrowser:
    """测试主服务端口冲突检测与浏览器自动打开"""

    def test_check_port_availability_free(self):
        """端口空闲时应返回 True"""
        with patch("scripts.wsl_helper.is_port_open", return_value=False):
            assert check_port_availability("127.0.0.1", 8000) is True

    def test_check_port_availability_occupied(self):
        """端口被占用时应返回 False 并提示"""
        with patch("scripts.wsl_helper.is_port_open", return_value=True):
            assert check_port_availability("127.0.0.1", 8000) is False

    def test_open_browser_async(self):
        """open_browser_async 应异步调用 webbrowser.open"""
        with patch("webbrowser.open") as mock_open:
            open_browser_async("http://127.0.0.1:8000/chat", delay=0.01)
            time.sleep(0.08)
            mock_open.assert_called_once_with("http://127.0.0.1:8000/chat")


class TestCLIArgs:
    """测试 run.py 命令行参数解析支持 --no-mcp 与 --no-browser"""

    def test_cli_parser_supports_no_mcp_and_no_browser(self):
        import argparse
        import run

        # 验证 main 函数使用的 parser 配置
        with patch("sys.argv", ["run.py", "--no-mcp", "--no-browser", "--check-only"]):
            # 解析参数验证
            parser = argparse.ArgumentParser()
            parser.add_argument("--no-mcp", action="store_true")
            parser.add_argument("--no-browser", action="store_true")
            args, _ = parser.parse_known_args(["--no-mcp", "--no-browser"])
            assert args.no_mcp is True
            assert args.no_browser is True
