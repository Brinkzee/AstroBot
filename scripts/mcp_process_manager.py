"""MCP 独立子进程服务管理器。

负责统一拉起、端口复用探测、就绪状态监控及主进程退出时的级联终止清理。
"""

import atexit
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def is_port_open(host: str = "127.0.0.1", port: int = 8001, timeout: float = 0.2) -> bool:
    """快速探测指定主机端口是否处于监听就绪状态。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def is_local_host(host: str) -> bool:
    """判断目标主机地址是否为本机地址。"""
    normalized = str(host or "").strip().lower()
    return normalized in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _terminate_process(proc: subprocess.Popen) -> None:
    """安全终止单个子进程并回收退出状态，防止僵尸进程与句柄竞争。"""
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
    except Exception:
        pass


class MCPServerManager:
    """管理外部 MCP Server 独立子进程生命周期的管理器。"""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_files: List[Any] = []
        atexit.register(self.stop_all)

    def _get_server_configs(self) -> List[Dict[str, Any]]:
        """获取物流与售后 MCP 服务的配置列表。"""
        logistics_port = 8001
        aftersale_port = 8002
        logistics_host = "127.0.0.1"
        aftersale_host = "127.0.0.1"

        try:
            from app.config import settings
            l_url = urlparse(settings.MCP_LOGISTICS_SERVER_URL)
            a_url = urlparse(settings.MCP_AFTERSALE_SERVER_URL)
            if l_url.hostname:
                logistics_host = l_url.hostname
            if l_url.port:
                logistics_port = l_url.port
            if a_url.hostname:
                aftersale_host = a_url.hostname
            if a_url.port:
                aftersale_port = a_url.port
        except Exception:
            pass

        return [
            {
                "name": "logistics",
                "label": "物流 MCP 服务",
                "script": "scripts/mcp_logistics_server.py",
                "host": logistics_host,
                "port": logistics_port,
                "log_file": "log/mcp_logistics.log",
            },
            {
                "name": "aftersale",
                "label": "售后 MCP 服务",
                "script": "scripts/mcp_aftersale_server.py",
                "host": aftersale_host,
                "port": aftersale_port,
                "log_file": "log/mcp_aftersale.log",
            },
        ]

    def start_servers(self, wait_timeout: float = 3.0, verbose: bool = True) -> Dict[str, bool]:
        """启动所有 MCP 独立服务子进程。

        若端口已在线则智能复用；若未启动则在独立子进程拉起并重定向日志到 log/ 目录。
        """
        status: Dict[str, bool] = {}
        servers = self._get_server_configs()

        for srv in servers:
            name = srv["name"]
            label = srv["label"]
            host = srv["host"]
            port = srv["port"]
            script_path = PROJECT_ROOT / srv["script"]
            raw_log_path = Path(srv["log_file"])
            log_path = raw_log_path if raw_log_path.is_absolute() else (PROJECT_ROOT / raw_log_path)

            # 1. 端口已被监听，直接复用外部已启动实例
            if is_port_open(host, port, timeout=0.2):
                if verbose:
                    print(f"  • {label} ({host}:{port}): 端口已在线，直接复用现有服务 ✅")
                status[name] = True
                continue

            # 1.5 非本机地址不尝试拉起本地子进程
            if not is_local_host(host):
                if verbose:
                    print(f"  ⚠️ {label} ({host}:{port}) 为远端地址且当前不可达，跳过本地子进程拉起")
                status[name] = False
                continue

            # 2. 端口未被监听，自动拉起独立子进程
            if verbose:
                print(f"  ▶ 正在自动拉起 {label} (端口: {port})...")

            # 若此前已记录同名旧进程，先清理防止泄漏孤儿进程
            existing_proc = self.processes.pop(name, None)
            if existing_proc is not None:
                _terminate_process(existing_proc)

            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)

                creation_flags = 0
                if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                    creation_flags |= subprocess.CREATE_NO_WINDOW

                cmd = [
                    sys.executable,
                    str(script_path),
                    "--host",
                    str(host),
                    "--port",
                    str(port),
                ]
                with open(log_path, "a", encoding="utf-8") as f_log:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.DEVNULL,
                        stdout=f_log,
                        stderr=subprocess.STDOUT,
                        creationflags=creation_flags,
                        cwd=str(PROJECT_ROOT),
                    )
                self.processes[name] = proc

                # 3. 轮询等待端口就绪
                start_time = time.time()
                ready = False
                while time.time() - start_time < wait_timeout:
                    if proc.poll() is not None:
                        # 进程提前异常退出
                        self.processes.pop(name, None)
                        break
                    if is_port_open(host, port, timeout=0.1):
                        ready = True
                        break
                    time.sleep(0.05)

                if ready:
                    status[name] = True
                    if verbose:
                        print(f"    ↳ {label} 已成功就绪 (PID: {proc.pid}) ✅")
                else:
                    status[name] = False
                    if verbose:
                        print(f"    ⚠️ {label} 在 {wait_timeout:.1f}s 内未完全就绪，将在后台继续启动或降级运行")
            except Exception as e:
                status[name] = False
                if verbose:
                    print(f"    ❌ 拉起 {label} 异常: {e}")

        return status

    def stop_all(self):
        """级联停止并清理所有由本管理器启动的子进程与日志句柄。"""
        for _name, proc in list(self.processes.items()):
            _terminate_process(proc)
        self.processes.clear()

        for f in self.log_files:
            try:
                f.close()
            except Exception:
                pass
        self.log_files.clear()
