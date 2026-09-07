import atexit
import socket
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_keepalive_proc = None

def is_port_open(host: str = "127.0.0.1", port: int = 3306) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except (OSError, ConnectionRefusedError):
        return False

def ensure_mysql_ready(verbose: bool = True):
    """
    确保 WSL2 中的 MySQL 容器就绪并保持 Windows 端 wsl 进程保活，
    避免 WSL2 在 8~10 秒空闲时自动休眠导致 MySQL 连接被切断。
    """
    global _keepalive_proc

    # 1. 启动 Windows 下持有 WSL2 会话的保活进程
    if sys.platform == "win32" and _keepalive_proc is None:
        try:
            creation_flags = 0
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                creation_flags = subprocess.CREATE_NO_WINDOW
            _keepalive_proc = subprocess.Popen(
                ["wsl", "-d", "Ubuntu", "sleep", "86400"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags
            )
            def cleanup():
                global _keepalive_proc
                if _keepalive_proc and _keepalive_proc.poll() is None:
                    _keepalive_proc.terminate()
                    _keepalive_proc = None
            atexit.register(cleanup)
        except Exception as e:
            if verbose:
                print(f"[提示] 启动 WSL 保活进程跳过或异常: {e}")

    # 2. 检查 3306 端口是否就绪
    if not is_port_open("127.0.0.1", 3306):
        if verbose:
            print("检测到 MySQL 端口 (127.0.0.1:3306) 未开放，正在自动拉起 Docker 容器...")
        try:
            subprocess.run(
                ["wsl", "-d", "Ubuntu", "docker", "compose", "-f", "/mnt/d/PycharmProjects/AstroBot/docker-compose.yml", "up", "-d"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            if verbose:
                print(f"[提示] 拉起容器命令执行: {e}")

        # 最多轮询等待 30 秒
        for _ in range(30):
            time.sleep(1)
            if is_port_open("127.0.0.1", 3306):
                time.sleep(2)  # 给 MySQL 内部握手缓冲
                if verbose:
                    print("[OK] MySQL 服务就绪，端口 3306 连接正常！")
                return
        if verbose:
            print("[WARN] 警告: 等待 30 秒后 MySQL 端口仍未开放，请检查 Docker 容器状态。")
    else:
        if verbose:
            print("[OK] MySQL 服务正在运行 (127.0.0.1:3306 已就绪)")
