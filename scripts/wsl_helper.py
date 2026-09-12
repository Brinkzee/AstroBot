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
_mysql_ensured = False

def is_port_open(host: str = "127.0.0.1", port: int = 3306, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False

def ensure_mysql_ready(verbose: bool = True):
    """
    确保 WSL2 中的 MySQL 容器就绪并保持 Windows 端 wsl 进程保活，
    避免 WSL2 在 8~10 秒空闲时自动休眠导致 MySQL 连接被切断。
    """
    global _keepalive_proc, _mysql_ensured

    # 1. 确保 Windows 端持有 1 个 WSL2 会话保活进程（单例，避免 WSL2 在 8~10 秒空闲时自动挂起）
    if sys.platform == "win32" and _keepalive_proc is None:
        try:
            creation_flags = 0
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                creation_flags |= subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "DETACHED_PROCESS"):
                creation_flags |= subprocess.DETACHED_PROCESS
            _keepalive_proc = subprocess.Popen(
                ["wsl", "-d", "Ubuntu", "sleep", "86400"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags
            )
        except Exception as e:
            if verbose:
                print(f"[提示] 启动 WSL 保活进程跳过或异常: {e}")

    # 2. 检查 3306 端口是否已经就绪，已就绪则直接极速返回
    if is_port_open("127.0.0.1", 3306, timeout=1.0):
        _mysql_ensured = True
        return

    # 3. 检查 3306 端口是否就绪（重试 3 次，避免瞬时网络抖动误判）
    for _ in range(3):
        if is_port_open("127.0.0.1", 3306, timeout=1.0):
            _mysql_ensured = True
            if verbose:
                print("[OK] MySQL 服务正在运行 (127.0.0.1:3306 已就绪)")
            return
        time.sleep(0.3)

    # 3. 端口仍未开放，检查 Docker 容器是否已在运行
    container_running = False
    try:
        proc = subprocess.run(
            ["wsl", "-d", "Ubuntu", "docker", "ps", "--filter", "name=astro_bot_mysql", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "Up" in proc.stdout:
            container_running = True
    except Exception:
        pass

    if not container_running:
        if verbose:
            print("检测到 MySQL 容器未运行，正在自动拉起 Docker 容器...")
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
        if is_port_open("127.0.0.1", 3306, timeout=1.0):
            time.sleep(1)  # 给 MySQL 内部握手缓冲
            _mysql_ensured = True
            if verbose:
                print("[OK] MySQL 服务就绪，端口 3306 连接正常！")
            return

    if verbose:
        print("[WARN] 警告: 等待 30 秒后 MySQL 端口仍未开放，请检查 Docker 容器状态。")
