"""Process and PID governance script for Chapter 10 lightweight classifier service.

Commands:
  start   - Start the service in the background and poll /healthz until healthy
  stop    - Safely terminate the service and its process tree, then clean up PID file
  status  - Check PID liveness and query /healthz endpoint
  restart - Stop then start the service
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, Optional
import urllib.request
import urllib.error

import psutil

DEFAULT_PID_FILE = "data/ch10/classifier.pid"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8110
DEFAULT_TIMEOUT = 30.0


def write_pid(pid_file: Path | str, pid: int) -> None:
    """Write PID to specified pid_file, creating parent directories if needed."""
    path = Path(pid_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid), encoding="utf-8")


def read_pid(pid_file: Path | str) -> Optional[int]:
    """Read PID from pid_file if it exists, returning None otherwise."""
    path = Path(pid_file)
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
        return int(content)
    except (ValueError, OSError):
        return None


def remove_pid(pid_file: Path | str) -> None:
    """Remove pid_file safely without raising if it does not exist."""
    path = Path(pid_file)
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def is_pid_alive(pid: int) -> bool:
    """Check if process with given PID exists and is not a zombie."""
    if pid <= 0:
        return False
    try:
        if not psutil.pid_exists(pid):
            return False
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    except Exception:
        # Fallback to os.kill
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def kill_process_tree(pid: int, timeout: float = 5.0) -> bool:
    """Safely terminate a process and all its child processes (anti-orphan)."""
    if not is_pid_alive(pid):
        return True

    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        procs_to_kill = children + [parent]

        for p in procs_to_kill:
            try:
                p.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        gone, alive = psutil.wait_procs(procs_to_kill, timeout=timeout)
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return True
    except psutil.NoSuchProcess:
        return True
    except Exception:
        # Platform-specific fallback
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
            )
        else:
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        return True


def query_healthz(host: str, port: int, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    """Query /healthz endpoint via HTTP."""
    url = f"http://{host}:{port}/healthz"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "classifier_service_governance"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                body = response.read().decode("utf-8")
                return json.loads(body)
    except Exception:
        return None
    return None


def get_service_status(
    pid_file: Path | str = DEFAULT_PID_FILE,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> Dict[str, Any]:
    """Retrieve detailed service status."""
    pid = read_pid(pid_file)
    if pid is None:
        return {
            "running": False,
            "pid": None,
            "health": None,
            "message": "Service is STOPPED (no PID file)",
        }

    if not is_pid_alive(pid):
        return {
            "running": False,
            "pid": pid,
            "stale": True,
            "health": None,
            "message": f"Service is STOPPED (stale PID file: {pid})",
        }

    # Process is alive, probe healthz
    health_data = query_healthz(host, port)
    if health_data and health_data.get("status") == "ok":
        return {
            "running": True,
            "pid": pid,
            "health": "OK",
            "health_data": health_data,
            "message": f"Service is RUNNING (PID: {pid}, port: {port}, health: OK)",
        }

    return {
        "running": True,
        "pid": pid,
        "health": "UNHEALTHY",
        "message": f"Service is UNHEALTHY (PID: {pid}, port: {port}, health check failed)",
    }


def start_service(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    pid_file: Path | str = DEFAULT_PID_FILE,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """Start the service in the background and wait for /healthz to be ready."""
    pid_path = Path(pid_file)
    existing_pid = read_pid(pid_path)

    if existing_pid and is_pid_alive(existing_pid):
        health = query_healthz(host, port)
        if health and health.get("status") == "ok":
            print(f"[INFO] Service already running with PID {existing_pid} (healthy).")
            return True
        else:
            print(f"[WARN] Process {existing_pid} is running but unhealthy. Restarting...")
            stop_service(pid_file=pid_path)
    elif existing_pid:
        print(f"[INFO] Removing stale PID file for PID {existing_pid}.")
        remove_pid(pid_path)

    # Launch uvicorn process
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.services.classifier_service.server:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    print(f"[INFO] Starting classifier service on {host}:{port}...")
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    write_pid(pid_path, proc.pid)
    print(f"[INFO] Background process spawned with PID {proc.pid}. Waiting for /healthz...")

    # Poll /healthz
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_pid_alive(proc.pid):
            print(f"[ERROR] Service process {proc.pid} exited prematurely.")
            remove_pid(pid_path)
            return False

        health = query_healthz(host, port, timeout=1.0)
        if health and health.get("status") == "ok":
            print(f"[SUCCESS] Service started successfully with PID {proc.pid}.")
            return True

        time.sleep(0.5)

    print(f"[ERROR] Timeout ({timeout}s) waiting for service to become healthy. Terminating...")
    kill_process_tree(proc.pid)
    remove_pid(pid_path)
    return False


def stop_service(
    pid_file: Path | str = DEFAULT_PID_FILE,
    timeout: float = 5.0,
) -> bool:
    """Stop the running service and remove PID file."""
    pid_path = Path(pid_file)
    pid = read_pid(pid_path)

    if pid is None:
        print("[INFO] No PID file found. Service is not running.")
        return True

    print(f"[INFO] Stopping service with PID {pid}...")
    kill_process_tree(pid, timeout=timeout)
    remove_pid(pid_path)

    if not is_pid_alive(pid):
        print(f"[SUCCESS] Service with PID {pid} stopped successfully.")
        return True
    else:
        print(f"[WARN] Failed to terminate process {pid}.")
        return False


def restart_service(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    pid_file: Path | str = DEFAULT_PID_FILE,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """Restart the service."""
    print("[INFO] Restarting classifier service...")
    stop_service(pid_file=pid_file)
    time.sleep(1.0)
    return start_service(host=host, port=port, pid_file=pid_file, timeout=timeout)


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Chapter 10 Classifier Service PID Governance",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start
    start_parser = subparsers.add_parser("start", help="Start the classifier service")
    start_parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind (default: 127.0.0.1)")
    start_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8110)")
    start_parser.add_argument("--pid-file", default=DEFAULT_PID_FILE, help="Path to PID file")
    start_parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Startup timeout in seconds")

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop the classifier service")
    stop_parser.add_argument("--pid-file", default=DEFAULT_PID_FILE, help="Path to PID file")
    stop_parser.add_argument("--timeout", type=float, default=5.0, help="Termination timeout in seconds")

    # status
    status_parser = subparsers.add_parser("status", help="Get status of the classifier service")
    status_parser.add_argument("--host", default=DEFAULT_HOST, help="Host to query (default: 127.0.0.1)")
    status_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to query (default: 8110)")
    status_parser.add_argument("--pid-file", default=DEFAULT_PID_FILE, help="Path to PID file")

    # restart
    restart_parser = subparsers.add_parser("restart", help="Restart the classifier service")
    restart_parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind (default: 127.0.0.1)")
    restart_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8110)")
    restart_parser.add_argument("--pid-file", default=DEFAULT_PID_FILE, help="Path to PID file")
    restart_parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Startup timeout in seconds")

    return parser.parse_args(args)


def main():
    """Main CLI execution routine."""
    args = parse_args()
    if args.command == "start":
        ok = start_service(
            host=args.host,
            port=args.port,
            pid_file=args.pid_file,
            timeout=args.timeout,
        )
        sys.exit(0 if ok else 1)
    elif args.command == "stop":
        ok = stop_service(pid_file=args.pid_file, timeout=args.timeout)
        sys.exit(0 if ok else 1)
    elif args.command == "status":
        status = get_service_status(
            pid_file=args.pid_file,
            host=args.host,
            port=args.port,
        )
        print(status["message"])
        sys.exit(0 if status["running"] and status.get("health") == "OK" else 1)
    elif args.command == "restart":
        ok = restart_service(
            host=args.host,
            port=args.port,
            pid_file=args.pid_file,
            timeout=args.timeout,
        )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
