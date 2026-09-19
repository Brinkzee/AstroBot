import asyncio
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from httpx import AsyncClient, ASGITransport

from app.core.jobs import (
    JobRegistry,
    JobAlreadyRunningError,
    job_runner,
    start_job,
    get_job_status,
    get_job_logs,
    stop_job,
)
from main import app

pytestmark = pytest.mark.asyncio

ROOT_DIR = Path(__file__).resolve().parent.parent


async def test_job_runner_whitelist_and_reentrancy():
    """1. 验证白名单过滤与拒绝非法作业。"""
    # 拒绝非白名单作业
    with pytest.raises(ValueError, match="not in whitelist"):
        await start_job("malicious_command; rm -rf /")

    # 验证 8 个标准作业均在白名单中
    expected_jobs = [
        "classifier-up",
        "classifier-down",
        "classify-pool",
        "eval-ch10",
        "export-onnx",
        "train-ch10",
        "data-prep-ch10",
        "replay-threshold",
    ]
    for j in expected_jobs:
        assert j in JobRegistry.WHITELIST


async def test_heavy_jobs_flag():
    """2. 验证繁重任务标记 (heavy=True)。"""
    assert JobRegistry.is_heavy("train-ch10") is True
    assert JobRegistry.is_heavy("data-prep-ch10") is True
    assert JobRegistry.is_heavy("export-onnx") is True

    assert JobRegistry.is_heavy("classify-pool") is False
    assert JobRegistry.is_heavy("eval-ch10") is False
    assert JobRegistry.is_heavy("classifier-up") is False
    assert JobRegistry.is_heavy("classifier-down") is False
    assert JobRegistry.is_heavy("replay-threshold") is False


async def test_cross_platform_recipe_resolution():
    """3. 验证跨平台配方解析（有 make 时优先 make，无 make 时自适应 python 脚本）。"""
    # 当系统存在 make 时
    with patch("shutil.which", return_value="/usr/bin/make"):
        cmd = JobRegistry.get_command("classify-pool")
        assert cmd == ["make", "classify-pool"]

    # 当系统不存在 make 时（如普通 Windows 环境）
    with patch("shutil.which", return_value=None):
        cmd = JobRegistry.get_command("classify-pool")
        assert cmd[0] == sys.executable
        assert any("classify_pool.py" in arg for arg in cmd)
        assert "--min-batch" in cmd
        assert "10" in cmd


async def test_anti_reentrancy_mechanism():
    """4. 验证同名作业处于 running 状态时的严格防重入机制。"""
    # 模拟一个长期运行的进程，使用事件保持 running 状态
    stop_event = threading.Event()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # 处于 running
    mock_proc.pid = 99999

    def mock_readline():
        stop_event.wait(timeout=5)
        return ""

    mock_proc.stdout.readline.side_effect = mock_readline

    with patch("subprocess.Popen", return_value=mock_proc):
        # 首次启动
        job_info = await start_job("classify-pool")
        job_id = job_info["job_id"]
        assert job_info["status"] == "running"

        # 再次尝试启动同名作业，必须抛出 JobAlreadyRunningError
        with pytest.raises(JobAlreadyRunningError, match="already running"):
            await start_job("classify-pool")

        # 停止该作业，避免影响其他测试
        await stop_job(job_id)
        stop_event.set()


async def test_log_file_persistence_and_reading(tmp_path):
    """5. 验证日志落盘至 log/acceptance/<job>.log 与读取。"""
    log_dir = ROOT_DIR / "log" / "acceptance"
    log_dir.mkdir(parents=True, exist_ok=True)
    test_log_file = log_dir / "replay-threshold.log"

    # 运行一个快速的 python 命令来模拟作业写入日志
    test_cmd = [sys.executable, "-c", "print('hello ch10 acceptance test'); print('second line')"]
    with patch.object(JobRegistry, "get_command", return_value=test_cmd):
        job_info = await start_job("replay-threshold")
        job_id = job_info["job_id"]

        # 等待作业执行完毕
        for _ in range(50):
            status = await get_job_status(job_id)
            if status and status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.1)

        assert status["status"] == "completed"
        assert test_log_file.exists()

        # 读取日志内容
        logs_info = await get_job_logs(job_id)
        assert logs_info is not None
        logs_content = "\n".join(logs_info["logs"])
        assert "hello ch10 acceptance test" in logs_content
        assert "second line" in logs_content

        # 验证 tail 参数
        tail_logs = await get_job_logs(job_id, tail=1)
        assert len(tail_logs["logs"]) == 1


async def test_process_tree_termination():
    """6. 验证 stop_job 停止作业与底层真实 OS 进程树清理。"""
    # 启动一个真实的后台 sleep 进程作为子进程
    # 在 Windows / Linux 跨平台启动 python -c "import time; time.sleep(30)"
    sleep_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    with patch.object(JobRegistry, "get_command", return_value=sleep_cmd):
        job_info = await start_job("classifier-up")
        job_id = job_info["job_id"]

        # 等待进程启动并获取真实 pid
        pid = None
        for _ in range(50):
            job = job_runner._jobs.get(job_id)
            if job and job.pid:
                pid = job.pid
                break
            await asyncio.sleep(0.1)

        assert pid is not None
        import psutil
        assert psutil.pid_exists(pid)

        # 确认已经处于 running
        status = await get_job_status(job_id)
        assert status["status"] == "running"

        # 执行停止
        stop_res = await stop_job(job_id)
        assert stop_res["status"] == "stopped"

        # 确认状态已变更
        after_status = await get_job_status(job_id)
        assert after_status["status"] == "stopped"

        # 断言真实底层 OS 进程已被终止
        await asyncio.sleep(0.5)
        assert not psutil.pid_exists(pid) or (job.proc and job.proc.poll() is not None)


async def test_jobs_api_endpoints_contract():
    """7. 验证 /api/jobs 端点契约（启动、防重入、状态查询、日志查询、停止）。"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. 尝试启动非法作业 -> 400
        res = await client.post("/api/jobs/start", json={"job_name": "rm_rf_hack"})
        assert res.status_code == 400
        assert "白名单" in res.json()["detail"] or "whitelist" in res.json()["detail"]

        # 2. 启动合法作业 (使用 sleep 模拟)
        sleep_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
        with patch.object(JobRegistry, "get_command", return_value=sleep_cmd):
            res_start = await client.post("/api/jobs/start", json={"job_name": "train-ch10"})
            assert res_start.status_code == 200
            data = res_start.json()
            assert "job_id" in data
            assert data["status"] == "running"
            assert data["heavy"] is True
            job_id = data["job_id"]

            # 3. 防重入测试：在 train-ch10 处于 running 时再次启动 -> 409
            res_conflict = await client.post("/api/jobs/start", json={"job_name": "train-ch10"})
            assert res_conflict.status_code == 409
            assert "already running" in res_conflict.json()["detail"] or "正在运行" in res_conflict.json()["detail"]

            # 4. 查询状态 -> 200
            res_status = await client.get(f"/api/jobs/{job_id}")
            assert res_status.status_code == 200
            assert res_status.json()["status"] == "running"
            assert res_status.json()["heavy"] is True

            # 5. 查询日志 -> 200
            res_logs = await client.get(f"/api/jobs/{job_id}/logs?tail=10")
            assert res_logs.status_code == 200
            assert "logs" in res_logs.json()

            # 6. 停止作业 -> 200
            res_stop = await client.post(f"/api/jobs/{job_id}/stop")
            assert res_stop.status_code == 200
            assert res_stop.json()["status"] == "stopped"

            # 7. 查询不存在的作业 -> 404
            res_404 = await client.get("/api/jobs/non_existent_job_123")
            assert res_404.status_code == 404


async def test_sliding_window_deque_logs():
    """8. 验证内存日志滑动窗口 deque 机制，确保 tail 在长任务中不冻结。"""
    original_max = job_runner._max_logs_per_job
    job_runner._max_logs_per_job = 5
    try:
        test_cmd = [sys.executable, "-c", "for i in range(10): print(f'line_{i}')"]
        with patch.object(JobRegistry, "get_command", return_value=test_cmd):
            job_info = await start_job("classify-pool")
            job_id = job_info["job_id"]

            for _ in range(50):
                status = await get_job_status(job_id)
                if status and status["status"] in ["completed", "failed"]:
                    break
                await asyncio.sleep(0.1)

            logs_res = await get_job_logs(job_id)
            assert logs_res is not None
            # 内存日志总数受 deque maxlen (5) 限制
            assert len(logs_res["logs"]) <= 5
            # 滑动窗口必须保留最新产生的日志，而不是被冻结在最早的几行
            assert any("line_9" in line for line in logs_res["logs"])

            # tail=2 应该返回最后 2 行
            tail_res = await get_job_logs(job_id, tail=2)
            assert len(tail_res["logs"]) == 2
            assert any("line_9" in line or "完成" in line for line in tail_res["logs"])
    finally:
        job_runner._max_logs_per_job = original_max


async def test_stop_before_execute_race_condition():
    """9. 验证在子进程实际 Popen 前调用 stop_job 绝不遗留孤儿进程。"""
    sleep_cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    with patch.object(JobRegistry, "get_command", return_value=sleep_cmd):
        # 模拟在 _execute_job 执行前，人为将状态标记为 stopped
        job_info = await start_job("eval-ch10")
        job_id = job_info["job_id"]

        # 立即停止作业
        stop_res = await stop_job(job_id)
        assert stop_res["status"] == "stopped"

        # 等待后台工作线程走完
        await asyncio.sleep(0.5)
        final_status = await get_job_status(job_id)
        assert final_status["status"] == "stopped"
        # 确认没有孤儿进程遗留为 running
        job = job_runner._jobs[job_id]
        if job.pid:
            import psutil
            assert not psutil.pid_exists(job.pid) or (job.proc and job.proc.poll() is not None)

