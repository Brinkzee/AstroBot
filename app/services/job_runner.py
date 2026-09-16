"""后台安全白名单作业运行器服务 (JobRunner)。

安全规范：
1. 严格白名单机制：只允许执行在 WHITELIST_JOBS 中注册的预定义作业（例如 eval-rag）。
2. 绝对防注入：不接受任何外部命令行参数注入或拼接字符串。
3. 线程隔离与行缓冲输出：支持实时捕获 stdout/stderr 至作业内存日志缓冲区。
4. 作业生命周期状态管理：pending -> running -> completed / failed。
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

# 项目根路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class JobRecord:
    job_id: str
    job_name: str
    status: str = "pending"  # pending, running, completed, failed
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    return_code: Optional[int] = None
    logs: List[str] = field(default_factory=list)


class JobRunner:
    """安全白名单后台作业执行器"""

    # 预定义安全白名单作业映射
    WHITELIST_JOBS = {
        "eval-rag": [
            sys.executable,
            str(ROOT_DIR / "scripts" / "run_ch04_evaluation.py"),
            "--mock",
        ],
        "cost-analysis": [sys.executable, str(ROOT_DIR / "scripts" / "check_intent_costs.py"), "--mock"],
        "eval-pipeline": [sys.executable, str(ROOT_DIR / "scripts" / "run_eval_pipeline.py"), "--samples", "10"],
        "calibrate-confidence": [sys.executable, str(ROOT_DIR / "scripts" / "calibrate_confidence_gate.py")],
        "flywheel-pipeline": [sys.executable, str(ROOT_DIR / "scripts" / "run_flywheel_pipeline.py")]
    }

    def __init__(self, max_logs_per_job: int = 2000):
        self._jobs: Dict[str, JobRecord] = {}
        self._max_logs_per_job = max_logs_per_job
        self._lock = threading.Lock()

    def get_whitelist(self) -> List[str]:
        return list(self.WHITELIST_JOBS.keys())

    def start_job(self, job_name: str) -> Dict[str, str]:
        """启动白名单作业"""
        if job_name not in self.WHITELIST_JOBS:
            raise ValueError(f"作业不在安全白名单内: {job_name}。合法作业列表: {self.get_whitelist()}")

        job_id = f"job_{uuid.uuid4().hex[:10]}"
        job = JobRecord(job_id=job_id, job_name=job_name)

        with self._lock:
            self._jobs[job_id] = job

        # 启动后台线程执行进程
        thread = threading.Thread(
            target=self._execute_job,
            args=(job_id, self.WHITELIST_JOBS[job_name]),
            daemon=True,
            name=f"JobWorker-{job_id}",
        )
        thread.start()

        return {
            "job_id": job.job_id,
            "job_name": job.job_name,
            "status": job.status,
            "created_at": job.created_at,
        }

    def _execute_job(self, job_id: str, cmd_args: List[str]):
        """后台线程实际运行进程并收集日志"""
        job = self._jobs.get(job_id)
        if not job:
            return

        with self._lock:
            job.status = "running"
            job.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job.logs.append(f"[{job.started_at}] 作业启动: {' '.join(cmd_args)}")

        try:
            # 传递 UTF-8 环境变量，避免 Windows 默认代码页 (CP936/GBK) 导致子进程控制台输出乱码
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUTF8"] = "1"

            # 使用行缓冲方式捕获日志
            proc = subprocess.Popen(
                cmd_args,
                cwd=str(ROOT_DIR),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
            )

            if proc.stdout:
                for raw_line in iter(proc.stdout.readline, ""):
                    line = raw_line.rstrip("\r\n")
                    if not line and proc.poll() is not None:
                        break
                    with self._lock:
                        if len(job.logs) < self._max_logs_per_job:
                            job.logs.append(line)
                        elif len(job.logs) == self._max_logs_per_job:
                            job.logs.append("... [日志已达最大存储行数限制，后续输出已截断] ...")

            proc.wait()
            return_code = proc.returncode

            with self._lock:
                job.return_code = return_code
                job.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if return_code == 0:
                    job.status = "completed"
                    job.logs.append(f"[{job.finished_at}] 作业执行成功完成 (exit code 0)")
                else:
                    job.status = "failed"
                    job.logs.append(f"[{job.finished_at}] 作业执行失败 (exit code {return_code})")

        except Exception as e:
            logger.exception(f"Job execution failed for {job_id}: {e}")
            with self._lock:
                job.status = "failed"
                job.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                job.logs.append(f"[{job.finished_at}] 执行异常: {str(e)}")

    def get_job(self, job_id: str) -> Optional[Dict]:
        """获取作业当前状态元信息"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return {
                "job_id": job.job_id,
                "job_name": job.job_name,
                "status": job.status,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "return_code": job.return_code,
                "log_count": len(job.logs),
            }

    def get_logs(self, job_id: str, tail_lines: Optional[int] = None) -> Optional[List[str]]:
        """获取作业运行日志"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if tail_lines and tail_lines > 0:
                return job.logs[-tail_lines:]
            return list(job.logs)


# 全局单例作业运行器
job_runner = JobRunner()
