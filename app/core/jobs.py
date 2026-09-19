"""白名单作业调度器与生命周期管理器 (JobRunner & JobRegistry)。

核心规范：
1. 零 Shell 注入安全：所有作业名称硬编码白名单映射，前端仅传 job_name，无法传入任何参数、命令片段或路径。
2. 配方一致性与跨平台容错：优先调用 make <target>；当在 Windows 环境无原生 make.exe 时，自适应执行 Makefile 对应的官方命令，确保配方严格一致且不报 FileNotFoundError。
3. 日志落盘：日志自动写入 log/acceptance/<job>.log，支持多行实时追加与前端尾部轮询。
4. 繁重任务标记：将 train-ch10, data-prep-ch10, export-onnx 标为 heavy=True。
5. 防重入：同名作业处于 running 状态时拒绝重复启动（返回错误/409）。
6. 进程生命周期治理：start_new_session=True（或 Windows CREATE_NEW_PROCESS_GROUP），停止作业时递归终止整棵进程树（避免孤儿进程）。
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional
import uuid

import psutil

logger = logging.getLogger(__name__)

# 项目根路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class JobAlreadyRunningError(Exception):
    """同名作业处于运行状态时触发防重入异常"""
    pass


class JobRegistry:
    """作业注册表与跨平台配方解析器"""

    RECIPES: Dict[str, Dict[str, Any]] = {
        # 第十章标准配方
        "classifier-up": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "classifier_service.py"), "start"],
            "heavy": False,
            "desc": "启动分类器独立推理服务 (:8110)",
        },
        "classifier-down": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "classifier_service.py"), "stop"],
            "heavy": False,
            "desc": "停止分类器独立推理服务",
        },
        "classify-pool": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "classify_pool.py"), "--min-batch", "10"],
            "heavy": False,
            "desc": "旁路批量归类待审池问题",
        },
        "eval-ch10": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "evaluate_classifier.py")],
            "heavy": False,
            "desc": "评测多标签分类器指标",
        },
        "export-onnx": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "export_onnx.py")],
            "heavy": True,
            "desc": "导出 ONNX 紧凑模型与元数据",
        },
        "train-ch10": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "train_classifier.py")],
            "heavy": True,
            "desc": "全量训练多标签分类器",
        },
        "data-prep-ch10": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "data_prep_ch10.py")],
            "heavy": True,
            "desc": "准备第十章训练与评测数据",
        },
        "replay-threshold": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "scan_threshold_replay.py")],
            "heavy": False,
            "desc": "阈值扫描复算与选线复现",
        },
        # 兼容历史章节配方
        "eval-rag": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "run_ch04_evaluation.py"), "--mock"],
            "heavy": False,
            "desc": "RAG 策略评测作业",
        },
        "cost-analysis": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "check_intent_costs.py"), "--mock"],
            "heavy": False,
            "desc": "意图成本分析作业",
        },
        "eval-pipeline": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "run_eval_pipeline.py"), "--samples", "10"],
            "heavy": False,
            "desc": "评测流水线作业",
        },
        "calibrate-confidence": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "calibrate_confidence_gate.py")],
            "heavy": False,
            "desc": "置信度校准作业",
        },
        "flywheel-pipeline": {
            "cmd": [sys.executable, str(ROOT_DIR / "scripts" / "run_flywheel_pipeline.py")],
            "heavy": False,
            "desc": "飞轮流水线作业",
        },
    }

    WHITELIST: List[str] = list(RECIPES.keys())

    @classmethod
    def is_heavy(cls, job_name: str) -> bool:
        """判断是否为耗时繁重作业"""
        recipe = cls.RECIPES.get(job_name)
        if not recipe:
            return False
        return recipe.get("heavy", False)

    @classmethod
    def get_command(cls, job_name: str) -> List[str]:
        """获取作业执行命令。优先使用 make <target>，若系统无 make 则自适应官方 Python 命令。"""
        if job_name not in cls.RECIPES:
            raise ValueError(f"Job '{job_name}' is not in whitelist (作业不在安全白名单内)。")

        make_executable = shutil.which("make")
        if make_executable:
            return ["make", job_name]

        return list(cls.RECIPES[job_name]["cmd"])


@dataclass
class JobRecord:
    """单次作业执行生命周期与元数据记录"""
    job_id: str
    job_name: str
    status: str = "pending"  # pending, running, completed, failed, stopped
    heavy: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    return_code: Optional[int] = None
    logs: List[str] = field(default_factory=list)
    pid: Optional[int] = None
    log_file: Optional[str] = None
    proc: Optional[subprocess.Popen] = None


class JobRunner:
    """白名单安全作业执行器与生命周期管理器"""

    WHITELIST_JOBS = JobRegistry.RECIPES

    def __init__(self, max_logs_per_job: int = 5000):
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._max_logs_per_job = max_logs_per_job

    def get_whitelist(self) -> List[str]:
        return list(JobRegistry.WHITELIST)

    def start_job(self, job_name: str) -> Dict[str, Any]:
        """启动白名单作业，严格防注入与防重入"""
        if job_name not in JobRegistry.WHITELIST:
            raise ValueError(
                f"Job '{job_name}' is not in whitelist (作业不在安全白名单内)。"
                f"合法作业列表: {JobRegistry.WHITELIST}"
            )

        with self._lock:
            # 防重入检查：同一作业在 running 状态时拒绝重复启动
            for j in self._jobs.values():
                if j.job_name == job_name and j.status == "running":
                    raise JobAlreadyRunningError(
                        f"Job '{job_name}' is already running (同名作业正在运行中，job_id: {j.job_id})"
                    )

            job_id = f"job_{uuid.uuid4().hex[:10]}"
            heavy = JobRegistry.is_heavy(job_name)

            log_dir = ROOT_DIR / "log" / "acceptance"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{job_name}.log"

            job = JobRecord(
                job_id=job_id,
                job_name=job_name,
                status="running",
                heavy=heavy,
                log_file=str(log_path),
            )
            self._jobs[job_id] = job

        cmd_args = JobRegistry.get_command(job_name)

        thread = threading.Thread(
            target=self._execute_job,
            args=(job_id, cmd_args, log_path),
            daemon=True,
            name=f"JobWorker-{job_id}",
        )
        thread.start()

        return {
            "job_id": job.job_id,
            "job_name": job.job_name,
            "status": job.status,
            "heavy": job.heavy,
            "created_at": job.created_at,
        }

    def _execute_job(self, job_id: str, cmd_args: List[str], log_path: Path):
        """后台线程实际运行进程并实时写入日志与进程树控制"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            start_msg = f"[{job.started_at}] 作业启动: {' '.join(cmd_args)}"
            job.logs.append(start_msg)

        try:
            with open(log_path, "a", encoding="utf-8") as f_log:
                f_log.write(f"\n=== [{job.started_at}] Job '{job.job_name}' (ID: {job_id}) Started ===\n")
                f_log.write(f"{start_msg}\n")
                f_log.flush()

                child_env = os.environ.copy()
                child_env["PYTHONIOENCODING"] = "utf-8"
                child_env["PYTHONUTF8"] = "1"

                kwargs: Dict[str, Any] = {
                    "cwd": str(ROOT_DIR),
                    "env": child_env,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "bufsize": 1,
                    "universal_newlines": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                }

                # 严格独立进程组：Windows 使用 CREATE_NEW_PROCESS_GROUP，Unix 使用 start_new_session=True
                if sys.platform == "win32":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    kwargs["start_new_session"] = True

                proc = subprocess.Popen(cmd_args, **kwargs)
                with self._lock:
                    job.proc = proc
                    job.pid = proc.pid

                if proc.stdout:
                    for raw_line in iter(proc.stdout.readline, ""):
                        line = raw_line.rstrip("\r\n")
                        if not line and proc.poll() is not None:
                            break
                        with self._lock:
                            if len(job.logs) < self._max_logs_per_job:
                                job.logs.append(line)
                            elif len(job.logs) == self._max_logs_per_job:
                                job.logs.append("... [日志已达最大存储限制，后续输出截断] ...")
                        f_log.write(line + "\n")
                        f_log.flush()

                proc.wait()
                return_code = proc.returncode

                with self._lock:
                    if job.status != "stopped":
                        job.return_code = return_code
                        job.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if return_code == 0:
                            job.status = "completed"
                            end_msg = f"[{job.finished_at}] 作业执行成功完成 (exit code 0)"
                        else:
                            job.status = "failed"
                            end_msg = f"[{job.finished_at}] 作业执行失败 (exit code {return_code})"
                        job.logs.append(end_msg)
                        f_log.write(f"{end_msg}\n")
                        f_log.flush()

        except Exception as e:
            logger.exception(f"Job execution failed for {job_id}: {e}")
            with self._lock:
                if job.status != "stopped":
                    job.status = "failed"
                    job.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    job.logs.append(f"[{job.finished_at}] 执行异常: {str(e)}")

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        """安全停止正在运行的作业并递归终止整棵进程树"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(f"Job '{job_id}' not found")
            if job.status != "running":
                return {"job_id": job_id, "status": job.status, "message": "Job is not running"}

            job.status = "stopped"
            job.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pid = job.pid
            proc = job.proc

        if pid:
            self._terminate_process_tree(pid)

        if proc:
            try:
                proc.wait(timeout=2)
            except Exception:
                pass

        stop_msg = f"[{job.finished_at}] 作业已被手动终止 (stopped)"
        with self._lock:
            job.logs.append(stop_msg)

        if job.log_file:
            try:
                with open(job.log_file, "a", encoding="utf-8") as f_log:
                    f_log.write(f"{stop_msg}\n")
                    f_log.flush()
            except Exception:
                pass

        return {"job_id": job_id, "status": "stopped"}

    def _terminate_process_tree(self, pid: int):
        """递归清理整棵进程树，杜绝孤儿进程"""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            parent.terminate()
            gone, alive = psutil.wait_procs([parent] + children, timeout=3)
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        except Exception as e:
            logger.warning(f"Error terminating process tree for pid {pid}: {e}")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取作业状态与元数据"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return {
                "job_id": job.job_id,
                "job_name": job.job_name,
                "status": job.status,
                "heavy": job.heavy,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "return_code": job.return_code,
                "log_count": len(job.logs),
            }

    def get_logs(self, job_id: str, tail_lines: Optional[int] = None) -> Optional[List[str]]:
        """获取作业最新日志或尾部切片"""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            logs = list(job.logs)
            if tail_lines and tail_lines > 0:
                logs = logs[-tail_lines:]
            return logs


# 全局单例作业运行器
job_runner = JobRunner()


async def start_job(job_name: str) -> Dict[str, Any]:
    return job_runner.start_job(job_name)


async def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    return job_runner.get_job(job_id)


async def get_job_logs(job_id: str, tail: Optional[int] = None) -> Optional[Dict[str, Any]]:
    job = job_runner.get_job(job_id)
    if not job:
        return None
    raw_logs = job_runner.get_logs(job_id, tail_lines=tail)
    if isinstance(raw_logs, list):
        logs = raw_logs
    elif isinstance(raw_logs, dict):
        logs = raw_logs.get("logs", [])
    else:
        logs = []
    return {
        "job_id": job_id,
        "job_name": job.get("job_name", ""),
        "status": job.get("status", ""),
        "logs": logs,
        "log_count": job.get("log_count", len(logs)),
    }


async def stop_job(job_id: str) -> Dict[str, Any]:
    return job_runner.stop_job(job_id)
