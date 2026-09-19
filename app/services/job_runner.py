"""后台安全白名单作业运行器服务 (JobRunner)。

注意：核心调度器已在第十章收敛至 app.core.jobs。
本文件保留作为向前兼容导出，指向 app.core.jobs 中的全局单例与类定义。
"""

from app.core.jobs import (
    ROOT_DIR,
    JobRecord,
    JobRegistry,
    JobRunner,
    job_runner,
)

__all__ = [
    "ROOT_DIR",
    "JobRecord",
    "JobRegistry",
    "JobRunner",
    "job_runner",
]
