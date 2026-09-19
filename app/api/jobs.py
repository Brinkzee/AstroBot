"""中立白名单作业管理 API 路由 (/api/jobs)。

端点契约：
- POST /api/jobs/start: 启动指定白名单作业，支持防重入（409）与非法作业拒绝（400）。
- GET /api/jobs/{job_id}: 查询作业执行状态、退出码与元数据。
- GET /api/jobs/{job_id}/logs: 查询作业实时日志或尾部切片。
- POST /api/jobs/{job_id}/stop: 安全停止正在运行的作业并回收进程树。
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.jobs import (
    JobRegistry,
    JobAlreadyRunningError,
    start_job,
    get_job_status,
    get_job_logs,
    stop_job,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobStartRequest(BaseModel):
    job_name: str = Field(..., description="白名单作业名称，如 classify-pool, train-ch10")


@router.post("")
@router.post("/")
@router.post("/start")
async def api_start_job(request: JobStartRequest):
    """启动安全白名单作业"""
    try:
        job_info = await start_job(request.job_name)
        return job_info
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except JobAlreadyRunningError as e:
        raise HTTPException(
            status_code=409,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"启动作业异常: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"启动作业异常: {str(e)}",
        )


@router.get("/{job_id}")
async def api_get_job_status(job_id: str):
    """查询作业执行状态与元数据"""
    job = await get_job_status(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"作业不存在: {job_id}",
        )
    return job


@router.get("/{job_id}/logs")
async def api_get_job_logs(
    job_id: str,
    tail: Optional[int] = Query(None, ge=1, le=5000, description="读取最近 N 行日志"),
):
    """获取作业运行输出日志及当前执行状态"""
    logs_info = await get_job_logs(job_id, tail=tail)
    if not logs_info:
        raise HTTPException(
            status_code=404,
            detail=f"作业不存在: {job_id}",
        )
    return logs_info


@router.post("/{job_id}/stop")
async def api_stop_job(job_id: str):
    """安全停止正在运行的作业并递归清理进程树"""
    try:
        result = await stop_job(job_id)
        return result
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"作业不存在: {job_id}",
        )
    except Exception as e:
        logger.exception(f"停止作业异常: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"停止作业异常: {str(e)}",
        )
