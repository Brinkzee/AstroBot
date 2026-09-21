"""RAG 评估工作台与白名单作业管理 API 路由。

严格遵循规范：
1. 报告单一事实源：严格只读 reports/ch04_evaluation_report.md，绝不读取或生成 json，也不重算 MRR。
2. 产物缺失是正常状态：文件不存在或无法解析时返回 present=false 与 suggested_job='eval-rag'。
3. 编造个案处置闭环：更新状态时强制要求提供非空 resolution 处置说明。
4. 白名单作业运行器：仅允许启动 eval-rag，绝对防命令注入。
"""

from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.faith_case import FaithCase
from app.services.job_runner import job_runner
from app.services.rag.evaluator import parse_evaluation_report_md

logger = logging.getLogger(__name__)

# 单一真实源评估报告路径 (供测试打 patch)
REPORT_PATH = Path("reports/ch04_evaluation_report.md")

router = APIRouter()


# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class FaithCaseResolveRequest(BaseModel):
    status: str = Field(..., description="处置状态：已解决 / 无需解决 / 未解决")
    resolution: str = Field(..., description="处置说明，详细说明修改措施或判定理由")


class JobStartRequest(BaseModel):
    job_name: str = Field(..., description="白名单作业名称，如 eval-rag")


# ==============================================================================
# 1. 评估报告查询接口 (GET /api/rag-eval/report)
# ==============================================================================

@router.get("/rag-eval/report")
async def get_rag_eval_report():
    """获取最新 RAG 评估报告数据。

    若产物文件不存在或解析失败，返回 present=false 与建议作业 suggested_job='eval-rag'。
    """
    if not REPORT_PATH.exists():
        return {
            "present": False,
            "suggested_job": "eval-rag",
            "message": "尚未生成评估报告，请先运行 RAG 评估作业",
        }

    try:
        content = REPORT_PATH.read_text(encoding="utf-8")
        parsed = parse_evaluation_report_md(content)
        return {
            "present": True,
            "meta": parsed.get("meta", {}),
            "kpis": parsed.get("kpis", {}),
            "retrieval": parsed.get("retrieval", {}),
            "evidence_coverage": parsed.get("evidence_coverage", {}),
            "generation": parsed.get("generation"),
            "full_table": parsed.get("full_table", []),
            "raw_markdown": content,
        }
    except Exception as e:
        logger.warning(f"Failed to parse evaluation report {REPORT_PATH}: {e}")
        return {
            "present": False,
            "suggested_job": "eval-rag",
            "message": f"评估报告解析异常: {str(e)}",
        }


# ==============================================================================
# 2. 编造个案台账查询与处置接口 (GET/POST /api/rag-eval/faith-cases)
# ==============================================================================

@router.get("/rag-eval/faith-cases")
async def get_faith_cases(
    status: Optional[str] = Query(None, description="状态筛选：未解决 / 已解决 / 无需解决"),
    bucket: Optional[str] = Query(None, description="分桶筛选：A_policy / B_model / C_colloquial / E_multi"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
    db: AsyncSession = Depends(get_db),
):
    """分页获取编造个案台账列表，并返回双口径幻觉率与台账总体统计指标。"""
    try:
        # 1. 过滤总数
        count_stmt = select(func.count(FaithCase.id))
        if status:
            count_stmt = count_stmt.where(FaithCase.status == status)
        if bucket:
            count_stmt = count_stmt.where(FaithCase.bucket == bucket)
        count_res = await db.execute(count_stmt)
        total_filtered = count_res.scalar() or 0

        # 2. 过滤分页数据
        list_stmt = select(FaithCase)
        if status:
            list_stmt = list_stmt.where(FaithCase.status == status)
        if bucket:
            list_stmt = list_stmt.where(FaithCase.bucket == bucket)
        list_stmt = (
            list_stmt.order_by(FaithCase.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        list_res = await db.execute(list_stmt)
        records = list_res.scalars().all()

        # 3. 台账全表统计 (严格按测试预设的 side_effect 顺序)
        total_cnt_res = await db.execute(select(func.count(FaithCase.id)))
        total_cnt = total_cnt_res.scalar() or 0

        unres_res = await db.execute(
            select(func.count(FaithCase.id)).where(FaithCase.status == "未解决")
        )
        unres_cnt = unres_res.scalar() or 0

        res_res = await db.execute(
            select(func.count(FaithCase.id)).where(FaithCase.status == "已解决")
        )
        res_cnt = res_res.scalar() or 0

        wont_res = await db.execute(
            select(func.count(FaithCase.id)).where(FaithCase.status == "无需解决")
        )
        wont_cnt = wont_res.scalar() or 0
    except Exception as e:
        logger.warning(f"Database query failed for faith cases: {e}")
        return {
            "metrics": {
                "current_round_judge_rate": 0.0,
                "current_round_human_confirmed_rate": 0.0,
                "ledger_summary": {
                    "total": 0,
                    "unresolved": 0,
                    "resolved": 0,
                    "wont_resolve": 0,
                },
            },
            "total": 0,
            "page": page,
            "page_size": page_size,
            "items": [],
            "warning": f"数据库连接不可用或离线，编造个案台账暂无法加载: {str(e)}",
        }

    # 4. 获取评估集基数计算双口径幻觉率
    eval_set_size = 300
    try:
        if REPORT_PATH.exists():
            report_txt = REPORT_PATH.read_text(encoding="utf-8")
            parsed_meta = parse_evaluation_report_md(report_txt).get("meta", {})
            eval_set_size = parsed_meta.get("eval_set_size", 300)
    except Exception:
        pass

    # 双口径指标：裁判判出率 vs 人工确认幻觉率
    current_round_judge_rate = (
        round((total_cnt / eval_set_size) * 100, 2) if eval_set_size > 0 else 0.0
    )
    current_round_human_confirmed_rate = (
        round((unres_cnt / eval_set_size) * 100, 2) if eval_set_size > 0 else 0.0
    )

    # 5. 组装条目，提取引用角标 cited_indices
    items = []
    for r in records:
        raw_ans = r.answer or ""
        # 提取形如 [1] [2] 的角标序号并保序去重
        found_indices = [int(n) for n in re.findall(r"\[(\d+)\]", raw_ans)]
        cited_indices = list(dict.fromkeys(found_indices))

        first_seen_str = (
            r.first_seen_at.strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(r.first_seen_at, "strftime") and r.first_seen_at
            else (str(r.first_seen_at) if r.first_seen_at else None)
        )
        last_seen_str = (
            r.last_seen_at.strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(r.last_seen_at, "strftime") and r.last_seen_at
            else (str(r.last_seen_at) if r.last_seen_at else None)
        )
        resolved_str = (
            r.resolved_at.strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(r.resolved_at, "strftime") and r.resolved_at
            else (str(r.resolved_at) if r.resolved_at else None)
        )

        is_recurrent = (r.status == "未解决" and r.resolved_at is not None)

        items.append({
            "id": r.id,
            "eval_id": r.eval_id,
            "bucket": r.bucket,
            "query": r.query,
            "strategy": r.strategy,
            "answer": r.answer,
            "reason": r.reason,
            "citations": r.citations or [],
            "cited_indices": cited_indices,
            "judge_model": r.judge_model,
            "status": r.status,
            "seen_count": r.seen_count,
            "first_seen_at": first_seen_str,
            "last_seen_at": last_seen_str,
            "resolution": r.resolution,
            "resolved_at": resolved_str,
            "is_recurrent": is_recurrent,
        })

    return {
        "metrics": {
            "current_round_judge_rate": current_round_judge_rate,
            "current_round_human_confirmed_rate": current_round_human_confirmed_rate,
            "ledger_summary": {
                "total": total_cnt,
                "unresolved": unres_cnt,
                "resolved": res_cnt,
                "wont_resolve": wont_cnt,
            },
        },
        "total": total_filtered,
        "page": page,
        "page_size": page_size,
        "items": items,
    }


@router.post("/rag-eval/faith-cases/{case_id}/resolve")
async def resolve_faith_case(
    case_id: int,
    request: FaithCaseResolveRequest,
    db: AsyncSession = Depends(get_db),
):
    """处置编造个案，更新其状态为已解决或无需解决，强制填写处置说明。"""
    clean_resolution = (request.resolution or "").strip()
    if not clean_resolution:
        raise HTTPException(status_code=400, detail="处置说明不能为空，请详细说明修改或判定理由")

    if request.status not in ["已解决", "无需解决", "未解决"]:
        raise HTTPException(status_code=400, detail="非法处置状态")

    try:
        stmt = select(FaithCase).where(FaithCase.id == case_id)
        result = await db.execute(stmt)
        case = result.scalar_one_or_none()
        if not case:
            raise HTTPException(status_code=404, detail="未找到该编造个案记录")

        case.status = request.status
        case.resolution = clean_resolution
        if request.status in ["已解决", "无需解决"]:
            case.resolved_at = datetime.now()

        await db.commit()
        await db.refresh(case)

        return {
            "success": True,
            "id": case.id,
            "eval_id": case.eval_id,
            "status": case.status,
            "resolution": case.resolution,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to resolve faith case: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"数据库暂不可用，无法保存处置记录: {str(e)}",
        )


# ==============================================================================
# 3. 白名单作业运行器接口 (/api/jobs/...)
# ==============================================================================

@router.post("/jobs")
@router.post("/jobs/start")
async def start_job(request: JobStartRequest):
    """触发安全白名单作业"""
    if request.job_name not in job_runner.get_whitelist():
        raise HTTPException(
            status_code=400,
            detail=f"作业不在安全白名单内: '{request.job_name}'。仅允许触发安全白名单作业: {job_runner.get_whitelist()}",
        )

    try:
        job_info = job_runner.start_job(request.job_name)
        return job_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动作业异常: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """查询作业执行状态与元数据"""
    job = job_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"作业不存在: {job_id}")
    return job


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    tail: Optional[int] = Query(None, ge=1, le=2000, description="读取最近 N 行日志"),
):
    """获取作业运行输出日志及当前执行状态"""
    job = job_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"作业不存在: {job_id}")
    logs = job_runner.get_logs(job_id, tail_lines=tail)
    return {
        "job_id": job_id,
        "status": job["status"],
        "logs": logs or [],
    }

