import json
import logging
from pathlib import Path
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.services.eval.eval_pipeline import EvalPipelineService
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/observability", tags=["observability"])

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

@router.get("/overview")
async def get_overview(db: AsyncSession = Depends(get_db)):
    cost_file = ROOT_DIR / "reports" / "cost_by_intent.json"
    calibration_file = ROOT_DIR / "reports" / "confidence_calibration.json"

    # Block 1: Cost
    cost_block = {
        "present": False,
        "job_name": "cost-analysis",
        "hint": "暂无意图成本报表，请先进行对话积累 Trace 后点击重跑",
        "data": None
    }
    if cost_file.exists():
        try:
            with open(cost_file, "r", encoding="utf-8") as f:
                cost_data = json.load(f)
                cost_block["present"] = True
                cost_block["hint"] = ""
                cost_block["data"] = cost_data
        except Exception as e:
            logger.error(f"Failed to read cost report: {e}")

    # Block 2: Eval Trend
    eval_trend_block = {
        "present": False,
        "job_name": "eval-pipeline",
        "hint": "暂无评测轮次记录，点击重跑即可生成趋势的第一个数据点",
        "runs": []
    }
    try:
        service = EvalPipelineService()
        runs = await service.get_recent_runs_with_deltas(db, limit=10)
        if runs:
            eval_trend_block["present"] = True
            eval_trend_block["hint"] = ""
            eval_trend_block["runs"] = runs
    except Exception as e:
        logger.error(f"Failed to read eval runs: {e}")

    # Block 3: Calibration
    calibration_block = {
        "present": False,
        "job_name": "calibrate-confidence",
        "hint": "暂无置信度校准报表，点击重跑即可在评估集上扫描最佳阈值",
        "data": None
    }
    if calibration_file.exists():
        try:
            with open(calibration_file, "r", encoding="utf-8") as f:
                calib_data = json.load(f)
                recommended = calib_data.get("recommended_threshold")
                active = settings.evidence_confidence_threshold
                is_mismatched = False
                if recommended is not None:
                    is_mismatched = abs(recommended - active) > 0.001
                
                calib_data["active_threshold"] = active
                calib_data["is_mismatched"] = is_mismatched
                
                calibration_block["present"] = True
                calibration_block["hint"] = ""
                calibration_block["data"] = calib_data
        except Exception as e:
            logger.error(f"Failed to read calibration report: {e}")

    return {
        "cost_block": cost_block,
        "eval_trend_block": eval_trend_block,
        "calibration_block": calibration_block
    }
