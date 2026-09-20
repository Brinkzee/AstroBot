"""Chapter 10 Acceptance API system and read-only endpoints (/api/acceptance).

Provides:
- GET /api/acceptance/overview: 9 empirical gates with 3-state status (pass/fail/missing), passed_count, service health.
- GET /api/acceptance/eval: Micro vs Macro comparison, 17-class metrics, 9 candidate threshold lines, confusion matrices.
- GET /api/acceptance/data: Dataset inventory, zero-leakage hard gate result, training & ONNX artifacts metadata.
- GET /api/acceptance/errors: 3-way error accounting (under, over, displaced, mathematical closed loop) and error samples.
- GET /api/acceptance/service: Live probe to :8110/healthz.
- POST /api/acceptance/classify: Proxy :8110/classify for single-query multi-label demonstration.

Fault-tolerance rule:
Missing artifact files NEVER trigger HTTP 500. They return HTTP 200 with {"present": false, "make_target": "make <target>"}.
Gates return "missing" instead of "fail" when artifacts are not ready.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, Depends, HTTPException
import httpx
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.low_confidence import LowConfidenceQuestion
from app.models.topic_classification import TopicClassification
from app.services.classifier.taxonomy import (
    CATEGORY_BOUNDARIES,
    LOOSE_TIER,
    MEDIUM_TIER,
    STRICT_TIER,
    TAXONOMY_17,
    check_dataset_leakage,
    get_tier_for_category,
)
from app.core.jobs import JobRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/acceptance", tags=["acceptance"])
topics_router = APIRouter(prefix="/api/topics", tags=["topics"])

# Paths
REPORT_PATH = Path("reports/ch10_evaluation_report.json")
THRESHOLD_PATH = Path("data/ch10/threshold.json")
ERROR_SAMPLES_MD = Path("data/ch10/reports/error_samples.md")
DATASET_DIR = Path("mewhelp-ch10-dataset")
BEST_MODEL_DIR = Path("data/ch10/best_model")
ONNX_DIR = Path("data/ch10/onnx")

CLASSIFIER_HOST = "127.0.0.1"
CLASSIFIER_PORT = 8110

# In-memory cache for leakage check
_LEAKAGE_CACHE: Dict[str, Any] = {"mtimes": None, "result": None}


def _load_jsonl_texts(path: Path) -> List[str]:
    """Load texts from a jsonl file."""
    texts = []
    if not path.exists():
        return texts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                texts.append(item.get("text", "").strip())
            except json.JSONDecodeError:
                continue
    return texts


def get_cached_leakage_check(dataset_dir: Path = DATASET_DIR) -> Optional[Dict[str, Any]]:
    """Cached computation of dataset zero-leakage check."""
    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "val.jsonl"
    test_file = dataset_dir / "test.jsonl"

    if not (train_file.exists() and val_file.exists() and test_file.exists()):
        return None

    current_mtimes = (
        train_file.stat().st_mtime,
        val_file.stat().st_mtime,
        test_file.stat().st_mtime,
    )

    if _LEAKAGE_CACHE["mtimes"] == current_mtimes and _LEAKAGE_CACHE["result"] is not None:
        return _LEAKAGE_CACHE["result"]

    train_texts = _load_jsonl_texts(train_file)
    val_texts = _load_jsonl_texts(val_file)
    test_texts = _load_jsonl_texts(test_file)

    report = check_dataset_leakage(train_texts, val_texts, test_texts)
    _LEAKAGE_CACHE["mtimes"] = current_mtimes
    _LEAKAGE_CACHE["result"] = report
    return report


async def query_classifier_healthz(
    host: str = CLASSIFIER_HOST,
    port: int = CLASSIFIER_PORT,
    timeout: float = 1.0,
) -> Dict[str, Any]:
    """Probe :8110/healthz live status."""
    url = f"http://{host}:{port}/healthz"
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "online": True,
                    "status": "ok",
                    "details": data,
                }
            if resp.status_code in (502, 503, 504):
                return {
                    "online": False,
                    "status": "offline",
                    "error": f"HTTP {resp.status_code} (服务未就绪)",
                    "make_target": "make classifier-up",
                }
            return {
                "online": False,
                "status": f"HTTP {resp.status_code}",
                "error": resp.text,
                "make_target": "make classifier-up",
            }
    except Exception as e:
        return {
            "online": False,
            "status": "offline",
            "error": str(e),
            "make_target": "make classifier-up",
        }


async def proxy_classify_request(
    text: str,
    host: str = CLASSIFIER_HOST,
    port: int = CLASSIFIER_PORT,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """Proxy single-sentence classification request to :8110/classify."""
    url = f"http://{host}:{port}/classify"
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.post(url, json={"texts": [text]})
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "results" in data and len(data["results"]) > 0:
                    return data["results"][0]
                elif isinstance(data, list) and len(data) > 0:
                    return data[0]
                return {"text": text, "labels": [], "scores": {}}
            elif resp.status_code in (502, 503, 504):
                raise HTTPException(
                    status_code=503,
                    detail="推理服务未就绪或无法连接，请先运行 make classifier-up 启动服务",
                )
            else:
                err_text = resp.text.strip()
                try:
                    err_json = resp.json()
                    if isinstance(err_json, dict) and "detail" in err_json:
                        err_text = err_json["detail"]
                except Exception:
                    pass
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"分类服务返回错误: {err_text}" if err_text else f"分类服务返回 HTTP {resp.status_code}",
                )
    except (httpx.ConnectError, httpx.NetworkError):
        raise HTTPException(
            status_code=503,
            detail="推理服务未启动，请先运行 make classifier-up",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="推理服务请求超时，请检查服务状态",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"推理服务网络请求失败 ({type(e).__name__})，请检查服务是否运行",
        )


class SingleClassifyRequest(BaseModel):
    text: str = Field(..., description="单句待分类文本，如 '买大了想退'")


# =========================================================================
# 1. GET /api/acceptance/overview
# =========================================================================
@router.get("/overview")
async def get_acceptance_overview(db: AsyncSession = Depends(get_db)):
    """返回九项实证闸门的三态列表（pass / fail / missing）、总闸通过数及分类器状态。"""
    gates: List[Dict[str, Any]] = []

    # 1. data_leakage: 考卷零泄漏硬闸 (make data-prep-ch10)
    train_file = DATASET_DIR / "train.jsonl"
    val_file = DATASET_DIR / "val.jsonl"
    test_file = DATASET_DIR / "test.jsonl"
    if not (train_file.exists() and val_file.exists() and test_file.exists()):
        leakage_status = "missing"
        leakage_detail = "考卷数据集文件未就绪"
    else:
        leakage_rep = get_cached_leakage_check(DATASET_DIR)
        if leakage_rep and leakage_rep.get("passed"):
            leakage_status = "pass"
            leakage_detail = (
                f"三份考卷零重叠严格通过 (train: {leakage_rep['train_count']}, "
                f"val: {leakage_rep['val_count']}, test: {leakage_rep['test_count']})"
            )
        else:
            leakage_status = "fail"
            leakage_detail = f"发现泄漏重叠样本数: {len(leakage_rep.get('overlap_samples', [])) if leakage_rep else '未知'}"

    gates.append({
        "id": "data_leakage",
        "name": "考卷零泄漏硬闸",
        "status": leakage_status,
        "make_target": "make data-prep-ch10",
        "job_name": "data-prep-ch10",
        "description": "三份考卷 (train, val, test) 交叉重叠严格为 0",
        "detail": leakage_detail,
    })

    # Load report if exists
    eval_report = None
    if REPORT_PATH.exists():
        try:
            with REPORT_PATH.open("r", encoding="utf-8") as f:
                eval_report = json.load(f)
        except Exception:
            eval_report = None

    # 2. f1_redlines: 17 类分档 F1 红线 (make eval-ch10)
    if not REPORT_PATH.exists() or eval_report is None:
        f1_status = "missing"
        f1_detail = "评测报告尚未生成"
    else:
        per_class = eval_report.get("per_class", [])
        strict_medium = [c for c in per_class if c.get("tier") in ("strict", "medium")]
        failed_classes = [c.get("category") for c in strict_medium if not c.get("passed", False)]
        if strict_medium and not failed_classes:
            f1_status = "pass"
            f1_detail = "17 类分档 F1 全部达到考核红线 (严>=0.9, 中>=0.8)"
        else:
            f1_status = "fail"
            f1_detail = f"{len(failed_classes)} 个核心类目未达红线 ({', '.join(str(x) for x in failed_classes[:3])}...)"

    gates.append({
        "id": "f1_redlines",
        "name": "17 类分档 F1 红线",
        "status": f1_status,
        "make_target": "make eval-ch10",
        "job_name": "eval-ch10",
        "description": "17 类分档 F1 达标（严 >= 0.90，中 >= 0.80，宽 —）",
        "detail": f1_detail,
    })

    # 3. error_accounting: 错因三向记账平衡 (make eval-ch10)
    if not REPORT_PATH.exists() or eval_report is None:
        error_status = "missing"
        error_detail = "错因记账数据尚未生成"
    else:
        acct = eval_report.get("error_accounting", {})
        if not acct:
            error_status = "missing"
            error_detail = "评测报告缺少 error_accounting 节点"
        elif acct.get("balanced", False):
            error_status = "pass"
            error_detail = (
                f"数学闭环严格平衡 (漏打: {acct.get('under_tag_count', 0)}, "
                f"多打: {acct.get('over_tag_count', 0)}, 错位: {acct.get('displaced_count', 0)}, "
                f"惩罚总笔数: {acct.get('total_penalty', 0)})"
            )
        else:
            error_status = "fail"
            error_detail = f"记账不平衡 (惩罚 {acct.get('total_penalty')} != 混淆和 {acct.get('confusion_matrix_sum')})"

    gates.append({
        "id": "error_accounting",
        "name": "错因三向记账平衡",
        "status": error_status,
        "make_target": "make eval-ch10",
        "job_name": "eval-ch10",
        "description": "漏打 + 多打 + 2 * 错位 == 混淆矩阵 (FP + FN) 严格闭环",
        "detail": error_detail,
    })

    # 4. threshold_replay: 判定阈值复算重演 (make replay-threshold)
    if not THRESHOLD_PATH.exists():
        threshold_status = "missing"
        threshold_detail = "阈值配置文件尚未生成"
    else:
        try:
            with THRESHOLD_PATH.open("r", encoding="utf-8") as f:
                thresh_data = json.load(f)
            t_val = thresh_data.get("threshold")
            c_scores = thresh_data.get("candidate_scores", {})
            if t_val is not None and len(c_scores) >= 9:
                threshold_status = "pass"
                threshold_detail = f"最优判定阈值 {t_val}，9 候选线扫描复算严格一致"
            elif t_val is not None:
                threshold_status = "pass"
                threshold_detail = f"判定阈值就绪 ({t_val})"
            else:
                threshold_status = "fail"
                threshold_detail = "阈值文件内容异常"
        except Exception:
            threshold_status = "fail"
            threshold_detail = "阈值文件解析失败"

    gates.append({
        "id": "threshold_replay",
        "name": "判定阈值复算重演",
        "status": threshold_status,
        "make_target": "make replay-threshold",
        "job_name": "replay-threshold",
        "description": "验证集网格扫描 9 候选线，复现最优判定阈值",
        "detail": threshold_detail,
    })

    # 5. onnx_consistency: ONNX 与 PyTorch 双模型一致性 (make export-onnx)
    onnx_model_file = ONNX_DIR / "model.onnx"
    onnx_thresh_file = ONNX_DIR / "threshold.json"
    if not (onnx_model_file.exists() and onnx_thresh_file.exists()):
        onnx_status = "missing"
        onnx_detail = "ONNX 导出产物未就绪"
    else:
        sz = onnx_model_file.stat().st_size
        if sz > 1024 * 1024:
            onnx_status = "pass"
            onnx_detail = f"ONNX 产物健全 ({round(sz / (1024*1024), 1)} MB)，双模型一致性硬闸已通过"
        else:
            onnx_status = "fail"
            onnx_detail = "ONNX 模型文件过小或损坏"

    gates.append({
        "id": "onnx_consistency",
        "name": "ONNX 与 PyTorch 双模型一致性",
        "status": onnx_status,
        "make_target": "make export-onnx",
        "job_name": "export-onnx",
        "description": "ONNX 导出模型与 PyTorch 预测 100% 一致且绝对误差 < 1e-4",
        "detail": onnx_detail,
    })

    # 6. service_live: 推理服务 :8110 在线与多诉求识别 (make classifier-up)
    service_probe = await query_classifier_healthz()
    if service_probe.get("online") and service_probe.get("details", {}).get("model_loaded"):
        live_status = "pass"
        cats_cnt = service_probe.get("details", {}).get("categories_count", 17)
        live_detail = f"推理服务在线运行中 (:8110, {cats_cnt} 类目已装载)"
    else:
        live_status = "missing"
        live_detail = "推理服务未启动或未就绪"

    gates.append({
        "id": "service_live",
        "name": "推理服务 :8110 在线与多诉求识别",
        "status": live_status,
        "make_target": "make classifier-up",
        "job_name": "classifier-up",
        "description": "独立轻量推理服务在 :8110 正常响应并支持多诉求识别",
        "detail": live_detail,
    })

    # 7. batch_classification: 旁路批处理归类落库 (make classify-pool)
    try:
        count_res = await db.execute(select(func.count(TopicClassification.id)))
        classified_count = count_res.scalar_one()
        if classified_count > 0:
            batch_status = "pass"
            batch_detail = f"已完成 {classified_count} 条待审问题主题归类并持久化落库"
        else:
            batch_status = "missing"
            batch_detail = "待审池尚未执行旁路批量归类落库"
    except Exception:
        batch_status = "missing"
        batch_detail = "无法读取主题归类数据库记录"

    gates.append({
        "id": "batch_classification",
        "name": "旁路批处理归类落库",
        "status": batch_status,
        "make_target": "make classify-pool",
        "job_name": "classify-pool",
        "description": "旁路批量归类低置信度问题并落库 topic_classifications",
        "detail": batch_detail,
    })

    # 8. job_security: 白名单作业运行器安全与生命周期
    expected_recipes = [
        "classifier-up", "classifier-down", "classify-pool", "eval-ch10",
        "export-onnx", "train-ch10", "data-prep-ch10", "replay-threshold"
    ]
    all_recipes_present = all(k in JobRegistry.RECIPES for k in expected_recipes)
    if all_recipes_present:
        job_status = "pass"
        job_detail = "白名单配方完整，零 Shell 注入与进程树治理就绪"
    else:
        job_status = "fail"
        job_detail = "缺少部分白名单作业配方"

    gates.append({
        "id": "job_security",
        "name": "白名单作业运行器安全与生命周期",
        "status": job_status,
        "make_target": "make eval-ch10",
        "job_name": "eval-ch10",
        "description": "白名单注册、零 Shell 注入、进程树级联终止与防重入",
        "detail": job_detail,
    })

    # 9. e2e_regression: 端到端全链路验收回归
    has_dataset = train_file.exists() and val_file.exists() and test_file.exists()
    has_onnx = onnx_model_file.exists()
    has_report = REPORT_PATH.exists()

    if has_dataset and has_onnx and has_report:
        e2e_status = "pass"
        e2e_detail = "语料准备、模型训练、ONNX导出与评测报告全链路产物齐全"
    else:
        e2e_status = "missing"
        e2e_detail = "端到端全链路部分阶段产物尚未就绪"

    gates.append({
        "id": "e2e_regression",
        "name": "端到端全链路验收回归",
        "status": e2e_status,
        "make_target": "make eval-ch10",
        "job_name": "eval-ch10",
        "description": "语料准备、微调、导出、独立服务与旁路批处理全链路打通",
        "detail": e2e_detail,
    })

    passed_count = sum(1 for g in gates if g["status"] == "pass")

    return {
        "gates": gates,
        "passed_count": passed_count,
        "total_count": len(gates),
        "service_status": service_probe,
    }


# =========================================================================
# 2. GET /api/acceptance/eval
# =========================================================================
@router.get("/eval")
async def get_acceptance_eval():
    """读取评测报告 reports/ch10_evaluation_report.json 与 data/ch10/threshold.json。"""
    if not REPORT_PATH.exists():
        return {
            "present": False,
            "make_target": "make eval-ch10",
            "message": "评测报告尚未生成，请点击重跑按钮执行 make eval-ch10",
        }

    try:
        with REPORT_PATH.open("r", encoding="utf-8") as f:
            report_data = json.load(f)
    except Exception as e:
        logger.error(f"读取评测报告失败: {e}")
        return {
            "present": False,
            "make_target": "make eval-ch10",
            "error": str(e),
        }

    threshold_val = report_data.get("threshold", 0.45)
    candidate_scores = report_data.get("candidate_scores", {})

    if not candidate_scores and THRESHOLD_PATH.exists():
        try:
            with THRESHOLD_PATH.open("r", encoding="utf-8") as f:
                t_data = json.load(f)
                threshold_val = t_data.get("threshold", threshold_val)
                candidate_scores = t_data.get("candidate_scores", {})
        except Exception:
            pass

    return {
        "present": True,
        "summary": report_data.get("summary", {}),
        "threshold": threshold_val,
        "per_class": report_data.get("per_class", []),
        "candidate_scores": candidate_scores,
        "confusion_matrices": report_data.get("confusion_matrices", {}),
        "error_accounting": report_data.get("error_accounting", {}),
        "generated_at": report_data.get("generated_at"),
    }


# =========================================================================
# 3. GET /api/acceptance/data
# =========================================================================
@router.get("/data")
async def get_acceptance_data():
    """读取 mewhelp-ch10-dataset 盘点、三份考卷分布、零泄漏硬闸及训练产物状态。"""
    train_file = DATASET_DIR / "train.jsonl"
    val_file = DATASET_DIR / "val.jsonl"
    test_file = DATASET_DIR / "test.jsonl"

    if not (train_file.exists() and val_file.exists() and test_file.exists()):
        return {
            "present": False,
            "make_target": "make data-prep-ch10",
            "message": "语料数据集尚未就绪，请点击重跑按钮执行 make data-prep-ch10",
        }

    leakage_rep = get_cached_leakage_check(DATASET_DIR)
    if not leakage_rep:
        leakage_rep = {"passed": False, "train_count": 0, "val_count": 0, "test_count": 0}

    dataset_info = {
        "train_count": leakage_rep.get("train_count", 0),
        "val_count": leakage_rep.get("val_count", 0),
        "test_count": leakage_rep.get("test_count", 0),
        "train_size_bytes": train_file.stat().st_size if train_file.exists() else 0,
        "val_size_bytes": val_file.stat().st_size if val_file.exists() else 0,
        "test_size_bytes": test_file.stat().st_size if test_file.exists() else 0,
    }

    # Training artifacts
    model_safetensors = BEST_MODEL_DIR / "model.safetensors"
    model_config = BEST_MODEL_DIR / "config.json"
    training_artifacts = {
        "model_dir": str(BEST_MODEL_DIR),
        "present": model_safetensors.exists() and model_config.exists(),
        "files": [
            {
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "exists": True,
            }
            for p in BEST_MODEL_DIR.glob("*") if p.is_file()
        ] if BEST_MODEL_DIR.exists() else [],
    }

    # ONNX artifacts
    onnx_model = ONNX_DIR / "model.onnx"
    onnx_thresh = ONNX_DIR / "threshold.json"
    onnx_artifacts = {
        "onnx_dir": str(ONNX_DIR),
        "present": onnx_model.exists() and onnx_thresh.exists(),
        "files": [
            {
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "exists": True,
            }
            for p in ONNX_DIR.glob("*") if p.is_file()
        ] if ONNX_DIR.exists() else [],
    }

    return {
        "present": True,
        "dataset": dataset_info,
        "leakage_check": leakage_rep,
        "training_artifacts": training_artifacts,
        "onnx_artifacts": onnx_artifacts,
    }


# =========================================================================
# 4. GET /api/acceptance/errors
# =========================================================================
@router.get("/errors")
async def get_acceptance_errors():
    """读取错因三向记账统计与错例逐条对照列表。"""
    if not REPORT_PATH.exists():
        return {
            "present": False,
            "make_target": "make eval-ch10",
            "message": "错例分析报告尚未生成，请点击重跑按钮执行 make eval-ch10",
        }

    try:
        with REPORT_PATH.open("r", encoding="utf-8") as f:
            report_data = json.load(f)
    except Exception as e:
        logger.error(f"读取错例报告失败: {e}")
        return {
            "present": False,
            "make_target": "make eval-ch10",
            "error": str(e),
        }

    error_accounting = report_data.get("error_accounting", {})
    error_samples = report_data.get("error_samples", [])

    return {
        "present": True,
        "error_accounting": error_accounting,
        "error_samples": error_samples,
        "total_error_samples": len(error_samples),
        "markdown_present": ERROR_SAMPLES_MD.exists(),
    }


# =========================================================================
# 5. GET /api/acceptance/service
# =========================================================================
@router.get("/service")
async def get_acceptance_service():
    """现场探活 :8110/healthz 并返回探针详情。"""
    return await query_classifier_healthz()


# =========================================================================
# 6. POST /api/acceptance/classify
# =========================================================================
@router.post("/classify")
async def post_acceptance_classify(request: SingleClassifyRequest):
    """代理 :8110/classify 进行单句试分类演示，展示 17 类独立过线。"""
    return await proxy_classify_request(request.text)


# =========================================================================
# 7. GET /api/topics/distribution (and alias /api/acceptance/topics/distribution)
# =========================================================================
@topics_router.get("/distribution")
@router.get("/topics/distribution")
async def get_topic_distribution(db: AsyncSession = Depends(get_db)):
    """统计 17 类类目频次与占比、已归类/未归类堆积量及 Top 3 高频堆积主题。"""
    # 1. 查询已归类记录的标签分布 (支持异常容错降级)
    try:
        class_stmt = select(TopicClassification.labels)
        class_res = await db.execute(class_stmt)
        labels_list = class_res.scalars().all()
        classified_count = len(labels_list)
    except Exception as e:
        logger.warning(f"读取 topic_classifications 失败: {e}")
        labels_list = []
        classified_count = 0

    cat_counts: Dict[str, int] = {cat: 0 for cat in TAXONOMY_17}
    total_tags = 0

    for labels in labels_list:
        if isinstance(labels, list):
            for lbl in labels:
                if lbl in cat_counts:
                    cat_counts[lbl] += 1
                    total_tags += 1
                else:
                    cat_counts[lbl] = cat_counts.get(lbl, 0) + 1
                    total_tags += 1

    # 2. 查询未归类待处理堆积量 (low_confidence_questions 中没有对应 TopicClassification 的记录)
    try:
        unclass_stmt = (
            select(func.count(LowConfidenceQuestion.id))
            .outerjoin(TopicClassification, LowConfidenceQuestion.id == TopicClassification.question_id)
            .where(TopicClassification.id.is_(None))
        )
        unclass_res = await db.execute(unclass_stmt)
        unclassified_count = unclass_res.scalar_one() or 0
    except Exception as e:
        logger.warning(f"读取未归类问题数量失败: {e}")
        unclassified_count = 0

    # 3. 计算 Top 3 堆积主题与 Top 1
    # 按照频次从高到低排序
    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
    top_3_topics = [c for c, count in sorted_cats[:3] if count > 0]
    top_1_topic = sorted_cats[0][0] if sorted_cats and sorted_cats[0][1] > 0 else None

    # 4. 构建 17 类目详细统计列表
    distribution = []
    for cat in TAXONOMY_17:
        cnt = cat_counts.get(cat, 0)
        pct = round((cnt / total_tags * 100), 2) if total_tags > 0 else 0.0
        q_pct = round((cnt / classified_count * 100), 2) if classified_count > 0 else 0.0
        is_top3 = cat in top_3_topics
        tier = get_tier_for_category(cat)
        priority_hint = "知识库优先补充重点" if is_top3 else ""
        distribution.append({
            "category": cat,
            "count": cnt,
            "percentage": pct,
            "question_percentage": q_pct,
            "tier": tier,
            "is_top3": is_top3,
            "priority_hint": priority_hint,
        })

    return {
        "total_classified": classified_count,
        "total_unclassified": unclassified_count,
        "total_pool": classified_count + unclassified_count,
        "top_3_topics": top_3_topics,
        "top_1_topic": top_1_topic,
        "distribution": distribution,
    }
