import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.services.workflow.nodes.gate import compute_evidence_confidence
from app.services.workflow.nodes.knowledge_node import _extract_retrieval_items

logger = logging.getLogger(__name__)


def _load_eval_items(dataset_path: str, sample: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载评估集 JSONL，按真实 should_refuse 字段提取正负样本标注。"""
    items: List[Dict[str, Any]] = []
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"评估集文件不存在: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            query = str(data.get("query") or data.get("question") or "").strip()
            if not query:
                continue
            is_positive = not bool(data.get("should_refuse", False))
            items.append({
                "id": data.get("id", ""),
                "bucket": data.get("bucket", ""),
                "query": query,
                "is_positive": is_positive,
            })

    if sample is not None and sample > 0:
        positives = [x for x in items if x["is_positive"]][:sample]
        negatives = [x for x in items if not x["is_positive"]][:sample]
        items = positives + negatives

    return items


async def _build_default_retriever(
    min_score: float = 0.1,
    offline: bool = False,
    milvus_uri: Optional[str] = None,
) -> Tuple[Any, List[str]]:
    """构造真实 AdvancedKnowledgeRetriever，并在外部 API 配额失效时快速探针降级，避免批量重试阻塞。"""
    from app.services.rag.advanced_retriever import AdvancedKnowledgeRetriever
    from app.services.rag.embedding import BGEEmbeddingClient
    from app.services.rag.milvus_client import MilvusKnowledgeStore
    from app.services.rag.query_processor import QueryProcessor
    from app.services.rag.reranker import BGERerankerClient

    degraded: List[str] = []
    store = MilvusKnowledgeStore(uri=milvus_uri) if milvus_uri is not None else None

    if offline or os.getenv("CALIBRATE_OFFLINE") == "1":
        retriever = AdvancedKnowledgeRetriever(
            store=store,
            min_score=min_score,
            query_processor=QueryProcessor(mock=True),
            embedding_client=BGEEmbeddingClient(mock=True),
            reranker_client=BGERerankerClient(mock=True),
        )
        return retriever, ["query_processor", "embedding_client", "reranker_client"]

    qp = QueryProcessor(max_retries=1)
    emb = BGEEmbeddingClient(max_retries=1, retry_delay=0.1)
    reranker = BGERerankerClient(max_retries=1, retry_delay=0.1)

    # 1. 探测 QueryProcessor 外部 LLM 是否可用
    if not qp.is_mock and getattr(qp, "_llm", None) is not None:
        try:
            from langchain_core.messages import HumanMessage
            await qp._llm.ainvoke([HumanMessage(content="退货政策")])
        except Exception as e:
            logger.warning(f"校准探针：QueryProcessor 外部模型不可用 ({e})，切换为原词直通模式")
            qp.is_mock = True
            qp._llm = None
            degraded.append("query_processor")

    # 2. 探测 BGEEmbeddingClient 外部推理端点是否可用
    if not emb.is_mock and getattr(emb, "client", None) is not None:
        try:
            await asyncio.to_thread(emb.client.feature_extraction, ["退货政策"], model=emb.model)
        except Exception as e:
            logger.warning(f"校准探针：BGE-M3 外部端点不可用 ({e})，切换为本地确定性向量降级模式")
            emb.is_mock = True
            emb.client = None
            degraded.append("embedding_client")

    # 3. 探测 BGERerankerClient 外部推理端点是否可用
    if not reranker.is_mock and getattr(reranker, "client", None) is not None:
        try:
            await asyncio.to_thread(
                reranker._call_hf_api,
                "退货政策",
                ["7天无理由退货政策"],
            )
        except Exception as e:
            logger.warning(f"校准探针：BGE-Reranker 外部端点不可用 ({e})，切换为本地语义打分降级模式")
            reranker.is_mock = True
            reranker.client = None
            degraded.append("reranker_client")

    retriever = AdvancedKnowledgeRetriever(
        store=store,
        min_score=min_score,
        query_processor=qp,
        embedding_client=emb,
        reranker_client=reranker,
    )
    return retriever, degraded


async def _retrieve_docs_for_query(
    retriever: Any,
    query: str,
    strategy: str = "hybrid_rerank",
    min_score: float = 0.1,
) -> List[Dict[str, Any]]:
    if hasattr(retriever, "retrieve_with_strategy"):
        try:
            res = await retriever.retrieve_with_strategy(
                query=query,
                strategy=strategy,
                min_score=min_score,
            )
        except TypeError:
            res = await retriever.retrieve_with_strategy(query=query, min_score=min_score)
        return _extract_retrieval_items(res)
    if hasattr(retriever, "retrieve"):
        res = await retriever.retrieve(query=query, top_k=5)
        return _extract_retrieval_items(res)
    raise ValueError("retriever 必须实现 retrieve_with_strategy 或 retrieve 方法")


def _write_json_report(output_path: str, report: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)


async def run_calibration_async(
    dataset_path: str = "tests/data/eval_ch04.jsonl",
    output_path: str = "reports/confidence_calibration.json",
    retriever: Optional[Any] = None,
    strategy: str = "hybrid_rerank",
    min_score: float = 0.1,
    sample: Optional[int] = None,
    offline: bool = False,
    milvus_uri: Optional[str] = None,
) -> Dict[str, Any]:
    eval_items = await asyncio.to_thread(_load_eval_items, dataset_path, sample)
    if not eval_items:
        raise ValueError(f"评估集为空，无法执行置信度阈值校准: {dataset_path}")

    degraded_components: List[str] = []
    active_retriever = retriever
    own_retriever = False
    if active_retriever is None:
        active_retriever, degraded_components = await _build_default_retriever(
            min_score=min_score,
            offline=offline,
            milvus_uri=milvus_uri,
        )
        own_retriever = True

    samples: List[Dict[str, Any]] = []
    try:
        for item in eval_items:
            docs = await _retrieve_docs_for_query(
                active_retriever,
                query=item["query"],
                strategy=strategy,
                min_score=min_score,
            )
            conf = compute_evidence_confidence(docs)
            samples.append({
                "id": item["id"],
                "is_positive": item["is_positive"],
                "confidence": conf,
            })
    finally:
        if own_retriever and active_retriever is not None and hasattr(active_retriever, "close"):
            try:
                active_retriever.close()
            except Exception as close_err:
                logger.debug(f"关闭校准检索器异常: {close_err}")

    positive_count = sum(1 for s in samples if s["is_positive"])
    negative_count = len(samples) - positive_count

    thresholds = [round(0.10 + i * 0.02, 2) for i in range(41)]

    best_key = (-2.0, -1.0)
    tied_best: List[Dict[str, Any]] = []
    grid_scan: List[Dict[str, float]] = []

    for t in thresholds:
        tp = sum(1 for s in samples if s["is_positive"] and s["confidence"] >= t)
        fp = sum(1 for s in samples if not s["is_positive"] and s["confidence"] >= t)
        tn = sum(1 for s in samples if not s["is_positive"] and s["confidence"] < t)
        fn = sum(1 for s in samples if s["is_positive"] and s["confidence"] < t)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = 1.0 - specificity
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = sensitivity
        accuracy = (tp + tn) / len(samples) if samples else 0.0

        youden_j = sensitivity + specificity - 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        row = {
            "threshold": t,
            "tpr": round(sensitivity, 4),
            "fpr": round(fpr, 4),
            "youden_j": round(youden_j, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "accuracy": round(accuracy, 4),
        }
        grid_scan.append(row)

        candidate_key = (round(youden_j, 6), round(f1, 6))
        candidate_entry = {
            "threshold": t,
            "youden_j": youden_j,
            "f1": f1,
            "metrics": {
                "sensitivity": round(sensitivity, 4),
                "specificity": round(specificity, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "accuracy": round(accuracy, 4),
            },
        }
        if candidate_key > best_key:
            best_key = candidate_key
            tied_best = [candidate_entry]
        elif candidate_key == best_key:
            tied_best.append(candidate_entry)

    # 若多个阈值并列达到最优 (Youden's J, F1)，取平台区间中位点以获得最大正负样本安全间隔
    chosen = tied_best[len(tied_best) // 2] if tied_best else {
        "threshold": 0.40,
        "youden_j": 0.0,
        "f1": 0.0,
        "metrics": {},
    }
    best_t = chosen["threshold"]
    best_j = chosen["youden_j"]
    best_f1 = chosen["f1"]
    best_metrics = chosen["metrics"]

    active_t = getattr(settings, "evidence_confidence_threshold", 0.40)

    report: Dict[str, Any] = {
        "calibrated_at": datetime.datetime.now().isoformat(),
        "dataset_path": dataset_path,
        "retrieval_strategy": strategy,
        "degraded_components": degraded_components,
        "sample_count": len(samples),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "recommended_threshold": round(best_t, 2),
        "best_youden_j": round(best_j, 4),
        "best_f1": round(best_f1, 4),
        "active_threshold": active_t,
        "threshold_mismatch": abs(best_t - active_t) > 0.01,
        "metrics": best_metrics,
        "grid_scan": grid_scan,
    }

    await asyncio.to_thread(_write_json_report, output_path, report)

    print(
        f"Calibration finished on {len(samples)} real samples "
        f"(pos={positive_count}, neg={negative_count}). "
        f"Recommended Threshold: {best_t:.2f} (J: {best_j:.4f}, F1: {best_f1:.4f})"
    )
    print(f"Report saved to {output_path}")
    return report


def run_calibration(
    dataset_path: str = "tests/data/eval_ch04.jsonl",
    output_path: str = "reports/confidence_calibration.json",
    retriever: Optional[Any] = None,
    strategy: str = "hybrid_rerank",
    min_score: float = 0.1,
    sample: Optional[int] = None,
    offline: bool = False,
    milvus_uri: Optional[str] = None,
) -> Dict[str, Any]:
    return asyncio.run(
        run_calibration_async(
            dataset_path=dataset_path,
            output_path=output_path,
            retriever=retriever,
            strategy=strategy,
            min_score=min_score,
            sample=sample,
            offline=offline,
            milvus_uri=milvus_uri,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于真实检索评估集的置信度闸门网格校准脚本")
    parser.add_argument("--dataset", type=str, default="tests/data/eval_ch04.jsonl", help="评估集 JSONL 路径")
    parser.add_argument("--output", type=str, default="reports/confidence_calibration.json", help="校准报表输出路径")
    parser.add_argument("--strategy", type=str, default="hybrid_rerank", help="检索策略")
    parser.add_argument("--min-score", type=float, default=0.1, help="检索最低过滤分数")
    parser.add_argument("--sample", type=int, default=None, help="正负样本各采样条数（默认 None 跑全量）")
    parser.add_argument("--offline", action="store_true", help="显式跳过外部 API 探针，直接使用本地 Milvus + 本地降级打分")
    parser.add_argument("--milvus-uri", type=str, default=None, help="可选的自定义 Milvus-Lite 本地路径或服务 URI")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_calibration(
        dataset_path=args.dataset,
        output_path=args.output,
        strategy=args.strategy,
        min_score=args.min_score,
        sample=args.sample,
        offline=args.offline,
        milvus_uri=args.milvus_uri,
    )
