import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.eval_run import EvalRun, TriggeredBy


def _load_jsonl_sync(dataset_path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


class EvalPipelineService:
    async def run_eval_round(
        self,
        db: AsyncSession,
        triggered_by: str = "手动",
        sample_limit: Optional[int] = None,
        dataset_path: Optional[Path] = None,
        retriever=None,
        judge=None,
    ) -> EvalRun:
        if dataset_path is None:
            dataset_path = Path("tests/data/eval_ch04.jsonl")

        samples = await asyncio.to_thread(_load_jsonl_sync, Path(dataset_path))

        if sample_limit is not None:
            samples = samples[:sample_limit]

        from app.services.rag.evaluator import (
            RAGEvaluator,
            compute_retrieval_metrics,
            evaluate_faithfulness,
        )

        recall_at_3 = 0.0
        recall_at_5 = 0.0
        recall_at_10 = 0.0
        mrr = 0.0
        faithfulness = 0.0
        graded_count = 0
        refusal_total = 0
        refusal_hits = 0

        mock_evaluator = RAGEvaluator(is_mock=True) if retriever is None else None

        for sample in samples:
            query = sample["query"]
            should_refuse = bool(sample.get("should_refuse", False))

            if retriever is not None:
                if hasattr(retriever, "retrieve_with_strategy"):
                    res = await retriever.retrieve_with_strategy(
                        query, strategy="hybrid_rerank"
                    )
                    if hasattr(res, "ranked_docs") and res.ranked_docs:
                        docs = res.ranked_docs
                    elif hasattr(res, "docs"):
                        docs = res.docs
                    else:
                        docs = res or []
                else:
                    res = await retriever.retrieve(query)
                    docs = res.docs if hasattr(res, "docs") else (res or [])
            else:
                docs, _ = mock_evaluator._mock_retrieval(sample, strategy="hybrid_rerank")

            if should_refuse:
                refusal_total += 1
                refused = len(docs) == 0
                if refused:
                    refusal_hits += 1
                    faithfulness += 1.0
                continue

            graded_count += 1
            r_metrics = compute_retrieval_metrics(
                docs,
                sample.get("expect_section") or [],
                sample.get("expect_sections_all"),
            )

            recall_at_3 += r_metrics.get("recall@3", 0.0)
            recall_at_5 += r_metrics.get("recall@5", 0.0)
            recall_at_10 += r_metrics.get("recall@10", 0.0)
            mrr += r_metrics.get("mrr", 0.0)

            # Simulated answer for faithfulness evaluation
            simulated_answer = "根据相关规定，这是系统生成的模拟回复。"
            citations = list(docs[:3]) if docs else []
            f_res = await asyncio.to_thread(
                evaluate_faithfulness,
                query=query,
                answer=simulated_answer,
                citations=citations,
                model=judge,
            )
            faithfulness += 1.0 if f_res.get("faithful", True) else 0.0

        n_graded = graded_count if graded_count > 0 else 1
        n_total = len(samples) if len(samples) > 0 else 1
        avg_recall_3 = recall_at_3 / n_graded
        avg_recall_5 = recall_at_5 / n_graded
        avg_recall_10 = recall_at_10 / n_graded
        avg_mrr = mrr / n_graded
        avg_faithfulness = faithfulness / n_total

        metrics: Dict[str, Any] = {
            "recall_at_3": round(avg_recall_3, 4),
            "recall_at_5": round(avg_recall_5, 4),
            "recall_at_10": round(avg_recall_10, 4),
            "mrr": round(avg_mrr, 4),
            "faithfulness": round(avg_faithfulness, 4),
        }
        if refusal_total > 0:
            metrics["refusal_rate"] = round(refusal_hits / refusal_total, 4)

        run = EvalRun(
            triggered_by=triggered_by,
            dataset_size=len(samples),
            metrics=metrics,
        )

        db.add(run)
        await db.commit()
        await db.refresh(run)

        return run

    async def get_recent_runs_with_deltas(
        self, db: AsyncSession, limit: int = 10
    ) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        stmt = select(EvalRun).order_by(EvalRun.id.desc()).limit(limit + 1)
        result = await db.execute(stmt)
        runs = list(result.scalars().all())

        output: List[Dict[str, Any]] = []
        display_count = min(len(runs), limit)
        for i in range(display_count):
            current = runs[i]
            current_metrics = current.metrics or {}

            run_dict: Dict[str, Any] = {
                "id": current.id,
                "triggered_by": current.triggered_by,
                "dataset_size": current.dataset_size,
                "metrics": current_metrics,
                "created_at": current.created_at,
                "deltas": {},
            }

            if i + 1 < len(runs):
                prior = runs[i + 1]
                prior_metrics = prior.metrics or {}
                for k, v in current_metrics.items():
                    if not isinstance(v, (int, float)):
                        continue
                    raw_prior = prior_metrics.get(k, 0.0)
                    prior_val = float(raw_prior) if isinstance(raw_prior, (int, float)) else 0.0
                    delta = round(float(v) - prior_val, 4)

                    if delta > 0:
                        trend = "↗ 上升"
                    elif delta < 0:
                        trend = "↘ 下滑"
                    else:
                        trend = "─ 持平"

                    run_dict["deltas"][k] = delta
                    run_dict["deltas"][f"{k}_trend"] = trend
            else:
                for k in current_metrics.keys():
                    run_dict["deltas"][k] = 0.0
                    run_dict["deltas"][f"{k}_trend"] = "─ 持平"

            output.append(run_dict)

        return output
