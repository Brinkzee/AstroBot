import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.eval_run import EvalRun, TriggeredBy

class EvalPipelineService:
    async def run_eval_round(
        self,
        db: AsyncSession,
        triggered_by: str = "手动",
        sample_limit: Optional[int] = None,
        dataset_path: Optional[Path] = None,
        retriever=None,
        judge=None
    ) -> EvalRun:
        if dataset_path is None:
            dataset_path = Path("tests/data/eval_ch04.jsonl")
        
        samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        if sample_limit is not None:
            samples = samples[:sample_limit]
            
        from app.services.rag.evaluator import compute_retrieval_metrics, evaluate_faithfulness, RAGEvaluator
        
        recall_at_3 = 0.0
        recall_at_5 = 0.0
        recall_at_10 = 0.0
        mrr = 0.0
        faithfulness = 0.0
        
        mock_evaluator = RAGEvaluator(is_mock=True) if retriever is None else None
        
        for sample in samples:
            query = sample["query"]
            
            if retriever is not None:
                if hasattr(retriever, "retrieve_with_strategy"):
                    # Assuming it returns docs
                    docs = await retriever.retrieve_with_strategy(query, strategy="hybrid_rerank")
                else:
                    docs = await retriever.retrieve(query)
            else:
                docs, _ = mock_evaluator._mock_retrieval(sample, strategy="hybrid_rerank")
                
            r_metrics = compute_retrieval_metrics(
                docs,
                sample.get("expect_section"),
                sample.get("expect_sections_all")
            )
            
            recall_at_3 += r_metrics.get("recall@3", 0.0)
            recall_at_5 += r_metrics.get("recall@5", 0.0)
            recall_at_10 += r_metrics.get("recall@10", 0.0)
            mrr += r_metrics.get("mrr", 0.0)
            
            # Simulated answer for faithfulness evaluation
            simulated_answer = "根据相关规定，这是系统生成的模拟回复。"
            citations = docs[:3] if docs else []
            f_res = evaluate_faithfulness(
                query=query,
                answer=simulated_answer,
                citations=citations,
                model=judge
            )
            faithfulness += 1.0 if f_res.get("faithful", True) else 0.0

        n = len(samples) if len(samples) > 0 else 1
        avg_recall_3 = recall_at_3 / n
        avg_recall_5 = recall_at_5 / n
        avg_recall_10 = recall_at_10 / n
        avg_mrr = mrr / n
        avg_faithfulness = faithfulness / n
        
        metrics = {
            "recall_at_3": round(avg_recall_3, 4),
            "recall_at_5": round(avg_recall_5, 4),
            "recall_at_10": round(avg_recall_10, 4),
            "mrr": round(avg_mrr, 4),
            "faithfulness": round(avg_faithfulness, 4),
        }
        
        run = EvalRun(
            triggered_by=triggered_by,
            dataset_size=len(samples),
            metrics=metrics
        )
        
        db.add(run)
        await db.commit()
        await db.refresh(run)
        
        return run

    async def get_recent_runs_with_deltas(self, db: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
        stmt = select(EvalRun).order_by(EvalRun.id.desc()).limit(limit)
        result = await db.execute(stmt)
        runs = list(result.scalars().all())
        
        output = []
        for i in range(len(runs)):
            current = runs[i]
            
            run_dict = {
                "id": current.id,
                "triggered_by": current.triggered_by,
                "dataset_size": current.dataset_size,
                "metrics": current.metrics,
                "created_at": current.created_at,
                "deltas": {}
            }
            
            if i < len(runs) - 1:
                prior = runs[i + 1]
                for k, v in current.metrics.items():
                    prior_val = prior.metrics.get(k, 0.0)
                    delta = round(v - prior_val, 4)
                    
                    if delta > 0:
                        trend = "↗ 上升"
                    elif delta < 0:
                        trend = "↘ 下滑"
                    else:
                        trend = "─ 持平"
                        
                    run_dict["deltas"][k] = delta
                    run_dict["deltas"][f"{k}_trend"] = trend
            else:
                for k in current.metrics.keys():
                    run_dict["deltas"][k] = 0.0
                    run_dict["deltas"][f"{k}_trend"] = "─ 持平"
            
            output.append(run_dict)
            
        return output
