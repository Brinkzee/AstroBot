import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to sys.path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import AsyncSessionLocal
from app.services.eval.eval_pipeline import EvalPipelineService

async def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Pipeline")
    parser.add_argument("--trigger", choices=["manual", "cron", "手动", "定时"], default="手动", help="Trigger source")
    parser.add_argument("--samples", type=int, default=None, help="Sample limit")
    args = parser.parse_args()

    trigger_val = args.trigger
    if trigger_val == "manual":
        trigger_val = "手动"
    elif trigger_val == "cron":
        trigger_val = "定时"

    service = EvalPipelineService()
    
    async with AsyncSessionLocal() as db:
        run = await service.run_eval_round(
            db=db,
            triggered_by=trigger_val,
            sample_limit=args.samples
        )
        
        runs = await service.get_recent_runs_with_deltas(db, limit=2)
        
        if not runs:
            print("No evaluation runs found.")
            return

        latest = runs[0]
        
        print("\n=== 评估结果对比报表 ===")
        print(f"{'Metric':<15} | {'Current Score':<15} | {'Previous Score':<15} | {'Delta':<10} | {'Trend'}")
        print("-" * 75)
        
        metrics = latest["metrics"]
        deltas = latest.get("deltas", {})
        prior = runs[1] if len(runs) > 1 else None
        
        for k, current_val in metrics.items():
            delta_val = deltas.get(k, 0.0)
            trend_str = deltas.get(f"{k}_trend", "─ 持平")
            prior_val = prior["metrics"].get(k, 0.0) if prior else current_val
            
            print(f"{k:<15} | {current_val:<15.4f} | {prior_val:<15.4f} | {delta_val:<10.4f} | {trend_str}")
            
            if delta_val < 0:
                print(f"[WARNING] {k} dropped by {delta_val:.4f}!")

if __name__ == "__main__":
    asyncio.run(main())
