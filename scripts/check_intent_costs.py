import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.observability.cost_analytics import CostAnalyticsService


def run_cli():
    parser = argparse.ArgumentParser(description="Generate cost analytics report grouped by intent.")
    parser.add_argument("--output", type=str, default="reports/cost_by_intent.json",
                        help="Path to save the JSON report")
    parser.add_argument("--mock", action="store_true",
                        help="Generate verifiable sample report if Langfuse server is not running")

    args = parser.parse_args()

    service = CostAnalyticsService()
    report = service.fetch_and_generate_report(output_file=Path(args.output), mock=args.mock)

    print("\n" + "="*85)
    print(f"{'Intent':<15} | {'Calls':<6} | {'Prompt Tokens':<14} | {'Completion Tokens':<18} | {'Total Tokens':<13} | {'Cost ($)':<9} | {'% Total':<8}")
    print("-" * 85)

    for i, item in enumerate(report["items"]):
        intent = item["intent"]
        if i == 0:
            intent = f"{intent} *" # highlight #1 most expensive

        calls = item["total_calls"]
        p_tokens = item["prompt_tokens"]
        c_tokens = item["completion_tokens"]
        t_tokens = item["total_tokens"]
        cost = f"{item['estimated_cost_usd']:.4f}"
        pct = f"{item['cost_percentage']:.1f}%"
        
        print(f"{intent:<15} | {calls:<6} | {p_tokens:<14} | {c_tokens:<18} | {t_tokens:<13} | {cost:<9} | {pct:<8}")
        
    print("="*85)
    print(f"* Highlighted as #1 most expensive intent.")
    print(f"Total Requests: {report['total_requests']}")
    print(f"Total Tokens:   {report['total_tokens']}")
    print(f"Total Cost:     ${report['total_cost_usd']:.4f}\n")


if __name__ == "__main__":
    run_cli()
