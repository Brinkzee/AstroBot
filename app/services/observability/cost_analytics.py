import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def _extract_tokens_from_usage(usage_obj: Any) -> Tuple[int, int]:
    """兼容 dict 或 Pydantic/对象形式的 usage / usage_details 提取 (prompt_tokens, completion_tokens)。"""
    if usage_obj is None:
        return 0, 0
    if isinstance(usage_obj, dict):
        p = (
            usage_obj.get("input")
            or usage_obj.get("prompt_tokens")
            or usage_obj.get("promptTokens")
            or 0
        )
        c = (
            usage_obj.get("output")
            or usage_obj.get("completion_tokens")
            or usage_obj.get("completionTokens")
            or 0
        )
        return int(p or 0), int(c or 0)
    p = (
        getattr(usage_obj, "input", None)
        or getattr(usage_obj, "prompt_tokens", None)
        or getattr(usage_obj, "promptTokens", None)
        or 0
    )
    c = (
        getattr(usage_obj, "output", None)
        or getattr(usage_obj, "completion_tokens", None)
        or getattr(usage_obj, "completionTokens", None)
        or 0
    )
    return int(p or 0), int(c or 0)


class CostAnalyticsService:
    def __init__(self, prompt_cost_per_m: float = 2.5, completion_cost_per_m: float = 10.0):
        self.prompt_cost_per_m = prompt_cost_per_m
        self.completion_cost_per_m = completion_cost_per_m

    def aggregate_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        intent_stats: Dict[str, Dict[str, Any]] = {}
        sum_requests = len(records)
        sum_tokens = 0
        sum_cost = 0.0

        for r in records:
            intent = r.get("intent") or "UNKNOWN"
            p_tokens = int(r.get("prompt_tokens") or 0)
            c_tokens = int(r.get("completion_tokens") or 0)
            t_tokens = p_tokens + c_tokens

            if intent not in intent_stats:
                intent_stats[intent] = {
                    "intent": intent,
                    "total_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "direct_cost_usd": 0.0,
                }

            stats = intent_stats[intent]
            stats["total_calls"] += 1
            stats["prompt_tokens"] += p_tokens
            stats["completion_tokens"] += c_tokens
            stats["total_tokens"] += t_tokens
            direct_cost = r.get("total_cost")
            if isinstance(direct_cost, (int, float)) and direct_cost > 0 and t_tokens == 0:
                stats["direct_cost_usd"] += float(direct_cost)

            sum_tokens += t_tokens

        # Pass 1: compute estimated cost per intent and sum_cost
        items = []
        for stats in intent_stats.values():
            p_tokens = stats["prompt_tokens"]
            c_tokens = stats["completion_tokens"]
            token_cost = (
                (p_tokens / 1_000_000 * self.prompt_cost_per_m)
                + (c_tokens / 1_000_000 * self.completion_cost_per_m)
            )
            cost = token_cost + stats.pop("direct_cost_usd", 0.0)
            sum_cost += cost
            stats["estimated_cost_usd"] = cost
            items.append(stats)

        # Pass 2: compute true cost share percentage (cost / sum_cost * 100)
        for stats in items:
            stats["cost_percentage"] = (
                (stats["estimated_cost_usd"] / sum_cost * 100.0) if sum_cost > 0 else 0.0
            )

        items.sort(
            key=lambda x: (x["estimated_cost_usd"], x["total_tokens"]),
            reverse=True,
        )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "total_requests": sum_requests,
            "total_tokens": sum_tokens,
            "total_cost_usd": sum_cost,
            "items": items,
        }

    def fetch_and_generate_report(
        self,
        output_file: Optional[Path] = None,
        mock: bool = False,
        fallback_to_mock: bool = False,
    ) -> Dict[str, Any]:
        is_mock = bool(mock)
        fetch_error: Optional[str] = None

        if mock:
            records = [
                {"intent": "refund", "prompt_tokens": 12000, "completion_tokens": 4000},
                {"intent": "refund", "prompt_tokens": 10000, "completion_tokens": 3000},
                {"intent": "knowledge", "prompt_tokens": 15000, "completion_tokens": 2000},
                {"intent": "chitchat", "prompt_tokens": 1000, "completion_tokens": 200},
                {"intent": "UNKNOWN", "prompt_tokens": 500, "completion_tokens": 50},
            ]
        else:
            lf = None
            try:
                from langfuse import Langfuse

                lf = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )

                api_client = getattr(lf, "api", None) or getattr(lf, "client", None)
                if api_client is None:
                    raise RuntimeError("Langfuse client exposes neither .api nor .client")

                page = 1
                records = []
                while True:
                    response = api_client.trace.list(page=page)
                    traces = getattr(response, "data", None) or []
                    for trace in traces:
                        meta_dict = getattr(trace, "metadata", None)
                        intent = (
                            meta_dict.get("intent")
                            if isinstance(meta_dict, dict)
                            else None
                        ) or "UNKNOWN"

                        p_tokens = int(getattr(trace, "promptTokens", 0) or 0)
                        c_tokens = int(getattr(trace, "completionTokens", 0) or 0)
                        if getattr(trace, "usage", None) is not None:
                            up, uc = _extract_tokens_from_usage(trace.usage)
                            if up or uc:
                                p_tokens, c_tokens = up, uc

                        # Langfuse v3/v4 stores token usage on observations rather than trace root
                        if (
                            p_tokens == 0
                            and c_tokens == 0
                            and hasattr(api_client, "observations")
                            and getattr(trace, "id", None)
                        ):
                            try:
                                obs_resp = api_client.observations.get_many(trace_id=trace.id)
                                for obs in getattr(obs_resp, "data", None) or []:
                                    op, oc = _extract_tokens_from_usage(
                                        getattr(obs, "usage_details", None)
                                        or getattr(obs, "usage", None)
                                    )
                                    p_tokens += op
                                    c_tokens += oc
                            except Exception as obs_err:
                                logger.debug(f"Skipping observations fetch for trace {trace.id}: {obs_err}")

                        records.append({
                            "intent": intent,
                            "prompt_tokens": p_tokens,
                            "completion_tokens": c_tokens,
                            "total_cost": getattr(trace, "total_cost", None),
                        })

                    meta = getattr(response, "meta", None)
                    total_pages = (
                        getattr(meta, "total_pages", None)
                        or getattr(meta, "totalPages", None)
                        or 1
                    )
                    if page >= int(total_pages):
                        break
                    page += 1
            except Exception as e:
                logger.warning(f"Failed to fetch cost analytics from Langfuse: {e}")
                if not fallback_to_mock:
                    raise
                is_mock = True
                fetch_error = str(e)
                records = [
                    {"intent": "refund", "prompt_tokens": 12000, "completion_tokens": 4000},
                    {"intent": "refund", "prompt_tokens": 10000, "completion_tokens": 3000},
                    {"intent": "knowledge", "prompt_tokens": 15000, "completion_tokens": 2000},
                    {"intent": "chitchat", "prompt_tokens": 1000, "completion_tokens": 200},
                    {"intent": "UNKNOWN", "prompt_tokens": 500, "completion_tokens": 50},
                ]
            finally:
                if lf is not None:
                    try:
                        if hasattr(lf, "shutdown"):
                            lf.shutdown()
                        elif hasattr(lf, "flush"):
                            lf.flush()
                    except Exception as close_err:
                        logger.debug(f"Failed to shutdown Langfuse client cleanly: {close_err}")

        report = self.aggregate_records(records)
        report["is_mock"] = is_mock
        if fetch_error is not None:
            report["error"] = fetch_error

        if output_file is None:
            output_file = Path("reports/cost_by_intent.json")

        # Never silently overwrite an existing live report with fallback mock data
        if not (is_mock and not mock and output_file.exists()):
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)

        return report
