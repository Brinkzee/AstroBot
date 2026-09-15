import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from app.services.observability.cost_analytics import CostAnalyticsService
from app.services.workflow.nodes.pre_nodes import intent_recognition_node
from scripts.check_intent_costs import run_cli


def test_aggregate_records_sorting_and_math():
    records = [
        {"intent": "refund", "prompt_tokens": 1000, "completion_tokens": 500},
        {"intent": "refund", "prompt_tokens": 2000, "completion_tokens": 500},
        {"intent": "chitchat", "prompt_tokens": 500, "completion_tokens": 100},
        {"intent": "knowledge", "prompt_tokens": 5000, "completion_tokens": 1000},
    ]

    service = CostAnalyticsService()
    report = service.aggregate_records(records)

    assert report["total_requests"] == 4
    assert report["total_tokens"] == 10600
    # Expected: refund: 4000, chitchat: 600, knowledge: 6000
    # knowledge should be first (6000 tokens)
    items = report["items"]
    assert len(items) == 3
    assert items[0]["intent"] == "knowledge"
    assert items[0]["total_tokens"] == 6000
    assert items[0]["total_calls"] == 1
    
    assert items[1]["intent"] == "refund"
    assert items[1]["total_tokens"] == 4000
    assert items[1]["total_calls"] == 2
    
    assert items[2]["intent"] == "chitchat"
    assert items[2]["total_tokens"] == 600
    assert items[2]["total_calls"] == 1

    # Check percentages
    assert items[0]["cost_percentage"] == pytest.approx(6000 / 10600 * 100)
    assert items[1]["cost_percentage"] == pytest.approx(4000 / 10600 * 100)
    assert items[2]["cost_percentage"] == pytest.approx(600 / 10600 * 100)

    # Check cost (assuming $2.5/1M prompt, $10/1M completion)
    # knowledge: 5000 * 2.5/1M + 1000 * 10.0/1M = 0.0125 + 0.01 = 0.0225
    assert items[0]["estimated_cost_usd"] == pytest.approx(0.0225)


@pytest.mark.asyncio
async def test_intent_node_updates_langfuse_trace():
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=MagicMock(content='{"intent": "售后", "confidence": 0.95}'))

    state = {"input_query": "东西坏了怎么修"}

    with patch("app.services.observability.langfuse_service.LangfuseManager.update_current_trace") as mock_update:
        res = await intent_recognition_node(state, model=mock_model)
        assert res["intent"] == "售后"
        
        mock_update.assert_called_once_with(metadata={"intent": "售后"}, tags=["售后"])


def test_check_intent_costs_cli(tmp_path):
    output_file = tmp_path / "cost_by_intent.json"
    
    # Run the mock CLI
    with patch("sys.argv", ["check_intent_costs.py", "--output", str(output_file), "--mock"]):
        run_cli()

    assert output_file.exists()
    
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert "total_requests" in data
    assert "items" in data
    assert len(data["items"]) > 0
