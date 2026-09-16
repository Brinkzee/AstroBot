import json
from unittest.mock import patch, MagicMock
import pytest
from httpx import AsyncClient, ASGITransport
from main import app
from app.services.job_runner import job_runner

pytestmark = pytest.mark.asyncio

async def test_job_runner_whitelist_includes_ch09_jobs():
    whitelist = job_runner.get_whitelist()
    assert "cost-analysis" in whitelist
    assert "eval-pipeline" in whitelist
    assert "calibrate-confidence" in whitelist
    assert "flywheel-pipeline" in whitelist

@patch("app.api.observability.EvalPipelineService")
@patch("pathlib.Path.exists")
@patch("builtins.open")
async def test_observability_overview_when_reports_present(mock_open, mock_exists, MockEvalService):
    mock_exists.return_value = True
    
    mock_file_data = {
        "cost_by_intent.json": '{"some": "data"}',
        "confidence_calibration.json": '{"recommended_threshold": 0.5}'
    }
    
    def open_side_effect(path, *args, **kwargs):
        filename = str(path).split("/")[-1].split("\\")[-1]
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = mock_file_data.get(filename, "{}")
        return mock_file
        
    mock_open.side_effect = open_side_effect
    
    mock_service_instance = MockEvalService.return_value
    async def mock_get_recent_runs(*args, **kwargs):
        return [{"id": 1, "metrics": {}}]
    mock_service_instance.get_recent_runs_with_deltas.side_effect = mock_get_recent_runs

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/observability/overview")
        
    assert response.status_code == 200
    data = response.json()
    
    assert data["cost_block"]["present"] is True
    assert data["cost_block"]["job_name"] == "cost-analysis"
    assert data["cost_block"]["data"] == {"some": "data"}
    
    assert data["eval_trend_block"]["present"] is True
    assert data["eval_trend_block"]["job_name"] == "eval-pipeline"
    assert data["eval_trend_block"]["runs"] == [{"id": 1, "metrics": {}}]
    
    assert data["calibration_block"]["present"] is True
    assert data["calibration_block"]["job_name"] == "calibrate-confidence"
    assert data["calibration_block"]["data"]["recommended_threshold"] == 0.5
    assert "is_mismatched" in data["calibration_block"]["data"]
    assert "active_threshold" in data["calibration_block"]["data"]

@patch("app.api.observability.EvalPipelineService")
@patch("pathlib.Path.exists")
async def test_observability_overview_when_reports_absent(mock_exists, MockEvalService):
    mock_exists.return_value = False
    
    mock_service_instance = MockEvalService.return_value
    async def mock_get_recent_runs_empty(*args, **kwargs):
        return []
    mock_service_instance.get_recent_runs_with_deltas.side_effect = mock_get_recent_runs_empty

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/observability/overview")
        
    assert response.status_code == 200
    data = response.json()
    
    assert data["cost_block"]["present"] is False
    assert data["cost_block"]["data"] is None
    assert "暂无意图成本报表" in data["cost_block"]["hint"]
    
    assert data["eval_trend_block"]["present"] is False
    assert data["eval_trend_block"]["runs"] == []
    assert "暂无评测轮次记录" in data["eval_trend_block"]["hint"]
    
    assert data["calibration_block"]["present"] is False
    assert data["calibration_block"]["data"] is None
    assert "暂无置信度校准报表" in data["calibration_block"]["hint"]
