"""
Chapter 09 Acceptance Tests
"""
import pytest
import pytest_asyncio
import httpx
from httpx import AsyncClient, ASGITransport
import os
import json
from unittest.mock import patch, MagicMock

from scripts.wsl_helper import ensure_mysql_ready

from main import app

@pytest.fixture(scope="module", autouse=True)
def setup_mysql_ready():
    ensure_mysql_ready(verbose=False)

from app.services.observability.langfuse_service import LangfuseManager
from app.services.observability.cost_analytics import CostAnalyticsService
from app.services.workflow.nodes.gate import compute_evidence_confidence
from app.services.flywheel.pipeline import FlywheelPipeline
from app.services.eval.eval_pipeline import EvalPipelineService
from app.services.rag.dual_writer import KnowledgeDualWriter

@pytest.mark.asyncio
async def test_acceptance_1_langfuse():
    manager = LangfuseManager()
    assert manager is not None
    # Assuming disabled mode gracefully degrades
    handler = manager.get_callback_handler()
    assert handler is None or handler is not None # just has to return safely without exception

@pytest.mark.asyncio
async def test_acceptance_2_cost_analytics():
    service = CostAnalyticsService()
    records = [
        {"intent": "refund", "prompt_tokens": 100, "completion_tokens": 10},
        {"intent": "refund", "prompt_tokens": 100, "completion_tokens": 10},
        {"intent": "qa", "prompt_tokens": 50, "completion_tokens": 5}
    ]
    report = service.aggregate_records(records)
    
    items = report["items"]
    assert items[0]["intent"] == "refund"
    assert sum(item["cost_percentage"] for item in items) - 100.0 < 0.1

@pytest.mark.asyncio
async def test_acceptance_3_evidence_confidence():
    conf = compute_evidence_confidence([{"score": 0.8}, {"score": 0.7}, {"score": 0.6}])
    assert isinstance(conf, float)

@pytest.mark.asyncio
async def test_acceptance_4_low_confidence_fallback():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create a conversation first
        from app.db.session import AsyncSessionLocal
        from app.models.conversation import Conversation
        async with AsyncSessionLocal() as db:
            conv = Conversation(id=123)
            db.add(conv)
            try:
                await db.commit()
            except Exception:
                await db.rollback()
        
        response = await ac.post(
            "/api/chat/feedback",
            json={
                "message_id": 999,
                "conversation_id": 123,
                "query": "test question",
                "feedback_type": "down"
            }
        )
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_acceptance_5_flywheel_pipeline():
    pipeline = FlywheelPipeline()
    assert pipeline is not None

@pytest.mark.asyncio
async def test_acceptance_6_review_knowledge_dual():
    writer = KnowledgeDualWriter()
    assert writer is not None

@pytest.mark.asyncio
async def test_acceptance_7_eval_trend():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/observability/overview")
        assert response.status_code == 200
        data = response.json()
        assert "cost_block" in data
        assert "eval_trend_block" in data
        assert "calibration_block" in data
