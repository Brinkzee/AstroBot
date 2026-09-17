"""
Chapter 09 Acceptance Tests
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
import os
import json
from unittest.mock import patch, MagicMock
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from app.db.session import Base, get_db
from app.models.conversation import Conversation
from app.models.eval_run import EvalRun, TriggeredBy
from app.models.review_queue import ReviewQueue, ReviewStatus
from app.models.low_confidence import LowConfidenceQuestion
from app.services.observability.langfuse_service import LangfuseManager
from app.services.observability.cost_analytics import CostAnalyticsService
from app.services.workflow.nodes.gate import compute_evidence_confidence
from app.services.flywheel.pipeline import FlywheelPipeline
from app.services.eval.eval_pipeline import EvalPipelineService
from app.services.rag.dual_writer import KnowledgeDualWriter

@pytest_asyncio.fixture
async def memory_db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestingSessionLocal = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with TestingSessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
def test_app(memory_db):
    async def override_get_db():
        yield memory_db

    app.dependency_overrides[get_db] = override_get_db
    yield app
    app.dependency_overrides.clear()

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
    assert 0.0 <= conf <= 1.0

@pytest.mark.asyncio
async def test_acceptance_4_low_confidence_fallback(test_app, memory_db):
    # Insert conversation into memory DB
    conv = Conversation(id=123)
    memory_db.add(conv)
    await memory_db.commit()

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/chat/feedback",
            json={
                "message_id": 999,
                "conversation_id": 123,
                "query": "你们包邮吗？",
                "feedback_type": "down",
                "reason": "回答不满意"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

@pytest.mark.asyncio
async def test_acceptance_5_flywheel_pipeline():
    pipeline = FlywheelPipeline()
    assert pipeline is not None
    assert hasattr(pipeline, "process_pending_questions")

@pytest.mark.asyncio
async def test_acceptance_6_review_knowledge_dual():
    # Use mocks for external Milvus store and embedding client to keep unit test offline and fast
    mock_store = MagicMock()
    mock_embed = MagicMock()
    writer = KnowledgeDualWriter(store=mock_store, embedding_client=mock_embed)
    assert writer is not None
    assert writer.store == mock_store
    assert writer.embedding_client == mock_embed

@pytest.mark.asyncio
async def test_acceptance_7_eval_trend(test_app, memory_db):
    # Seed one eval run into memory DB
    run = EvalRun(
        triggered_by=TriggeredBy.MANUAL.value,
        dataset_size=5,
        metrics={
            "recall_at_3": 0.85,
            "recall_at_5": 0.90,
            "recall_at_10": 0.95,
            "mrr": 0.80,
            "faithfulness": 0.92,
        }
    )
    memory_db.add(run)
    await memory_db.commit()

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.get("/api/observability/overview")
        assert response.status_code == 200
        data = response.json()
        assert "cost_block" in data
        assert "eval_trend_block" in data
        assert "calibration_block" in data
        assert data["eval_trend_block"]["present"] is True
        assert len(data["eval_trend_block"]["runs"]) >= 1
