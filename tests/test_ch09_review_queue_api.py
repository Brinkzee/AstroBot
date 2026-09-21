import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch

from main import app
from app.db.session import Base, get_db
from app.models.review_queue import ReviewQueue, ReviewStatus
from app.models.low_confidence import LowConfidenceQuestion
from app.api.review_queue_routes import review_queue_router

# In case main.py doesn't have it included yet, we include it directly in the test app setup
# to avoid failing if we haven't modified main.py yet, or we can just rely on main.py if we modify it first.
# Wait, the task asks us to modify main.py. So we will just test it on the actual app.

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
    # temporarily include it for the test if it's missing, though we should really add it to main
    return app

pytestmark = pytest.mark.asyncio

async def test_list_review_queue(memory_db: AsyncSession, test_app):
    queue1 = ReviewQueue(
        normalized_question="测试问题1",
        occurrence_count=5,
        review_status=ReviewStatus.PENDING.value
    )
    queue2 = ReviewQueue(
        normalized_question="测试问题2",
        occurrence_count=2,
        review_status=ReviewStatus.APPROVED.value
    )
    memory_db.add_all([queue1, queue2])
    await memory_db.commit()
    await memory_db.refresh(queue1)
    
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.get("/api/review-queue")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert data["items"][0]["id"] == queue1.id
        
        response = await ac.get("/api/review-queue?status=通过")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == queue2.id

async def test_get_review_queue_detail(memory_db: AsyncSession, test_app):
    queue1 = ReviewQueue(
        normalized_question="详情问题",
        occurrence_count=1,
        review_status=ReviewStatus.PENDING.value
    )
    memory_db.add(queue1)
    await memory_db.commit()
    await memory_db.refresh(queue1)
    
    lcq = LowConfidenceQuestion(
        raw_question="原话1",
        source="retrieval_low_conf",
        matched_review_id=queue1.id,
        retrieved_chunks=[{"chunk": "test", "score": 0.1}]
    )
    memory_db.add(lcq)
    await memory_db.commit()
    await memory_db.refresh(lcq)

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.get(f"/api/review-queue/{queue1.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == queue1.id
        assert len(data["raw_queries"]) == 1
        assert data["raw_queries"][0]["raw_question"] == "原话1"
        assert data["raw_queries"][0]["retrieved_chunks"][0]["chunk"] == "test"

async def test_reject_review_queue(memory_db: AsyncSession, test_app):
    queue1 = ReviewQueue(
        normalized_question="驳回问题",
        occurrence_count=1,
        review_status=ReviewStatus.PENDING.value
    )
    memory_db.add(queue1)
    await memory_db.commit()
    await memory_db.refresh(queue1)

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        response = await ac.post(f"/api/review-queue/{queue1.id}/reject")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        assert data["id"] == queue1.id
        
    await memory_db.refresh(queue1)
    assert queue1.review_status == ReviewStatus.REJECTED.value


async def test_approve_review_queue_dual_writes_to_kb(memory_db: AsyncSession, test_app):
    queue1 = ReviewQueue(
        normalized_question="通过问题",
        occurrence_count=1,
        review_status=ReviewStatus.PENDING.value
    )
    memory_db.add(queue1)
    await memory_db.commit()
    await memory_db.refresh(queue1)

    with patch("app.api.review_queue_routes.KnowledgeDualWriter") as MockWriter:
        mock_instance = MockWriter.return_value
        from unittest.mock import AsyncMock
        mock_instance.write_chunks = AsyncMock(return_value=[])
        
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
            response = await ac.post(f"/api/review-queue/{queue1.id}/approve", json={"approved_answer": "核准的答案"})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "approved"
            
            mock_instance.write_chunks.assert_called_once()
            called_args = mock_instance.write_chunks.call_args
            chunks = called_args[0][1]
            assert len(chunks) == 1
            assert chunks[0].questions == "通过问题"
            assert chunks[0].answer == "核准的答案"
            assert chunks[0].category == "常见问题"
            
        await memory_db.refresh(queue1)
        assert queue1.review_status == ReviewStatus.APPROVED.value
        assert queue1.approved_answer == "核准的答案"
