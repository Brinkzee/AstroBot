import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import select

from app.db.session import Base, get_db
from app.models.low_confidence import LowConfidenceQuestion
from app.models.conversation import Conversation
from app.models.message import Message
from app.services.workflow.nodes.knowledge_node import knowledge_fallback_node
from app.services.rag.generator import RAGControlledGenerator
from app.api.routes import router

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
    app = FastAPI()
    app.include_router(router)

    async def override_get_db():
        yield memory_db

    app.dependency_overrides[get_db] = override_get_db
    return app

@pytest.mark.asyncio
async def test_knowledge_fallback_stores_retrieved_chunks(memory_db):
    state = {
        "input_query": "测试没有证据",
        "conversation_id": 100,
        "retrieved_docs": [
            {"text": "这段文本超过了但是会被截断" * 30, "score": 0.1, "section_path": "a/b/c"},
            {"text": "第二段文本", "score": 0.2, "section": "d/e/f"},
        ]
    }
    
    await knowledge_fallback_node(state, db=memory_db)
    
    stmt = select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == 100)
    res = await memory_db.execute(stmt)
    lcq = res.scalar_one()
    
    assert lcq.source == "retrieval_low_conf"
    assert lcq.reason == "retrieval_score_below_threshold"
    assert len(lcq.retrieved_chunks) == 2
    assert "这段文本" in lcq.retrieved_chunks[0]["text"]
    assert len(lcq.retrieved_chunks[0]["text"]) == 300
    assert lcq.retrieved_chunks[0]["score"] == 0.1
    assert lcq.retrieved_chunks[0]["section"] == "a/b/c"

@pytest.mark.asyncio
async def test_generator_self_check_stores_retrieved_chunks(memory_db):
    generator = RAGControlledGenerator(model=None, stream_model=None)
    citations = [
        {"text": "长文本" * 100, "score": 0.9, "section_path": "path/a"},
        {"content": "另一个字段的文本", "score": 0.8, "section": "path/b"}
    ]
    
    # query contains a mock trigger word like '火星' to force failure in mock
    res = await generator.check_sufficiency(
        query="如何去火星",
        citations=citations,
        db=memory_db,
        conversation_id=200
    )
    
    assert res.useful is False
    
    stmt = select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == 200)
    q_res = await memory_db.execute(stmt)
    lcq = q_res.scalar_one()
    
    assert lcq.source == "self_check"
    assert len(lcq.retrieved_chunks) == 2
    assert len(lcq.retrieved_chunks[0]["text"]) == 300
    assert lcq.retrieved_chunks[1]["text"] == "另一个字段的文本"
    assert lcq.retrieved_chunks[1]["section"] == "path/b"

@pytest.mark.asyncio
async def test_user_feedback_thumbs_down_saves_to_pool(test_app, memory_db):
    conv = Conversation(id=300, user_id="u1")
    msg = Message(conversation_id=300, role="user", content="测试用户问题")
    memory_db.add(conv)
    memory_db.add(msg)
    await memory_db.commit()

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/api/chat/feedback", json={
            "conversation_id": 300,
            "feedback_type": "down",
            "query": "测试用户问题2"
        })
    
    assert resp.status_code == 200
    
    stmt = select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == 300)
    res = await memory_db.execute(stmt)
    lcq = res.scalar_one()
    
    assert lcq.source == "user_feedback"
    assert lcq.raw_question == "测试用户问题2"

@pytest.mark.asyncio
async def test_user_feedback_without_retrieval_has_null_chunks(test_app, memory_db):
    conv = Conversation(id=400, user_id="u1")
    msg = Message(conversation_id=400, role="user", content="我没有检索知识")
    memory_db.add(conv)
    memory_db.add(msg)
    await memory_db.commit()

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/api/chat/feedback", json={
            "conversation_id": 400,
            "feedback_type": "down"
        })
    
    assert resp.status_code == 200
    
    stmt = select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == 400)
    res = await memory_db.execute(stmt)
    lcq = res.scalar_one()
    
    assert lcq.source == "user_feedback"
    assert lcq.raw_question == "我没有检索知识"
    assert lcq.retrieved_chunks is None

@pytest.mark.asyncio
async def test_user_feedback_thumbs_down_with_retrieval(test_app, memory_db):
    from app.models.tool_audit_log import ToolAuditLog
    import json
    conv = Conversation(id=301, user_id="u2")
    msg_user = Message(conversation_id=301, role="user", content="退款政策是啥")
    msg_ast = Message(
        conversation_id=301, 
        role="assistant", 
        content="", 
        tool_calls='[{"name": "query_faq", "arguments": {"query": "退款"}}]'
    )
    
    audit_log = ToolAuditLog(
        conversation_id=301,
        tool_name="query_faq",
        tool_source="builtin",
        status="成功",
        arguments={"query": "退款"},
        result_summary=json.dumps([
            {"answer": "我们的退款政策是...", "distance": 0.85, "section_path": "faq/refund"}
        ]),
        duration_ms=100
    )
    
    memory_db.add_all([conv, msg_user, msg_ast, audit_log])
    await memory_db.commit()

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        resp = await client.post("/api/chat/feedback", json={
            "conversation_id": 301,
            "feedback_type": "down",
            "query": "退款政策是啥"
        })
    
    assert resp.status_code == 200
    
    stmt = select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == 301)
    res = await memory_db.execute(stmt)
    lcq = res.scalar_one()
    
    assert lcq.source == "user_feedback"
    assert lcq.raw_question == "退款政策是啥"
    assert len(lcq.retrieved_chunks) == 1
    assert lcq.retrieved_chunks[0]["text"] == "我们的退款政策是..."
    assert lcq.retrieved_chunks[0]["score"] == 0.85
    assert lcq.retrieved_chunks[0]["section"] == "faq/refund"

