import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.session import Base, get_db
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.ticket import Ticket
from app.models.tool_audit_log import ToolAuditLog
from app.models.summary import ConversationSummary
from app.models.low_confidence import LowConfidenceQuestion
from main import app


class AsyncSessionAdapter:
    """Async wrapper over SQLite synchronous session for API testing."""

    def __init__(self, sync_session: Session):
        self._sync = sync_session

    def add(self, obj):
        self._sync.add(obj)

    def add_all(self, objs):
        self._sync.add_all(objs)

    async def flush(self):
        self._sync.flush()

    async def commit(self):
        self._sync.commit()

    async def rollback(self):
        self._sync.rollback()

    async def refresh(self, obj):
        self._sync.refresh(obj)

    async def get(self, entity_cls, ident):
        return self._sync.get(entity_cls, ident)

    async def execute(self, stmt):
        return self._sync.execute(stmt)

    async def delete(self, obj):
        self._sync.delete(obj)

    async def close(self):
        self._sync.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.rollback()
        await self.close()


@pytest.fixture
def test_db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sync_sess = Session(engine, expire_on_commit=False)
    adapter = AsyncSessionAdapter(sync_sess)

    async def override_get_db():
        yield adapter

    app.dependency_overrides[get_db] = override_get_db
    yield sync_sess
    app.dependency_overrides.pop(get_db, None)
    sync_sess.close()
    engine.dispose()


@pytest.mark.asyncio
async def test_delete_conversation_not_found(test_db_session):
    """用例 1: 删除不存在的会话应返回 404 且 detail 包含'不存在'"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/conversations/99999")
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_delete_conversation_cascade_success(test_db_session):
    """用例 2: 创建完整外键图数据，调用 DELETE，assert 200，并直连数据库断言所有相关表均已级联清理或解绑"""
    conv = Conversation(user_id="test_user", status="进行中")
    test_db_session.add(conv)
    test_db_session.commit()

    msg = Message(conversation_id=conv.id, role="user", content="我想咨询售后问题")
    summary = ConversationSummary(
        conversation_id=conv.id,
        seq=1,
        from_msg_id=1,
        upto_msg_id=1,
        content="用户咨询售后",
    )
    ticket = Ticket(
        ticket_no="TK_DELETE_TEST_001",
        conversation_id=conv.id,
        description="用户申请换货",
        ticket_type="售后",
        status="待处理",
    )
    audit = ToolAuditLog(
        conversation_id=conv.id,
        tool_name="query_order",
        tool_source="builtin",
        status="成功",
    )
    lcq = LowConfidenceQuestion(
        conversation_id=conv.id,
        raw_question="这个衣服掉色吗？",
        source="retrieval_low_conf",
        reason="知识库未命中",
    )

    test_db_session.add_all([msg, summary, ticket, audit, lcq])
    test_db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete(f"/api/conversations/{conv.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["conversation_id"] == conv.id
        assert f"会话 #{conv.id} 已成功删除" in data["message"]

    # 直连数据库断言所有相关表均已级联清理或解绑
    assert test_db_session.get(Conversation, conv.id) is None
    assert test_db_session.execute(
        select(Message).where(Message.conversation_id == conv.id)
    ).scalars().all() == []
    assert test_db_session.execute(
        select(ConversationSummary).where(ConversationSummary.conversation_id == conv.id)
    ).scalars().all() == []
    assert test_db_session.execute(
        select(Ticket).where(Ticket.conversation_id == conv.id)
    ).scalars().all() == []
    assert test_db_session.execute(
        select(ToolAuditLog).where(ToolAuditLog.conversation_id == conv.id)
    ).scalars().all() == []

    # LowConfidenceQuestion 应该保留，但 conversation_id 被解绑设为 None
    refreshed_lcq = test_db_session.get(LowConfidenceQuestion, lcq.id)
    assert refreshed_lcq is not None
    assert refreshed_lcq.conversation_id is None


@pytest.mark.asyncio
async def test_delete_conversation_isolation(test_db_session):
    """用例 3: 创建会话 A 和会话 B，删除 A 后确认 B 及其消息依然完好"""
    conv_a = Conversation(user_id="user_a", status="进行中")
    conv_b = Conversation(user_id="user_b", status="进行中")
    test_db_session.add_all([conv_a, conv_b])
    test_db_session.commit()

    msg_a = Message(conversation_id=conv_a.id, role="user", content="会话 A 的消息")
    msg_b = Message(conversation_id=conv_b.id, role="user", content="会话 B 的消息")
    test_db_session.add_all([msg_a, msg_b])
    test_db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete(f"/api/conversations/{conv_a.id}")
        assert resp.status_code == 200

    # 确认 A 及其消息被删除
    assert test_db_session.get(Conversation, conv_a.id) is None
    assert test_db_session.execute(
        select(Message).where(Message.conversation_id == conv_a.id)
    ).scalars().all() == []

    # 确认 B 及其消息依然完好
    remained_conv_b = test_db_session.get(Conversation, conv_b.id)
    assert remained_conv_b is not None
    assert remained_conv_b.user_id == "user_b"
    b_messages = test_db_session.execute(
        select(Message).where(Message.conversation_id == conv_b.id)
    ).scalars().all()
    assert len(b_messages) == 1
    assert b_messages[0].content == "会话 B 的消息"
