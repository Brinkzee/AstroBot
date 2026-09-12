import json
from datetime import datetime, timedelta
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.session import Base, get_db
from app.models.conversation import Conversation
from app.models.message import Message
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
async def test_get_conversations_empty(test_db_session):
    """测试无会话时返回空列表"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/conversations")
        assert resp.status_code == 200
        assert resp.json() == []


@pytest.mark.asyncio
async def test_get_conversations_user_filtering_and_order(test_db_session):
    """测试按用户过滤和按更新时间倒序排序"""
    t0 = datetime(2026, 9, 12, 10, 0, 0)
    t1 = datetime(2026, 9, 12, 11, 0, 0)
    t2 = datetime(2026, 9, 12, 12, 0, 0)

    conv1 = Conversation(user_id="default_user", status="进行中", created_at=t0, updated_at=t0)
    conv2 = Conversation(user_id="default_user", status="已结束", created_at=t1, updated_at=t2)
    conv_other = Conversation(user_id="other_user", status="进行中", created_at=t1, updated_at=t1)

    test_db_session.add_all([conv1, conv2, conv_other])
    test_db_session.commit()

    msg1 = Message(conversation_id=conv1.id, role="user", content="会话1第一句话", created_at=t0)
    msg2 = Message(conversation_id=conv2.id, role="user", content="会话2最新提问", created_at=t1)
    test_db_session.add_all([msg1, msg2])
    test_db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 查询 default_user (默认)
        resp = await client.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        # conv2 updated_at (t2) > conv1 updated_at (t0) -> conv2 排在前面
        assert data[0]["id"] == conv2.id
        assert data[0]["title"] == "会话2最新提问"
        assert data[0]["status"] == "已结束"
        assert data[0]["message_count"] == 1

        assert data[1]["id"] == conv1.id
        assert data[1]["title"] == "会话1第一句话"
        assert data[1]["status"] == "进行中"
        assert data[1]["message_count"] == 1

        # 查询 other_user
        resp_other = await client.get("/api/conversations?user_id=other_user")
        assert resp_other.status_code == 200
        data_other = resp_other.json()
        assert len(data_other) == 1
        assert data_other[0]["id"] == conv_other.id
        assert data_other[0]["title"] == "新会话"
        assert data_other[0]["message_count"] == 0


@pytest.mark.asyncio
async def test_get_conversations_title_preview_and_summary_flags(test_db_session):
    """测试首问预览（长字符截取25字、无提问显示新会话）与 has_summary 标记"""
    now = datetime(2026, 9, 12, 14, 0, 0)

    # conv_long: 超长提问 (>25 字符)
    conv_long = Conversation(user_id="default_user", created_at=now, updated_at=now)
    # conv_summary_text: 带有 summary 文本
    conv_summary_text = Conversation(
        user_id="default_user",
        summary="这是历史摘要内容",
        summary_upto_msg_id=0,
        created_at=now,
        updated_at=now,
    )
    # conv_summary_id: 带有 summary_upto_msg_id > 0
    conv_summary_id = Conversation(
        user_id="default_user",
        summary=None,
        summary_upto_msg_id=8,
        created_at=now,
        updated_at=now,
    )
    # conv_no_summary: 无摘要标记
    conv_no_summary = Conversation(
        user_id="default_user",
        summary="",
        summary_upto_msg_id=0,
        created_at=now,
        updated_at=now,
    )

    test_db_session.add_all([conv_long, conv_summary_text, conv_summary_id, conv_no_summary])
    test_db_session.commit()

    # 构造 conv_long 消息：一条助手消息在前，一条超长提问在后（验证必须提取第一条 role="user" 的消息）
    m_asst = Message(
        conversation_id=conv_long.id,
        role="assistant",
        content="您好，我是智能客服小星！",
        created_at=now - timedelta(seconds=10),
    )
    long_text = "我想问一下你们这个加厚羽绒服的保暖充绒量是多少克？北方零下二十度能穿吗？"
    m_user = Message(
        conversation_id=conv_long.id,
        role="user",
        content=long_text,
        created_at=now,
    )
    test_db_session.add_all([m_asst, m_user])
    test_db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        item_map = {item["id"]: item for item in data}

        # 检查截取25字符
        assert item_map[conv_long.id]["title"] == long_text[:25]
        assert len(item_map[conv_long.id]["title"]) == 25
        assert item_map[conv_long.id]["message_count"] == 2

        # 检查 has_summary
        assert item_map[conv_summary_text.id]["has_summary"] is True
        assert item_map[conv_summary_id.id]["has_summary"] is True
        assert item_map[conv_no_summary.id]["has_summary"] is False

        # 检查无提问时的标题
        assert item_map[conv_no_summary.id]["title"] == "新会话"


@pytest.mark.asyncio
async def test_get_conversation_messages_not_found(test_db_session):
    """测试获取不存在的会话消息返回 404"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/conversations/99999/messages")
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_get_conversation_messages_empty_history(test_db_session):
    """测试存在的空会话返回空消息列表"""
    conv = Conversation(user_id="default_user")
    test_db_session.add(conv)
    test_db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/api/conversations/{conv.id}/messages")
        assert resp.status_code == 200
        assert resp.json() == []


@pytest.mark.asyncio
async def test_get_conversation_messages_full_flow(test_db_session):
    """测试消息历史按时间升序返回，并正确包含 user、assistant(tool_calls)、tool 等全部字段"""
    t0 = datetime(2026, 9, 12, 15, 0, 0)
    t1 = datetime(2026, 9, 12, 15, 0, 1)
    t2 = datetime(2026, 9, 12, 15, 0, 2)
    t3 = datetime(2026, 9, 12, 15, 0, 3)

    conv = Conversation(user_id="default_user")
    test_db_session.add(conv)
    test_db_session.commit()

    # 插入 4 条不同类型的消息，乱序插入以验证时间升序排列
    tool_calls_payload = [{"id": "call_order_1", "name": "query_order", "args": {"order_id": "OD888"}}]

    m3 = Message(
        conversation_id=conv.id,
        role="assistant",
        content="您的订单 OD888 已发货，正在派送中。",
        created_at=t3,
    )
    m1 = Message(
        conversation_id=conv.id,
        role="user",
        content="帮我查下订单 OD888 的物流",
        created_at=t0,
    )
    m2 = Message(
        conversation_id=conv.id,
        role="assistant",
        content=None,
        tool_calls=tool_calls_payload,
        created_at=t1,
    )
    m_tool = Message(
        conversation_id=conv.id,
        role="tool",
        content='{"order_id": "OD888", "status": "已发货"}',
        tool_call_id="call_order_1",
        created_at=t2,
    )

    test_db_session.add_all([m3, m1, m2, m_tool])
    test_db_session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/api/conversations/{conv.id}/messages")
        assert resp.status_code == 200
        msgs = resp.json()

        assert len(msgs) == 4
        # 验证正序排列
        assert msgs[0]["id"] == m1.id
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "帮我查下订单 OD888 的物流"
        assert msgs[0]["tool_calls"] is None
        assert msgs[0]["tool_call_id"] is None

        assert msgs[1]["id"] == m2.id
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] is None
        assert msgs[1]["tool_calls"] == tool_calls_payload
        assert msgs[1]["tool_call_id"] is None

        assert msgs[2]["id"] == m_tool.id
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["content"] == '{"order_id": "OD888", "status": "已发货"}'
        assert msgs[2]["tool_call_id"] == "call_order_1"

        assert msgs[3]["id"] == m3.id
        assert msgs[3]["role"] == "assistant"
        assert "OD888 已发货" in msgs[3]["content"]
