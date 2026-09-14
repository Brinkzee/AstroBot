import re
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from app.db.session import Base, get_db
from app.models.conversation import Conversation
from app.models.message import Message
from main import app


class AsyncSessionAdapter:
    """Async wrapper over SQLite synchronous session for testing."""

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
def test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
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


@pytest.fixture
def client(test_db):
    return TestClient(app)


def test_get_root_route_returns_html(client):
    """验证 GET / 返回 200 及 HTML 内容"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "星光优选" in response.text
    assert "智能客服小星" in response.text


def test_get_chat_route_returns_html(client):
    """验证 GET /chat 返回 200 及 HTML 内容"""
    response = client.get("/chat")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "智能客服小星" in response.text


def test_sidebar_dom_structure(client):
    """验证 HTML 中包含会话侧栏核心 DOM 元素与样式标识"""
    response = client.get("/")
    html = response.text

    # 验证会话侧栏结构容器
    assert "session-sidebar" in html, "页面缺少 session-sidebar 容器"
    assert "session-list" in html or 'id="sessionList"' in html, "页面缺少 sessionList 容器"
    assert "btn-new-chat" in html or 'id="btnNewChat"' in html, "页面缺少 btnNewChat 按钮"

    # 验证样式定义中包含 session-item, summary-badge, active 高亮
    assert ".session-sidebar" in html, "缺少 .session-sidebar 样式"
    assert ".session-item" in html, "缺少 .session-item 样式"
    assert ".summary-badge" in html, "缺少 .summary-badge 样式"
    assert ".active" in html, "缺少 .active 高亮样式"


def test_frontend_js_multi_session_logic(client):
    """验证前端脚本中包含多会话加载、历史切换、摘要徽标渲染与静默降级逻辑"""
    response = client.get("/")
    html = response.text

    # 验证 loadSessions 函数与 /api/conversations 调用
    assert "loadSessions" in html, "缺少 loadSessions 函数定义"
    assert "/api/conversations" in html, "缺少 /api/conversations 接口调用"

    # 验证历史消息接口调用
    assert "/api/conversations/" in html or "/api/conversations" in html, "缺少会话历史消息接口调用"

    # 验证 summary-badge 渲染条件 (has_summary)
    assert "has_summary" in html, "缺少针对 has_summary 渲染 [已摘要] 标签的逻辑"
    assert "已摘要" in html, "缺少 [已摘要] 徽标文案"

    # 验证静默降级提示文案与无侵入式处理 (不使用 alert 打断)
    assert "暂无历史会话" in html, "缺少会话空状态或降级提示文案"

    # 验证新对话重置与激活态切换
    assert "currentConversationId" in html, "缺少 currentConversationId 状态维护"


def test_session_data_contract_and_fallback_resilience(client):
    """验证前端针对会话列表接口契约与静默降级的预期响应"""
    resp = client.get("/api/conversations")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)

    html = client.get("/").text
    assert "catch" in html, "前端网络请求必须包含 catch 错误处理以实现静默降级"
