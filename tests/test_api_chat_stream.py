import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, ToolMessage

from main import app
from app.db.session import get_db
from app.schemas.chat import ChatStreamRequest
from app.services.chat_service import ChatService
from tests.test_chat_service import FakeAsyncSession


@pytest.fixture(autouse=True)
def mock_db_session():
    fake_db = FakeAsyncSession()
    async def override_get_db():
        yield fake_db
    app.dependency_overrides[get_db] = override_get_db
    yield fake_db
    app.dependency_overrides.pop(get_db, None)


def parse_sse_events(sse_text: str):
    """解析 SSE 响应文本为事件列表和 done 标记"""
    events = []
    has_done = False
    for line in sse_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                has_done = True
            else:
                events.append(json.loads(data_str))
    return events, has_done


def test_chat_stream_request_schema():
    """测试 ChatStreamRequest 模式及其 effective_conversation_id 属性"""
    # 1. 指定 conversation_id
    req1 = ChatStreamRequest(conversation_id=10, message="查询订单")
    assert req1.conversation_id == 10
    assert req1.session_id is None
    assert req1.effective_conversation_id == 10

    # 2. 指定 session_id 为数字字符串
    req2 = ChatStreamRequest(session_id="25", message="查询商品")
    assert req2.conversation_id is None
    assert req2.session_id == "25"
    assert req2.effective_conversation_id == 25

    # 3. session_id 为非纯数字字符串
    req3 = ChatStreamRequest(session_id="session_abc_123", message="你好")
    assert req3.effective_conversation_id is None

    # 4. 同时传入 conversation_id 与 session_id，优先以 conversation_id 为准
    req4 = ChatStreamRequest(conversation_id=7, session_id="99", message="查询物流")
    assert req4.effective_conversation_id == 7

    # 5. 均未传入
    req5 = ChatStreamRequest(message="新会话")
    assert req5.effective_conversation_id is None

    # 6. 空 message 触发校验异常
    with pytest.raises(Exception):
        ChatStreamRequest(message="")


def test_chat_stream_sse_tool_and_text_events():
    """测试 POST /api/chat/stream 对接 ChatService 吐出 tool_start, tool_end, text 与 [DONE]"""
    client = TestClient(app)

    async def fake_stream_chat(db, conversation_id, message):
        assert conversation_id == 101
        assert message == "查订单1001"
        yield {
            "event_type": "tool_start",
            "conversation_id": 101,
            "tool_name": "query_order",
            "tool_label": "查询订单",
            "args": {"order_id": "1001"},
        }
        yield {
            "event_type": "tool_end",
            "conversation_id": 101,
            "tool_name": "query_order",
            "success": True,
        }
        yield {
            "event_type": "text",
            "conversation_id": 101,
            "content": "您的订单1001已发货。",
        }

    with patch("app.api.routes.chat_service.stream_chat", side_effect=fake_stream_chat):
        res = client.post(
            "/api/chat/stream",
            json={"conversation_id": 101, "message": "查订单1001"},
        )

        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]
        assert res.headers.get("Cache-Control") == "no-cache"
        assert res.headers.get("X-Accel-Buffering") == "no"

        events, has_done = parse_sse_events(res.text)
        assert has_done is True
        assert len(events) == 3

        # tool_start
        assert events[0]["event_type"] == "tool_start"
        assert events[0]["conversation_id"] == 101
        assert events[0]["tool_name"] == "query_order"
        assert events[0]["tool_label"] == "查询订单"
        assert events[0]["args"] == {"order_id": "1001"}

        # tool_end
        assert events[1]["event_type"] == "tool_end"
        assert events[1]["conversation_id"] == 101
        assert events[1]["tool_name"] == "query_order"
        assert events[1]["success"] is True

        # text
        assert events[2]["event_type"] == "text"
        assert events[2]["conversation_id"] == 101
        assert events[2]["content"] == "您的订单1001已发货。"


def test_chat_stream_backward_compatible_session_id():
    """测试通过 session_id 别名调用并正确传递 effective_conversation_id"""
    client = TestClient(app)

    captured_conv_id = None

    async def fake_stream_chat(db, conversation_id, message):
        nonlocal captured_conv_id
        captured_conv_id = conversation_id
        yield {
            "event_type": "text",
            "conversation_id": conversation_id or 1,
            "content": "收到",
        }

    with patch("app.api.routes.chat_service.stream_chat", side_effect=fake_stream_chat):
        res = client.post(
            "/api/chat/stream",
            json={"session_id": "88", "message": "测试兼容"},
        )
        assert res.status_code == 200
        assert captured_conv_id == 88

        events, has_done = parse_sse_events(res.text)
        assert has_done is True
        assert len(events) == 1
        assert events[0]["event_type"] == "text"
        assert events[0]["content"] == "收到"


def test_chat_stream_error_event():
    """测试当生成过程抛出异常时，返回 event_type='error' 并以 [DONE] 结束"""
    client = TestClient(app)

    async def fake_stream_chat(db, conversation_id, message):
        yield {
            "event_type": "error",
            "conversation_id": conversation_id,
            "error": "服务暂时不可用",
        }

    with patch("app.api.routes.chat_service.stream_chat", side_effect=fake_stream_chat):
        res = client.post(
            "/api/chat/stream",
            json={"conversation_id": 55, "message": "测试异常"},
        )
        assert res.status_code == 200
        events, has_done = parse_sse_events(res.text)
        assert has_done is True
        assert len(events) == 1
        assert events[0]["event_type"] == "error"
        assert "服务暂时不可用" in events[0]["error"]


@pytest.mark.asyncio
async def test_chat_stream_emits_actions_for_complaint():
    """测试投诉流式输出下发 actions 事件"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/chat/stream",
            json={"message": "我要投诉你们平台", "conversation_id": None},
        )
        assert resp.status_code == 200
        text = resp.text
        assert "event_type\": \"actions\"" in text
        assert "transfer_agent" in text
        assert "create_ticket" in text


@pytest.mark.asyncio
async def test_chat_stream_workflow_tool_events_and_persistence():
    """测试在工作流模式下，工具调用事件(tool_start, tool_end)、文本及动作的完整回显与数据库落盘"""

    mock_engine = MagicMock()
    mock_final_state = {
        "conversation_id": 10,
        "input_query": "查订单1001物流",
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "query_logistics", "args": {"order_id": "1001"}}],
            ),
            ToolMessage(
                content='{"status": "派送中"}',
                tool_call_id="call_1",
                name="query_logistics",
            ),
        ],
        "response_text": "您的订单1001正在派件中，预计今日送达。",
        "suggested_actions": ["transfer_agent"],
    }
    mock_engine.run = AsyncMock(return_value=mock_final_state)

    fake_db = FakeAsyncSession()
    service = ChatService(workflow_engine=mock_engine)

    events = [e async for e in service.stream_chat(db=fake_db, conversation_id=10, message="查订单1001物流")]

    # 1. 验证事件流
    assert len(events) == 4
    assert events[0]["event_type"] == "tool_start"
    assert events[0]["tool_name"] == "query_logistics"
    assert events[0]["tool_label"] == "查询物流"
    assert events[0]["args"] == {"order_id": "1001"}

    assert events[1]["event_type"] == "tool_end"
    assert events[1]["tool_name"] == "query_logistics"
    assert events[1]["success"] is True

    assert events[2]["event_type"] == "text"
    assert "正在派件中" in events[2]["content"]

    assert events[3]["event_type"] == "actions"
    assert events[3]["actions"] == ["transfer_agent"]

    # 2. 验证数据库落盘 4 条消息: user, assistant(tool_calls), tool, assistant(final)
    assert len(fake_db.messages) == 4
    assert fake_db.messages[0].role == "user"
    assert fake_db.messages[0].content == "查订单1001物流"

    assert fake_db.messages[1].role == "assistant"
    assert len(fake_db.messages[1].tool_calls) == 1
    assert fake_db.messages[1].tool_calls[0]["name"] == "query_logistics"

    assert fake_db.messages[2].role == "tool"
    assert fake_db.messages[2].tool_call_id == "call_1"
    assert "派送中" in fake_db.messages[2].content

    assert fake_db.messages[3].role == "assistant"
    assert "正在派件中" in fake_db.messages[3].content


