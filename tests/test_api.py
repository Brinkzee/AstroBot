import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest
from app.schemas.after_sale import AfterSaleTicket
from app.services.session_manager import session_manager

def test_health_endpoint():
    from main import app
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_chat_page_endpoint():
    from main import app
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "智能客服小星" in res.text

    res_chat = client.get("/chat")
    assert res_chat.status_code == 200
    assert "text/html" in res_chat.headers["content-type"]

@patch("app.api.routes.extract_after_sale_ticket")
def test_after_sale_extract_endpoint(mock_extract):
    from main import app
    client = TestClient(app)

    mock_extract.return_value = AfterSaleTicket(
        order_id="TB20240901",
        issue_type="商品质量问题/破损",
        expected_solution="换货",
        raw_description="拉链坏了要换货"
    )

    res = client.post("/api/after-sale/extract", json={"description": "拉链坏了要换货，订单TB20240901"})
    assert res.status_code == 200
    data = res.json()
    assert data["order_id"] == "TB20240901"
    assert data["issue_type"] == "商品质量问题/破损"
    assert data["expected_solution"] == "换货"

@patch("app.api.routes.get_chat_model")
def test_chat_stream_endpoint(mock_get_model):
    from main import app
    client = TestClient(app)

    # 模拟流式 chunks
    async def fake_astream(messages):
        class Chunk:
            def __init__(self, c):
                self.content = c
        yield Chunk("您好！")
        yield Chunk("我是小星。")

    mock_llm = MagicMock()
    mock_llm.astream = fake_astream
    mock_get_model.return_value = mock_llm

    session_id = "test-stream-session-1"
    session_manager.clear_session(session_id)

    res = client.post("/api/chat/stream", json={"session_id": session_id, "message": "你好呀"})
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    body = res.text
    assert "data: " in body
    assert "[DONE]" in body
    assert "您好！" in body
    assert "我是小星。" in body

    # 验证历史会话已被记录
    history = session_manager.get_history(session_id)
    assert len(history) == 2
    assert history[0].content == "你好呀"
    assert history[1].content == "您好！我是小星。"

@patch("app.api.routes.get_chat_model")
def test_multi_turn_context_retained(mock_get_model):
    """测试连续两轮对话，第二轮输入时历史上下文正确传递给 Prompt"""
    from main import app
    client = TestClient(app)

    calls = []
    async def fake_astream(messages):
        calls.append(messages)
        class Chunk:
            def __init__(self, c):
                self.content = c
        yield Chunk("收到。")

    mock_llm = MagicMock()
    mock_llm.astream = fake_astream
    mock_get_model.return_value = mock_llm

    sid = "multi-turn-test"
    session_manager.clear_session(sid)

    # 第一轮
    res1 = client.post("/api/chat/stream", json={"session_id": sid, "message": "我想买羽绒服"})
    assert res1.status_code == 200
    assert len(calls) == 1

    # 第二轮
    res2 = client.post("/api/chat/stream", json={"session_id": sid, "message": "保暖效果好吗？"})
    assert res2.status_code == 200
    assert len(calls) == 2

    # 第二次调用的 messages 中应包含第一轮的问答历史
    round2_messages = calls[1]
    history_texts = [m.content for m in round2_messages]
    assert any("我想买羽绒服" in t for t in history_texts)
    assert any("收到。" in t for t in history_texts)
