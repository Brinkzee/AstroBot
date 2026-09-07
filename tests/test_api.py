import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest
from app.schemas.after_sale import AfterSaleTicket

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

@patch("app.api.routes.chat_service.stream_chat")
def test_chat_stream_endpoint(mock_stream_chat):
    from main import app
    client = TestClient(app)

    async def fake_stream_chat(db, conversation_id, message):
        yield {
            "event_type": "text",
            "conversation_id": conversation_id or 1,
            "content": "您好！我是小星。"
        }

    mock_stream_chat.side_effect = fake_stream_chat

    session_id = "1"
    res = client.post("/api/chat/stream", json={"session_id": session_id, "message": "你好呀"})
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    body = res.text
    assert "data: " in body
    assert "[DONE]" in body
    assert "您好！我是小星。" in body
    assert '"event_type": "text"' in body

@patch("app.api.routes.chat_service.stream_chat")
def test_multi_turn_context_retained(mock_stream_chat):
    """测试连续两轮对话，conversation_id 正确在连续请求中传递"""
    from main import app
    client = TestClient(app)

    calls = []
    async def fake_stream_chat(db, conversation_id, message):
        calls.append({"conversation_id": conversation_id, "message": message})
        yield {
            "event_type": "text",
            "conversation_id": conversation_id,
            "content": "收到。"
        }

    mock_stream_chat.side_effect = fake_stream_chat

    cid = 99

    # 第一轮
    res1 = client.post("/api/chat/stream", json={"conversation_id": cid, "message": "我想买羽绒服"})
    assert res1.status_code == 200
    assert len(calls) == 1
    assert calls[0]["conversation_id"] == 99
    assert calls[0]["message"] == "我想买羽绒服"

    # 第二轮
    res2 = client.post("/api/chat/stream", json={"conversation_id": cid, "message": "保暖效果好吗？"})
    assert res2.status_code == 200
    assert len(calls) == 2
    assert calls[1]["conversation_id"] == 99
    assert calls[1]["message"] == "保暖效果好吗？"
