import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from main import app
from app.db.session import get_db
from app.models import Conversation
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


MOCK_SUGGESTED_ORDERS = [
    {
        "order_id": "1001",
        "product_name": "高端智能降噪耳机",
        "amount": "1299.00",
        "status": "已完成",
        "order_time": "2024-03-01 10:00:00",
    },
    {
        "order_id": "1002",
        "product_name": "极简智能手表",
        "amount": "899.00",
        "status": "已发货",
        "order_time": "2024-03-05 14:30:00",
    },
    {
        "order_id": "1003",
        "product_name": "快充移动电源",
        "amount": "199.00",
        "status": "已完成",
        "order_time": "2024-03-08 09:15:00",
    },
]


@pytest.mark.asyncio
async def test_stream_chat_emits_order_selector_when_need_order_selection():
    """测试用例 1: 模拟工作流返回 status='need_order_selection' 与 suggested_orders，
    断言 stream_chat 输出 order_selector 事件且包含 3 个订单项，随后输出引导 text 事件"""
    mock_engine = MagicMock()
    mock_final_state = {
        "conversation_id": 100,
        "status": "need_order_selection",
        "suggested_orders": MOCK_SUGGESTED_ORDERS,
        "response_text": "为您查询退款政策前，请先选择您需要咨询的订单：",
        "suggested_actions": [],
    }
    mock_engine.run = AsyncMock(return_value=mock_final_state)

    fake_db = FakeAsyncSession()
    fake_db.conversations[100] = Conversation(id=100, user_id="default_user", status="进行中")
    service = ChatService(workflow_engine=mock_engine)

    events = [
        e
        async for e in service.stream_chat(
            db=fake_db,
            conversation_id=100,
            message="我想办理退货",
        )
    ]

    # 断言首个事件为 order_selector
    assert len(events) >= 2
    assert events[0]["event_type"] == "order_selector"
    assert events[0]["conversation_id"] == 100
    assert len(events[0]["orders"]) == 3
    assert events[0]["orders"][0]["order_id"] == "1001"
    assert events[0]["orders"][1]["order_id"] == "1002"
    assert events[0]["orders"][2]["order_id"] == "1003"

    # 断言随后输出引导 text 事件
    assert events[1]["event_type"] == "text"
    assert events[1]["conversation_id"] == 100
    assert "请先选择您需要咨询的订单" in events[1]["content"]


@pytest.mark.asyncio
async def test_stream_chat_emits_actions_containing_apply_refund():
    """测试用例 2: 模拟退款工作流返回判定通过与 suggested_actions=['apply_refund']，
    断言 stream_chat 输出 actions 事件包含 'apply_refund'，且不输出 order_selector 事件"""
    mock_engine = MagicMock()
    mock_final_state = {
        "conversation_id": 101,
        "order_id": "1001",
        "status": "completed",
        "response_text": "您的订单1001符合七天无理由退货条件，已为您开启退款申请入口。",
        "suggested_actions": ["apply_refund"],
    }
    mock_engine.run = AsyncMock(return_value=mock_final_state)

    fake_db = FakeAsyncSession()
    fake_db.conversations[101] = Conversation(id=101, user_id="default_user", status="进行中")
    service = ChatService(workflow_engine=mock_engine)

    events = [
        e
        async for e in service.stream_chat(
            db=fake_db,
            conversation_id=101,
            message="订单1001可以退款吗",
        )
    ]

    # 断言不发射 order_selector
    event_types = [e["event_type"] for e in events]
    assert "order_selector" not in event_types

    # 断言发射 text 事件与 actions 事件
    assert "text" in event_types
    assert "actions" in event_types

    action_event = next(e for e in events if e["event_type"] == "actions")
    assert action_event["conversation_id"] == 101
    assert "apply_refund" in action_event["actions"]


@pytest.mark.asyncio
async def test_stream_chat_standard_queries_without_order_selector():
    """测试用例 3: 模拟普通闲聊或无 suggested_orders 查询，断言不发射 order_selector 事件"""
    mock_engine = MagicMock()
    mock_final_state = {
        "conversation_id": 102,
        "status": "chitchat",
        "response_text": "您好！我是智能客服助手小星，很高兴为您服务！",
        "suggested_actions": [],
    }
    mock_engine.run = AsyncMock(return_value=mock_final_state)

    fake_db = FakeAsyncSession()
    fake_db.conversations[102] = Conversation(id=102, user_id="default_user", status="进行中")
    service = ChatService(workflow_engine=mock_engine)

    events = [
        e
        async for e in service.stream_chat(
            db=fake_db,
            conversation_id=102,
            message="你好",
        )
    ]

    event_types = [e["event_type"] for e in events]
    assert "order_selector" not in event_types
    assert "text" in event_types
    assert events[0]["event_type"] == "text"
    assert "智能客服助手小星" in events[0]["content"]


def test_api_chat_stream_e2e_sse_events(mock_db_session):
    """测试用例 4: 结合 FastAPI 端点 /api/chat/stream，通过 TestClient 发起真实 SSE 请求，
    解析 data: ... 行并验证 JSON 格式以及 order_selector / actions 事件下发"""
    mock_db_session.conversations[201] = Conversation(id=201, user_id="default_user", status="进行中")
    mock_db_session.conversations[202] = Conversation(id=202, user_id="default_user", status="进行中")

    client = TestClient(app)

    # 1. 测试缺单挂起下发 order_selector SSE 流
    mock_engine_order_selector = MagicMock()
    mock_engine_order_selector.run = AsyncMock(
        return_value={
            "conversation_id": 201,
            "status": "need_order_selection",
            "suggested_orders": MOCK_SUGGESTED_ORDERS,
            "response_text": "请选择需要退款的订单：",
            "suggested_actions": [],
        }
    )

    with patch("app.api.routes.chat_service.workflow_engine", mock_engine_order_selector):
        res1 = client.post(
            "/api/chat/stream",
            json={"conversation_id": 201, "message": "我想退货"},
        )
        assert res1.status_code == 200
        assert "text/event-stream" in res1.headers["content-type"]
        assert res1.headers.get("Cache-Control") == "no-cache"

        events1, has_done1 = parse_sse_events(res1.text)
        assert has_done1 is True
        assert len(events1) >= 2

        # 校验 order_selector 事件结构
        order_sel = events1[0]
        assert order_sel["event_type"] == "order_selector"
        assert order_sel["conversation_id"] == 201
        assert len(order_sel["orders"]) == 3
        assert order_sel["orders"][0]["order_id"] == "1001"

        # 校验 text 事件结构
        text_event = events1[1]
        assert text_event["event_type"] == "text"
        assert text_event["conversation_id"] == 201
        assert text_event["content"] == "请选择需要退款的订单："

    # 2. 测试退款判定通过下发 apply_refund 动作事件 SSE 流
    mock_engine_actions = MagicMock()
    mock_engine_actions.run = AsyncMock(
        return_value={
            "conversation_id": 202,
            "status": "completed",
            "order_id": "1001",
            "response_text": "订单1001符合退货退款政策。",
            "suggested_actions": ["apply_refund"],
        }
    )

    with patch("app.api.routes.chat_service.workflow_engine", mock_engine_actions):
        res2 = client.post(
            "/api/chat/stream",
            json={"conversation_id": 202, "message": "订单1001退款"},
        )
        assert res2.status_code == 200
        events2, has_done2 = parse_sse_events(res2.text)
        assert has_done2 is True

        event_types2 = [e["event_type"] for e in events2]
        assert "order_selector" not in event_types2
        assert "actions" in event_types2

        act_ev = next(e for e in events2 if e["event_type"] == "actions")
        assert act_ev["actions"] == ["apply_refund"]
