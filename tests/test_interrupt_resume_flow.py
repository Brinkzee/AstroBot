import asyncio
import json
import re
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import select

from main import app
from app.db.session import AsyncSessionLocal, engine, get_db
from app.models.ticket import Ticket
from app.models.tool_audit_log import ToolAuditLog
from app.models.conversation import Conversation
from app.schemas.chat import ChatResumeRequest
from app.services.workflow.engine import WorkflowEngine
from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.agent_node import main_agent_node
from scripts.wsl_helper import ensure_mysql_ready
from tests.test_chat_service import FakeAsyncSession


@pytest.fixture(scope="module", autouse=True)
def setup_mysql_ready():
    """保证 WSL2 中的 MySQL 容器已拉起且保活进程处于激活状态"""
    ensure_mysql_ready(verbose=False)


@pytest.fixture(autouse=True)
async def reset_engine_connections():
    """避免 pytest-asyncio 在多个测试函数间切换 event loop 时复用已关闭 loop 的连接与单例"""
    import gc
    from app.llm import clear_llm_cache
    from app.tools import business_tools

    clear_llm_cache()
    if hasattr(business_tools, "_retriever") and business_tools._retriever is not None:
        if hasattr(business_tools._retriever, "close"):
            business_tools._retriever.close()
        business_tools._retriever = None
    await engine.dispose()
    gc.collect()

    yield

    clear_llm_cache()
    if hasattr(business_tools, "_retriever") and business_tools._retriever is not None:
        if hasattr(business_tools._retriever, "close"):
            business_tools._retriever.close()
        business_tools._retriever = None
    await engine.dispose()
    gc.collect()


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


@pytest.fixture
def clean_db():
    async def _clean(conv_id: int):
        async with AsyncSessionLocal() as session:
            tickets = (await session.execute(select(Ticket).where(Ticket.conversation_id == conv_id))).scalars().all()
            for t in tickets:
                await session.delete(t)
            audits = (await session.execute(select(ToolAuditLog).where(ToolAuditLog.conversation_id == conv_id))).scalars().all()
            for a in audits:
                await session.delete(a)
            await session.commit()
    return _clean


# ==============================================================================
# Test 1: User says "帮我建工单" with NO description -> Agent responds asking for details, does NOT call create_ticket
# ==============================================================================
@pytest.mark.asyncio
async def test_agent_asks_for_details_when_no_description():
    """Test 1: User says "帮我建工单" with NO description -> Agent responds asking for details, does NOT call create_ticket."""
    state = create_initial_state(
        conversation_id=901,
        query="帮我建工单",
    )
    state["intent"] = "售后"

    # 1. 模拟遵循提示词的模型返回追问话术
    mock_model_ask = MagicMock()
    mock_model_ask.bind_tools.return_value = mock_model_ask
    mock_model_ask.ainvoke = AsyncMock(
        return_value=AIMessage(content="请问您具体遇到了什么问题？请提供详细情况以便我们为您建立工单联系专员跟进处理。")
    )

    result = await main_agent_node(state, model=mock_model_ask)
    assert "请问您具体遇到了什么问题" in result["response_text"] or "详细情况" in result["response_text"]
    assert "__interrupt__" not in result
    tool_msgs = [m for m in result.get("messages", []) if isinstance(m, ToolMessage) or getattr(m, "tool_calls", None)]
    assert len(tool_msgs) == 0

    # 2. 模拟模型企图以空描述直接调 create_ticket，节点安全拦截并主动追问
    mock_model_hallucinate = MagicMock()
    mock_model_hallucinate.bind_tools.return_value = mock_model_hallucinate
    mock_model_hallucinate.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[{"name": "create_ticket", "args": {"description": ""}, "id": "call_empty"}]
        )
    )

    result2 = await main_agent_node(state, model=mock_model_hallucinate)
    assert "请问您具体遇到了什么问题" in result2["response_text"] or "详细情况" in result2["response_text"]
    assert "__interrupt__" not in result2


# ==============================================================================
# Test 2: User provides description -> Agent calls create_ticket, triggers interrupt, graph state captures interrupt
# ==============================================================================
@pytest.mark.asyncio
async def test_agent_triggers_interrupt_when_description_provided(clean_db):
    """Test 2: User provides description -> Agent calls create_ticket, triggers interrupt({"event_type": "ticket_preview", ...}), graph state captures the interrupt."""
    conv_id = 902
    await clean_db(conv_id)

    engine = WorkflowEngine()
    fake_intent = {"intent": "人工", "confidence": 0.98, "intent_reason": "用户申请建工单转人工专员处理"}

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_ticket",
                    "args": {
                        "ticket_type": "售后",
                        "description": "羽绒服袖口开线破损严重，请专员处理",
                    },
                    "id": "call_ticket_902",
                }
            ],
        )
    )

    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)):
        with patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model):
            final_state = await engine.run(
                conversation_id=conv_id,
                query="帮我建工单，我买的羽绒服袖口开线破损严重，请专员处理",
            )

    # 验证触发 LangGraph interrupt 并捕获 ticket_preview
    assert "__interrupt__" in final_state
    interrupts = final_state["__interrupt__"]
    assert len(interrupts) > 0
    val = getattr(interrupts[0], "value", interrupts[0])
    if isinstance(val, dict) and "value" in val:
        val = val["value"]
    assert val.get("event_type") == "ticket_preview"
    assert val.get("ticket_type") == "售后"
    assert "袖口开线" in val.get("description", "")
    assert val.get("conversation_id") == conv_id


# ==============================================================================
# Test 3: resume(conv_id, action="confirm") -> create_ticket executed, audit "成功", ticket in DB, response contains T...
# ==============================================================================
@pytest.mark.asyncio
async def test_resume_confirm_flow(clean_db):
    """Test 3: resume(conv_id, action="confirm") executes create_ticket, records "成功" in tool_audit_logs, ticket record is created in tickets table, response contains ticket number (T...)."""
    conv_id = 903
    await clean_db(conv_id)

    engine = WorkflowEngine()
    fake_intent = {"intent": "人工", "confidence": 0.98, "intent_reason": "用户申请建工单转人工专员处理"}

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    # 第一次 LLM 返回建单 tool_call，恢复后第二次 LLM 返回总结话术
    mock_model.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_ticket",
                        "args": {
                            "ticket_type": "售后",
                            "description": "羽绒服袖口开线破损严重，请专员处理",
                        },
                        "id": "call_ticket_903",
                    }
                ],
            ),
            AIMessage(content="已为您成功创建售后服务工单，售后专员将在24小时内与您联系，请保持电话畅通。"),
        ]
    )

    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)):
        with patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model):
            # 1. 运行首轮，触发挂起
            state1 = await engine.run(
                conversation_id=conv_id,
                query="帮我建工单，我买的羽绒服袖口开线破损严重，请专员处理",
            )
            assert "__interrupt__" in state1

            # 2. 恢复挂起，用户确认
            resumed_state = await engine.resume(conversation_id=conv_id, action="confirm")

    # 3. 验证回复包含工单编号 (T...)
    resp_text = resumed_state.get("response_text", "")
    assert re.search(r"T\d+", resp_text) is not None

    # 4. 验证 tickets 表已插入记录
    async with AsyncSessionLocal() as session:
        t_stmt = select(Ticket).where(Ticket.conversation_id == conv_id)
        t_res = await session.execute(t_stmt)
        ticket = t_res.scalar_one_or_none()
        assert ticket is not None
        assert ticket.ticket_no.startswith("T")
        assert "袖口开线" in ticket.description

        # 5. 验证 tool_audit_logs 表已落盘且状态为「成功」
        a_stmt = select(ToolAuditLog).where(
            ToolAuditLog.conversation_id == conv_id,
            ToolAuditLog.tool_name == "create_ticket",
        )
        a_res = await session.execute(a_stmt)
        audit_log = a_res.scalar_one_or_none()
        assert audit_log is not None
        assert audit_log.status == "成功"


# ==============================================================================
# Test 4: resume(conv_id, action="cancel") -> rejects create_ticket, audit "权限拒绝", NO ticket in DB, friendly cancellation
# ==============================================================================
@pytest.mark.asyncio
async def test_resume_cancel_flow(clean_db):
    """Test 4: resume(conv_id, action="cancel") rejects create_ticket, records "权限拒绝" in tool_audit_logs, NO ticket is created in tickets table, response contains friendly cancellation message."""
    conv_id = 904
    await clean_db(conv_id)

    engine = WorkflowEngine()
    fake_intent = {"intent": "人工", "confidence": 0.98, "intent_reason": "用户申请建工单转人工专员处理"}

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_ticket",
                    "args": {
                        "ticket_type": "售后",
                        "description": "羽绒服袖口开线破损严重，请专员处理",
                    },
                    "id": "call_ticket_904",
                }
            ],
        )
    )

    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)):
        with patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model):
            # 1. 运行首轮，触发挂起
            state1 = await engine.run(
                conversation_id=conv_id,
                query="帮我建工单，我买的羽绒服袖口开线破损严重，请专员处理",
            )
            assert "__interrupt__" in state1

            # 2. 恢复挂起，用户取消
            resumed_state = await engine.resume(conversation_id=conv_id, action="cancel")

    # 3. 验证回复为友好取消提示
    resp_text = resumed_state.get("response_text", "")
    assert "取消" in resp_text

    # 4. 验证 tickets 表未创建任何工单
    async with AsyncSessionLocal() as session:
        t_stmt = select(Ticket).where(Ticket.conversation_id == conv_id)
        t_res = await session.execute(t_stmt)
        ticket = t_res.scalar_one_or_none()
        assert ticket is None

        # 5. 验证 tool_audit_logs 表已落盘且状态为「权限拒绝」
        a_stmt = select(ToolAuditLog).where(
            ToolAuditLog.conversation_id == conv_id,
            ToolAuditLog.tool_name == "create_ticket",
        )
        a_res = await session.execute(a_stmt)
        audit_log = a_res.scalar_one_or_none()
        assert audit_log is not None
        assert audit_log.status == "权限拒绝"


# ==============================================================================
# Test 5: POST /api/chat/resume with action="confirm" streams SSE events
# ==============================================================================
def test_api_chat_resume_confirm_sse():
    """Test 5: POST /api/chat/resume with action="confirm" streams SSE events: yields tool_start, tool_end, text (containing ticket no), and [DONE]."""
    conv_id = 905
    client = TestClient(app)

    mock_resume_events = [
        {
            "event_type": "tool_start",
            "conversation_id": conv_id,
            "tool_name": "create_ticket",
            "tool_label": "创建人工工单",
            "args": {"ticket_type": "售后", "description": "羽绒服破损"},
        },
        {
            "event_type": "tool_end",
            "conversation_id": conv_id,
            "tool_name": "create_ticket",
            "success": True,
        },
        {
            "event_type": "text",
            "conversation_id": conv_id,
            "content": "已为您成功创建售后工单，工单编号为：T20260914193000123，专员将尽快跟进。",
        },
    ]

    async def fake_resume_chat(*args, **kwargs):
        for e in mock_resume_events:
            yield e

    with patch("app.api.routes.chat_service.resume_chat", side_effect=fake_resume_chat):
        res = client.post(
            "/api/chat/resume",
            json={"conversation_id": conv_id, "action": "confirm"},
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        events, has_done = parse_sse_events(res.text)
        assert has_done is True
        event_types = [e["event_type"] for e in events]
        assert event_types == ["tool_start", "tool_end", "text"]

        assert events[0]["tool_name"] == "create_ticket"
        assert events[1]["success"] is True
        assert "T20260914" in events[2]["content"]


# ==============================================================================
# Test 6: POST /api/chat/resume with action="cancel" streams SSE events
# ==============================================================================
def test_api_chat_resume_cancel_sse():
    """Test 6: POST /api/chat/resume with action="cancel" streams SSE events: yields text (cancellation message), and [DONE]."""
    conv_id = 906
    client = TestClient(app)

    mock_resume_events = [
        {
            "event_type": "text",
            "conversation_id": conv_id,
            "content": "已为您取消工单创建。如果您有其他问题，欢迎随时咨询。",
        },
    ]

    async def fake_resume_chat(*args, **kwargs):
        for e in mock_resume_events:
            yield e

    with patch("app.api.routes.chat_service.resume_chat", side_effect=fake_resume_chat):
        res = client.post(
            "/api/chat/resume",
            json={"conversation_id": conv_id, "action": "cancel"},
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        events, has_done = parse_sse_events(res.text)
        assert has_done is True
        event_types = [e["event_type"] for e in events]
        assert event_types == ["text"]
        assert "tool_start" not in event_types
        assert "取消" in events[0]["content"]


# ==============================================================================
# Test 7: Multi-turn test: User creates ticket -> cancel -> NEXT turn normal question
# ==============================================================================
@pytest.mark.asyncio
async def test_multiturn_ticket_then_normal_question(clean_db):
    """Test 7: Multi-turn test: User creates ticket -> cancel -> In the NEXT turn,
    user asks a normal question ("运费险怎么退"). Verify that _pending_agent_states
    has NO leftover state for conv_id, LLM is called normally, and normal response is returned
    (no replaying previous ticket).
    """
    conv_id = 907
    await clean_db(conv_id)

    from app.services.workflow.nodes.agent_node import _pending_agent_states

    engine = WorkflowEngine()
    fake_intent_ticket = {"intent": "人工", "confidence": 0.98, "intent_reason": "用户申请建工单转人工专员处理"}
    fake_intent_qa = {"intent": "订单", "confidence": 0.95, "intent_reason": "运费险咨询"}

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_ticket",
                        "args": {
                            "ticket_type": "售后",
                            "description": "羽绒服袖口开线破损严重，请专员处理",
                        },
                        "id": "call_ticket_907",
                    }
                ],
            ),
            AIMessage(content="运费险可以在订单详情页申请退货退款，卖家同意后由系统自动发起运费险理赔。"),
        ]
    )

    with patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model):
        with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", side_effect=[fake_intent_ticket, fake_intent_qa]):
            # Turn 1: 用户发起建单，触发挂起
            state1 = await engine.run(
                conversation_id=conv_id,
                query="帮我建工单，我买的羽绒服袖口开线破损严重，请专员处理",
            )
            assert "__interrupt__" in state1
            # 挂起期间，缓存中存在待恢复状态
            assert conv_id in _pending_agent_states

            # 用户选择取消建单
            resumed_state = await engine.resume(conversation_id=conv_id, action="cancel")
            assert "取消" in resumed_state.get("response_text", "")

            # 关键断言 1: 挂起恢复后，_pending_agent_states 中必须彻底清理，禁止状态遗留泄漏
            assert conv_id not in _pending_agent_states

            # Turn 2: 下一轮对话，用户咨询常规业务问题
            state2 = await engine.run(
                conversation_id=conv_id,
                query="运费险怎么退",
            )

            # 关键断言 2: 第二轮正常执行，无遗留挂起状态，LLM 正常调用且正常返回
            assert conv_id not in _pending_agent_states
            assert "__interrupt__" not in state2
            assert "运费险" in state2.get("response_text", "")
            assert "工单" not in state2.get("response_text", "")


# ==============================================================================
# Test 8: Direct test of chat_service.resume_chat (confirm and cancel)
# ==============================================================================
@pytest.mark.asyncio
async def test_chat_service_resume_chat_direct(clean_db):
    """Test 8: Direct test of chat_service.resume_chat (without mocking resume_chat at router level)
    to verify that resume_chat yields tool_start/tool_end (or text for cancel), saves message to db, and completes.
    """
    from app.services.chat_service import ChatService

    # 1. 测试 action="confirm": 产出 tool_start, tool_end, text，保存消息至 db 并完成
    conv_id_confirm = 908
    await clean_db(conv_id_confirm)

    engine_confirm = WorkflowEngine()
    fake_intent = {"intent": "人工", "confidence": 0.98, "intent_reason": "用户申请建工单转人工专员处理"}

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_ticket",
                        "args": {
                            "ticket_type": "售后",
                            "description": "羽绒服袖口开线破损严重，请专员处理",
                        },
                        "id": "call_ticket_908",
                    }
                ],
            ),
            AIMessage(content="已为您成功创建工单，工单编号为：T20260914193000908，人工客服专员将尽快为您跟进处理。"),
        ]
    )

    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)):
        with patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model):
            # 触发中断挂起
            state_suspended = await engine_confirm.run(
                conversation_id=conv_id_confirm,
                query="帮我建工单，我买的羽绒服袖口开线破损严重，请专员处理",
            )
            assert "__interrupt__" in state_suspended

            # 使用 FakeAsyncSession 作为 db 调用 chat_service.resume_chat
            fake_db = FakeAsyncSession()
            chat_service = ChatService(workflow_engine=engine_confirm)

            events = [
                event
                async for event in chat_service.resume_chat(
                    db=fake_db,
                    conversation_id=conv_id_confirm,
                    action="confirm",
                )
            ]

            # 验证事件序列：tool_start -> tool_end -> text
            event_types = [e["event_type"] for e in events]
            assert "tool_start" in event_types
            assert "tool_end" in event_types
            assert "text" in event_types

            # 验证工具详情
            tool_start_evt = next(e for e in events if e["event_type"] == "tool_start")
            assert tool_start_evt["tool_name"] == "create_ticket"
            assert tool_start_evt["conversation_id"] == conv_id_confirm

            tool_end_evt = next(e for e in events if e["event_type"] == "tool_end")
            assert tool_end_evt["tool_name"] == "create_ticket"
            assert tool_end_evt["success"] is True

            text_evt = next(e for e in events if e["event_type"] == "text")
            assert "T20260914" in text_evt["content"]

            # 验证消息已正确持久化到 db
            assert len(fake_db.messages) >= 3
            saved_roles = [m.role for m in fake_db.messages]
            assert "assistant" in saved_roles
            assert "tool" in saved_roles

    # 2. 测试 action="cancel": 产出 text，保存消息至 db 并完成
    conv_id_cancel = 909
    await clean_db(conv_id_cancel)

    engine_cancel = WorkflowEngine()
    mock_model_cancel = MagicMock()
    mock_model_cancel.bind_tools.return_value = mock_model_cancel
    mock_model_cancel.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_ticket",
                    "args": {
                        "ticket_type": "售后",
                        "description": "羽绒服袖口开线破损严重，请专员处理",
                    },
                    "id": "call_ticket_909",
                }
            ],
        )
    )

    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)):
        with patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model_cancel):
            # 触发挂起
            state_suspended2 = await engine_cancel.run(
                conversation_id=conv_id_cancel,
                query="帮我建工单，我买的羽绒服袖口开线破损严重，请专员处理",
            )
            assert "__interrupt__" in state_suspended2

            fake_db_cancel = FakeAsyncSession()
            chat_service_cancel = ChatService(workflow_engine=engine_cancel)

            cancel_events = [
                event
                async for event in chat_service_cancel.resume_chat(
                    db=fake_db_cancel,
                    conversation_id=conv_id_cancel,
                    action="cancel",
                )
            ]

            cancel_types = [e["event_type"] for e in cancel_events]
            assert "text" in cancel_types
            assert "tool_start" not in cancel_types
            assert "取消" in cancel_events[0]["content"]

            # 验证取消消息持久化至 db
            assert len(fake_db_cancel.messages) == 1
            assert fake_db_cancel.messages[0].role == "assistant"
            assert "取消" in fake_db_cancel.messages[0].content

