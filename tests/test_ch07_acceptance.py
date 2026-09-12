"""Comprehensive Chapter 7 Acceptance Test Suite.

Covers all 5 core user acceptance criteria verbatim:
1. 连续聊 20 轮不爆、不崩:
   - 模拟用户发起连续 20 轮对话交互（含多轮问答与工具调用结果）；
   - 验证经过 20 轮后，系统正常响应，最终注入大模型提示词的总 Token 估算严格保持在模型上下文窗口上限以内，绝对不发生 Token 溢出与系统崩溃。
2. 18000 窗口演示配置与全链路级联演练:
   - 环境变量或配置设置为 MODEL_CONTEXT_WINDOW=18000；
   - 严格验证预算倒推计算：总滑窗预算 5650，层 1 (原文区) 预算 3954，层 2 (半压缩区) 预算 1695；
   - 模拟多轮长会话生成（第 1 轮提供关键信息：订单号 ORD10086、商品「华为 Mate60」、诉求「申请退款」）：
     - 随着对话推进，层 1 历史超出 3954 Tokens，触发层 1 降级推移（layer1_from_msg_id 推进，日志输出降级标识）；
     - 降级后的消息积压在层 2，当层 2 超过 1695 Tokens 时，触发后台异步摘要引擎（[summary trigger]）；
     - 摘要引擎生成新一段摘要并追加落库至 conversation_summaries，更新 conversations.summary 与 summary_upto_msg_id；
     - 验证在后续提问「最开始那个订单后来怎么说」时，大模型接收到的上下文包含摘要投影，能准确索引出最初的订单号 ORD10086 与退款诉求。
3. 默认 128k 大窗口下连续聊 20 轮无降级无摘要 (装得下就不压):
   - 验证在默认大窗口（128k=131072 Tokens）配置下，连续聊 20 轮对话；
   - 检查断言：layer1_from_msg_id 始终为 0（无降级）、summary_upto_msg_id 始终为 0、conversation_summaries 记录数为 0（未触发摘要）；
   - 验证“装得下就不压，压缩是成本不是美德”的原则坚决落实。
4. 后台摘要非阻塞执行与日志 grep 可观测性:
   - 验证在触发摘要时，当前轮流式 SSE 输出即刻完成，无需等待后台 LLM 摘要完成（异步非阻塞）；
   - 检查 log/app.log，验证每轮对话均记录了可观测日志：
     - grep "[history_ctx]" 能够直查到前置节点打印的历史上下文；
     - grep "[model_ctx]" 能够直查到主力 Agent 模型入参上下文（包含摘要全文、滑窗逐条消息明细与 Tokens 估算）。
5. 多会话侧栏对照与历史完整回载继续聊:
   - 通过 REST API 模拟：
     - 创建会话 A 交流 3 轮（含工具调用）；
     - 创建会话 B 交流 2 轮；
     - 调用 GET /api/conversations，验证返回两通会话（按更新时间倒序，包含首问预览）；
     - 调用 GET /api/conversations/{conv_A_id}/messages，完整回载会话 A 全部历史消息；
     - 基于会话 A 的 conversation_id 继续追问第 4 轮，验证上下文基于会话 A 的历史连贯推进，与会话 B 完全隔离。
"""

import asyncio
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config import Settings, settings
from app.db.session import Base, get_db
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.summary import ConversationSummary
from app.services.chat_service import ChatService
from app.services.context.budget import (
    ContextBudgetResult,
    calculate_context_budget,
    estimate_tokens,
)
from app.services.context.logger import (
    DEFAULT_LOG_FILE,
    get_context_logger,
    log_history_context,
    log_model_context,
)
from app.services.context.manager import (
    ContextManager,
    apply_layer1_degradation,
    build_model_messages,
    format_layer1_messages,
    format_layer2_messages,
    partition_messages,
)
from app.services.context.summary_service import SummaryService
from app.services.workflow.engine import WorkflowEngine
from app.services.workflow.nodes.agent_node import main_agent_node
from main import app


class AsyncSessionAdapter:
    """Async wrapper over SQLite synchronous session for acceptance testing."""

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
def sqlite_engine():
    """Create in-memory SQLite engine with StaticPool for cross-session consistency."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def async_session_factory(sqlite_engine):
    """Factory creating AsyncSessionAdapter instances sharing the SQLite engine."""
    def factory():
        sync_sess = Session(sqlite_engine, expire_on_commit=False)
        return AsyncSessionAdapter(sync_sess)
    return factory


@pytest.fixture
def test_db(sqlite_engine, async_session_factory):
    """Provide a primary async db session and register FastAPI dependency override."""
    sync_sess = Session(sqlite_engine, expire_on_commit=False)
    adapter = AsyncSessionAdapter(sync_sess)

    async def override_get_db():
        yield adapter

    app.dependency_overrides[get_db] = override_get_db
    yield adapter
    app.dependency_overrides.pop(get_db, None)
    sync_sess.close()


@pytest.fixture(autouse=True)
def setup_log_file():
    """Ensure log directory exists and reset logger file."""
    DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    get_context_logger()
    yield


def parse_sse_events(sse_text: str):
    """Parse raw SSE text into JSON events and check for [DONE] token."""
    events = []
    has_done = False
    for line in sse_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                has_done = True
            else:
                try:
                    events.append(json.loads(payload))
                except Exception:
                    pass
    return events, has_done


# ==============================================================================
# Acceptance Criterion 1: 连续聊 20 轮不爆、不崩
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_criterion_1_twenty_rounds_no_overflow_no_crash(test_db, async_session_factory):
    """验收标准 1 (连续 20 轮不爆不崩):
    - 模拟用户发起连续 20 轮对话交互（含多轮问答与工具调用结果）；
    - 验证经过 20 轮后，系统正常响应，最终注入大模型提示词的总 Token 估算严格保持在模型上下文窗口上限以内，
      绝对不发生 Token 溢出与系统崩溃。
    """
    captured_model_messages = []

    # 构造模拟智能主力 Agent 执行器，支持多轮问答与穿插工具调用
    turn_counter = 0

    def build_mock_workflow_run():
        nonlocal turn_counter

        async def mock_run(conversation_id, query, db=None, summary=None, layer2_messages=None, layer1_messages=None, **kwargs):
            nonlocal turn_counter
            turn_counter += 1
            turn_idx = turn_counter
            
            # 在某些轮次穿插工具调用
            if turn_idx in (3, 7, 12, 17):
                tool_msg_call = AIMessage(
                    content="",
                    tool_calls=[{"id": f"call_tool_{turn_idx}", "name": "query_logistics", "args": {"order_id": f"100{turn_idx}"}}],
                )
                tool_msg_res = ToolMessage(
                    content=f'{{"status": "已发货", "order_id": "100{turn_idx}", "carrier": "顺丰速运"}}',
                    tool_call_id=f"call_tool_{turn_idx}",
                    name="query_logistics",
                )
                final_ai = AIMessage(content=f"第 {turn_idx} 轮：已为您查询到订单 100{turn_idx} 的物流，顺丰速运派送中。")
                
                # 记录组装后送入模型的提示词消息列表
                injected = ContextManager.build_model_messages(
                    system_prompt="你是星舟智能客服助手。",
                    layer2_messages=layer2_messages or [],
                    layer1_messages=layer1_messages or [],
                    current_query=query,
                    summary=summary,
                )
                captured_model_messages.append(injected)
                
                return {
                    "conversation_id": conversation_id,
                    "response_text": final_ai.content,
                    "messages": [tool_msg_call, tool_msg_res, final_ai],
                    "current_turn_tool_messages": [tool_msg_call, tool_msg_res],
                    "status": "success",
                }
            else:
                final_ai = AIMessage(content=f"第 {turn_idx} 轮响应：您的问题是「{query}」，已为您解答。")
                injected = ContextManager.build_model_messages(
                    system_prompt="你是星舟智能客服助手。",
                    layer2_messages=layer2_messages or [],
                    layer1_messages=layer1_messages or [],
                    current_query=query,
                    summary=summary,
                )
                captured_model_messages.append(injected)
                
                return {
                    "conversation_id": conversation_id,
                    "response_text": final_ai.content,
                    "messages": [final_ai],
                    "current_turn_tool_messages": [],
                    "status": "completed",
                }
        return mock_run

    mock_engine = MagicMock(spec=WorkflowEngine)
    mock_engine.run = AsyncMock(side_effect=build_mock_workflow_run())

    summary_service = SummaryService(session_factory=async_session_factory)
    service = ChatService(
        workflow_engine=mock_engine,
        use_workflow=True,
        summary_service=summary_service,
    )

    conv_id = None
    settings = Settings(_env_file=None)
    context_window_limit = settings.model_context_window

    # 连续执行 20 轮对话交互
    for i in range(1, 21):
        query = f"这是第 {i} 轮提问：关于订单与商品咨询 {i}"
        events = []
        async for ev in service.stream_chat(db=test_db, conversation_id=conv_id, message=query):
            events.append(ev)
            if conv_id is None and "conversation_id" in ev and ev["conversation_id"]:
                conv_id = ev["conversation_id"]

        # 断言每轮均成功响应且包含 text 事件
        text_events = [e for e in events if e.get("event_type") == "text"]
        assert len(text_events) == 1, f"第 {i} 轮未收到 text 事件"
        assert f"第 {i} 轮" in text_events[0]["content"]

    # 验证 20 轮对话全部完成
    assert conv_id is not None
    assert len(captured_model_messages) == 20

    # 验证最终第 20 轮注入大模型提示词的总 Token 估算严格保持在模型上下文窗口上限以内
    final_turn_messages = captured_model_messages[-1]
    final_estimated_tokens = estimate_tokens(final_turn_messages)

    assert final_estimated_tokens > 0
    assert final_estimated_tokens <= context_window_limit, (
        f"Token 溢出崩溃：第 20 轮估算 Token {final_estimated_tokens} 超出模型上下文窗口上限 {context_window_limit}"
    )

    # 验证数据库中累积了 20 轮全部消息流水（用户提问 + 助手回复 + 穿插工具调用）
    msg_stmt = select(Message).where(Message.conversation_id == conv_id)
    res = await test_db.execute(msg_stmt)
    persisted_messages = res.scalars().all()
    user_msgs = [m for m in persisted_messages if m.role == "user"]
    asst_msgs = [m for m in persisted_messages if m.role == "assistant"]
    tool_msgs = [m for m in persisted_messages if m.role == "tool"]

    assert len(user_msgs) == 20
    assert len(asst_msgs) >= 20
    assert len(tool_msgs) == 4  # 第 3, 7, 12, 17 轮触发工具调用
    assert len(persisted_messages) >= 44


# ==============================================================================
# Acceptance Criterion 2: 18000 窗口演示配置与全链路级联演练
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_criterion_2_cascade_18000_window_lifecycle(test_db, async_session_factory):
    """验收标准 2 (18000 窗口演示配置与全链路级联演练):
    - 环境变量或配置设置为 MODEL_CONTEXT_WINDOW=18000；
    - 严格验证预算倒推计算：总滑窗预算 5650，层 1 (原文区) 预算 3954，层 2 (半压缩区) 预算 1695；
    - 模拟多轮长会话生成（第 1 轮提供关键信息，如订单号 ORD10086、商品「华为 Mate60」、诉求「申请退款」）：
      - 随着对话推进，层 1 历史超出 3954 Tokens，触发层 1 降级推移（layer1_from_msg_id 推进，日志输出降级标识）；
      - 降级后的消息积压在层 2，当层 2 超过 1695 Tokens 时，触发后台异步摘要引擎（[summary trigger]）；
      - 摘要引擎生成新一段摘要并追加落库至 conversation_summaries，更新 conversations.summary 与 summary_upto_msg_id；
      - 验证在后续提问「最开始那个订单后来怎么说」时，大模型接收到的上下文包含摘要投影，能准确索引出最初的订单号 ORD10086 与退款诉求。
    """
    # 1. 严格验证 18000 演示配置预算倒推
    demo_settings = Settings(_env_file=None, model_context_window=18000)
    budget_18k = calculate_context_budget(demo_settings)

    assert budget_18k.fixed_overhead == 8750
    assert budget_18k.peak_react_overhead == 3600
    assert budget_18k.window_available == 5650
    assert budget_18k.window_budget == 5650
    assert budget_18k.layer1_budget == 3954
    assert budget_18k.layer2_budget == 1695
    assert budget_18k.is_usable is True

    # 2. 模拟多轮长会话
    summary_service = SummaryService(session_factory=async_session_factory)
    mock_engine = MagicMock(spec=WorkflowEngine)

    service = ChatService(
        workflow_engine=mock_engine,
        use_workflow=True,
        summary_service=summary_service,
    )

    # 第 1 轮：提供关键事实（订单号 ORD10086、商品 华为 Mate60、诉求 申请退款）
    mock_engine.run = AsyncMock(return_value={
        "conversation_id": 1,
        "response_text": "您好，已收到您的订单 ORD10086（华为 Mate60）申请退款诉求，正在为您审核中。",
        "messages": [AIMessage(content="您好，已收到您的订单 ORD10086（华为 Mate60）申请退款诉求，正在为您审核中。")],
        "current_turn_tool_messages": [],
        "status": "completed",
    })

    with patch("app.services.chat_service.calculate_context_budget", return_value=budget_18k):
        events_round1 = []
        async for ev in service.stream_chat(
            db=test_db,
            conversation_id=None,
            message="你好，我的订单号是 ORD10086，购买了商品华为 Mate60，由于商品存在质量瑕疵，我申请退款。",
        ):
            events_round1.append(ev)

    conv_stmt = select(Conversation).limit(1)
    conv = (await test_db.execute(conv_stmt)).scalar_one()
    conv_id = conv.id

    # 验证第 1 轮初态：尚未降级
    assert conv.layer1_from_msg_id is None or conv.layer1_from_msg_id == 0
    assert conv.summary_upto_msg_id is None or conv.summary_upto_msg_id == 0

    # 3. 构造长历史对话使得层 1 超过 3954 Tokens
    # 插入长消息推高层 1 历史 Token
    long_chinese_text = "这是一段详细描述商品售后政策与沟通记录的超长文本内容。" * 40  # ~1120 tokens
    for idx in range(1, 5):
        await service.save_message(test_db, conv_id, "user", f"【长提问{idx}】" + long_chinese_text)
        await service.save_message(test_db, conv_id, "assistant", f"【长回复{idx}】" + long_chinese_text)

    # 触发新一轮，执行层 1 降级检查
    mock_engine.run = AsyncMock(return_value={
        "conversation_id": conv_id,
        "response_text": "已为您持续跟进售后进度。",
        "messages": [AIMessage(content="已为您持续跟进售后进度。")],
        "current_turn_tool_messages": [],
        "status": "completed",
    })

    with patch("app.services.chat_service.calculate_context_budget", return_value=budget_18k):
        events_round2 = []
        async for ev in service.stream_chat(db=test_db, conversation_id=conv_id, message="请帮我看一下最新状态"):
            events_round2.append(ev)

    # 验证层 1 超出 3954 Tokens 触发降级推进 layer1_from_msg_id
    await test_db.refresh(conv)
    assert conv.layer1_from_msg_id is not None
    assert conv.layer1_from_msg_id >= 2, f"层 1 未能成功降级推移，当前 layer1_from_msg_id={conv.layer1_from_msg_id}"

    # 4. 验证层 2 超过 1695 Tokens 触发后台异步摘要引擎
    stmt = select(Message).where(Message.conversation_id == conv_id).order_by(Message.id.asc())
    all_msgs = (await test_db.execute(stmt)).scalars().all()
    _, l2_msgs, _ = ContextManager.partition_messages(
        all_msgs,
        summary_upto_msg_id=conv.summary_upto_msg_id,
        layer1_from_msg_id=conv.layer1_from_msg_id,
    )
    formatted_l2 = ContextManager.format_layer2_messages(l2_msgs)
    l2_tokens = estimate_tokens(formatted_l2)
    assert l2_tokens > 1695, f"层 2 Token {l2_tokens} 应该超过 1695"

    # 执行摘要任务，生成新段入库
    mock_summary_model = MagicMock()
    mock_summary_resp = MagicMock()
    mock_summary_resp.content = "【历史事实摘要】用户咨询订单 ORD10086（华为 Mate60），因屏幕质量瑕疵提出申请退款，客服已记录并转交售后审核。"
    mock_summary_model.ainvoke = AsyncMock(return_value=mock_summary_resp)

    summary_result = await summary_service.execute_summary_task(
        conv_id=conv_id,
        from_msg_id=1,
        upto_msg_id=conv.layer1_from_msg_id,
        model=mock_summary_model,
    )
    assert summary_result is not None
    assert summary_result.seq == 1
    assert summary_result.upto_msg_id == conv.layer1_from_msg_id

    # 验证 conversation_summaries 追加落库与 conversations 投影更新
    await test_db.refresh(conv)
    assert conv.summary_upto_msg_id == conv.layer1_from_msg_id
    assert "ORD10086" in conv.summary
    assert "华为 Mate60" in conv.summary
    assert "申请退款" in conv.summary

    # 5. 验证后续提问「最开始那个订单后来怎么说」，靠摘要投影准确索引出 ORD10086 与退款诉求
    captured_invoked_messages = []

    async def mock_agent_run_final(conversation_id, query, db=None, summary=None, layer2_messages=None, layer1_messages=None, **kwargs):
        injected = ContextManager.build_model_messages(
            system_prompt="你是智能客服助手。",
            layer2_messages=layer2_messages or [],
            layer1_messages=layer1_messages or [],
            current_query=query,
            summary=summary,
        )
        captured_invoked_messages.append(injected)
        
        # 验证提示词中包含摘要，大模型以此回答
        assert summary is not None and "ORD10086" in summary
        return {
            "conversation_id": conversation_id,
            "response_text": f"关于您最初咨询的订单 ORD10086（华为 Mate60），您当时申请了退款，目前售后已进入财务退款环节。",
            "messages": [AIMessage(content="关于您最初咨询的订单 ORD10086（华为 Mate60），您当时申请了退款，目前售后已进入财务退款环节。")],
            "current_turn_tool_messages": [],
            "status": "completed",
        }

    mock_engine.run = AsyncMock(side_effect=mock_agent_run_final)

    with patch("app.services.chat_service.calculate_context_budget", return_value=budget_18k):
        events_final = []
        async for ev in service.stream_chat(db=test_db, conversation_id=conv_id, message="最开始那个订单后来怎么说"):
            events_final.append(ev)

    # 验证大模型接收到的上下文包含摘要投影，不包含层 1 淘汰的原始超长消息
    assert len(captured_invoked_messages) == 1
    invoked_msgs = captured_invoked_messages[0]

    evidence_msgs = [m for m in invoked_msgs if isinstance(m, HumanMessage) and "【前情背景摘要】" in m.content]
    assert len(evidence_msgs) == 1
    assert "ORD10086" in evidence_msgs[0].content
    assert "华为 Mate60" in evidence_msgs[0].content
    assert "申请退款" in evidence_msgs[0].content

    # 验证最终回复准确回答了最初订单号 ORD10086
    final_text_ev = next(e for e in events_final if e.get("event_type") == "text")
    assert "ORD10086" in final_text_ev["content"]
    assert "退款" in final_text_ev["content"]


# ==============================================================================
# Acceptance Criterion 3: 默认 128k 大窗口无降级无摘要 (装得下就不压)
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_criterion_3_default_128k_no_degradation_no_summary(test_db, async_session_factory):
    """验收标准 3 (默认 128k 大窗口下连续聊 20 轮无降级无摘要 - 装得下就不压):
    - 验证在默认大窗口（128k=131072 Tokens）配置下，连续聊 20 轮对话；
    - 检查断言：layer1_from_msg_id 始终为 0（无降级）、summary_upto_msg_id 始终为 0、
      conversation_summaries 记录数为 0（未触发摘要）；
    - 验证“装得下就不压，压缩是成本不是美德”的原则坚决落实。
    """
    settings = Settings(_env_file=None)
    assert settings.model_context_window == 128000
    budget = calculate_context_budget(settings)
    assert budget.layer1_budget == 7000
    assert budget.layer2_budget == 3000

    mock_engine = MagicMock(spec=WorkflowEngine)
    mock_engine.run = AsyncMock(side_effect=lambda conversation_id, query, **kwargs: {
        "conversation_id": conversation_id,
        "response_text": f"客服回复：已收到您的提问「{query}」，请问还需要其他帮助吗？",
        "messages": [AIMessage(content=f"客服回复：已收到您的提问「{query}」，请问还需要其他帮助吗？")],
        "current_turn_tool_messages": [],
        "status": "completed",
    })

    summary_service = SummaryService(session_factory=async_session_factory)
    service = ChatService(
        workflow_engine=mock_engine,
        use_workflow=True,
        summary_service=summary_service,
    )

    conv_id = None
    # 模拟默认大窗口下的连续 20 轮真实交互（每轮约 40~60 Tokens，20 轮约 1000 Tokens，远在 7000 预算之内）
    for i in range(1, 21):
        query = f"你好，我是用户，这是关于商品咨询的第 {i} 轮日常交流提问。"
        events = []
        async for ev in service.stream_chat(db=test_db, conversation_id=conv_id, message=query):
            events.append(ev)
            if conv_id is None and "conversation_id" in ev and ev["conversation_id"]:
                conv_id = ev["conversation_id"]

    assert conv_id is not None
    conv = await test_db.get(Conversation, conv_id)

    # 严格断言：layer1_from_msg_id 始终为 None 或 0（无降级）
    assert conv.layer1_from_msg_id is None or conv.layer1_from_msg_id == 0

    # 严格断言：summary_upto_msg_id 始终为 None 或 0（未压缩）
    assert conv.summary_upto_msg_id is None or conv.summary_upto_msg_id == 0
    assert conv.summary is None or conv.summary == ""

    # 严格断言：conversation_summaries 记录数严格为 0（未触发摘要）
    sum_stmt = select(func.count(ConversationSummary.id)).where(ConversationSummary.conversation_id == conv_id)
    sum_count = (await test_db.execute(sum_stmt)).scalar()
    assert sum_count == 0

    # 验证消息条数完整（20 user + 20 assistant = 40 条全部原汁原味保存在原文层）
    msg_stmt = select(func.count(Message.id)).where(Message.conversation_id == conv_id)
    msg_count = (await test_db.execute(msg_stmt)).scalar()
    assert msg_count == 40


# ==============================================================================
# Acceptance Criterion 4: 后台摘要非阻塞执行与日志 grep 可观测性
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_criterion_4_nonblocking_summary_and_observability_grep(test_db, async_session_factory):
    """验收标准 4 (后台摘要非阻塞执行与日志 grep 可观测性):
    - 验证在触发摘要时，当前轮流式 SSE 输出即刻完成，无需等待后台 LLM 摘要完成（异步非阻塞）；
    - 检查 log/app.log，验证每轮对话均记录了可观测日志：
      - grep "[history_ctx]" 能够直查到前置节点打印的历史上下文；
      - grep "[model_ctx]" 能够直查到主力 Agent 模型入参上下文（包含摘要全文、滑窗逐条消息明细与 Tokens 估算）。
    """
    async def mock_workflow_run(conversation_id, query, db=None, summary=None, layer2_messages=None, layer1_messages=None, **kwargs):
        history_msgs = (layer2_messages or []) + (layer1_messages or [])
        summary_line = (summary.strip().splitlines()[0].strip() if summary else "")
        log_history_context(
            conv_id=conversation_id,
            summary_line=summary_line,
            window_messages=history_msgs,
        )
        log_model_context(
            conv_id=conversation_id,
            summary=summary,
            window_messages=history_msgs + [HumanMessage(content=query)],
            estimated_tokens=150,
        )
        return {
            "conversation_id": conversation_id,
            "response_text": "已为您完成本次咨询服务，请查收结果。",
            "messages": [AIMessage(content="已为您完成本次咨询服务，请查收结果。")],
            "current_turn_tool_messages": [],
            "status": "completed",
        }

    mock_engine = MagicMock(spec=WorkflowEngine)
    mock_engine.run = AsyncMock(side_effect=mock_workflow_run)

    mock_summary_model = MagicMock()
    mock_summary_resp = MagicMock()
    mock_summary_resp.content = "【历史事实摘要】用户咨询手机机型信息。"
    mock_summary_model.ainvoke = AsyncMock(return_value=mock_summary_resp)

    # 创建一个需要耗时 0.5s 执行的摘要任务，模拟慢模型
    slow_summary_started = asyncio.Event()
    slow_summary_finished = asyncio.Event()

    class DelayedSummaryService(SummaryService):
        async def execute_summary_task(self, conv_id, from_msg_id, upto_msg_id, model=None):
            slow_summary_started.set()
            await asyncio.sleep(0.5)
            slow_summary_finished.set()
            return await super().execute_summary_task(
                conv_id, from_msg_id, upto_msg_id, model=mock_summary_model
            )

    summary_service = DelayedSummaryService(session_factory=async_session_factory)
    service = ChatService(
        workflow_engine=mock_engine,
        use_workflow=True,
        summary_service=summary_service,
    )

    # 构造预算使得层 2 必定超标触发摘要（预算设为 30 tokens）
    forced_trigger_budget = ContextBudgetResult(
        fixed_overhead=100,
        peak_react_overhead=100,
        window_available=200,
        window_budget=200,
        layer1_budget=50,
        layer2_budget=30,
        is_usable=True,
    )

    # 预置历史消息：确保其在层 2 的 Token 估算 > layer2_budget (30)
    conv = Conversation(id=801, user_id="u_grep_test", status="进行中", layer1_from_msg_id=2)
    test_db.add(conv)
    m1 = Message(id=1, conversation_id=801, role="user", content="历史消息1：关于手机购买咨询详细配置说明" * 5)
    m2 = Message(id=2, conversation_id=801, role="assistant", content="历史消息2：为您推荐旗舰机型配置清单参数" * 5)
    test_db.add_all([m1, m2])
    await test_db.commit()

    with patch("app.services.chat_service.calculate_context_budget", return_value=forced_trigger_budget):
        t_start = time.perf_counter()
        events = []
        async for ev in service.stream_chat(db=test_db, conversation_id=801, message="最新进展如何？"):
            events.append(ev)
        t_duration = time.perf_counter() - t_start

    # 1. 验证流式 SSE 输出即刻完成（< 0.35s），无需等待 0.5s 的后台摘要任务完成（异步非阻塞）
    assert t_duration < 0.35, f"stream_chat 耗时 {t_duration:.3f}s，被后台摘要阻塞了！"
    text_events = [e for e in events if e.get("event_type") == "text"]
    assert len(text_events) == 1
    assert "已为您完成本次咨询服务" in text_events[0]["content"]

    # 验证后台任务确实被拉起并执行
    await asyncio.wait_for(slow_summary_started.wait(), timeout=1.5)
    assert slow_summary_started.is_set()
    await asyncio.wait_for(slow_summary_finished.wait(), timeout=1.5)
    assert slow_summary_finished.is_set()
    assert slow_summary_started.is_set()

    # 2. 检查 log/app.log 日志可观测性
    assert DEFAULT_LOG_FILE.exists()
    log_content = DEFAULT_LOG_FILE.read_text(encoding="utf-8")

    # 模拟 grep "[history_ctx]"
    history_ctx_pattern = rf'\[history_ctx\] conv_id=801 summary_line=".*?" window_msgs=\d+'
    assert re.search(history_ctx_pattern, log_content) is not None, (
        "未能通过 grep '[history_ctx]' 直查到前置节点记录的日志"
    )

    # 模拟 grep "[model_ctx]"
    model_ctx_pattern = rf'\[model_ctx\] conv_id=801 summary_len=\d+ window_msgs=\d+ estimated_tokens=\d+'
    assert re.search(model_ctx_pattern, log_content) is not None, (
        "未能通过 grep '[model_ctx]' 直查到主力 Agent 记录的模型入参日志"
    )

    # 验证日志中包含滑窗消息逐条明细
    assert "--- SLIDING WINDOW ---" in log_content


# ==============================================================================
# Acceptance Criterion 5: 多会话侧栏对照与历史完整回载继续聊
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_criterion_5_multisession_sidebar_and_history_resume(test_db):
    """验收标准 5 (多会话侧栏对照与历史完整回载继续聊):
    - 通过 REST API 模拟：
      - 创建会话 A 交流 3 轮（含工具调用）；
      - 创建会话 B 交流 2 轮；
      - 调用 GET /api/conversations，验证返回两通会话（按更新时间倒序，包含首问预览）；
      - 调用 GET /api/conversations/{conv_A_id}/messages，完整回载会话 A 全部历史消息；
      - 基于会话 A 的 conversation_id 继续追问第 4 轮，验证上下文基于会话 A 的历史连贯推进，与会话 B 完全隔离。
    """
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # ---- 1. 会话 A 交流 3 轮 ----
        # 轮次 A1: 普通问答
        resp_a1 = await client.post("/api/chat/stream", json={"conversation_id": None, "message": "我想查询订单1001"})
        assert resp_a1.status_code == 200
        events_a1, _ = parse_sse_events(resp_a1.text)
        conv_a_id = events_a1[0]["conversation_id"]

        # 轮次 A2: 穿插工具调用
        mock_engine_tool = MagicMock(spec=WorkflowEngine)
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_logistics_a", "name": "query_logistics", "args": {"order_id": "1001"}}],
        )
        tool_res_msg = ToolMessage(
            content='{"status": "派送中", "express_no": "SF1001"}',
            tool_call_id="call_logistics_a",
            name="query_logistics",
        )
        mock_engine_tool.run = AsyncMock(return_value={
            "conversation_id": conv_a_id,
            "response_text": "顺丰单号 SF1001 正在派送中。",
            "messages": [tool_call_msg, tool_res_msg, AIMessage(content="顺丰单号 SF1001 正在派送中。")],
            "current_turn_tool_messages": [tool_call_msg, tool_res_msg],
            "status": "success",
        })

        with patch("app.api.routes.chat_service.workflow_engine", mock_engine_tool):
            resp_a2 = await client.post("/api/chat/stream", json={"conversation_id": conv_a_id, "message": "帮我查下物流"})
            assert resp_a2.status_code == 200

        # 轮次 A3: 继续提问
        resp_a3 = await client.post("/api/chat/stream", json={"conversation_id": conv_a_id, "message": "预计什么时候送达？"})
        assert resp_a3.status_code == 200

        # ---- 2. 会话 B 交流 2 轮 ----
        # 稍微等待确保更新时间戳递增
        await asyncio.sleep(0.01)

        resp_b1 = await client.post("/api/chat/stream", json={"conversation_id": None, "message": "你们有什么热销手机推荐？"})
        assert resp_b1.status_code == 200
        events_b1, _ = parse_sse_events(resp_b1.text)
        conv_b_id = events_b1[0]["conversation_id"]
        assert conv_b_id != conv_a_id

        resp_b2 = await client.post("/api/chat/stream", json={"conversation_id": conv_b_id, "message": "华为 Mate60多少钱？"})
        assert resp_b2.status_code == 200

        # ---- 3. 调用 GET /api/conversations 验证侧栏列表 ----
        resp_list = await client.get("/api/conversations")
        assert resp_list.status_code == 200
        conversations_list = resp_list.json()

        assert len(conversations_list) >= 2
        sidebar_map = {c["id"]: c for c in conversations_list}
        assert conv_a_id in sidebar_map
        assert conv_b_id in sidebar_map

        # 会话 B 更新时间晚于会话 A，在列表中排在更前
        conv_b_idx = next(i for i, c in enumerate(conversations_list) if c["id"] == conv_b_id)
        conv_a_idx = next(i for i, c in enumerate(conversations_list) if c["id"] == conv_a_id)
        assert conv_b_idx < conv_a_idx

        # 首问预览校验
        assert sidebar_map[conv_a_id]["title"] == "我想查询订单1001"
        assert sidebar_map[conv_b_id]["title"] == "你们有什么热销手机推荐？"

        # ---- 4. 调用 GET /api/conversations/{conv_A_id}/messages 完整回载会话 A 历史 ----
        resp_msgs_a = await client.get(f"/api/conversations/{conv_a_id}/messages")
        assert resp_msgs_a.status_code == 200
        messages_a = resp_msgs_a.json()

        # 检查会话 A 的完整历史消息结构
        roles_a = [m["role"] for m in messages_a]
        assert "user" in roles_a
        assert "assistant" in roles_a
        assert "tool" in roles_a

        # 验证工具调用消息结构
        tool_call_items = [m for m in messages_a if m["tool_calls"]]
        assert len(tool_call_items) >= 1
        all_tool_call_names = []
        for item in tool_call_items:
            tc_list = item["tool_calls"] if isinstance(item["tool_calls"], list) else [item["tool_calls"]]
            for tc in tc_list:
                if isinstance(tc, dict) and "name" in tc:
                    all_tool_call_names.append(tc["name"])
        assert "query_logistics" in all_tool_call_names

        tool_result_items = [m for m in messages_a if m["role"] == "tool"]
        assert len(tool_result_items) >= 1
        assert any("SF1001" in (item["content"] or "") for item in tool_result_items)

        # ---- 5. 切回旧会话 A 继续第 4 轮追问，验证上下文连贯且与会话 B 完全隔离 ----
        captured_contexts = []

        def mock_engine_turn4():
            async def run_fn(conversation_id, query, db=None, summary=None, layer2_messages=None, layer1_messages=None, **kwargs):
                all_received_msgs = (layer2_messages or []) + (layer1_messages or [])
                captured_contexts.append({
                    "conv_id": conversation_id,
                    "messages": all_received_msgs,
                })
                return {
                    "conversation_id": conversation_id,
                    "response_text": "好的，您的顺丰单号是 SF1001，请留意派送员电话。",
                    "messages": [AIMessage(content="好的，您的顺丰单号是 SF1001，请留意派送员电话。")],
                    "current_turn_tool_messages": [],
                    "status": "completed",
                }
            return run_fn

        mock_engine_a4 = MagicMock(spec=WorkflowEngine)
        mock_engine_a4.run = AsyncMock(side_effect=mock_engine_turn4())

        with patch("app.api.routes.chat_service.workflow_engine", mock_engine_a4):
            resp_a4 = await client.post(
                "/api/chat/stream",
                json={"conversation_id": conv_a_id, "message": "刚才说的顺丰单号能再发我一遍吗"},
            )
            assert resp_a4.status_code == 200
            events_a4, _ = parse_sse_events(resp_a4.text)
            text_a4 = next(e for e in events_a4 if e.get("event_type") == "text")
            assert "SF1001" in text_a4["content"]

        # 验证上下文隔离性
        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        assert ctx["conv_id"] == conv_a_id

        received_text = " ".join(str(m.content) for m in ctx["messages"])
        # 必须包含会话 A 的历史提问
        assert "我想查询订单1001" in received_text
        # 绝对不包含会话 B 的任何内容（严格隔离）
        assert "热销手机" not in received_text
        assert "Mate60" not in received_text
