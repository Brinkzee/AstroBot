"""Comprehensive regression tests for WF-01..WF-18, VER-02, VER-10 fixes."""

import asyncio
import json
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.config import settings
from app.prompts.qa_extraction import parse_qa_extraction_output, qa_extraction_prompt
from app.prompts.query_understanding import parse_query_understanding_output
from app.prompts.rag_qa import parse_self_check_response
from app.services.context.manager import ContextManager
from app.services.workflow.engine import WorkflowEngine
from app.services.workflow.nodes import agent_node
from app.services.workflow.nodes.agent_node import (
    _pending_agent_states,
    is_refund_approved,
    main_agent_node,
)
from app.services.workflow.nodes.pre_nodes import coreference_resolution_node
from app.services.workflow.nodes.refund_nodes import (
    extract_order_id,
    refund_expansion_retrieval_node,
    refund_order_check_node,
)
from app.services.workflow.state import AgentWorkflowState, create_initial_state
from app.tools.audit import AuditLogger
from app.tools.business_tools import create_ticket, query_faq, query_order
from app.tools.executor import ToolExecutor, _is_query_missed
from app.tools.permission import ToolPermissionGuard
from app.tools.registry import ToolRegistry
from scripts.bare_agent_loop import run_bare_agent_loop
from scripts.mcp_process_manager import MCPServerManager


# ==============================================================================
# WF-01: qa_extraction_prompt unescaped braces KeyError
# ==============================================================================
def test_wf01_qa_extraction_prompt_formats_with_braces():
    assert qa_extraction_prompt.input_variables == ["dialogue_text"]
    messages = qa_extraction_prompt.format_messages(
        dialogue_text="用户：退货运费{怎么算}？\n客服：质量问题由商家承担。"
    )
    assert len(messages) == 2
    assert '"question"' in messages[0].content
    assert "{怎么算}" in messages[1].content


# ==============================================================================
# WF-02: Stale _pending_agent_states bleed & multi-tool interrupt/resume
# ==============================================================================
@pytest.mark.asyncio
async def test_wf02_abandoned_interrupt_does_not_hijack_next_turn():
    conv_id = 99001
    _pending_agent_states.pop(conv_id, None)

    engine = WorkflowEngine()
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_ticket",
                        "args": {"ticket_type": "售后", "description": "拉链坏了"},
                        "id": "call_abandoned_1",
                    }
                ],
            ),
            AIMessage(content="订单1001目前已发货，正在顺丰派送中。"),
        ]
    )

    fake_intents = [
        {"intent": "人工", "confidence": 0.99, "intent_reason": "建单"},
        {"intent": "订单", "confidence": 0.99, "intent_reason": "查单"},
    ]

    with patch("app.services.workflow.nodes.pre_nodes.coreference_resolution_node", side_effect=lambda s: {"resolved_query": s["input_query"]}), \
         patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", side_effect=fake_intents), \
         patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model), \
         patch("app.services.workflow.nodes.agent_node.default_tool_registry", ToolRegistry(mcp_client=None)):
        state1 = await engine.run(conversation_id=conv_id, query="帮我建工单，拉链坏了")
        assert "__interrupt__" in state1

        # User abandons preview and asks a fresh normal question on same conv_id
        state2 = await engine.run(conversation_id=conv_id, query="算了，查订单1001状态")
        assert "__interrupt__" not in state2
        assert "订单1001" in state2["response_text"]
        assert conv_id not in _pending_agent_states


@pytest.mark.asyncio
async def test_wf02_multitool_with_create_ticket_no_duplicate_exec_and_paired_tool_msgs():
    conv_id = 99002
    _pending_agent_states.pop(conv_id, None)

    order_call_count = 0

    @tool
    def count_order(order_id: str) -> str:
        """查询订单"""
        nonlocal order_call_count
        order_call_count += 1
        return json.dumps({"order_id": order_id, "订单状态": "已发货"}, ensure_ascii=False)

    @tool
    def fake_create_ticket(conversation_id: int = 0, description: str = "", ticket_type: str = "售后") -> str:
        """创建工单"""
        return json.dumps({"ticket_no": "T20260101000123", "conversation_id": conversation_id, "ticket_type": ticket_type, "status": "已创建"}, ensure_ascii=False)

    fake_create_ticket.name = "create_ticket"

    captured_second_call_msgs = []
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model

    async def fake_ainvoke(msgs):
        if len(captured_second_call_msgs) == 0:
            captured_second_call_msgs.append("first")
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "count_order", "args": {"order_id": "1001"}, "id": "call_before"},
                    {"name": "create_ticket", "args": {"ticket_type": "售后", "description": "袖口破损"}, "id": "call_ticket"},
                    {"name": "count_order", "args": {"order_id": "1002"}, "id": "call_after"},
                ],
            )
        captured_second_call_msgs.append(list(msgs))
        return AIMessage(content="已核实订单并为您创建工单 T20260101000123。")

    mock_model.ainvoke = AsyncMock(side_effect=fake_ainvoke)

    registry = ToolRegistry(tools=[count_order, fake_create_ticket], mcp_client=None)
    mock_audit = MagicMock(spec=AuditLogger)
    mock_audit.log_call = AsyncMock(return_value=None)
    executor = ToolExecutor(registry=registry, audit_logger=mock_audit)

    engine = WorkflowEngine()
    fake_intent = {"intent": "人工", "confidence": 0.99, "intent_reason": "建单"}

    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)), \
         patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model), \
         patch("app.services.workflow.nodes.agent_node.default_tool_registry", registry), \
         patch("app.services.workflow.nodes.agent_node.default_tool_executor", executor):
        state1 = await engine.run(conversation_id=conv_id, query="帮我查订单并建工单，袖口破损")
        assert "__interrupt__" in state1
        assert order_call_count == 1  # call_before executed once prior to interrupt

        resumed = await engine.resume(conversation_id=conv_id, action="confirm")
        assert "T20260101000123" in resumed["response_text"]
        # call_before must NOT re-execute on resume; call_after executes once after confirm => total 2
        assert order_call_count == 2

        # Every tool_call_id from resp must have a paired ToolMessage before second LLM call
        msgs_sent = captured_second_call_msgs[1]
        tool_msg_ids = {m.tool_call_id for m in msgs_sent if isinstance(m, ToolMessage)}
        assert tool_msg_ids == {"call_before", "call_ticket", "call_after"}


# ==============================================================================
# WF-03: Parallel tool calls must append resp to current_turn_tool_messages ONCE
# ==============================================================================
@pytest.mark.asyncio
async def test_wf03_parallel_tool_calls_single_resp_in_current_turn_tool_messages():
    resp1 = AIMessage(
        content="",
        tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"},
            {"name": "query_order", "args": {"order_id": "1002"}, "id": "c2"},
        ],
    )
    resp2 = AIMessage(content="订单1001已发货，订单1002待支付。")

    mock_model = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=[resp1, resp2])
    mock_model.bind_tools.return_value = bound

    state = create_initial_state(99003, "查订单1001和1002")
    with patch("app.tools.executor.default_audit_logger.log_call", new=AsyncMock(return_value=None)):
        updated = await main_agent_node(state, model=mock_model, tools=[query_order])

    ctm = updated["current_turn_tool_messages"]
    ai_msgs = [m for m in ctm if isinstance(m, AIMessage)]
    tool_msgs = [m for m in ctm if isinstance(m, ToolMessage)]
    assert len(ai_msgs) == 1
    assert len(tool_msgs) == 2
    assert len(ctm) == 3


# ==============================================================================
# WF-04: Empty description cleanup, unexpected resume action, resp2 token usage
# ==============================================================================
@pytest.mark.asyncio
async def test_wf04_empty_ticket_description_leaves_no_dangling_tool_calls():
    mock_model = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[{"name": "create_ticket", "args": {"description": ""}, "id": "call_empty_desc"}],
        )
    )
    mock_model.bind_tools.return_value = bound

    state = create_initial_state(99004, "帮我建工单")
    state["intent"] = "人工"
    updated = await main_agent_node(state, model=mock_model, tools=[create_ticket])

    assert "具体遇到了什么问题" in updated["response_text"]
    for m in updated["messages"]:
        assert not getattr(m, "tool_calls", None), "Dangling tool_calls must be removed on empty description"


@pytest.mark.asyncio
async def test_wf04_unexpected_resume_action_handled_cleanly_and_resp2_tokens_counted():
    conv_id = 99005
    _pending_agent_states.pop(conv_id, None)

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_ticket",
                    "args": {"ticket_type": "售后", "description": "商品有划痕"},
                    "id": "call_unexpected_act",
                }
            ],
            response_metadata={"token_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
        )
    )

    engine = WorkflowEngine()
    fake_intent = {"intent": "人工", "confidence": 0.99, "intent_reason": "建单"}
    mock_audit = MagicMock(spec=AuditLogger)
    mock_audit.log_call = AsyncMock(return_value=None)
    executor = ToolExecutor(audit_logger=mock_audit)

    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)), \
         patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model), \
         patch("app.services.workflow.nodes.agent_node.default_tool_executor", executor):
        s1 = await engine.run(conversation_id=conv_id, query="帮我建工单，商品有划痕")
        assert "__interrupt__" in s1

        # Unexpected resume action must NOT loop back to LLM with dangling tool_call
        s2 = await engine.resume(conversation_id=conv_id, action="unexpected_action")
        assert "取消" in s2["response_text"]
        assert mock_model.ainvoke.await_count == 1
        for m in s2["messages"]:
            assert not getattr(m, "tool_calls", None)

    # Now verify resp2 token usage is accumulated on confirm
    conv_id2 = 99006
    _pending_agent_states.pop(conv_id2, None)

    @tool
    def ok_ticket(conversation_id: int = 0, description: str = "", ticket_type: str = "售后") -> str:
        """创建工单"""
        return json.dumps({"ticket_no": "T20260101000456", "conversation_id": conversation_id, "ticket_type": ticket_type, "status": "ok"}, ensure_ascii=False)

    ok_ticket.name = "create_ticket"
    reg = ToolRegistry(tools=[ok_ticket], mcp_client=None)
    exec2 = ToolExecutor(registry=reg, audit_logger=mock_audit)

    mock_model2 = MagicMock()
    mock_model2.bind_tools.return_value = mock_model2
    mock_model2.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[{"name": "create_ticket", "args": {"description": "破损"}, "id": "call_tok"}],
                response_metadata={"token_usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}},
            ),
            AIMessage(
                content="工单 T20260101000456 创建完成。",
                response_metadata={"token_usage": {"prompt_tokens": 80, "completion_tokens": 15, "total_tokens": 95}},
            ),
        ]
    )

    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)), \
         patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model2), \
         patch("app.services.workflow.nodes.agent_node.default_tool_registry", reg), \
         patch("app.services.workflow.nodes.agent_node.default_tool_executor", exec2):
        await engine.run(conversation_id=conv_id2, query="帮我建工单，破损")
        s_conf = await engine.resume(conversation_id=conv_id2, action="confirm")
        assert s_conf["token_usage"]["prompt_tokens"] >= 80
        assert s_conf["token_usage"]["completion_tokens"] >= 15
        assert s_conf["token_usage"]["total_tokens"] >= 95


# ==============================================================================
# WF-05: create_ticket DB failure must fail cleanly & redact secrets/DSNs
# ==============================================================================
@pytest.mark.asyncio
async def test_wf05_create_ticket_db_failure_not_reported_as_success_and_redacts_dsn():
    mock_session = AsyncMock()
    mock_session.get = AsyncMock(
        side_effect=RuntimeError("mysql+aiomysql://root:SuperSecretPwd123@10.0.0.5:3306/astro_bot refused [SQL: INSERT INTO tickets VALUES (1)]")
    )
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None

    mock_audit = MagicMock(spec=AuditLogger)
    mock_audit.log_call = AsyncMock(return_value=None)
    registry = ToolRegistry(tools=[create_ticket], mcp_client=None)
    executor = ToolExecutor(registry=registry, audit_logger=mock_audit)

    with patch("app.tools.business_tools.AsyncSessionLocal", MagicMock(return_value=mock_ctx)):
        res = await executor.execute(
            {"name": "create_ticket", "args": {"conversation_id": 1, "description": "破损"}, "id": "call_db_fail"},
            context={"user_query": "帮我建工单", "confirmed": True},
        )
        assert res["success"] is False
        assert res["status"] == "失败"
        assert "SuperSecretPwd123" not in res["output"]
        assert "SuperSecretPwd123" not in (res["error"] or "")
        mock_audit.log_call.assert_awaited_once()
        assert mock_audit.log_call.call_args[1]["status"] == "失败"

    # Also test ToolExecutor redacts raw DSN/password/SQL from arbitrary tool exceptions
    @tool
    def leaky_tool(x: str) -> str:
        """测试脱敏"""
        raise RuntimeError(
            "connect failed mysql+aiomysql://admin:TopSecret99@127.0.0.1:3306/db [SQL: SELECT * FROM users] api_key=sk-1234567890abcdef"
        )

    reg2 = ToolRegistry(tools=[leaky_tool], mcp_client=None)
    exec2 = ToolExecutor(registry=reg2, audit_logger=mock_audit)
    res2 = await exec2.execute({"name": "leaky_tool", "args": {"x": "1"}, "id": "call_leak"})
    assert res2["success"] is False
    assert "TopSecret99" not in res2["output"]
    assert "sk-1234567890abcdef" not in res2["output"]
    assert "SELECT * FROM users" not in res2["output"]

    # Also verify agent_node never prepends "已为您成功创建工单" if tool_result failed even if JSON has ticket_no
    conv_id = 99007
    _pending_agent_states.pop(conv_id, None)

    @tool
    def failing_json_ticket(conversation_id: int = 0, description: str = "", ticket_type: str = "售后") -> str:
        """返回错误JSON"""
        return json.dumps({"error": "写入失败", "ticket_no": "T999999", "status": "创建失败"}, ensure_ascii=False)

    failing_json_ticket.name = "create_ticket"
    reg3 = ToolRegistry(tools=[failing_json_ticket], mcp_client=None)
    exec3 = ToolExecutor(registry=reg3, audit_logger=mock_audit)

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content="", tool_calls=[{"name": "create_ticket", "args": {"description": "坏了"}, "id": "c_fail_json"}]),
            AIMessage(content="非常抱歉，当前工单系统异常，创建失败，请稍后重试。"),
        ]
    )
    engine = WorkflowEngine()
    fake_intent = {"intent": "人工", "confidence": 0.99, "intent_reason": "建单"}
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)), \
         patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model), \
         patch("app.services.workflow.nodes.agent_node.default_tool_registry", reg3), \
         patch("app.services.workflow.nodes.agent_node.default_tool_executor", exec3):
        await engine.run(conversation_id=conv_id, query="帮我建工单，坏了")
        resumed = await engine.resume(conversation_id=conv_id, action="confirm")
        assert "已为您成功创建工单" not in resumed["response_text"]
        assert "T999999" not in resumed["response_text"]


# ==============================================================================
# WF-06: WorkflowEngine.run propagates db into knowledge_fallback_node
# ==============================================================================
@pytest.mark.asyncio
async def test_wf06_workflow_engine_forwards_db_to_knowledge_fallback():
    engine = WorkflowEngine()
    mock_db = MagicMock()
    fake_intent = {"intent": "商品咨询", "confidence": 0.95, "intent_reason": "商品咨询"}
    mock_retriever = MagicMock()
    mock_retriever.retrieve_with_strategy = AsyncMock(return_value=SimpleNamespace(docs=[]))

    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)), \
         patch("app.services.workflow.nodes.knowledge_node.get_retriever", return_value=mock_retriever), \
         patch("app.models.low_confidence.record_low_confidence", new=AsyncMock(return_value=None)) as mock_rec:
        final_state = await engine.run(conversation_id=99008, query="太空飞船保修吗", db=mock_db)

    assert final_state["status"] == "fallback"
    assert final_state["confidence_passed"] is False
    mock_rec.assert_awaited_once()
    assert mock_rec.call_args.kwargs["db"] is mock_db
    assert mock_rec.call_args.kwargs["source"] == "retrieval_low_conf"


# ==============================================================================
# WF-07: refund_expansion_retrieval_node normalizes Milvus hits & strips entity
# ==============================================================================
@pytest.mark.asyncio
async def test_wf07_refund_expansion_normalizes_milvus_docs_and_strips_entity():
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content='{"queries": ["退货政策"]}'))

    # Object has empty hits=[] AND non-empty docs=[...] (Milvus shape: answer, rerank_score, entity)
    milvus_result = SimpleNamespace(
        hits=[],
        docs=[
            {
                "id": 1,
                "answer": "支持7天无理由退货，商品需保持吊牌完好。",
                "section_path": "售后政策 > 退货",
                "rerank_score": 0.91,
                "distance": 0.65,
                "entity": {"embedding": [0.123456] * 1024},
            }
        ],
    )
    mock_retriever = MagicMock()
    mock_retriever.retrieve_with_strategy = AsyncMock(return_value=milvus_result)

    state = create_initial_state(99009, "订单1001能退吗")
    state["order_id"] = "1001"
    res = await refund_expansion_retrieval_node(state, model=mock_model, retriever=mock_retriever)

    docs = res["retrieved_docs"]
    assert len(docs) == 1
    doc = docs[0]
    assert "entity" not in doc
    assert doc["score"] == 0.91
    assert doc["text"] == "支持7天无理由退货，商品需保持吊牌完好。"
    assert doc["content"] == "支持7天无理由退货，商品需保持吊牌完好。"

    # Verify ContextManager formats clean policy text without dict repr or vector floats
    built = ContextManager.build_model_messages(
        system_prompt="sys",
        layer2_messages=[],
        layer1_messages=[],
        current_query="订单1001能退吗",
        retrieved_docs=docs,
    )
    evidence_msg = built[-1].content
    assert "支持7天无理由退货" in evidence_msg
    assert "0.123456" not in evidence_msg
    assert "'entity'" not in evidence_msg


# ==============================================================================
# WF-08: MCP read-only allowlist & 2-turn confirmed create_ticket
# ==============================================================================
def test_wf08_mcp_readonly_allowlist_blocks_unlisted_and_mutating_tools():
    guard = ToolPermissionGuard()

    # Known read-only MCP tools & approved prefixes allowed
    for name in ("query_logistics", "check_warranty", "query_return_progress", "calculate_repair_quote", "evaluate_recycling_value"):
        allowed, reason = guard.check_permission(tool_name=name, tool_source="mcp", is_write=False)
        assert allowed is True, f"{name} should be allowed: {reason}"

    # Mutating or unlisted MCP tools blocked even when server claims is_write=False
    for bad_name in ("delete_order", "issue_refund", "update_address", "query_and_delete", "arbitrary_unknown_tool"):
        allowed, reason = guard.check_permission(tool_name=bad_name, tool_source="mcp", is_write=False)
        assert allowed is False, f"{bad_name} must be denied"
        assert reason is not None


@pytest.mark.asyncio
async def test_wf08_two_turn_confirmed_ticket_allowed_without_unconfirmed_bypass():
    guard = ToolPermissionGuard()

    # Unconfirmed direct call with no intent and no ticket keywords MUST be denied
    allowed, _ = guard.check_permission(
        tool_name="create_ticket",
        tool_source="builtin",
        is_write=True,
        context={"user_query": "羽绒服多少钱", "intent": None, "confirmed": True},
    )
    assert allowed is False

    # 2-turn flow: Turn 1 requested ticket -> Turn 2 gave details (classified as 售后) -> user confirms preview
    conv_id = 99010
    _pending_agent_states.pop(conv_id, None)

    @tool
    def stub_ticket(conversation_id: int = 0, description: str = "", ticket_type: str = "售后") -> str:
        """创建工单"""
        return json.dumps({"ticket_no": "T20260101000777", "conversation_id": conversation_id, "ticket_type": ticket_type, "status": "ok"}, ensure_ascii=False)

    stub_ticket.name = "create_ticket"
    reg = ToolRegistry(tools=[stub_ticket], mcp_client=None)
    mock_audit = MagicMock(spec=AuditLogger)
    mock_audit.log_call = AsyncMock(return_value=None)
    executor = ToolExecutor(registry=reg, permission_guard=guard, audit_logger=mock_audit)

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content="", tool_calls=[{"name": "create_ticket", "args": {"description": "拉链坏了"}, "id": "c_2turn"}]),
            AIMessage(content="已为您创建工单 T20260101000777。"),
        ]
    )

    engine = WorkflowEngine()
    fake_intent = {"intent": "订单", "confidence": 0.95, "intent_reason": "订单问题"}
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)), \
         patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model), \
         patch("app.services.workflow.nodes.agent_node.default_tool_registry", reg), \
         patch("app.services.workflow.nodes.agent_node.default_tool_executor", executor):
        s1 = await engine.run(
            conversation_id=conv_id,
            query="我的羽绒服拉链坏了",
            layer1_messages=[
                HumanMessage(content="帮我建个工单"),
                AIMessage(content="请问您具体遇到了什么问题？"),
            ],
        )
        assert "__interrupt__" in s1
        s2 = await engine.resume(conversation_id=conv_id, action="confirm")
        assert "T20260101000777" in s2["response_text"]
        assert mock_audit.log_call.call_args[1]["status"] == "成功"


# ==============================================================================
# WF-09 & VER-10: MCP CLI host/port, remote skip, handle closure, wait after kill
# ==============================================================================
def test_wf09_mcp_servers_cli_args_and_process_manager_lifecycle(tmp_path):
    from scripts.mcp_aftersale_server import parse_args as parse_aftersale_args
    from scripts.mcp_logistics_server import parse_args as parse_logistics_args

    l_args = parse_logistics_args(["--host", "127.0.0.1", "--port", "18001"])
    assert l_args.host == "127.0.0.1"
    assert l_args.port == 18001

    a_args = parse_aftersale_args(["--host", "127.0.0.1", "--port", "18002"])
    assert a_args.host == "127.0.0.1"
    assert a_args.port == 18002

    # 1. Custom local port passed via CLI + remote host skipped + parent log handle closed + wait() after kill()
    mock_proc = MagicMock()
    mock_proc.pid = 4321
    mock_proc.poll.return_value = None
    mock_proc.wait.side_effect = [__import__("subprocess").TimeoutExpired("cmd", 1.0), 0]

    port_calls = [False, True]

    def fake_port(host, port, timeout=0.2):
        if host == "remote.mcp.internal":
            return False
        return port_calls.pop(0) if port_calls else True

    manager = MCPServerManager()
    custom_configs = [
        {
            "name": "logistics",
            "label": "物流 MCP 服务",
            "script": "scripts/mcp_logistics_server.py",
            "host": "127.0.0.1",
            "port": 18001,
            "log_file": str(tmp_path / "mcp_logistics.log"),
        },
        {
            "name": "aftersale",
            "label": "售后 MCP 服务",
            "script": "scripts/mcp_aftersale_server.py",
            "host": "remote.mcp.internal",
            "port": 18002,
            "log_file": str(tmp_path / "mcp_aftersale.log"),
        },
    ]

    opened_files = []
    real_open = open

    def tracking_open(*args, **kwargs):
        f = real_open(*args, **kwargs)
        opened_files.append(f)
        return f

    with patch.object(manager, "_get_server_configs", return_value=custom_configs), \
         patch("scripts.mcp_process_manager.is_port_open", side_effect=fake_port), \
         patch("builtins.open", side_effect=tracking_open), \
         patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        status = manager.start_servers(wait_timeout=0.5, verbose=False)

    assert status["logistics"] is True
    assert status["aftersale"] is False  # remote host skipped, not spawned
    assert mock_popen.call_count == 1
    cmd = mock_popen.call_args[0][0]
    assert "--host" in cmd and "127.0.0.1" in cmd
    assert "--port" in cmd and "18001" in cmd
    assert mock_popen.call_args.kwargs.get("stdin") == __import__("subprocess").DEVNULL
    assert len(opened_files) == 1 and opened_files[0].closed is True

    # stop_all must call kill() AND wait() after TimeoutExpired
    manager.stop_all()
    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()
    assert mock_proc.wait.call_count == 2


# ==============================================================================
# WF-10 & VER-02: evidence_confidence & confidence_passed persisted in workflow state
# ==============================================================================
@pytest.mark.asyncio
async def test_wf10_evidence_confidence_and_confidence_passed_in_workflow_state():
    assert "evidence_confidence" in AgentWorkflowState.__annotations__
    init_state = create_initial_state(99011, "退货规则")
    assert init_state["evidence_confidence"] is None

    engine = WorkflowEngine()
    fake_intent = {"intent": "商品咨询", "confidence": 0.97, "intent_reason": "商品咨询"}
    mock_retriever = MagicMock()
    mock_retriever.retrieve_with_strategy = AsyncMock(
        return_value=SimpleNamespace(docs=[{"text": "7天无理由退货", "rerank_score": 0.92}])
    )
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="支持7天无理由退货。"))

    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)), \
         patch("app.services.workflow.nodes.knowledge_node.get_retriever", return_value=mock_retriever), \
         patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_model):
        final_state = await engine.run(conversation_id=99011, query="怎么退货")

    assert final_state["evidence_confidence"] is not None
    assert final_state["evidence_confidence"] >= getattr(settings, "evidence_confidence_threshold", 0.35)
    assert final_state["confidence_passed"] is True


# ==============================================================================
# WF-11: Executor tool_type=None, non-dict JSON args, retry_count on miss, AuditLogger
# ==============================================================================
@pytest.mark.asyncio
async def test_wf11_executor_edge_cases_and_audit_logger_sanitization():
    # 1. tool_type=None must not raise AttributeError
    class ToolWithNoneType:
        name = "none_type_tool"
        tool_type = None
        args_schema = None

        async def ainvoke(self, args):
            return "ok"

    reg = ToolRegistry(tools=[], mcp_client=None)
    reg.register(ToolWithNoneType())
    mock_audit = MagicMock(spec=AuditLogger)
    mock_audit.log_call = AsyncMock(return_value=None)
    executor = ToolExecutor(registry=reg, audit_logger=mock_audit)

    r1 = await executor.execute({"name": "none_type_tool", "args": {}, "id": "c_none_type"})
    assert r1["success"] is True

    # 2. Non-dict JSON string "[1, 2]" -> INVALID_ARGS / 校验拦下
    r2 = await executor.execute({"name": "none_type_tool", "args": "[1, 2, 3]", "id": "c_list_json"})
    assert r2["success"] is False
    assert r2["status"] == "校验拦下"
    assert r2["error_type"] == "INVALID_ARGS"

    # 3. Attempt 1 timeout + Attempt 2 QUERY_MISSED must accurately report retry_count == 1
    attempts = 0

    @tool
    async def flaky_miss(order_id: str) -> str:
        """先超时再落空"""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise asyncio.TimeoutError("net blip")
        return json.dumps({"found": False, "message": "未找到订单"}, ensure_ascii=False)

    reg.register(flaky_miss)
    exec_retry = ToolExecutor(registry=reg, max_retries=1, audit_logger=mock_audit)
    r3 = await exec_retry.execute({"name": "flaky_miss", "args": {"order_id": "999"}, "id": "c_retry_miss"})
    assert r3["success"] is True
    assert r3["error_type"] == "QUERY_MISSED"
    assert r3["retry_count"] == 1
    assert mock_audit.log_call.call_args[1]["retry_count"] == 1

    # 4. AuditLogger JSON-sanitizes datetime & truncates long IDs/names
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None
    audit = AuditLogger(session_factory=MagicMock(return_value=mock_ctx))

    entry = await audit.log_call(
        tool_name="t" * 200,
        tool_source="MCP",
        tool_call_id="id_" + "x" * 100,
        mcp_server="srv_" + "y" * 100,
        arguments={"ts": datetime(2026, 9, 1, 12, 0, 0), "tags": {"a"}},
    )
    assert entry is not None
    assert len(entry.tool_name) == 128
    assert len(entry.tool_call_id) == 64
    assert len(entry.mcp_server) == 64
    assert entry.tool_source == "mcp"
    # Must be pure JSON-serializable primitives
    json.dumps(entry.arguments)


# ==============================================================================
# WF-12: ToolRegistry concurrent discovery & cache retention on partial failure
# ==============================================================================
@pytest.mark.asyncio
async def test_wf12_registry_concurrent_discovery_and_preserves_cache_on_transient_error():
    @tool
    def query_logistics(order_id: str) -> str:
        """查物流"""
        return "ok"

    @tool
    def check_warranty(order_id: str) -> str:
        """查质保"""
        return "ok"

    fail_logistics = False

    async def fake_get_tools(server_name=None):
        await asyncio.sleep(0.15)
        if server_name == "logistics":
            if fail_logistics:
                raise ConnectionResetError("transient failure")
            return [query_logistics]
        if server_name == "aftersale":
            return [check_warranty]
        return [query_logistics, check_warranty]

    mock_client = MagicMock()
    mock_client.connections = {
        "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:8001/mcp"},
        "aftersale": {"transport": "streamable_http", "url": "http://127.0.0.1:8002/mcp"},
    }
    mock_client.get_tools = AsyncMock(side_effect=fake_get_tools)

    reg = ToolRegistry(tools=[], mcp_client=mock_client)

    # First fetch must run concurrently (< 0.28s for two 0.15s servers)
    t0 = time.perf_counter()
    tools1 = await reg.get_all_tools()
    elapsed = time.perf_counter() - t0
    assert {t.name for t in tools1} == {"query_logistics", "check_warranty"}
    assert elapsed < 0.28, f"Expected concurrent MCP fetch, took {elapsed:.3f}s"

    # Transient failure on logistics must NOT wipe already-cached query_logistics
    fail_logistics = True
    tools2 = await reg.get_all_tools()
    assert reg.get_tool("query_logistics") is not None
    assert "check_warranty" in {t.name for t in tools2}


# ==============================================================================
# WF-13: Prompt output parsers handle None, string booleans, and JSON null
# ==============================================================================
def test_wf13_prompt_parsers_edge_cases():
    assert parse_self_check_response('{"useful": "false", "reason": "证据未提及"}')["useful"] is False
    assert parse_self_check_response('{"useful": "true", "reason": "充分"}')["useful"] is True
    assert parse_self_check_response(None)["useful"] is False

    qu = parse_query_understanding_output('{"standard_query": null, "expanded_keywords": []}')
    assert qu["standard_query"] == ""

    qa = parse_qa_extraction_output('[{"question": null, "answer": "有效答案"}, {"question": "有效问题", "answer": null}]')
    assert qa == []


# ==============================================================================
# WF-14: ORDER_PATTERNS rejects mobile phones & SKUs; async non-blocking check
# ==============================================================================
@pytest.mark.asyncio
async def test_wf14_extract_order_id_rejects_phones_and_skus_and_async_offload():
    assert extract_order_id("我手机号13800138000想申请退货") is None
    assert extract_order_id("我的订单PRO-X99想退") is None
    assert extract_order_id("订单PRO-X99能退吗") is None
    assert extract_order_id("手机号13800138000，订单1001想退货") == "1001"
    assert extract_order_id("订单号: TB987654") == "TB987654"

    state = create_initial_state(99014, "订单9999可以退货吗")
    with patch("app.services.workflow.nodes.refund_nodes.asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
        res = await refund_order_check_node(state)
    assert res["status"] == "order_fetched"
    assert res["order_id"] == "9999"
    mock_to_thread.assert_awaited_once()


# ==============================================================================
# WF-15: coreference includes Layer 3 summary; is_refund_approved double negation
# ==============================================================================
@pytest.mark.asyncio
async def test_wf15_coreference_uses_summary_and_refund_double_negation():
    mock_model = MagicMock()
    captured = []

    async def capture(msgs):
        captured.extend(msgs)
        return AIMessage(content="订单1001极简保暖羽绒服能退吗")

    mock_model.ainvoke = AsyncMock(side_effect=capture)

    state = create_initial_state(
        99015,
        "它能退吗",
        summary="用户此前咨询过订单1001极简保暖羽绒服。",
        layer2_messages=[],
        layer1_messages=[],
    )
    res = await coreference_resolution_node(state, model=mock_model)
    assert res["resolved_query"] == "订单1001极简保暖羽绒服能退吗"
    assert any("订单1001极简保暖羽绒服" in str(m.content) for m in captured)

    # Double negation approval must return True
    text = "经核实，该商品不属于不支持退货的特殊品类，符合7天无理由退货条件，您可点击下方“申请退款”按钮提交退款申请单。"
    assert is_refund_approved(text) is True
    # Real rejection still returns False
    assert is_refund_approved("经核实，您的订单已签收超过7天，不支持退货退款。") is False


# ==============================================================================
# WF-16: bare_agent_loop supports BaseTool, JSON string args, and error recovery
# ==============================================================================
@pytest.mark.asyncio
async def test_wf16_bare_agent_loop_basetool_json_args_and_error_recovery():
    @tool
    def exploding_tool(x: str) -> str:
        """会抛异常的工具"""
        raise ValueError("boom")

    mock_llm = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(
        side_effect=[
            SimpleNamespace(
                content="",
                tool_calls=[
                    {"name": "query_order", "args": json.dumps({"order_id": "1001"}), "id": "c_bt"},
                    {"name": "exploding_tool", "args": {"x": "1"}, "id": "c_err"},
                ],
            ),
            AIMessage(content="订单1001已查到，另一项查询暂时出错。"),
        ]
    )
    mock_llm.bind_tools.return_value = bound

    result = await run_bare_agent_loop(
        llm=mock_llm,
        tools={"query_order": query_order, "exploding_tool": exploding_tool},
        query="帮我查订单1001",
    )
    assert "订单1001" in result["answer"]
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2
    assert "极简保暖羽绒服" in tool_msgs[0].content
    assert "boom" in tool_msgs[1].content


# ==============================================================================
# WF-17 & WF-18: business_tools conv status & LIKE escape; _is_query_missed; MAX_STEPS
# ==============================================================================
@pytest.mark.asyncio
async def test_wf17_create_ticket_updates_existing_conv_and_query_faq_escapes_like():
    # 1. Existing conversation status updated to 已转人工
    existing_conv = SimpleNamespace(id=55, user_id="u55", status="进行中")
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.get = AsyncMock(return_value=existing_conv)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None

    with patch("app.tools.business_tools.AsyncSessionLocal", MagicMock(return_value=mock_ctx)):
        await create_ticket.ainvoke({"conversation_id": 55, "description": "破损", "ticket_type": "售后"})
    assert existing_conv.status == "已转人工"

    # 2. query_faq escapes % and _ in SQL LIKE
    captured_stmts = []
    mock_faq_session = AsyncMock()
    mock_res = MagicMock()
    mock_res.scalars.return_value.all.return_value = []

    async def capture_execute(stmt):
        captured_stmts.append(stmt)
        return mock_res

    mock_faq_session.execute = AsyncMock(side_effect=capture_execute)
    mock_faq_ctx = MagicMock()
    mock_faq_ctx.__aenter__.return_value = mock_faq_session
    mock_faq_ctx.__aexit__.return_value = None
    mock_ret = MagicMock()
    mock_ret.retrieve = AsyncMock(return_value=[])

    with patch("app.tools.business_tools.get_retriever", return_value=mock_ret), \
         patch("app.tools.business_tools.AsyncSessionLocal", MagicMock(return_value=mock_faq_ctx)):
        await query_faq.ainvoke({"keyword": "100%_test"})

    compiled = str(captured_stmts[0].compile(compile_kwargs={"literal_binds": True}))
    assert "\\%" in compiled and "\\_" in compiled


@pytest.mark.asyncio
async def test_wf18_is_query_missed_policy_false_positive_and_max_steps_config(monkeypatch):
    # Multi-sentence valid policy answer mentioning 不在保修期 must NOT be flagged as QUERY_MISSED
    policy_answer = (
        "1. 问：商品过保后如何维修？\n"
        "   答：如果您的商品不在保修期内，仍可享受官方付费维修服务，具体费用以售后检测报价为准。"
    )
    assert _is_query_missed(policy_answer, policy_answer) is False

    # Short actual miss still returns True
    assert _is_query_missed("未找到与【火星票】相关的常见问题解答。", "未找到与【火星票】相关的常见问题解答。") is True

    # MAX_STEPS respects settings.max_agent_steps by default and remains monkeypatchable
    tool_resp = AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c_step"}])
    summary_resp = AIMessage(content="总结完成。")

    mock_model = MagicMock()
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=tool_resp)
    mock_model.bind_tools.return_value = bound
    mock_model.ainvoke = AsyncMock(return_value=summary_resp)

    monkeypatch.setattr(settings, "max_agent_steps", 2)
    with patch("app.tools.executor.default_audit_logger.log_call", new=AsyncMock(return_value=None)):
        updated = await main_agent_node(create_initial_state(99018, "查单"), model=mock_model, tools=[query_order])
    assert updated["steps_taken"] == 2

    # Module-level override takes precedence
    monkeypatch.setattr(agent_node, "MAX_STEPS", 4)
    with patch("app.tools.executor.default_audit_logger.log_call", new=AsyncMock(return_value=None)):
        updated2 = await main_agent_node(create_initial_state(99019, "查单"), model=mock_model, tools=[query_order])
    assert updated2["steps_taken"] == 4
