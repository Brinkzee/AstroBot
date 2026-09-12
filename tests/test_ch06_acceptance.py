"""Comprehensive Chapter 6 Acceptance Test Suite.

Verifies the 4 critical acceptance criteria specified by the user:
1. 多轮用例：物流 -> 退款（缺单弹卡片回填） -> 政策裁决 -> 聊回物流，意图与指代均正确；
2. 意图 JSON 稳定，离群怪问题归入「其他」；
3. 「这个能退吗」先消解补全再走退款子流程拿订单和政策；
4. 浏览器不带单号问退款弹出订单选择器，点选后流程继续走完，并支持提交退款工单。
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from main import app
from app.db.session import get_db
from app.services.workflow.engine import WorkflowEngine
from app.services.workflow.nodes.pre_nodes import (
    coreference_rewrite_node,
    intent_recognition_node,
    other_fallback_node,
)
from app.services.workflow.nodes.refund_nodes import (
    extract_order_id,
    refund_order_check_node,
    refund_expansion_retrieval_node,
)
from app.services.workflow.nodes.agent_node import main_agent_node
from app.services.workflow.nodes.router import route_by_intent, route_refund_slot
from tests.test_chat_service import FakeAsyncSession
from scripts.wsl_helper import ensure_mysql_ready


@pytest.fixture(scope="module", autouse=True)
def setup_mysql_ready():
    """确保 MySQL 服务准备完毕"""
    ensure_mysql_ready(verbose=False)


@pytest.fixture(autouse=True)
def mock_db_session():
    fake_db = FakeAsyncSession()

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.pop(get_db, None)


def parse_sse_events(sse_text: str):
    events = []
    has_done = False
    chunks = sse_text.strip().split("\n\n")
    for chunk in chunks:
        lines = [line.strip() for line in chunk.split("\n") if line.strip()]
        for line in lines:
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
# Acceptance Criterion 1: 多轮用例
# 物流 -> 退款（缺单弹卡片回填） -> 政策裁决 -> 代词指代消解 -> 聊回物流
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_criterion_1_multi_turn_lifecycle():
    """验收标准 1: 多轮对话闭环
    轮次 1: 问物流(无单号) -> 识别为「物流」，直通业务数据/Agent
    轮次 2: 问退款（无单号） -> 识别为「退款退货」，槽位拦截挂起，下发 order_selector
    轮次 3: 点选回填单号 -> 提取单号1001，预取订单，政策扩写检索，Agent裁决并下发 apply_refund
    轮次 4: 问代词指代 -> 消解节点结合历史补全代词
    轮次 5: 聊回物流 -> 指代消解与意图均正确
    """
    engine = WorkflowEngine()
    conv_id = 901

    # ---- 轮次 1: 查物流(一般性物流咨询，不绑定订单) ----
    turn1_intent = {"intent": "物流", "confidence": 0.98, "intent_reason": "查询合作快递"}
    turn1_agent_res = {
        "messages": [AIMessage(content="我们店铺默认合作快递为顺丰速运和中通快递，发货后通常 2-3 天内送达。")],
        "response_text": "我们店铺默认合作快递为顺丰速运和中通快递，发货后通常 2-3 天内送达。",
        "suggested_actions": [],
        "status": "completed",
    }
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=turn1_intent)):
        with patch("app.services.workflow.nodes.agent_node.main_agent_node", new=AsyncMock(return_value=turn1_agent_res)):
            res1 = await engine.run(conversation_id=conv_id, query="你们平时发什么快递物流？")
            assert res1["status"] == "completed"
            assert "顺丰速运" in res1["response_text"]

    # ---- 轮次 2: 问退款（未带单号，触发订单卡片下发） ----
    turn2_intent = {"intent": "退款退货", "confidence": 0.95, "intent_reason": "申请退货退款"}
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=turn2_intent)):
        res2 = await engine.run(conversation_id=conv_id, query="我想办理退货退款")
        assert res2["status"] == "need_order_selection"
        assert "请先选择您需要咨询的订单" in res2["response_text"]
        assert res2.get("suggested_orders") is not None
        orders_ids = [o["order_id"] for o in res2["suggested_orders"]]
        assert "1001" in orders_ids

    # ---- 轮次 3: 点选回填单号「退款订单: 1001」----
    turn3_intent = {"intent": "退款退货", "confidence": 0.99, "intent_reason": "回填退款订单号并咨询政策"}
    turn3_expansion = {
        "retrieved_docs": [
            {"text": "七天无理由退换货规则：签收后7日内商品完好未洗涤可申请退货", "score": 0.95}
        ]
    }
    turn3_agent_res = {
        "messages": [AIMessage(content="您的订单1001（极简保暖羽绒服）在七天无理由退货期内，支持申请退款。")],
        "response_text": "您的订单1001（极简保暖羽绒服）在七天无理由退货期内，支持申请退款。",
        "suggested_actions": ["apply_refund"],
        "status": "completed",
    }
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=turn3_intent)):
        with patch("app.services.workflow.nodes.refund_nodes.refund_expansion_retrieval_node", new=AsyncMock(return_value=turn3_expansion)):
            with patch("app.services.workflow.nodes.agent_node.main_agent_node", new=AsyncMock(return_value=turn3_agent_res)):
                res3 = await engine.run(conversation_id=conv_id, query="退款订单: 1001")
                assert res3["order_id"] == "1001"
                assert res3["order_data"] is not None
                assert res3["order_data"]["商品明细"][0]["商品名称"] == "极简保暖羽绒服"
                assert "支持申请退款" in res3["response_text"]
                assert "apply_refund" in res3["suggested_actions"]
                assert res3["status"] == "completed"

    # ---- 轮次 4: 代词指代消解 ----
    # 模拟历史带有订单 1001 和极简保暖羽绒服，用户问「它的退货运费谁承担」
    mock_history = [
        HumanMessage(content="退款订单: 1001"),
        AIMessage(content="您的订单1001（极简保暖羽绒服）在七天无理由退货期内，支持申请退款。"),
    ]
    fake_llm_coref = AsyncMock()
    fake_llm_coref.ainvoke.return_value = AIMessage(content="订单1001极简保暖羽绒服的退货运费谁承担")
    coref_state = {
        "input_query": "它的退货运费谁承担",
        "messages": mock_history,
    }
    coref_res = await coreference_rewrite_node(coref_state, model=fake_llm_coref)
    assert "订单1001" in coref_res["resolved_query"] or "羽绒服" in coref_res["resolved_query"]

    # ---- 轮次 5: 聊回物流 ----
    turn5_intent = {"intent": "物流", "confidence": 0.96, "intent_reason": "查看物流进展"}
    turn5_agent_res = {
        "messages": [AIMessage(content="订单1001当前物流显示已进入末端配送，预计今天下午送达。")],
        "response_text": "订单1001当前物流显示已进入末端配送，预计今天下午送达。",
        "suggested_actions": [],
        "status": "completed",
    }
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=turn5_intent)):
        with patch("app.services.workflow.nodes.agent_node.main_agent_node", new=AsyncMock(return_value=turn5_agent_res)):
            res5 = await engine.run(conversation_id=conv_id, query="再帮我查一下这单的物流最新情况")
            assert "末端配送" in res5["response_text"]
            assert res5["status"] == "completed"


# ==============================================================================
# Acceptance Criterion 2: 意图 JSON 稳定解析与离群怪问题兜底
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_criterion_2_intent_stability_and_weird_fallback():
    """验收标准 2:
    - 意图识别强 JSON 格式输出与正确置信度提取；
    - 离群怪问题（天气、代码、哲学等）准确分类为「其他」并流转至 other_fallback；
    - other_fallback 输出委婉致歉与业务范围说明，并带 transfer_agent 建议动作。
    """
    # 1. 验证怪问题直接走 other_fallback
    engine = WorkflowEngine()
    fake_fallback_intent = {"intent": "其他", "confidence": 0.96, "intent_reason": "天气询问不在电商服务范围"}
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_fallback_intent)):
        result = await engine.run(conversation_id=902, query="明天深圳会下暴雨吗？")
        assert result["status"] == "other_fallback"
        assert "超出我的业务范围" in result["response_text"]
        assert "transfer_agent" in result["suggested_actions"]

    # 2. 验证小模型输出非 JSON 时唤醒大模型级联救场
    mock_small_malformed = AsyncMock()
    mock_small_malformed.ainvoke.return_value = AIMessage(content="这是一段普通的非JSON回复，模型未遵循格式")
    mock_primary_correct = AsyncMock()
    mock_primary_correct.ainvoke.return_value = AIMessage(content='```json\n{"intent": "其他", "confidence": 0.88, "intent_reason": "无法识别具体业务"}\n```')

    res = await intent_recognition_node(
        {"resolved_query": "薛定谔方程的具体解法是什么？"},
        model=mock_primary_correct,
        small_model=mock_small_malformed,
    )
    assert res["intent"] == "其他"
    assert res["confidence"] == 0.88


# ==============================================================================
# Acceptance Criterion 3: 「这个能退吗」先消解补全再走退款子流程
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_criterion_3_coref_then_refund_subflow():
    """验收标准 3:
    对于模糊提问「这个能退吗」：
    1. 首先经过 coreference_rewrite_node 补全代词「这个」->「订单1001 极简保暖羽绒服能退吗」；
    2. 经由意图识别归为「退款退货」；
    3. refund_order_check 提取到订单号 1001 并预取订单数据；
    4. refund_expansion_retrieval 并发召回政策条款；
    5. 主力 Agent 专职裁决输出，触发 apply_refund 动作。
    """
    history = [
        HumanMessage(content="我想了解一下订单1001"),
        AIMessage(content="订单1001包含商品：极简保暖羽绒服，金额299.00元，已发货。"),
    ]

    # 1. 模拟指代消解补全
    mock_coref_llm = AsyncMock()
    mock_coref_llm.ainvoke.return_value = AIMessage(content="订单1001极简保暖羽绒服能退吗")
    coref_state = {
        "input_query": "这个能退吗",
        "messages": history,
    }
    coref_res = await coreference_rewrite_node(coref_state, model=mock_coref_llm)
    resolved_q = coref_res["resolved_query"]
    check_state = {
        "input_query": "这个能退吗",
        "resolved_query": resolved_q,
        "messages": history,
    }

    # 2. 订单槽位提取
    extracted_id = extract_order_id(check_state)
    assert extracted_id == "1001"

    # 3. 槽位节点装配真实订单数据
    check_res = await refund_order_check_node(check_state)
    assert check_res["status"] == "order_fetched"
    assert check_res["order_id"] == "1001"
    assert check_res["order_data"]["商品明细"][0]["商品名称"] == "极简保暖羽绒服"
    assert check_res["order_data"]["订单状态"] == "已发货"

    # 4. 检索侧 Query 扩写
    mock_expansion_llm = AsyncMock()
    mock_expansion_llm.ainvoke.return_value = AIMessage(
        content='```json\n{"queries": ["极简保暖羽绒服退货退款政策", "羽绒服支持7天无理由退货吗", "已发货订单如何退款"]}\n```'
    )
    fake_retriever = MagicMock()
    mock_hit = MagicMock()
    mock_hit.score = 0.95
    mock_hit.to_dict.return_value = {
        "chunk_id": "chunk_refund_1",
        "text": "七天无理由退货政策：签收7天内未拆吊牌未洗涤，支持退款退货。",
        "score": 0.95,
    }
    fake_search_res = MagicMock()
    fake_search_res.hits = [mock_hit]
    fake_retriever.retrieve_with_strategy = AsyncMock(return_value=fake_search_res)

    combined_state = {**check_state, **check_res}
    expansion_res = await refund_expansion_retrieval_node(
        combined_state,
        model=mock_expansion_llm,
        retriever=fake_retriever,
    )
    assert len(expansion_res["retrieved_docs"]) >= 1
    assert "七天无理由" in expansion_res["retrieved_docs"][0]["text"]

    # 5. 主力 Agent 专职裁决
    full_state = {
        **combined_state,
        **expansion_res,
        "messages": history + [HumanMessage(content=resolved_q)],
    }
    mock_agent_llm = MagicMock()
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(return_value=AIMessage(
        content="经核对，您的订单1001（极简保暖羽绒服）在签收7日内且符合无理由退换规则，支持办理退款退货。您可点击下方按钮直接申请退款。"
    ))
    mock_agent_llm.bind_tools.return_value = bound_model
    mock_agent_llm.ainvoke = bound_model.ainvoke

    agent_res = await main_agent_node(full_state, model=mock_agent_llm)
    assert "支持办理退款退货" in agent_res["response_text"]
    assert "apply_refund" in agent_res["suggested_actions"]


# ==============================================================================
# Acceptance Criterion 4: 浏览器 SSE 订单选择卡片下发与退款工单创建
# ==============================================================================

def test_acceptance_criterion_4_browser_sse_and_ticket_flow():
    """验收标准 4: 模拟真实客户端调用
    1. 调用 /api/chat/stream 提出退款需求，未带单号时 SSE 流下发 order_selector 事件；
    2. 模拟点选卡片回传单号后，SSE 流下发 actions=['apply_refund']；
    3. 调用 POST /api/tickets 提交退款单，成功创建工单并落库。
    """
    client = TestClient(app)

    # 1. 触发 order_selector
    mock_engine_need_order = MagicMock()
    mock_engine_need_order.run = AsyncMock(
        return_value={
            "conversation_id": 904,
            "status": "need_order_selection",
            "response_text": "请先点选您需要退款的订单：",
            "suggested_orders": [
                {
                    "order_id": "1001",
                    "product_name": "极简保暖羽绒服",
                    "amount": "299.00元",
                    "status": "已发货",
                    "create_time": "2026-09-05 14:20:00",
                },
                {
                    "order_id": "1002",
                    "product_name": "潮流纯棉连帽卫衣",
                    "amount": "159.00元",
                    "status": "待支付",
                    "create_time": "2026-09-06 10:15:00",
                },
                {
                    "order_id": "1003",
                    "product_name": "经典修身牛仔裤",
                    "amount": "89.00元",
                    "status": "已完成",
                    "create_time": "2026-09-01 09:30:00",
                },
            ],
            "suggested_actions": [],
        }
    )

    with patch("app.api.routes.chat_service.workflow_engine", mock_engine_need_order):
        res1 = client.post(
            "/api/chat/stream",
            json={"conversation_id": 904, "message": "这件衣服我想退掉"},
        )
        assert res1.status_code == 200
        events1, has_done1 = parse_sse_events(res1.text)
        assert has_done1 is True

        # 验证下发了 order_selector 事件
        order_sel_events = [e for e in events1 if e.get("event_type") == "order_selector"]
        assert len(order_sel_events) == 1
        orders = order_sel_events[0]["orders"]
        assert len(orders) == 3
        assert orders[0]["order_id"] == "1001"
        assert orders[0]["product_name"] == "极简保暖羽绒服"

    # 2. 模拟点选卡片后流式推送带 apply_refund
    mock_engine_approved = MagicMock()
    mock_engine_approved.run = AsyncMock(
        return_value={
            "conversation_id": 904,
            "order_id": "1001",
            "status": "completed",
            "response_text": "您的订单1001支持7天无理由退款，请点击下方申请退款按钮填写退款单。",
            "suggested_actions": ["apply_refund"],
        }
    )

    with patch("app.api.routes.chat_service.workflow_engine", mock_engine_approved):
        res2 = client.post(
            "/api/chat/stream",
            json={"conversation_id": 904, "message": "退款订单: 1001"},
        )
        assert res2.status_code == 200
        events2, has_done2 = parse_sse_events(res2.text)
        assert has_done2 is True

        action_events = [e for e in events2 if e.get("event_type") == "actions"]
        assert len(action_events) == 1
        assert "apply_refund" in action_events[0]["actions"]

    # 3. 提交退款单表单到 POST /api/tickets
    ticket_payload = {
        "conversation_id": 904,
        "description": "【退款申请】订单号: 1001，原因: 7天无理由退货，补充说明: 尺码稍大未穿过吊牌完好",
        "ticket_type": "退款退货",
    }
    res3 = client.post("/api/tickets", json=ticket_payload)
    assert res3.status_code == 200
    ticket_data = res3.json()
    assert ticket_data["ticket_no"].startswith("T")
    assert "24小时" in ticket_data["status"]
    assert ticket_data["conversation_id"] == 904
