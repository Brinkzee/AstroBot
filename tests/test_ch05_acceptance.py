"""Chapter 05 端到端综合业务验收测试 (五大核心验收标准系统化验证)

验收标准 1（强制知识检索）: 问政策类问题（如“退货退款政策是什么”），意图识别为知识类（退款退货或商品咨询），工作流强制流经知识检索节点（retrieved_docs 存在于 state 中）
验收标准 2（Agent 自调工具）: 问“订单 1001 的物流到哪了”，意图识别为物流直通主力 Agent，Agent 自主调用工具并完成回答（回答中包含物流关键信息如“顺丰”/“派送”/“1001”）
验收标准 3（投诉建议与动作解耦）: 说“我要投诉”，返回安抚话术，附带独立双按钮 suggested_actions = ["transfer_agent", "create_ticket"]；独立调用 POST /api/tickets 端点能够成功创建工单并返回 ticket_no 且以 T 开头并落库
验收标准 4（闲聊零消耗）: 闲聊（如“你好呀”）命中固定话术，返回“智能客服助手”，且模型消耗 Token 严格为 0
验收标准 5（多步 ReAct 推演）: 复合问题（如“我买的羽绒服物流到哪了，帮我查查订单和轨迹”），主力 Agent 自动执行多步工具推演（先查订单再查物流或多步分析），步数 steps_taken >= 2
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import AsyncClient, ASGITransport
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from sqlalchemy import select

from app.db.session import AsyncSessionLocal, engine
from app.models.ticket import Ticket
from app.services.workflow.engine import WorkflowEngine
from main import app
from scripts.wsl_helper import ensure_mysql_ready


@pytest.fixture(scope="module", autouse=True)
def setup_mysql_ready():
    """保证 WSL2 MySQL 容器已拉起且保活进程处于激活状态"""
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


class DeterministicAcceptanceLLM:
    """提供高拟真、确定性、零网络依赖的端到端 LLM 模拟器。

    精确支撑五大验收场景中的意图识别与主力 ReAct Agent 多步决策循环。
    """

    def bind_tools(self, tools: List[Any]) -> "DeterministicAcceptanceLLM":
        return self

    async def ainvoke(self, messages: List[Any]) -> AIMessage:
        sys_msg = ""
        user_msg = ""
        tool_msgs: List[ToolMessage] = []

        for m in messages:
            if isinstance(m, SystemMessage):
                sys_msg = str(m.content)
            elif isinstance(m, HumanMessage):
                user_msg = str(m.content)
            elif isinstance(m, ToolMessage):
                tool_msgs.append(m)

        # 1. 意图识别节点 (INTENT_SYSTEM_PROMPT)
        if "电商客服意图识别专家" in sys_msg or "七类之一" in sys_msg:
            if any(k in user_msg for k in ("退款", "退货", "政策", "运费")):
                return AIMessage(
                    content=json.dumps(
                        {"intent": "退款退货", "reason": "用户咨询退换货与退款政策规范"},
                        ensure_ascii=False,
                    )
                )
            elif "1001" in user_msg or "物流" in user_msg or "羽绒服" in user_msg or "轨迹" in user_msg:
                return AIMessage(
                    content=json.dumps(
                        {"intent": "物流", "reason": "用户查询订单物流履约轨迹"},
                        ensure_ascii=False,
                    )
                )
            elif "投诉" in user_msg:
                return AIMessage(
                    content=json.dumps(
                        {"intent": "投诉", "reason": "用户表达强烈服务不满与投诉意向"},
                        ensure_ascii=False,
                    )
                )
            elif any(k in user_msg for k in ("你好", "早", "嗨", "您好")):
                return AIMessage(
                    content=json.dumps(
                        {"intent": "闲聊", "reason": "用户日常寒暄与打招呼"},
                        ensure_ascii=False,
                    )
                )
            else:
                return AIMessage(
                    content=json.dumps(
                        {"intent": "售后", "reason": "业务数据兜底分流"},
                        ensure_ascii=False,
                    )
                )

        # 2. 主力 ReAct Agent 节点 (AGENT_SYSTEM_BASE)
        # 场景 A: 复合问题（多步 ReAct 推演：先查订单再查物流）
        if "羽绒服" in user_msg or ("订单" in user_msg and "轨迹" in user_msg):
            if len(tool_msgs) == 0:
                # 步 1: 发现需要先确认羽绒服对应的订单详情与编号
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "query_order",
                        "args": {"order_id": "1001"},
                        "id": "call_order_1001",
                    }],
                    response_metadata={"token_usage": {"prompt_tokens": 120, "completion_tokens": 15, "total_tokens": 135}},
                )
            elif len(tool_msgs) == 1:
                # 步 2: 获得订单 1001 信息后，发起物流轨迹追踪
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "query_logistics",
                        "args": {"order_id": "1001"},
                        "id": "call_logistics_1001",
                    }],
                    response_metadata={"token_usage": {"prompt_tokens": 180, "completion_tokens": 15, "total_tokens": 195}},
                )
            else:
                # 步 3: 汇总推演成果并自然语言作答
                return AIMessage(
                    content="您购买的羽绒服（订单号 1001）由顺丰速运承运，目前包裹处于派送中状态，快递员正在派送，请保持手机畅通！",
                    response_metadata={"token_usage": {"prompt_tokens": 240, "completion_tokens": 45, "total_tokens": 285}},
                )

        # 场景 B: 物流直通自调工具查询（标准 2）
        if "1001" in user_msg or "物流到哪了" in user_msg:
            if len(tool_msgs) == 0:
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "query_logistics",
                        "args": {"order_id": "1001"},
                        "id": "call_logistics_1001",
                    }],
                    response_metadata={"token_usage": {"prompt_tokens": 110, "completion_tokens": 15, "total_tokens": 125}},
                )
            else:
                return AIMessage(
                    content="订单 1001 的物流状态为：顺丰速运正在派送中，快递员已出发，请保持电话畅通。",
                    response_metadata={"token_usage": {"prompt_tokens": 160, "completion_tokens": 35, "total_tokens": 195}},
                )

        # 场景 C: 政策类知识库检索后主 Agent 结合上下文作答（标准 1 放行场景）
        if "退货" in user_msg or "退款" in user_msg or "政策" in user_msg:
            return AIMessage(
                content="根据官方售后退换货规范，商品自签收之日起支持7天无理由退货，请保持商品完好及包装配件齐全。",
                response_metadata={"token_usage": {"prompt_tokens": 130, "completion_tokens": 30, "total_tokens": 160}},
            )

        # 场景 D: 超限总结或常规文本作答
        return AIMessage(
            content="已根据相关信息为您办理完成。",
            response_metadata={"token_usage": {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65}},
        )


@pytest.fixture(autouse=True)
def mock_acceptance_environment():
    """提供全流程确定性 LLM 与知识检索 Mock 环境，确保端到端测试 100% 确定性、无外部网络依赖与秒级执行。"""
    mock_llm = DeterministicAcceptanceLLM()

    mock_retriever = MagicMock()
    mock_hit = MagicMock()
    mock_hit.score = 0.90
    mock_hit.text = "【售后退换货规范】商品自签收次日起7天内支持无理由退货退款，退货需保持原样且不影响二次销售。"
    mock_hit.to_dict.return_value = {
        "chunk_id": 101,
        "text": "【售后退换货规范】商品自签收次日起7天内支持无理由退货退款，退货需保持原样且不影响二次销售。",
        "score": 0.90,
        "section_path": "售后服务 > 7天退货退款政策",
    }
    mock_res = MagicMock()
    mock_res.hits = [mock_hit]
    mock_retriever.retrieve_with_strategy = AsyncMock(return_value=mock_res)

    with patch("app.services.workflow.nodes.pre_nodes.get_chat_model", return_value=mock_llm), \
         patch("app.services.workflow.nodes.agent_node.get_chat_model", return_value=mock_llm), \
         patch("app.services.workflow.nodes.knowledge_node.get_retriever", return_value=mock_retriever):
        yield


# ==============================================================================
# 验收标准 1: 强制知识检索节点验证
# ==============================================================================

@pytest.mark.asyncio
async def test_criterion_1_policy_mandatory_retrieval():
    """验收标准 1：问政策类问题，工作流中能看到强制知识检索节点被走到"""
    engine = WorkflowEngine()
    state = await engine.run(conversation_id=501, query="退货退款政策是什么")

    # 1. 意图分类为知识检索类
    assert state["intent"] in ("退款退货", "商品咨询")

    # 2. retrieved_docs 存在于 state 中且非空
    assert "retrieved_docs" in state
    assert len(state["retrieved_docs"]) > 0
    top_doc = state["retrieved_docs"][0]
    assert "退货退款" in top_doc["text"] or "7天" in top_doc["text"]
    assert top_doc["score"] >= 0.35

    # 3. 闸门放行至 Agent 完成回答
    assert "7天" in state["response_text"] or "退货" in state["response_text"]


# ==============================================================================
# 验收标准 2: 物流直通主力 Agent 自主调用工具作答
# ==============================================================================

@pytest.mark.asyncio
async def test_criterion_2_logistics_agent_tool_calling():
    """验收标准 2：问「订单 1001 的物流到哪了」，Agent 自己调工具作答"""
    engine = WorkflowEngine()
    state = await engine.run(conversation_id=502, query="订单 1001 的物流到哪了")

    # 1. 意图直通物流
    assert state["intent"] == "物流"

    # 2. 回答中包含物流关键信息
    assert any(key in state["response_text"] for key in ("顺丰", "派送", "1001"))

    # 3. Agent 真正自主触发了工具调用并回填了 ToolMessage
    assert state["steps_taken"] >= 1
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1
    tool_output = tool_messages[0].content
    assert any(k in tool_output for k in ("顺丰速运", "派送中", "1001"))


# ==============================================================================
# 验收标准 3: 投诉安抚话术、推荐动作解耦与创建工单写库
# ==============================================================================

@pytest.mark.asyncio
async def test_criterion_3_complaint_decoupled_actions():
    """验收标准 3：说「我要投诉」，返回安抚话术，附带独立双按钮；建工单成功写 tickets 表"""
    engine = WorkflowEngine()
    state = await engine.run(conversation_id=503, query="我要投诉你们态度太差了")

    # 1. 意图命中投诉
    assert state["intent"] == "投诉"

    # 2. 状态与安抚话术
    assert state["status"] == "complaint"
    assert "非常抱歉" in state["response_text"]

    # 3. 下发独立动作推荐双按钮（解耦：工作流本身不直接操作工单库）
    assert state["suggested_actions"] == ["transfer_agent", "create_ticket"]

    # 4. 验证用户自选触发的独立建工单 API 端点 (POST /api/tickets)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/api/tickets",
            json={
                "conversation_id": 503,
                "description": "我要投诉你们态度太差了",
                "ticket_type": "投诉",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["ticket_no"].startswith("T")
        assert data["conversation_id"] == 503
        assert data["ticket_type"] == "投诉"
        assert "24小时" in data["status"]

    # 5. 校验数据库 tickets 表中已成功持久化记录
    async with AsyncSessionLocal() as session:
        stmt = select(Ticket).where(Ticket.ticket_no == data["ticket_no"])
        result = await session.execute(stmt)
        db_ticket = result.scalar_one_or_none()
        assert db_ticket is not None
        assert db_ticket.ticket_no == data["ticket_no"]
        assert db_ticket.conversation_id == 503
        assert db_ticket.ticket_type == "投诉"
        assert "投诉" in db_ticket.description
        assert db_ticket.status == "待处理"


# ==============================================================================
# 验收标准 4: 闲聊命中固定话术且 Token 消耗为零
# ==============================================================================

@pytest.mark.asyncio
async def test_criterion_4_chitchat_fixed_response():
    """验收标准 4：闲聊拿到固定话术，不花模型 Token"""
    engine = WorkflowEngine()
    state = await engine.run(conversation_id=504, query="你好呀")

    # 1. 意图识别为闲聊
    assert state["intent"] == "闲聊"

    # 2. 命中固定友好话术
    assert "智能客服助手" in state["response_text"]

    # 3. 闲聊无推荐按钮且无模型 Token 消耗
    assert state["suggested_actions"] == []
    assert state["status"] == "chitchat"
    assert state["token_usage"]["total_tokens"] == 0
    assert state["token_usage"]["prompt_tokens"] == 0
    assert state["token_usage"]["completion_tokens"] == 0


# ==============================================================================
# 验收标准 5: 复合问题触发主力 Agent 多步 ReAct 推演 (steps >= 2)
# ==============================================================================

@pytest.mark.asyncio
async def test_criterion_5_complex_multistep_react():
    """验收标准 5：复合问题（先查订单再查物流）ReAct 走了不止一步"""
    engine = WorkflowEngine()
    state = await engine.run(conversation_id=505, query="我买的羽绒服物流到哪了，帮我查查订单和轨迹")

    # 1. 步数严格满足 >= 2
    assert state["steps_taken"] >= 2

    # 2. 验证多步 ReAct 实际执行了多个不同的业务工具 (如 query_order 与 query_logistics)
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 2

    # 3. 最终作答融合了多步工具的推演信息
    assert any(k in state["response_text"] for k in ("顺丰", "派送", "1001", "羽绒服"))
    assert state["status"] == "success"
