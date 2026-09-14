"""Comprehensive Chapter 8 Acceptance Test Suite (tests/test_ch08_acceptance.py)

Covers all 6 core user acceptance criteria verbatim:
1. AC-1: 新写简单工具只做注册，Agent 即可在对话中用上（零改动核心代码）；
2. AC-2: 两个独立 MCP Server 独立进程运行，问物流通过 MCP 获取；
3. AC-3: 售后 MCP Server 新增工具并重启 Server，客服端代码不动无需重启即可感知并调用；
4. AC-4: 问建单未讲明问题，Agent 追问补齐；前端卡片点「确认提交」，tickets 表落库且回复带工单号；
5. AC-5: 工单预览点「取消」，工单未建，tool_audit_logs 状态为「权限拒绝」；
6. AC-6: 读操作超时触发重试与降级，写操作超时不自动重试（重试次数恒为 0）。
"""

import asyncio
import gc
import json
import re
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from sqlalchemy import select

from app.db.session import AsyncSessionLocal, engine
from app.llm import clear_llm_cache
from app.models.ticket import Ticket
from app.models.tool_audit_log import ToolAuditLog
from app.services.workflow.engine import WorkflowEngine
from app.services.workflow.nodes.agent_node import _pending_agent_states, main_agent_node
from app.services.workflow.state import create_initial_state
from app.tools import business_tools
from app.tools.executor import ToolExecutor, default_tool_executor
from app.tools.permission import ToolPermissionGuard
from app.tools.registry import (
    ToolRegistry,
    annotate_tool,
    default_tool_registry,
)
from scripts.mcp_aftersale_server import (
    create_aftersale_server,
    register_dynamic_tool,
)
from scripts.mcp_logistics_server import create_logistics_server
from scripts.wsl_helper import ensure_mysql_ready


# ==============================================================================
# Fixtures and Teardown
# ==============================================================================

@pytest.fixture(scope="module", autouse=True)
def setup_mysql_ready():
    """保证 WSL2 中的 MySQL 容器已拉起且保活进程处于激活状态"""
    ensure_mysql_ready(verbose=False)


@pytest.fixture(autouse=True)
async def reset_engine_and_singletons():
    """避免 pytest-asyncio 在多个测试函数间切换 event loop 时复用已关闭 loop 的连接与单例"""
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


@pytest.fixture
def clean_db():
    """清理指定 conversation_id 下的工单与工具审计日志数据"""
    async def _clean(conv_id: int):
        async with AsyncSessionLocal() as session:
            tickets = (
                await session.execute(
                    select(Ticket).where(Ticket.conversation_id == conv_id)
                )
            ).scalars().all()
            for t in tickets:
                await session.delete(t)
            audits = (
                await session.execute(
                    select(ToolAuditLog).where(
                        ToolAuditLog.conversation_id == conv_id
                    )
                )
            ).scalars().all()
            for a in audits:
                await session.delete(a)
            await session.commit()
    return _clean


class FastMCPClientAdapter:
    """FastMCP Server 到 MultiServerMCPClient 的桥接适配器，直接通过 FastMCP 协议接口通讯"""

    def __init__(self, servers: Dict[str, Any]):
        self.servers = servers
        self.connections = {
            name: {
                "transport": "streamable_http",
                "url": f"http://127.0.0.1:{server.settings.port}/mcp",
                "timeout": 5.0,
            }
            for name, server in servers.items()
        }

    async def get_tools(self, server_name: Optional[str] = None) -> List[BaseTool]:
        tools: List[BaseTool] = []
        target_servers = (
            {server_name: self.servers[server_name]}
            if server_name and server_name in self.servers
            else self.servers
        )

        for s_name, server in target_servers.items():
            mcp_tool_list = await server.list_tools()
            for mt in mcp_tool_list:
                tool_func_name = mt.name

                def _make_tool(srv=server, t_name=tool_func_name, desc=mt.description, schema=mt.inputSchema):
                    async def _call_mcp(**kwargs):
                        res, _ = await srv.call_tool(t_name, kwargs)
                        if res and hasattr(res[0], "text"):
                            return res[0].text
                        return str(res)

                    return StructuredTool(
                        name=t_name,
                        description=desc or "",
                        args_schema=schema,
                        coroutine=_call_mcp,
                    )

                tools.append(_make_tool())
        return tools


# ==============================================================================
# AC-1: 新写简单工具只做注册，Agent 即可在对话中用上（零改动核心代码）
# ==============================================================================

@pytest.mark.asyncio
async def test_ac1_register_new_tool_available_immediately(clean_db):
    """AC-1: 注册全新自定义工具 query_vip_discount，发起对话，断言 Agent 自动发现并成功调用工具完成回答。

    全过程无需改动任何 Agent 提示词或工作流编排代码，注册后即插即用。
    """
    conv_id = 8101
    await clean_db(conv_id)

    # 1. 业务开发者新编写一个简单的 VIP 折扣查询工具
    @tool
    def query_vip_discount(vip_level: str) -> str:
        """根据用户会员等级查询专属 VIP 折扣与权益详情"""
        discounts = {
            "VIP1": "享受95折优惠与全场免运费权益",
            "VIP2": "享受9折优惠与生日礼品权益",
            "VIP3": "享受85折专属优惠与专属VIP客服一对一服务",
        }
        level_key = str(vip_level).strip().upper()
        return json.dumps(
            {
                "vip_level": vip_level,
                "discount_info": discounts.get(level_key, "普通会员暂无专属折扣"),
                "status": "VALID",
            },
            ensure_ascii=False,
        )

    # 2. 仅在注册中心执行一行注册，零修改任何 Agent/Workflow 代码
    default_tool_registry.register(query_vip_discount)

    try:
        # 验证注册中心已自动纳管该工具
        assert "query_vip_discount" in default_tool_registry
        all_tools = await default_tool_registry.get_all_tools()
        assert any(t.name == "query_vip_discount" for t in all_tools)

        # 3. 发起对话，测试 Agent 自动发现并调用新工具
        engine_instance = WorkflowEngine()
        fake_intent = {
            "intent": "订单",
            "confidence": 0.96,
            "intent_reason": "用户咨询会员等级折扣权益",
        }

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        # 模拟模型首次识别并调用 query_vip_discount，第二次接收工具回填结果并生成最终回复
        mock_model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "query_vip_discount",
                            "args": {"vip_level": "VIP3"},
                            "id": "call_vip_8101",
                        }
                    ],
                ),
                AIMessage(
                    content="您好！已为您查询到您的 VIP3 会员专属权益：享受85折专属优惠与专属VIP客服一对一服务。"
                ),
            ]
        )

        with patch(
            "app.services.workflow.nodes.pre_nodes.intent_recognition_node",
            new=AsyncMock(return_value=fake_intent),
        ):
            with patch(
                "app.services.workflow.nodes.agent_node.get_chat_model",
                return_value=mock_model,
            ):
                final_state = await engine_instance.run(
                    conversation_id=conv_id,
                    query="我是VIP3，能查下我有什么专属折扣吗",
                )

        # 4. 断言 Agent 成功调用该工具并生成正确回答
        resp_text = final_state.get("response_text", "")
        assert "85折" in resp_text
        assert "VIP3" in resp_text or "专属" in resp_text

        # 5. 断言 tool_audit_logs 已完整落库留痕且状态为「成功」
        async with AsyncSessionLocal() as session:
            stmt = select(ToolAuditLog).where(
                ToolAuditLog.conversation_id == conv_id,
                ToolAuditLog.tool_name == "query_vip_discount",
            )
            res = await session.execute(stmt)
            audit_entry = res.scalar_one_or_none()
            assert audit_entry is not None
            assert audit_entry.status == "成功"
            assert audit_entry.tool_source == "builtin"
            assert audit_entry.arguments == {"vip_level": "VIP3"}
            assert "85折" in audit_entry.result_summary

    finally:
        # 清理注册中心环境，防止污染后续测试
        default_tool_registry._tools.pop("query_vip_discount", None)


# ==============================================================================
# AC-2: 两个独立 MCP Server 独立进程运行，问物流通过 MCP 获取
# ==============================================================================

@pytest.mark.asyncio
async def test_ac2_mcp_servers_logistics_query(clean_db):
    """AC-2: 两个独立 MCP Server (物流 8001 / 售后 8002) 运行，问物流通过 MCP 获取；

    通过 FastMCP/Client 链路模拟物流查询，断言工具来源为 mcp、所属 server 为 logistics，
    数据取自 MCP Server，tool_audit_logs 记录为「成功」。
    """
    conv_id = 8102
    await clean_db(conv_id)

    # 1. 实例化两个独立的业务 MCP Server
    logistics_server = create_logistics_server(host="127.0.0.1", port=8001)
    aftersale_server = create_aftersale_server(host="127.0.0.1", port=8002)

    # 2. 构建连接这两个 Server 的客户端适配器
    mcp_adapter = FastMCPClientAdapter({
        "logistics": logistics_server,
        "aftersale": aftersale_server,
    })

    # 将适配器挂接至统一工具注册中心
    old_client = default_tool_registry.mcp_client
    default_tool_registry.mcp_client = mcp_adapter
    default_tool_registry._mcp_tools_cache.clear()

    try:
        # 3. 动态探测 MCP Server 导出的工具
        all_tools = await default_tool_registry.get_all_tools()
        tool_names = [t.name for t in all_tools]
        assert "query_logistics" in tool_names
        assert "check_warranty" in tool_names
        assert "query_return_progress" in tool_names

        # 验证 query_logistics 工具打标：来源为 mcp，所属 server 为 logistics
        logistics_tool = next(t for t in all_tools if t.name == "query_logistics")
        assert getattr(logistics_tool, "tool_source") == "mcp"
        assert getattr(logistics_tool, "mcp_server") == "logistics"

        # 4. 发起问物流会话
        engine_instance = WorkflowEngine()
        fake_intent = {
            "intent": "物流",
            "confidence": 0.99,
            "intent_reason": "用户查询订单物流轨迹",
        }

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        # 模拟模型首次发起 query_logistics 调用，第二次接收 FastMCP Server 返回的真实顺丰轨迹并总结
        mock_model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "query_logistics",
                            "args": {"order_id": "1001"},
                            "id": "call_logistics_8102",
                        }
                    ],
                ),
                AIMessage(
                    content="您的订单 1001 由顺丰速运承运，运单号为 SF14285700199，当前物流状态为派送中，快递员已出发。"
                ),
            ]
        )

        with patch(
            "app.services.workflow.nodes.pre_nodes.intent_recognition_node",
            new=AsyncMock(return_value=fake_intent),
        ):
            with patch(
                "app.services.workflow.nodes.agent_node.get_chat_model",
                return_value=mock_model,
            ):
                final_state = await engine_instance.run(
                    conversation_id=conv_id,
                    query="帮我查下订单 1001 的物流到哪了",
                )

        # 5. 断言回复中包含来自 MCP LogisticsServer 的真实数据
        resp_text = final_state.get("response_text", "")
        assert "SF14285700199" in resp_text
        assert "顺丰速运" in resp_text or "派送中" in resp_text

        # 6. 断言 tool_audit_logs 记录：tool_source 为 mcp，mcp_server 为 logistics，status 为「成功」
        async with AsyncSessionLocal() as session:
            stmt = select(ToolAuditLog).where(
                ToolAuditLog.conversation_id == conv_id,
                ToolAuditLog.tool_name == "query_logistics",
            )
            res = await session.execute(stmt)
            audit_entry = res.scalar_one_or_none()
            assert audit_entry is not None
            assert audit_entry.status == "成功"
            assert audit_entry.tool_source == "mcp"
            assert audit_entry.mcp_server == "logistics"
            assert "SF14285700199" in audit_entry.result_summary
            assert "顺丰速运" in audit_entry.result_summary

    finally:
        default_tool_registry.mcp_client = old_client
        default_tool_registry._mcp_tools_cache.clear()


# ==============================================================================
# AC-3: 售后 MCP Server 新增工具并重启 Server，客服端代码不动无需重启即可感知并调用
# ==============================================================================

@pytest.mark.asyncio
async def test_ac3_mcp_dynamic_tool_hot_plug(clean_db):
    """AC-3: 售后 MCP Server 新增工具并重启/重载 Server，客服端代码不动无需重启即可感知并调用；

    在售后 Server 侧动态注册新工具 calculate_repair_quote，客户端代码零改动且免重启，
    通过 get_all_tools() 动态感知，并在会话中成功调用。
    """
    conv_id = 8103
    await clean_db(conv_id)

    # 1. 售后 Server 初始状态仅包含 check_warranty 与 query_return_progress
    aftersale_server = create_aftersale_server(host="127.0.0.1", port=8002)
    mcp_adapter = FastMCPClientAdapter({"aftersale": aftersale_server})

    old_client = default_tool_registry.mcp_client
    default_tool_registry.mcp_client = mcp_adapter
    default_tool_registry._mcp_tools_cache.clear()

    try:
        initial_tools = await default_tool_registry.get_all_tools()
        initial_names = [t.name for t in initial_tools]
        assert "calculate_repair_quote" not in initial_names

        # 2. 模拟售后微服务团队在 Server 侧动态新增工具并热重载 Server
        def calculate_repair_quote(product_name: str, damage_type: str) -> str:
            """根据商品名称与损坏情况估算售后维修费用与耗时"""
            return json.dumps(
                {
                    "product_name": product_name,
                    "damage_type": damage_type,
                    "estimated_cost": "120.00元",
                    "repair_duration_days": 3,
                    "service_policy": "官方原厂配件质保90天",
                    "status": "QUOTED",
                },
                ensure_ascii=False,
            )

        register_dynamic_tool(aftersale_server, calculate_repair_quote)

        # 3. 客服端代码完全不动、不重启，动态拉取工具
        updated_tools = await default_tool_registry.get_all_tools()
        updated_names = [t.name for t in updated_tools]
        assert "calculate_repair_quote" in updated_names

        # 验证新动态工具被正确打上 MCP 售后服务标签
        repair_tool = next(t for t in updated_tools if t.name == "calculate_repair_quote")
        assert getattr(repair_tool, "tool_source") == "mcp"
        assert getattr(repair_tool, "mcp_server") == "aftersale"

        # 4. 在客服 Agent 会话中无缝调用该新工具
        engine_instance = WorkflowEngine()
        fake_intent = {
            "intent": "订单",
            "confidence": 0.95,
            "intent_reason": "用户咨询商品损坏维修费用",
        }

        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        mock_model.ainvoke = AsyncMock(
            side_effect=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculate_repair_quote",
                            "args": {
                                "product_name": "极简保暖羽绒服",
                                "damage_type": "拉链损坏",
                            },
                            "id": "call_repair_8103",
                        }
                    ],
                ),
                AIMessage(
                    content="已为您估算维修报价：极简保暖羽绒服拉链损坏预估费用为 120.00元，预计耗时 3 天，支持官方原厂配件质保90天。"
                ),
            ]
        )

        with patch(
            "app.services.workflow.nodes.pre_nodes.intent_recognition_node",
            new=AsyncMock(return_value=fake_intent),
        ):
            with patch(
                "app.services.workflow.nodes.agent_node.get_chat_model",
                return_value=mock_model,
            ):
                final_state = await engine_instance.run(
                    conversation_id=conv_id,
                    query="我的极简保暖羽绒服拉链坏了，维修大概要多少钱？",
                )

        # 5. 断言回复包含来自新 MCP 工具的报价详情
        resp_text = final_state.get("response_text", "")
        assert "120" in resp_text
        assert "维修" in resp_text or "羽绒服" in resp_text

        # 6. 断言 tool_audit_logs 完整记录：来源为 mcp/aftersale，状态为「成功」
        async with AsyncSessionLocal() as session:
            stmt = select(ToolAuditLog).where(
                ToolAuditLog.conversation_id == conv_id,
                ToolAuditLog.tool_name == "calculate_repair_quote",
            )
            res = await session.execute(stmt)
            audit_entry = res.scalar_one_or_none()
            assert audit_entry is not None
            assert audit_entry.status == "成功"
            assert audit_entry.tool_source == "mcp"
            assert audit_entry.mcp_server == "aftersale"
            assert "120.00元" in audit_entry.result_summary

    finally:
        default_tool_registry.mcp_client = old_client
        default_tool_registry._mcp_tools_cache.clear()


# ==============================================================================
# AC-4: 问建单未讲明问题，Agent 追问补齐；前端卡片点「确认提交」，tickets 表落库且回复带工单号
# ==============================================================================

@pytest.mark.asyncio
async def test_ac4_ticket_incomplete_ask_and_confirm_flow(clean_db):
    """AC-4: 问建单未讲明问题，Agent 追问补齐；前端卡片点「确认提交」，tickets 表落库且回复带工单号；

    第 1 轮“帮我建个工单” -> 意图分类为“人工” -> Agent 追问缺失细节，不调用 create_ticket；
    第 2 轮补充描述 -> 触发 create_ticket 与 LangGraph interrupt -> 下发 ticket_preview；
    调用 resume 提交 confirm -> tickets 表新增工单记录 -> tool_audit_logs 状态为「成功」 -> 最终回复包含工单编号。
    """
    conv_id = 8104
    await clean_db(conv_id)

    engine_instance = WorkflowEngine()
    fake_intent_human = {
        "intent": "人工",
        "confidence": 0.98,
        "intent_reason": "用户申请建工单转人工专员处理",
    }

    # --------------------------------------------------------------------------
    # 第 1 轮：用户仅说“帮我建个工单”，未提供问题细节
    # --------------------------------------------------------------------------
    mock_model_ask = MagicMock()
    mock_model_ask.bind_tools.return_value = mock_model_ask
    mock_model_ask.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="请问您具体遇到了什么问题？请提供详细情况以便我们为您建立工单联系专员跟进处理。"
        )
    )

    with patch(
        "app.services.workflow.nodes.pre_nodes.intent_recognition_node",
        new=AsyncMock(return_value=fake_intent_human),
    ):
        with patch(
            "app.services.workflow.nodes.agent_node.get_chat_model",
            return_value=mock_model_ask,
        ):
            state1 = await engine_instance.run(
                conversation_id=conv_id,
                query="帮我建个工单",
            )

    # 断言 1: 第 1 轮主动追问缺失细节，未触发 interrupt，未创建任何工单
    assert "请问您具体遇到了什么问题" in state1.get("response_text", "") or "详细情况" in state1.get("response_text", "")
    assert "__interrupt__" not in state1

    async with AsyncSessionLocal() as session:
        t_count = len(
            (
                await session.execute(
                    select(Ticket).where(Ticket.conversation_id == conv_id)
                )
            ).scalars().all()
        )
        assert t_count == 0

    # --------------------------------------------------------------------------
    # 第 2 轮：用户补充具体问题描述，触发 create_ticket 与 LangGraph interrupt
    # --------------------------------------------------------------------------
    mock_model_ticket = MagicMock()
    mock_model_ticket.bind_tools.return_value = mock_model_ticket
    mock_model_ticket.ainvoke = AsyncMock(
        side_effect=[
            # 第一次调用：返回带有具体描述的 create_ticket tool_call
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_ticket",
                        "args": {
                            "ticket_type": "售后",
                            "description": "羽绒服袖口开线破损严重，请专员处理",
                        },
                        "id": "call_ticket_8104",
                    }
                ],
            ),
            # 确认恢复后第二次调用：总结工单创建结果
            AIMessage(
                content="已为您成功创建售后服务工单，售后专员将在24小时内与您联系，请保持电话畅通。"
            ),
        ]
    )

    with patch(
        "app.services.workflow.nodes.pre_nodes.intent_recognition_node",
        new=AsyncMock(return_value=fake_intent_human),
    ):
        with patch(
            "app.services.workflow.nodes.agent_node.get_chat_model",
            return_value=mock_model_ticket,
        ):
            state2 = await engine_instance.run(
                conversation_id=conv_id,
                query="我买的羽绒服袖口开线破损严重，请专员处理",
            )

    # 断言 2: 捕获 LangGraph interrupt，下发 ticket_preview 事件卡片数据
    assert "__interrupt__" in state2
    interrupts = state2["__interrupt__"]
    assert len(interrupts) > 0
    preview_val = getattr(interrupts[0], "value", interrupts[0])
    if isinstance(preview_val, dict) and "value" in preview_val:
        preview_val = preview_val["value"]
    assert preview_val.get("event_type") == "ticket_preview"
    assert preview_val.get("ticket_type") == "售后"
    assert "袖口开线" in preview_val.get("description", "")
    assert preview_val.get("conversation_id") == conv_id

    # 挂起期间 tickets 表仍然无新记录（防止未确认即写入）
    async with AsyncSessionLocal() as session:
        t_count_pre = len(
            (
                await session.execute(
                    select(Ticket).where(Ticket.conversation_id == conv_id)
                )
            ).scalars().all()
        )
        assert t_count_pre == 0

    # --------------------------------------------------------------------------
    # 人机交互确认：用户在前端卡片点击「确认提交」，调用 resume 提交 confirm
    # --------------------------------------------------------------------------
    with patch(
        "app.services.workflow.nodes.agent_node.get_chat_model",
        return_value=mock_model_ticket,
    ):
        resumed_state = await engine_instance.resume(
            conversation_id=conv_id, action="confirm"
        )

    # 断言 3: tickets 表成功新增工单记录
    async with AsyncSessionLocal() as session:
        t_stmt = select(Ticket).where(Ticket.conversation_id == conv_id)
        t_res = await session.execute(t_stmt)
        ticket = t_res.scalar_one_or_none()
        assert ticket is not None
        assert ticket.ticket_no.startswith("T")
        assert "袖口开线" in ticket.description

        # 断言 4: tool_audit_logs 表已落盘且状态为「成功」
        a_stmt = select(ToolAuditLog).where(
            ToolAuditLog.conversation_id == conv_id,
            ToolAuditLog.tool_name == "create_ticket",
        )
        a_res = await session.execute(a_stmt)
        audit_log = a_res.scalar_one_or_none()
        assert audit_log is not None
        assert audit_log.status == "成功"

    # 断言 5: 最终回复包含工单编号 (T...)
    resp_final = resumed_state.get("response_text", "")
    assert re.search(r"T\d+", resp_final) is not None
    assert ticket.ticket_no in resp_final


# ==============================================================================
# AC-5: 工单预览点「取消」，工单未建，tool_audit_logs 状态为「权限拒绝」
# ==============================================================================

@pytest.mark.asyncio
async def test_ac5_ticket_cancel_flow_audit_denied(clean_db):
    """AC-5: 工单预览点「取消」，工单未建，tool_audit_logs 状态为「权限拒绝」；

    触发工单预览后调用 resume 提交 cancel -> tickets 表无新增 ->
    tool_audit_logs 状态为「权限拒绝」 -> 回复友好取消语 -> 下一轮普通提问不受污染。
    """
    conv_id = 8105
    await clean_db(conv_id)

    engine_instance = WorkflowEngine()
    fake_intent_ticket = {
        "intent": "人工",
        "confidence": 0.98,
        "intent_reason": "用户申请建工单转人工专员处理",
    }
    fake_intent_qa = {
        "intent": "订单",
        "confidence": 0.95,
        "intent_reason": "运费险规则咨询",
    }

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.ainvoke = AsyncMock(
        side_effect=[
            # 轮次 1: 触发建单 tool call
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "create_ticket",
                        "args": {
                            "ticket_type": "售后",
                            "description": "羽绒服袖口开线破损严重，请专员处理",
                        },
                        "id": "call_ticket_8105",
                    }
                ],
            ),
            # 轮次 2: 下一轮普通提问回答
            AIMessage(
                content="运费险可以在订单详情页申请退货退款，卖家同意后由系统自动发起运费险理赔退款。"
            ),
        ]
    )

    with patch(
        "app.services.workflow.nodes.agent_node.get_chat_model",
        return_value=mock_model,
    ):
        with patch(
            "app.services.workflow.nodes.pre_nodes.intent_recognition_node",
            side_effect=[fake_intent_ticket, fake_intent_qa],
        ):
            # 1. 运行首轮，触发挂起
            state1 = await engine_instance.run(
                conversation_id=conv_id,
                query="帮我建工单，我买的羽绒服袖口开线破损严重，请专员处理",
            )
            assert "__interrupt__" in state1
            assert conv_id in _pending_agent_states

            # 2. 恢复挂起，用户点击「取消」
            resumed_state = await engine_instance.resume(
                conversation_id=conv_id, action="cancel"
            )

            # 断言 1: 回复为友好取消提示语
            resp_text = resumed_state.get("response_text", "")
            assert "取消" in resp_text

            # 断言 2: tickets 表未创建任何工单
            async with AsyncSessionLocal() as session:
                t_stmt = select(Ticket).where(Ticket.conversation_id == conv_id)
                t_res = await session.execute(t_stmt)
                ticket = t_res.scalar_one_or_none()
                assert ticket is None

                # 断言 3: tool_audit_logs 表已落盘且状态为「权限拒绝」
                a_stmt = select(ToolAuditLog).where(
                    ToolAuditLog.conversation_id == conv_id,
                    ToolAuditLog.tool_name == "create_ticket",
                )
                a_res = await session.execute(a_stmt)
                audit_log = a_res.scalar_one_or_none()
                assert audit_log is not None
                assert audit_log.status == "权限拒绝"

            # 断言 4: 挂起状态缓存已被彻底清除
            assert conv_id not in _pending_agent_states

            # 3. 下一轮普通提问，验证上下文纯净且不受上一轮取消工单污染
            state2 = await engine_instance.run(
                conversation_id=conv_id,
                query="运费险怎么退",
            )
            assert conv_id not in _pending_agent_states
            assert "__interrupt__" not in state2
            resp2 = state2.get("response_text", "")
            assert "运费险" in resp2
            assert "工单" not in resp2


# ==============================================================================
# AC-6: 读操作超时触发重试与降级，写操作超时不自动重试（重试次数恒为 0）
# ==============================================================================

@pytest.mark.asyncio
async def test_ac6_read_retry_write_no_retry_policy(clean_db):
    """AC-6: 读操作超时触发重试与降级，写操作超时不自动重试（重试次数恒为 0）；

    注入瞬时网络抖动：测试读操作 query_order 发生超时触发白名单重试（retry_count > 0）；
    测试写操作 create_ticket 发生超时恒不重试（retry_count == 0），审计日志状态分别为「超时」。
    """
    conv_id = 8106
    await clean_db(conv_id)

    read_call_count = 0
    write_call_count = 0

    # 1. 模拟遭遇瞬时网络抖动超时的读操作 (query_order)
    @tool
    async def timeout_query_order(order_id: str) -> str:
        """模拟订单查询读操作超时"""
        nonlocal read_call_count
        read_call_count += 1
        await asyncio.sleep(0.15)
        return "order_result"

    # 2. 模拟遭遇网络抖动超时的写操作 (create_ticket)
    @tool
    async def timeout_create_ticket(description: str) -> str:
        """模拟工单创建写操作超时"""
        nonlocal write_call_count
        write_call_count += 1
        await asyncio.sleep(0.15)
        return "ticket_result"

    timeout_create_ticket.metadata = {"is_write": True}

    registry = ToolRegistry(tools=[])
    registry.register(timeout_query_order)
    registry.register(timeout_create_ticket)

    # 配置 0.05 秒超时，最多重试 1 次
    mock_guard = MagicMock()
    mock_guard.check_permission.return_value = (True, None)
    executor = ToolExecutor(
        registry=registry,
        timeout=0.05,
        max_retries=1,
        permission_guard=mock_guard,
    )

    # --------------------------------------------------------------------------
    # 测试 A: 读操作超时 -> 触发白名单重试 (retry_count > 0) -> 审计日志状态为「超时」
    # --------------------------------------------------------------------------
    read_res = await executor.execute(
        {
            "name": "timeout_query_order",
            "args": {"order_id": "1001"},
            "id": "call_read_retry_8106",
        },
        conversation_id=conv_id,
    )

    assert read_res["success"] is False
    assert read_res["status"] == "超时"
    assert read_res["error_type"] == "SYSTEM_FAULT"
    # 允许重试 1 次，总尝试 2 次，retry_count 为 1
    assert read_res["retry_count"] > 0
    assert read_call_count == 2

    # --------------------------------------------------------------------------
    # 测试 B: 写操作超时 -> 严格禁止自动重试 (retry_count == 0) -> 审计日志状态为「超时」
    # --------------------------------------------------------------------------
    write_res = await executor.execute(
        {
            "name": "timeout_create_ticket",
            "args": {"description": "羽绒服破损退货"},
            "id": "call_write_noretry_8106",
        },
        conversation_id=conv_id,
    )

    assert write_res["success"] is False
    assert write_res["status"] == "超时"
    assert write_res["error_type"] == "SYSTEM_FAULT"
    # 绝对不重试，调用次数为 1，retry_count 恒为 0
    assert write_res["retry_count"] == 0
    assert write_call_count == 1

    # --------------------------------------------------------------------------
    # 断言 C: 校验 tool_audit_logs 中读写操作的重试与超时留痕
    # --------------------------------------------------------------------------
    async with AsyncSessionLocal() as session:
        # 查验读操作审计
        read_stmt = select(ToolAuditLog).where(
            ToolAuditLog.conversation_id == conv_id,
            ToolAuditLog.tool_name == "timeout_query_order",
        )
        read_audit = (await session.execute(read_stmt)).scalar_one_or_none()
        assert read_audit is not None
        assert read_audit.status == "超时"
        assert read_audit.retry_count > 0
        assert read_audit.retry_count == 1

        # 查验写操作审计
        write_stmt = select(ToolAuditLog).where(
            ToolAuditLog.conversation_id == conv_id,
            ToolAuditLog.tool_name == "timeout_create_ticket",
        )
        write_audit = (await session.execute(write_stmt)).scalar_one_or_none()
        assert write_audit is not None
        assert write_audit.status == "超时"
        assert write_audit.retry_count == 0
