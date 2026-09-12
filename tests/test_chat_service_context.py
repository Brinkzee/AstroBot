import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.models import Conversation, Message
from app.services.chat_service import ChatService
from app.services.context.budget import ContextBudgetResult, estimate_tokens
from app.services.context.manager import ContextManager
from app.services.context.summary_service import SummaryService
from app.services.workflow.engine import WorkflowEngine
from app.services.workflow.nodes.agent_node import main_agent_node
from app.services.workflow.nodes.pre_nodes import coreference_resolution_node
from app.services.workflow.state import AgentWorkflowState, create_initial_state


class FakeAsyncSession:
    """In-memory AsyncSession mock for unit testing."""

    def __init__(self):
        self.conversations = {}
        self.messages = []
        self._conv_id_counter = 1
        self._msg_id_counter = 1

    def add(self, obj):
        if isinstance(obj, Conversation):
            if getattr(obj, "id", None) is None:
                obj.id = self._conv_id_counter
                self._conv_id_counter += 1
            if getattr(obj, "created_at", None) is None:
                obj.created_at = datetime.now()
            if getattr(obj, "updated_at", None) is None:
                obj.updated_at = datetime.now()
            self.conversations[obj.id] = obj
        elif isinstance(obj, Message):
            if getattr(obj, "id", None) is None:
                obj.id = self._msg_id_counter
                self._msg_id_counter += 1
            if getattr(obj, "created_at", None) is None:
                obj.created_at = datetime.now()
            self.messages.append(obj)

    async def commit(self):
        pass

    async def refresh(self, obj):
        pass

    async def get(self, entity_cls, ident):
        if entity_cls is Conversation:
            return self.conversations.get(ident)
        return None

    async def execute(self, stmt):
        stmt_str = str(stmt)
        if "conversations" in stmt_str:
            target_id = None
            for criterion in getattr(stmt, "_where_criteria", ()):
                if hasattr(criterion, "right") and hasattr(criterion.right, "value"):
                    target_id = criterion.right.value
            conv = self.conversations.get(target_id)

            class MockConvResult:
                def scalar_one_or_none(self):
                    return conv

                def scalars(self):
                    class MockScalars:
                        def all(self):
                            return [conv] if conv else []
                    return MockScalars()

            return MockConvResult()

        elif "messages" in stmt_str:
            target_conv_id = None
            for criterion in getattr(stmt, "_where_criteria", ()):
                if hasattr(criterion, "right") and hasattr(criterion.right, "value"):
                    target_conv_id = criterion.right.value

            matched = [
                m for m in self.messages
                if target_conv_id is None or m.conversation_id == target_conv_id
            ]

            class MockMsgResult:
                def scalars(self):
                    class MockScalars:
                        def all(self):
                            return matched
                    return MockScalars()

            return MockMsgResult()

        class DefaultMockResult:
            def scalar_one_or_none(self):
                return None

            def scalars(self):
                class MockScalars:
                    def all(self):
                        return []
                return MockScalars()

        return DefaultMockResult()


# =========================================================================
# 1. 测试多轮流式对话中消息进 messages 表
# =========================================================================

@pytest.mark.asyncio
async def test_multiturn_messages_persisted_to_messages_table():
    """测试多轮流式对话中，用户提问与工作流回答均持续写入 messages 表"""
    db = FakeAsyncSession()
    mock_engine = MagicMock(spec=WorkflowEngine)

    # 模拟两轮工作流回答
    mock_engine.run = AsyncMock(side_effect=[
        {
            "conversation_id": 1,
            "response_text": "您好！我是星舟智能客服，请问有什么可以帮您？",
            "messages": [
                HumanMessage(content="你好"),
                AIMessage(content="您好！我是星舟智能客服，请问有什么可以帮您？"),
            ],
            "status": "chitchat",
            "suggested_actions": [],
        },
        {
            "conversation_id": 1,
            "response_text": "订单1001正在顺丰速运派送中。",
            "messages": [
                HumanMessage(content="查一下订单1001"),
                AIMessage(content="订单1001正在顺丰速运派送中。"),
            ],
            "status": "success",
            "suggested_actions": [],
        },
    ])

    service = ChatService(workflow_engine=mock_engine, use_workflow=True)

    # 第一轮
    events_round1 = []
    async for ev in service.stream_chat(db, conversation_id=None, message="你好"):
        events_round1.append(ev)

    # 第二轮
    events_round2 = []
    async for ev in service.stream_chat(db, conversation_id=1, message="查一下订单1001"):
        events_round2.append(ev)

    # 验证 messages 表落盘
    assert len(db.messages) == 4
    roles = [m.role for m in db.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    contents = [m.content for m in db.messages]
    assert contents[0] == "你好"
    assert "智能客服" in contents[1]
    assert contents[2] == "查一下订单1001"
    assert "派送中" in contents[3]
    assert all(m.conversation_id == 1 for m in db.messages)


# =========================================================================
# 2. 测试超预算时触发层 1 降级并更新 conversations.layer1_from_msg_id
# =========================================================================

@pytest.mark.asyncio
async def test_layer1_degradation_on_budget_exceeded():
    """测试多轮历史导致层 1 Token 超过 layer1_budget 时，自动执行降级并推进 layer1_from_msg_id"""
    db = FakeAsyncSession()
    conv = Conversation(id=1, summary_upto_msg_id=None, layer1_from_msg_id=None)
    db.conversations[1] = conv

    # 预置 4 条较长消息入库，总 Token 约 100
    for i in range(1, 5):
        db.add(Message(
            id=i,
            conversation_id=1,
            role="user" if i % 2 == 1 else "assistant",
            content="这是一段测试较长对话内容用于测试降级推移逻辑" * 2,  # 每条约 46 tokens
        ))

    mock_engine = MagicMock(spec=WorkflowEngine)
    mock_engine.run = AsyncMock(return_value={
        "conversation_id": 1,
        "response_text": "已处理您的请求",
        "messages": [AIMessage(content="已处理您的请求")],
        "status": "success",
    })

    service = ChatService(workflow_engine=mock_engine, use_workflow=True)

    # mock 预算结果：设置较小的 layer1_budget 使得降级必须触发
    mock_budget = ContextBudgetResult(
        fixed_overhead=100,
        peak_react_overhead=100,
        window_available=200,
        window_budget=200,
        layer1_budget=50,  # 仅能容纳 1~2 条消息
        layer2_budget=50,
        is_usable=True,
    )

    with patch("app.services.chat_service.calculate_context_budget", return_value=mock_budget):
        events = []
        async for ev in service.stream_chat(db, conversation_id=1, message="新问题"):
            events.append(ev)

    # 验证 conv.layer1_from_msg_id 得到推移更新
    assert conv.layer1_from_msg_id is not None
    assert conv.layer1_from_msg_id >= 2

    # 验证 workflow_engine.run 接收到的 layer2_messages 和 layer1_messages 经过切分
    mock_engine.run.assert_called_once()
    call_kwargs = mock_engine.run.call_args.kwargs
    assert "layer2_messages" in call_kwargs
    assert "layer1_messages" in call_kwargs
    assert len(call_kwargs["layer2_messages"]) > 0


# =========================================================================
# 3. 测试调模型时传入的是裁过的 build_model_messages，而 State 维护完整流水
# =========================================================================

@pytest.mark.asyncio
async def test_main_agent_node_uses_build_model_messages_and_preserves_state():
    """测试 main_agent_node 优先使用 layer2_messages 与 layer1_messages 组装 prefix cache 保护入参"""
    l2_msgs = [
        HumanMessage(content="历史用户问题A"),
        AIMessage(content="历史客服回答A"),
    ]
    l1_msgs = [
        HumanMessage(content="订单1001是什么时候下的"),
        AIMessage(content="订单1001于9月1日下单"),
    ]

    state = create_initial_state(
        conversation_id=1,
        query="它能退货吗",
        user_id="user_123",
        summary="前序摘要：用户咨询过退货政策与订单1001",
        layer2_messages=l2_msgs,
        layer1_messages=l1_msgs,
    )
    state["resolved_query"] = "订单1001极简羽绒服支持退货吗"
    state["retrieved_docs"] = [{"text": "七天无理由退货政策：签收7日内支持退换", "score": 0.95}]

    mock_llm = MagicMock()
    mock_resp = AIMessage(content="订单1001在七天内支持退货。")
    bound_mock = MagicMock()
    bound_mock.ainvoke = AsyncMock(return_value=mock_resp)
    mock_llm.bind_tools.return_value = bound_mock
    mock_llm.ainvoke = bound_mock.ainvoke

    result = await main_agent_node(state, model=mock_llm, tools=[])

    # 验证调用模型时的输入参数符合 build_model_messages 组装规范
    bound_mock.ainvoke.assert_called_once()
    invoked_messages = bound_mock.ainvoke.call_args[0][0]

    # 1. 首条必须是纯净 SystemMessage (严禁混入摘要或动态证据)
    assert isinstance(invoked_messages[0], SystemMessage)
    assert "前情背景摘要" not in invoked_messages[0].content
    assert "七天无理由退货" not in invoked_messages[0].content

    # 2. 依次是 layer2, layer1
    # 3. 接着是当前提问
    assert any(m.content == "订单1001极简羽绒服支持退货吗" for m in invoked_messages)

    # 4. 动态证据包作为单条 HumanMessage 挂在提问之后
    user_msgs = [m for m in invoked_messages if isinstance(m, HumanMessage)]
    last_evidence_msg = user_msgs[-1]
    assert "【前情背景摘要】" in last_evidence_msg.content
    assert "【参考政策与业务证据】" in last_evidence_msg.content

    # 5. state 输出正确且消息流水更新
    assert "支持退货" in result["response_text"]
    assert len(result["messages"]) > 0


# =========================================================================
# 4. 测试层 2 超出预算时触发后台摘要非阻塞运行
# =========================================================================

@pytest.mark.asyncio
async def test_layer2_over_budget_triggers_async_summary_nonblocking():
    """测试当层 2 Token 超过 layer2_budget 时触发后台摘要调度，且不阻塞流式输出"""
    db = FakeAsyncSession()
    conv = Conversation(id=1, summary_upto_msg_id=None, layer1_from_msg_id=3)
    db.conversations[1] = conv

    # 预置消息：id 1..3 属于层 2，id 4 属于层 1
    for i in range(1, 5):
        db.add(Message(
            id=i,
            conversation_id=1,
            role="user" if i % 2 == 1 else "assistant",
            content="这是一段非常长的层2半压缩测试消息用于测试摘要触发阈值" * 5,  # ~150 tokens
        ))

    mock_engine = MagicMock(spec=WorkflowEngine)
    mock_engine.run = AsyncMock(return_value={
        "conversation_id": 1,
        "response_text": "已查询到您的最新信息",
        "messages": [AIMessage(content="已查询到您的最新信息")],
        "status": "success",
    })

    mock_summary_service = MagicMock(spec=SummaryService)
    # 模拟判定超标触发
    mock_summary_service.should_trigger_summary = MagicMock(return_value=True)
    fake_task = MagicMock()
    mock_summary_service.trigger_async_summary = MagicMock(return_value=fake_task)

    service = ChatService(
        workflow_engine=mock_engine,
        use_workflow=True,
    )
    service.summary_service = mock_summary_service

    events = []
    async for ev in service.stream_chat(db, conversation_id=1, message="最新进展"):
        events.append(ev)

    # 验证流式文本正常输出未被阻塞
    text_events = [e for e in events if e.get("event_type") == "text"]
    assert len(text_events) == 1
    assert "最新信息" in text_events[0]["content"]

    # 验证触发了异步摘要调度
    mock_summary_service.should_trigger_summary.assert_called_once()
    mock_summary_service.trigger_async_summary.assert_called_once()
    call_args = mock_summary_service.trigger_async_summary.call_args
    assert call_args[0][0] == 1  # conv.id
    assert call_args.kwargs.get("from_msg_id") == 0
    assert call_args.kwargs.get("upto_msg_id") == conv.layer1_from_msg_id


# =========================================================================
# 5. 测试 pre_nodes 使用历史上下文 (summary + 精简滑窗)
# =========================================================================

@pytest.mark.asyncio
async def test_pre_nodes_utilizes_context_state():
    """测试 coreference_resolution_node 读取 state 中的 summary 与 layer2/layer1 滑窗消息"""
    l2_msgs = [HumanMessage(content="我买了件羽绒服")]
    l1_msgs = [AIMessage(content="好的，已为您记录")]

    state = create_initial_state(
        conversation_id=1,
        query="它到哪了",
        summary="用户咨询羽绒服物流",
        layer2_messages=l2_msgs,
        layer1_messages=l1_msgs,
    )

    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="羽绒服物流到哪了"))

    result = await coreference_resolution_node(state, model=mock_model)
    assert "羽绒服" in result["resolved_query"]
    mock_model.ainvoke.assert_called_once()
    prompt_msgs = mock_model.ainvoke.call_args[0][0]
    prompt_human = prompt_msgs[-1].content
    # 对话历史应包含滑窗内容
    assert "我买了件羽绒服" in prompt_human


# =========================================================================
# 6. 测试多轮工具调用下 SSE 事件流与 DB 消息不重复
# =========================================================================

@pytest.mark.asyncio
async def test_multiturn_tool_call_no_duplicate_sse_and_db():
    """测试第一轮产生工具调用，第二轮为普通对话时，第二轮不产生历史工具事件且 DB 无重复工具消息"""
    db = FakeAsyncSession()
    mock_engine = MagicMock(spec=WorkflowEngine)

    round1_tool_msg = AIMessage(
        content="",
        tool_calls=[{"id": "call_track_1", "name": "track_logistics", "args": {"order_id": "1001"}}],
    )
    round1_tool_res = ToolMessage(
        content="顺丰速运派送中，快递员：张师傅",
        tool_call_id="call_track_1",
        name="track_logistics",
    )

    mock_engine.run = AsyncMock(side_effect=[
        # Round 1: 触发工具调用
        {
            "conversation_id": 1,
            "response_text": "您的订单1001正在顺丰速运派送中。",
            "messages": [
                HumanMessage(content="帮我查下订单1001"),
                round1_tool_msg,
                round1_tool_res,
                AIMessage(content="您的订单1001正在顺丰速运派送中。"),
            ],
            "current_turn_tool_messages": [round1_tool_msg, round1_tool_res],
            "status": "success",
        },
        # Round 2: 普通对话，不调用工具
        {
            "conversation_id": 1,
            "response_text": "不客气，很高兴为您服务！",
            "messages": [
                HumanMessage(content="好的，谢谢"),
                AIMessage(content="不客气，很高兴为您服务！"),
            ],
            "current_turn_tool_messages": [],
            "status": "chitchat",
        },
    ])

    service = ChatService(workflow_engine=mock_engine, use_workflow=True)

    # 第一轮执行
    events_round1 = []
    async for ev in service.stream_chat(db, conversation_id=None, message="帮我查下订单1001"):
        events_round1.append(ev)

    # 断言第一轮产生了工具事件
    r1_tool_start = [e for e in events_round1 if e.get("event_type") == "tool_start"]
    r1_tool_end = [e for e in events_round1 if e.get("event_type") == "tool_end"]
    assert len(r1_tool_start) == 1
    assert len(r1_tool_end) == 1
    assert r1_tool_start[0]["tool_name"] == "track_logistics"

    # 断言第一轮入库了 4 条消息：user, assistant(tool_call), tool, assistant(final)
    assert len(db.messages) == 4
    assert [m.role for m in db.messages] == ["user", "assistant", "tool", "assistant"]

    # 第二轮执行
    events_round2 = []
    async for ev in service.stream_chat(db, conversation_id=1, message="好的，谢谢"):
        events_round2.append(ev)

    # 断言第二轮未产生任何工具事件
    r2_tool_start = [e for e in events_round2 if e.get("event_type") == "tool_start"]
    r2_tool_end = [e for e in events_round2 if e.get("event_type") == "tool_end"]
    assert len(r2_tool_start) == 0
    assert len(r2_tool_end) == 0

    # 断言第二轮正常输出了文本和完成事件
    r2_text = [e for e in events_round2 if e.get("event_type") == "text"]
    assert len(r2_text) == 1
    assert "不客气" in r2_text[0]["content"]

    # 断言 DB 中消息总数严格为 6 条，无重复工具消息
    assert len(db.messages) == 6
    roles = [m.role for m in db.messages]
    assert roles == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    tool_messages_in_db = [m for m in db.messages if m.role == "tool"]
    assert len(tool_messages_in_db) == 1

