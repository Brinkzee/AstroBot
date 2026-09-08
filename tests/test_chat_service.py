import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.models import Conversation, Message
from app.services.chat_service import ChatService, TOOL_LABELS


class FakeAsyncSession:
    """In-memory AsyncSession mock for unit tests without live MySQL dependency."""

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


class MockChunk:
    def __init__(self, content: str):
        self.content = content


def test_tool_labels_mapping():
    """验证 5 大业务工具的中文标签映射对齐"""
    expected = {
        "query_order": "查询订单",
        "query_product": "查询商品",
        "query_logistics": "查询物流",
        "query_faq": "查询常见问题",
        "create_ticket": "创建人工工单",
    }
    for tool_name, label in expected.items():
        assert TOOL_LABELS.get(tool_name) == label


@pytest.mark.asyncio
async def test_get_or_create_conversation():
    """测试创建新会话与获取已存在会话"""
    db = FakeAsyncSession()
    service = ChatService()

    # 1. 传入 None，自动新建会话
    conv1 = await service.get_or_create_conversation(db, conversation_id=None, user_id="user_test_1")
    assert conv1 is not None
    assert conv1.id == 1
    assert conv1.user_id == "user_test_1"
    assert conv1.status == "进行中"

    # 2. 传入已存在的 conversation_id，返回现有会话
    conv2 = await service.get_or_create_conversation(db, conversation_id=conv1.id)
    assert conv2.id == conv1.id
    assert conv2.user_id == "user_test_1"

    # 3. 传入不存在的 conversation_id，自动新建会话
    conv3 = await service.get_or_create_conversation(db, conversation_id=9999, user_id="user_test_2")
    assert conv3 is not None
    assert conv3.id != 9999
    assert conv3.user_id == "user_test_2"


@pytest.mark.asyncio
async def test_save_message():
    """测试保存 user、assistant、tool 等不同角色的消息落盘"""
    db = FakeAsyncSession()
    service = ChatService()
    conv = await service.get_or_create_conversation(db)

    # 1. 保存 user 消息
    msg_user = await service.save_message(
        db,
        conversation_id=conv.id,
        role="user",
        content="订单1001到哪了？",
    )
    assert msg_user.id is not None
    assert msg_user.conversation_id == conv.id
    assert msg_user.role == "user"
    assert msg_user.content == "订单1001到哪了？"
    assert msg_user.tool_calls is None
    assert msg_user.tool_call_id is None

    # 2. 保存 assistant 带 tool_calls 消息
    tool_calls_data = [{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call_1"}]
    msg_ai_tool = await service.save_message(
        db,
        conversation_id=conv.id,
        role="assistant",
        content=None,
        tool_calls=tool_calls_data,
    )
    assert msg_ai_tool.role == "assistant"
    assert msg_ai_tool.content is None
    assert msg_ai_tool.tool_calls == tool_calls_data

    # 3. 保存 tool 结果消息
    msg_tool = await service.save_message(
        db,
        conversation_id=conv.id,
        role="tool",
        content='{"status": "派送中"}',
        tool_call_id="call_1",
    )
    assert msg_tool.role == "tool"
    assert msg_tool.content == '{"status": "派送中"}'
    assert msg_tool.tool_call_id == "call_1"

    # 4. 保存 assistant 最终回答文本消息
    msg_final = await service.save_message(
        db,
        conversation_id=conv.id,
        role="assistant",
        content="您的包裹正在派送中，请注意查收。",
    )
    assert msg_final.role == "assistant"
    assert msg_final.content == "您的包裹正在派送中，请注意查收。"


@pytest.mark.asyncio
async def test_load_conversation_messages():
    """测试将数据库中的消息还原转换为 LangChain BaseMessage 序列"""
    db = FakeAsyncSession()
    service = ChatService()
    conv = await service.get_or_create_conversation(db)

    await service.save_message(db, conv.id, role="user", content="你好")
    await service.save_message(
        db,
        conv.id,
        role="assistant",
        content=None,
        tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "call_0"}],
    )
    await service.save_message(
        db,
        conv.id,
        role="tool",
        content='{"order_id": "1001", "status": "已发货"}',
        tool_call_id="call_0",
    )
    await service.save_message(db, conv.id, role="assistant", content="您的订单已发货。")

    messages = await service.load_conversation_messages(db, conv.id)
    assert len(messages) == 4

    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "你好"

    assert isinstance(messages[1], AIMessage)
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0]["name"] == "query_order"
    assert messages[1].tool_calls[0]["args"] == {"order_id": "1001"}
    assert messages[1].tool_calls[0]["id"] == "call_0"

    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "call_0"
    assert "已发货" in messages[2].content

    assert isinstance(messages[3], AIMessage)
    assert messages[3].content == "您的订单已发货。"


@pytest.mark.asyncio
async def test_stream_chat_pure_text():
    """测试无工具调用时的纯对话分支：生成 text 事件并落盘 user 与 assistant 消息"""
    db = FakeAsyncSession()

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="您好，我是星光优选客服小星！请问有什么可以帮您？",
            tool_calls=[],
        )
    )
    mock_llm.bind_tools.return_value = mock_bound

    service = ChatService(model=mock_llm)

    events = [e async for e in service.stream_chat(db, conversation_id=None, message="你好呀")]

    # 验证事件流
    assert len(events) >= 1
    text_events = [e for e in events if e["event_type"] == "text"]
    assert len(text_events) == 1
    assert "我是星光优选客服小星" in text_events[0]["content"]

    # 验证无 tool 事件
    assert not any(e["event_type"] in ("tool_start", "tool_end") for e in events)

    # 验证数据库落盘两条消息 (user + assistant)
    assert len(db.messages) == 2
    assert db.messages[0].role == "user"
    assert db.messages[0].content == "你好呀"
    assert db.messages[1].role == "assistant"
    assert "我是星光优选客服小星" in db.messages[1].content


@pytest.mark.asyncio
async def test_stream_chat_tool_call_convergence():
    """测试工具调用场景与单轮收敛：
    1. 首轮识别工具调用并推 tool_start
    2. 执行工具并推 tool_end
    3. 回灌模型流式吐出 text
    4. 单轮收敛，验证保存 4 条消息 (user, assistant tool_call, tool, final assistant)
    """
    db = FakeAsyncSession()

    # 模拟第一次决策返回 tool_calls
    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call_logistics_1"}],
        )
    )
    mock_llm.bind_tools.return_value = mock_bound

    # 模拟回灌后流式输出
    mock_stream_llm = MagicMock()
    async def fake_astream(messages):
        # 验证回灌给流式模型的上下文包含 system, history, human, ai(tool_calls), tool
        roles = [getattr(m, "type", type(m).__name__) for m in messages]
        assert "tool" in roles or any(isinstance(m, ToolMessage) for m in messages)
        yield MockChunk("订单 1001 的物流状态显示：")
        yield MockChunk("顺丰速运正在派件中，请保持电话畅通。")

    mock_stream_llm.astream = fake_astream

    service = ChatService(model=mock_llm, stream_model=mock_stream_llm)

    events = [e async for e in service.stream_chat(db, conversation_id=None, message="帮我查下订单1001的物流")]

    # 验证产生 tool_start 事件
    tool_start_events = [e for e in events if e["event_type"] == "tool_start"]
    assert len(tool_start_events) == 1
    assert tool_start_events[0]["tool_name"] == "query_logistics"
    assert tool_start_events[0]["tool_label"] == "查询物流"
    assert tool_start_events[0]["args"] == {"order_id": "1001"}

    # 验证产生 tool_end 事件
    tool_end_events = [e for e in events if e["event_type"] == "tool_end"]
    assert len(tool_end_events) == 1
    assert tool_end_events[0]["tool_name"] == "query_logistics"
    assert tool_end_events[0]["success"] is True

    # 验证产生 text 事件
    text_events = [e for e in events if e["event_type"] == "text"]
    assert len(text_events) == 2
    full_text = "".join(e["content"] for e in text_events)
    assert "顺丰速运正在派件中" in full_text

    # 验证严格单轮收敛：数据库精确保存 4 条消息
    assert len(db.messages) == 4
    # 1: user
    assert db.messages[0].role == "user"
    assert db.messages[0].content == "帮我查下订单1001的物流"
    # 2: assistant with tool_calls
    assert db.messages[1].role == "assistant"
    assert len(db.messages[1].tool_calls) == 1
    assert db.messages[1].tool_calls[0]["name"] == "query_logistics"
    assert db.messages[1].tool_calls[0]["args"] == {"order_id": "1001"}
    assert db.messages[1].tool_calls[0]["id"] == "call_logistics_1"
    # 3: tool result
    assert db.messages[2].role == "tool"
    assert db.messages[2].tool_call_id == "call_logistics_1"
    assert "顺丰速运" in db.messages[2].content
    # 4: final assistant
    assert db.messages[3].role == "assistant"
    assert "顺丰速运正在派件中" in db.messages[3].content


@pytest.mark.asyncio
async def test_stream_chat_fatal_error_handling():
    """测试当发生未捕获致命异常时，优雅发射 error SSE 帧"""
    db = FakeAsyncSession()

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(side_effect=RuntimeError("Upstream LLM timeout"))
    mock_llm.bind_tools.return_value = mock_bound

    service = ChatService(model=mock_llm)

    events = [e async for e in service.stream_chat(db, conversation_id=None, message="测试错误")]

    error_events = [e for e in events if e["event_type"] == "error"]
    assert len(error_events) == 1
    assert "Upstream LLM timeout" in error_events[0]["error"]


@pytest.mark.asyncio
async def test_stream_chat_create_ticket_forces_active_conversation_id():
    """测试当大模型调用 create_ticket 时误填或幻觉 conversation_id (如订单号 1001)，
    stream_chat 能够强制将其纠正覆盖为当前实际活跃的会话 ID (conv.id)。
    """
    db = FakeAsyncSession()

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="",
            tool_calls=[{
                "name": "create_ticket",
                "args": {"conversation_id": 1001, "description": "商品划痕申请售后", "ticket_type": "售后"},
                "id": "call_ticket_1"
            }],
        )
    )
    mock_llm.bind_tools.return_value = mock_bound

    mock_stream_llm = MagicMock()
    async def fake_astream(messages):
        yield MockChunk("工单已为您创建成功。")
    mock_stream_llm.astream = fake_astream

    service = ChatService(model=mock_llm, stream_model=mock_stream_llm)

    service.executor.execute = AsyncMock(return_value={
        "success": True,
        "tool_name": "create_ticket",
        "tool_call_id": "call_ticket_1",
        "output": '{"ticket_no": "T123", "status": "工单已创建"}',
        "error": None,
    })

    events = [e async for e in service.stream_chat(db, conversation_id=1, message="商品有划痕转人工")]

    tool_start_events = [e for e in events if e["event_type"] == "tool_start"]
    assert len(tool_start_events) == 1
    # 核心断言：必须被纠正为当前会话 ID (1)，而不是模型幻觉传进来的 1001！
    assert tool_start_events[0]["args"]["conversation_id"] == 1

    called_tool_call = service.executor.execute.call_args[0][0]
    assert called_tool_call["args"]["conversation_id"] == 1

