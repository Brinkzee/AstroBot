from typing import List
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.models.conversation import Conversation
from app.models.message import Message
from app.services.context.budget import estimate_tokens
from app.services.context.manager import (
    ContextManager,
    apply_layer1_degradation,
    build_history_context_text,
    build_model_messages,
    format_layer1_messages,
    format_layer2_messages,
    partition_messages,
)


class TestPartitionMessages:
    """测试三层上下文消息切分：Layer 3 (摘要归档), Layer 2 (半压缩), Layer 1 (原文)"""

    def test_empty_messages(self):
        l3, l2, l1 = partition_messages([], summary_upto_msg_id=None, layer1_from_msg_id=None)
        assert l3 == []
        assert l2 == []
        assert l1 == []

    def test_initial_state_all_in_layer1(self):
        # 初态：summary_upto_msg_id=None/0, layer1_from_msg_id=None/0 -> 所有历史消息在层 1
        msgs = [
            Message(id=1, role="user", content="你好"),
            Message(id=2, role="assistant", content="您好，请问有什么可以帮您？"),
            Message(id=3, role="user", content="我想查一下订单状态"),
        ]
        l3, l2, l1 = partition_messages(msgs, summary_upto_msg_id=None, layer1_from_msg_id=None)
        assert len(l3) == 0
        assert len(l2) == 0
        assert len(l1) == 3
        assert [m.id for m in l1] == [1, 2, 3]

    def test_three_tier_partitioning_with_ids(self):
        # 消息 id 1..8
        # summary_upto_msg_id=2 (id <= 2 进 Layer 3)
        # layer1_from_msg_id=5 (2 < id <= 5 进 Layer 2, id > 5 进 Layer 1)
        msgs = [Message(id=i, role="user", content=f"消息{i}") for i in range(1, 9)]

        l3, l2, l1 = partition_messages(msgs, summary_upto_msg_id=2, layer1_from_msg_id=5)

        assert [m.id for m in l3] == [1, 2]
        assert [m.id for m in l2] == [3, 4, 5]
        assert [m.id for m in l1] == [6, 7, 8]

    def test_partition_with_dict_and_none_ids(self):
        dict_msgs = [
            {"id": 1, "role": "user", "content": "一"},
            {"id": 2, "role": "assistant", "content": "二"},
            {"id": 3, "role": "user", "content": "三"},
            {"id": None, "role": "user", "content": "无ID新消息"},
        ]
        l3, l2, l1 = partition_messages(dict_msgs, summary_upto_msg_id=1, layer1_from_msg_id=2)
        assert len(l3) == 1 and l3[0]["id"] == 1
        assert len(l2) == 1 and l2[0]["id"] == 2
        # id=3 与无 id 消息都归入 layer 1
        assert len(l1) == 2
        assert l1[0]["id"] == 3


class TestLayer1Degradation:
    """测试层 1 超预算降级状态机与锚点推进"""

    def test_under_budget_no_degradation(self):
        conv = Conversation(id=1, summary_upto_msg_id=None, layer1_from_msg_id=None)
        msgs = [
            Message(id=1, role="user", content="你好"),
            Message(id=2, role="assistant", content="您好"),
        ]
        # 足够大的预算
        res = apply_layer1_degradation(conv, msgs, layer1_budget=1000)
        assert res is None
        assert conv.layer1_from_msg_id is None

    def test_over_budget_advances_layer1_anchor(self):
        # 构造 4 条消息，每条约 20 字（20 tokens）
        # 总 Token 约 80
        # 预算设为 35，则前 2~3 条必须被降级推移至层 2
        conv = Conversation(id=1, summary_upto_msg_id=None, layer1_from_msg_id=None)
        msgs = [
            Message(id=1, role="user", content="一二三四五六七八九十一二三四五六七八九十"),  # 20 tokens
            Message(id=2, role="assistant", content="一二三四五六七八九十一二三四五六七八九十"),  # 20 tokens
            Message(id=3, role="user", content="一二三四五六七八九十一二三四五六七八九十"),  # 20 tokens
            Message(id=4, role="assistant", content="一二三四五六七八九十一二三四五六七八九十"),  # 20 tokens
        ]
        budget = 35

        new_anchor = apply_layer1_degradation(conv, msgs, layer1_budget=budget)

        # 降级后 conv.layer1_from_msg_id 必须更新
        assert conv.layer1_from_msg_id == new_anchor
        assert new_anchor is not None
        assert new_anchor >= 2  # 至少前两道消息被推入层 2

        # 验证降级后层 1 的剩余消息 Token 严格不超过预算
        _, _, remaining_l1 = partition_messages(
            msgs,
            summary_upto_msg_id=conv.summary_upto_msg_id,
            layer1_from_msg_id=conv.layer1_from_msg_id,
        )
        assert estimate_tokens(remaining_l1) <= budget

    def test_degradation_with_existing_summary_upto(self):
        # 如果已经有 summary_upto_msg_id=2，只针对未进摘要的消息计算与降级
        conv = Conversation(id=1, summary_upto_msg_id=2, layer1_from_msg_id=2)
        msgs = [
            Message(id=1, role="user", content="已进摘要1"),
            Message(id=2, role="assistant", content="已进摘要2"),
            Message(id=3, role="user", content="一二三四五六七八九十一二三四五六七八九十"),  # 20
            Message(id=4, role="assistant", content="一二三四五六七八九十一二三四五六七八九十"),  # 20
        ]
        # 预算设为 25 (msg 4 单条 20 tokens 可容纳，两条合计 40 tokens 超标)，消息 3 必须降级
        new_anchor = apply_layer1_degradation(conv, msgs, layer1_budget=25)
        assert new_anchor == 3
        assert conv.layer1_from_msg_id == 3



class TestFormatLayer2Messages:
    """测试层 2 半压缩渲染规则"""

    def test_format_user_message_uncompressed(self):
        # 用户原话 100% 完整保留，一个字不改
        long_user_prompt = "这是一段非常长的用户退款诉求说明，包含详细的订单号SN99887766以及对商家的严正声明，绝对不能有任何删改。"
        user_msg = Message(role="user", content=long_user_prompt)
        res = format_layer2_messages([user_msg])

        assert len(res) == 1
        assert isinstance(res[0], HumanMessage)
        assert res[0].content == long_user_prompt

    def test_format_assistant_message_truncation(self):
        # 客服答复：前 60 字截断 + "..."；短回复不加省略号
        short_reply = "好的，已为您核实完毕。"
        long_reply = (
            "您好，关于您反馈的极简轻暖羽绒服退款问题，经过我们与售后技术团队的沟通核实，"
            "目前仓库质检确认吊牌完整且未发现穿着痕迹，我们已为您加急通过退款申请，款项将在1-3个工作日内原路退回。"
        )
        assert len(long_reply) > 60

        msgs = [
            Message(role="assistant", content=short_reply),
            Message(role="assistant", content=long_reply),
        ]
        res = format_layer2_messages(msgs)

        assert len(res) == 2
        assert isinstance(res[0], AIMessage)
        assert res[0].content == short_reply

        assert isinstance(res[1], AIMessage)
        assert len(res[1].content) == 63  # 60 字 + 3 个点 "..."
        assert res[1].content == long_reply[:60] + "..."

    def test_format_tool_message_semantic_replacement(self):
        # 工具结果：替换为单行语义标识，保留 tool_call_id
        huge_tool_content = '{"code": 200, "data": {"huge_list": [{"id": 1, "text": "log"}] * 500}}'
        tool_msg = Message(
            role="tool",
            content=huge_tool_content,
            tool_call_id="call_query_order_99",
        )
        res = format_layer2_messages([tool_msg])

        assert len(res) == 1
        assert isinstance(res[0], ToolMessage)
        assert res[0].tool_call_id == "call_query_order_99"
        assert res[0].content == "[工具查询结果已由客服消化]"

    def test_format_tool_message_with_name(self):
        # 带工具名称的 LangChain ToolMessage
        tool_msg = ToolMessage(
            content='{"status": "ok"}',
            tool_call_id="call_123",
            name="query_logistics",
        )
        res = format_layer2_messages([tool_msg])
        assert len(res) == 1
        assert isinstance(res[0], ToolMessage)
        assert res[0].tool_call_id == "call_123"
        assert "[工具查询" in res[0].content
        assert "query_logistics" in res[0].content
        assert "结果已由客服消化" in res[0].content


class TestFormatLayer1Messages:
    """测试层 1 原文格式化：原样保留不裁剪"""

    def test_layer1_messages_preserved(self):
        msgs = [
            Message(role="user", content="我想查订单12345"),
            Message(
                role="assistant",
                content="稍等，我为您查询订单信息。",
                tool_calls=[{"id": "call_1", "name": "query_order", "args": {"order_id": "12345"}}],
            ),
            Message(role="tool", content='{"order_id": "12345", "status": "已发货"}', tool_call_id="call_1"),
        ]
        res = format_layer1_messages(msgs)

        assert len(res) == 3
        assert isinstance(res[0], HumanMessage)
        assert res[0].content == "我想查订单12345"

        assert len(res[1].tool_calls) == 1
        assert res[1].tool_calls[0]["id"] == "call_1"
        assert res[1].tool_calls[0]["name"] == "query_order"
        assert res[1].tool_calls[0]["args"] == {"order_id": "12345"}

        assert isinstance(res[2], ToolMessage)
        assert res[2].content == '{"order_id": "12345", "status": "已发货"}'
        assert res[2].tool_call_id == "call_1"


class TestBuildModelMessages:
    """测试上下文固定拼装顺序与 Prefix Cache 保护"""

    def test_assembly_order_and_prefix_cache_protection(self):
        system_prompt = "你是星舟智能客服助手。遵守客服准则。"
        layer2_msgs = [
            HumanMessage(content="我之前买的衣服"),
            AIMessage(content="您好，请问有什么可以协助您？"),
        ]
        layer1_msgs = [
            HumanMessage(content="订单号是 98765"),
            AIMessage(content="已查询到您的订单。"),
        ]
        current_query = "可以帮我申请换货吗？"
        summary = "用户此前咨询过订单98765退换货政策。"
        retrieved_docs = ["退换货规则：签收7天内支持无理由换货。"]

        assembled = build_model_messages(
            system_prompt=system_prompt,
            layer2_messages=layer2_msgs,
            layer1_messages=layer1_msgs,
            current_query=current_query,
            summary=summary,
            retrieved_docs=retrieved_docs,
        )

        # 顺序校验：
        # 0: SystemMessage (纯净固定前缀)
        # 1..2: Layer 2 半压缩历史
        # 3..4: Layer 1 原文历史
        # 5: HumanMessage (当前提问)
        # 6: HumanMessage (动态上下文证据包)
        assert len(assembled) == 7
        assert isinstance(assembled[0], SystemMessage)
        assert assembled[0].content == system_prompt
        # 严禁将摘要或证据混入 SystemMessage
        assert "前情背景摘要" not in assembled[0].content
        assert "退换货规则" not in assembled[0].content

        assert assembled[1:3] == layer2_msgs
        assert assembled[3:5] == layer1_msgs

        assert isinstance(assembled[5], HumanMessage)
        assert assembled[5].content == current_query

        assert isinstance(assembled[6], HumanMessage)
        assert "【前情背景摘要】" in assembled[6].content
        assert summary in assembled[6].content
        assert "【参考政策与业务证据】" in assembled[6].content
        assert "退换货规则：签收7天内支持无理由换货。" in assembled[6].content

    def test_assembly_without_summary_and_docs(self):
        # 没有摘要和证据时，不生成第 5 项证据包
        system_prompt = "你是智能客服。"
        current_query = "你好"
        assembled = build_model_messages(
            system_prompt=system_prompt,
            layer2_messages=[],
            layer1_messages=[],
            current_query=current_query,
            summary=None,
            retrieved_docs=None,
        )
        assert len(assembled) == 2
        assert isinstance(assembled[0], SystemMessage)
        assert isinstance(assembled[1], HumanMessage)
        assert assembled[1].content == "你好"


class TestBuildHistoryContextText:
    """测试意图消解与指代消解历史文本提取"""

    def test_build_history_context_text(self):
        summary = "用户咨询羽绒服物流进度。\n后续客服协助联系快递。"
        l2 = [HumanMessage(content="你好"), AIMessage(content="请问有什么可以帮您？")]
        l1 = [HumanMessage(content="查一下订单"), AIMessage(content="好的，请提供单号。")]

        summary_line, window_text = build_history_context_text(
            layer2_messages=l2,
            layer1_messages=l1,
            summary=summary,
        )

        assert summary_line == "用户咨询羽绒服物流进度。"
        assert "用户: 你好" in window_text
        assert "客服: 请问有什么可以帮您？" in window_text
        assert "用户: 查一下订单" in window_text
        assert "客服: 好的，请提供单号。" in window_text

    def test_build_history_context_empty(self):
        summary_line, window_text = build_history_context_text([], [], summary=None)
        assert summary_line == ""
        assert window_text == ""
