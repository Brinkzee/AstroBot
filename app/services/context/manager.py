import logging
from typing import Any, List, Optional, Sequence, Tuple
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.services.context.budget import estimate_tokens

logger = logging.getLogger(__name__)


def _get_message_id(msg: Any) -> Optional[int]:
    """从 ORM Message、dict 或通用对象中安全提取整数 ID"""
    if hasattr(msg, "id") and getattr(msg, "id") is not None:
        try:
            return int(msg.id)
        except (ValueError, TypeError):
            pass
    if isinstance(msg, dict) and "id" in msg and msg["id"] is not None:
        try:
            return int(msg["id"])
        except (ValueError, TypeError):
            pass
    return None


def _extract_msg_info(m: Any) -> Tuple[str, str, Any, Optional[str], Optional[str]]:
    """提取消息的 (role, content, tool_calls, tool_call_id, tool_name)"""
    if isinstance(m, HumanMessage):
        return "user", str(m.content or ""), None, None, None
    elif isinstance(m, AIMessage):
        return "assistant", str(m.content or ""), getattr(m, "tool_calls", None), None, None
    elif isinstance(m, ToolMessage):
        return "tool", str(m.content or ""), None, getattr(m, "tool_call_id", None), getattr(m, "name", None)
    elif isinstance(m, SystemMessage):
        return "system", str(m.content or ""), None, None, None

    role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
    content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""
    tool_calls = getattr(m, "tool_calls", None) or (m.get("tool_calls") if isinstance(m, dict) else None)
    tool_call_id = getattr(m, "tool_call_id", None) or (m.get("tool_call_id") if isinstance(m, dict) else None)
    tool_name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
    return str(role), str(content), tool_calls, tool_call_id, tool_name


def _format_docs(docs: Any) -> str:
    """格式化检索召回的参考知识文档"""
    if not docs:
        return ""
    if isinstance(docs, str):
        return docs.strip()
    formatted = []
    for doc in docs:
        if hasattr(doc, "page_content"):
            formatted.append(str(doc.page_content).strip())
        elif isinstance(doc, dict):
            c = doc.get("content") or doc.get("page_content") or str(doc)
            formatted.append(str(c).strip())
        else:
            formatted.append(str(doc).strip())
    return "\n\n".join(f for f in formatted if f)


def partition_messages(
    messages: Sequence[Any],
    summary_upto_msg_id: Optional[int] = None,
    layer1_from_msg_id: Optional[int] = None,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    根据逻辑锚点将消息序列切分为三层：
    - 层 3 (归档摘要层): m.id <= (summary_upto_msg_id or 0)
    - 层 2 (半压缩层): (summary_upto_msg_id or 0) < m.id <= (layer1_from_msg_id or 0)
    - 层 1 (零损原文层): m.id > (layer1_from_msg_id or 0)
    """
    s_id = summary_upto_msg_id or 0
    l1_id = layer1_from_msg_id or 0

    layer3: List[Any] = []
    layer2: List[Any] = []
    layer1: List[Any] = []

    for m in messages:
        mid = _get_message_id(m)
        if mid is None:
            # 无 ID 消息默认归入当前原文层
            layer1.append(m)
            continue

        if mid <= s_id:
            layer3.append(m)
        elif mid <= l1_id:
            layer2.append(m)
        else:
            layer1.append(m)

    return layer3, layer2, layer1


def apply_layer1_degradation(
    conv: Any,
    messages: Sequence[Any],
    layer1_budget: int,
) -> Optional[int]:
    """
    层 1 降级状态机：
    - 计算层 1 (零损原文) 估算 Token 总量；
    - 若超过 layer1_budget：从头部逐条/逐轮向后移动 layer1_from_msg_id，直到剩余 Token 不超预算；
    - 更新 conv.layer1_from_msg_id 并返回 new_id；未超预算则保持不变。
    """
    s_id = (
        getattr(conv, "summary_upto_msg_id", None)
        or (conv.get("summary_upto_msg_id") if isinstance(conv, dict) else None)
        or 0
    )
    curr_l1_id = (
        getattr(conv, "layer1_from_msg_id", None)
        or (conv.get("layer1_from_msg_id") if isinstance(conv, dict) else None)
    )

    _, _, layer1_msgs = partition_messages(
        messages,
        summary_upto_msg_id=s_id,
        layer1_from_msg_id=curr_l1_id,
    )

    if estimate_tokens(layer1_msgs) <= layer1_budget:
        return curr_l1_id

    remaining_l1 = list(layer1_msgs)
    new_id = curr_l1_id

    while remaining_l1 and estimate_tokens(remaining_l1) > layer1_budget:
        popped = remaining_l1.pop(0)
        pid = _get_message_id(popped)
        if pid is not None:
            new_id = pid

    if isinstance(conv, dict):
        conv["layer1_from_msg_id"] = new_id
    else:
        setattr(conv, "layer1_from_msg_id", new_id)

    return new_id


def format_layer2_messages(messages: Sequence[Any]) -> List[BaseMessage]:
    """
    渲染层 2 (半压缩区) 消息：
    - 用户原话: 100% 完整保留
    - 客服答复: 截取前 60 字符，超长附加 ...
    - 工具结果: 替换为单行语义标识（抹平长 JSON）
    """
    result: List[BaseMessage] = []
    for m in messages:
        role, content, tool_calls, tool_call_id, tool_name = _extract_msg_info(m)

        if role in ("user", "human"):
            result.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            if len(content) > 60:
                trunc = content[:60] + "..."
            else:
                trunc = content
            if tool_calls:
                result.append(AIMessage(content=trunc, tool_calls=tool_calls))
            else:
                result.append(AIMessage(content=trunc))
        elif role == "tool":
            if tool_name:
                sem_content = f"[工具查询: {tool_name}, 结果已由客服消化]"
            else:
                sem_content = "[工具查询结果已由客服消化]"
            cid = tool_call_id or "call_digest"
            result.append(ToolMessage(content=sem_content, tool_call_id=cid, name=tool_name))
        elif role == "system":
            result.append(SystemMessage(content=content))
        else:
            result.append(HumanMessage(content=content))

    return result


def format_layer1_messages(messages: Sequence[Any]) -> List[BaseMessage]:
    """
    渲染层 1 (零损原文区) 消息：
    原样保留用户、客服、工具消息，不进行任何裁剪或压缩。
    """
    result: List[BaseMessage] = []
    for m in messages:
        if isinstance(m, BaseMessage):
            result.append(m)
            continue

        role, content, tool_calls, tool_call_id, tool_name = _extract_msg_info(m)

        if role in ("user", "human"):
            result.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            if tool_calls:
                result.append(AIMessage(content=content, tool_calls=tool_calls))
            else:
                result.append(AIMessage(content=content))
        elif role == "tool":
            cid = tool_call_id or "call_1"
            result.append(ToolMessage(content=content, tool_call_id=cid, name=tool_name))
        elif role == "system":
            result.append(SystemMessage(content=content))
        else:
            result.append(HumanMessage(content=content))

    return result


def build_model_messages(
    system_prompt: str,
    layer2_messages: Sequence[BaseMessage],
    layer1_messages: Sequence[BaseMessage],
    current_query: str,
    summary: Optional[str] = None,
    retrieved_docs: Optional[Sequence[Any]] = None,
) -> List[BaseMessage]:
    """
    固定顺序组装上下文（严防 Prefix Cache 击穿）：
    1. SystemMessage(content=system_prompt)（人设、红线纯净置顶）
    2. 层 2 半压缩历史消息序列
    3. 层 1 原文历史消息序列
    4. HumanMessage(content=current_query)（用户当前提问）
    5. 动态上下文证据包（若 summary 或 retrieved_docs 存在）：合成单条 HumanMessage 挂在提问之后
    """
    msgs: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    msgs.extend(layer2_messages)
    msgs.extend(layer1_messages)
    msgs.append(HumanMessage(content=current_query))

    evidence_parts = []
    if summary and summary.strip():
        evidence_parts.append(f"【前情背景摘要】\n{summary.strip()}")

    docs_str = _format_docs(retrieved_docs)
    if docs_str:
        evidence_parts.append(f"【参考政策与业务证据】\n{docs_str}")

    if evidence_parts:
        msgs.append(HumanMessage(content="\n\n".join(evidence_parts)))

    return msgs


def build_history_context_text(
    layer2_messages: Sequence[BaseMessage],
    layer1_messages: Sequence[BaseMessage],
    summary: Optional[str] = None,
) -> Tuple[str, str]:
    """
    为上游指代消解与意图识别输出 (summary_line, sliding_window_text)。
    """
    summary_line = ""
    if summary and summary.strip():
        summary_line = summary.strip().splitlines()[0].strip()

    lines: List[str] = []
    for m in list(layer2_messages) + list(layer1_messages):
        content = getattr(m, "content", "")
        if isinstance(m, HumanMessage) or getattr(m, "role", "") == "user":
            lines.append(f"用户: {content}")
        elif isinstance(m, AIMessage) or getattr(m, "role", "") == "assistant":
            lines.append(f"客服: {content}")
        elif isinstance(m, ToolMessage) or getattr(m, "role", "") == "tool":
            lines.append(f"工具: {content}")
        elif isinstance(m, SystemMessage) or getattr(m, "role", "") == "system":
            continue
        elif content:
            lines.append(str(content))

    sliding_window_text = "\n".join(lines).strip()
    return summary_line, sliding_window_text


class ContextManager:
    """三层会话上下文管理器与降级状态机"""

    partition_messages = staticmethod(partition_messages)
    apply_layer1_degradation = staticmethod(apply_layer1_degradation)
    format_layer2_messages = staticmethod(format_layer2_messages)
    format_layer1_messages = staticmethod(format_layer1_messages)
    build_model_messages = staticmethod(build_model_messages)
    build_history_context_text = staticmethod(build_history_context_text)
