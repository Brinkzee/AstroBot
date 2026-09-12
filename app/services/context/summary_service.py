import asyncio
import logging
import time
from typing import Any, List, Optional, Sequence, Set
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.llm import get_chat_model
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.summary import ConversationSummary
from app.prompts.summarizer import SUMMARY_SYSTEM_PROMPT, format_summary_prompt

logger = logging.getLogger(__name__)


def should_trigger_summary(layer2_tokens: int, layer2_budget: int) -> bool:
    """
    基于 Token 估算判定是否触发后台分段摘要：
    仅当层 2 估算 Token 严格大于层 2 预算时触发，杜绝依赖消息条数判定。
    """
    return layer2_tokens > layer2_budget


def format_dialogue_for_summary(messages: Sequence[Any]) -> str:
    """
    将待压缩区间内的消息序列格式化为紧凑对话文本，用于 LLM 事实提炼。
    """
    lines: List[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role, content = "user", str(m.content or "")
        elif isinstance(m, AIMessage):
            role, content = "assistant", str(m.content or "")
        elif isinstance(m, ToolMessage):
            role, content = "tool", str(m.content or "")
        else:
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "user")
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""

        role_lower = str(role).lower()
        if role_lower in ("user", "human"):
            lines.append(f"用户: {content}")
        elif role_lower in ("assistant", "ai"):
            lines.append(f"客服: {content}")
        elif role_lower == "tool":
            tool_name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
            if tool_name:
                lines.append(f"工具({tool_name}): {content}")
            else:
                lines.append(f"工具: {content}")
        elif role_lower == "system":
            continue
        elif content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines).strip()


async def summarize_dialogue(
    dialogue_text: str,
    existing_summary: Optional[str] = None,
    model: Optional[Any] = None,
) -> str:
    """
    调用 LLM 异步执行受控事实提炼。
    """
    if model is None:
        model = get_chat_model(temperature=0.0)

    user_prompt = format_summary_prompt(
        dialogue_text=dialogue_text,
        existing_summary=existing_summary,
    )
    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    resp = await model.ainvoke(messages)
    return getattr(resp, "content", str(resp)).strip()


def summarize_dialogue_sync(
    dialogue_text: str,
    existing_summary: Optional[str] = None,
    model: Optional[Any] = None,
) -> str:
    """
    调用 LLM 同步执行受控事实提炼。
    """
    if model is None:
        model = get_chat_model(temperature=0.0)

    user_prompt = format_summary_prompt(
        dialogue_text=dialogue_text,
        existing_summary=existing_summary,
    )
    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    resp = model.invoke(messages)
    return getattr(resp, "content", str(resp)).strip()


async def persist_summary_segment(
    session: AsyncSession,
    conv_id: int,
    from_id: int,
    to_id: int,
    content: str,
) -> ConversationSummary:
    """
    原子落盘摘要分段并更新 Conversation 全量投影：
    1. 插入 conversation_summaries 表（seq = max_seq + 1，一段一行只追加）；
    2. 原子更新 conversations 表（summary_upto_msg_id = to_id，summary 全量投影拼接）。
    """
    stmt = (
        select(ConversationSummary)
        .where(ConversationSummary.conversation_id == conv_id)
        .order_by(ConversationSummary.seq.asc())
    )
    res = await session.execute(stmt)
    existing_summaries = list(res.scalars().all())

    current_max_seq = max([s.seq for s in existing_summaries], default=0)
    new_seq = current_max_seq + 1

    new_summary = ConversationSummary(
        conversation_id=conv_id,
        seq=new_seq,
        from_msg_id=from_id,
        upto_msg_id=to_id,
        content=content,
    )
    session.add(new_summary)

    conv = await session.get(Conversation, conv_id)
    if conv:
        conv.summary_upto_msg_id = to_id
        all_contents = [s.content for s in existing_summaries] + [content]
        conv.summary = "\n\n".join(all_contents)

    await session.commit()
    await session.refresh(new_summary)
    return new_summary


class SummaryService:
    """
    后台异步分段摘要引擎：
    - 基于 Token 超标阈值判定触发（不依赖消息条数）；
    - 并发防重锁保护（同一会话单飞）；
    - 派发非阻塞后台协程任务，独立数据库 Session；
    - 记录全生命周期结构化日志 ([summary trigger], [summary start], [summary done], [summary skip], [summary fail])。
    """
    _active_tasks: Set[int] = set()

    def __init__(self, session_factory: Optional[Any] = None) -> None:
        self.session_factory = session_factory or AsyncSessionLocal
        self._active_tasks = SummaryService._active_tasks

    should_trigger_summary = staticmethod(should_trigger_summary)
    persist_summary_segment = staticmethod(persist_summary_segment)
    summarize_dialogue = staticmethod(summarize_dialogue)
    summarize_dialogue_sync = staticmethod(summarize_dialogue_sync)
    format_dialogue_for_summary = staticmethod(format_dialogue_for_summary)

    def trigger_async_summary(
        self,
        conv_id: int,
        from_msg_id: int,
        upto_msg_id: int,
        model: Optional[Any] = None,
        layer2_tokens: Optional[int] = None,
        layer2_budget: Optional[int] = None,
    ) -> Optional[asyncio.Task]:
        """
        派发非阻塞后台协程任务，立即返回，绝不阻塞用户当前对话流。
        若同一会话已有运行中的摘要任务，记录 [summary skip] 日志并跳过。
        """
        if conv_id in self._active_tasks:
            logger.info(f"[summary skip] conv_id={conv_id} 原因=该会话已有正在运行的摘要任务")
            return None

        self._active_tasks.add(conv_id)

        if layer2_tokens is not None and layer2_budget is not None:
            logger.info(
                f"[summary trigger] conv_id={conv_id} 层2 约 {layer2_tokens} token > 预算 {layer2_budget}, "
                f"range=({from_msg_id}..{upto_msg_id})"
            )
        else:
            logger.info(
                f"[summary trigger] conv_id={conv_id} range=({from_msg_id}..{upto_msg_id})"
            )

        task = asyncio.create_task(
            self.execute_summary_task(
                conv_id=conv_id,
                from_msg_id=from_msg_id,
                upto_msg_id=upto_msg_id,
                model=model,
            )
        )
        return task

    async def execute_summary_task(
        self,
        conv_id: int,
        from_msg_id: int,
        upto_msg_id: int,
        model: Optional[Any] = None,
    ) -> Optional[ConversationSummary]:
        """
        执行后台摘要提炼与落盘核心流程。
        """
        start_time = time.time()
        session_factory = getattr(self, "session_factory", AsyncSessionLocal)
        try:
            async with session_factory() as session:
                # 1. 查询已有摘要以确定当前段落序号
                stmt_summaries = (
                    select(ConversationSummary)
                    .where(ConversationSummary.conversation_id == conv_id)
                    .order_by(ConversationSummary.seq.asc())
                )
                res_summaries = await session.execute(stmt_summaries)
                existing_summaries = list(res_summaries.scalars().all())

                current_max_seq = max([s.seq for s in existing_summaries], default=0)
                new_seq = current_max_seq + 1

                logger.info(
                    f"[summary start] conv_id={conv_id} 第{new_seq}段 开始执行, "
                    f"msg_range=({from_msg_id}..{upto_msg_id})"
                )

                # 2. 查询待压缩历史消息闭区间
                stmt_msgs = (
                    select(Message)
                    .where(
                        Message.conversation_id == conv_id,
                        Message.id >= from_msg_id,
                        Message.id <= upto_msg_id,
                    )
                    .order_by(Message.id.asc())
                )
                res_msgs = await session.execute(stmt_msgs)
                messages = list(res_msgs.scalars().all())

                if not messages:
                    logger.warning(
                        f"[summary skip] conv_id={conv_id} 原因=消息区间({from_msg_id}..{upto_msg_id})内无有效消息"
                    )
                    return None

                # 3. 格式化对话文本并调用 LLM 提炼纯事实梗概
                dialogue_text = self.format_dialogue_for_summary(messages)
                existing_summary_text = (
                    "\n\n".join([s.content for s in existing_summaries])
                    if existing_summaries
                    else None
                )

                new_summary_content = await self.summarize_dialogue(
                    dialogue_text=dialogue_text,
                    existing_summary=existing_summary_text,
                    model=model,
                )

                # 4. 原子落盘写入 conversation_summaries 并更新 conversations 投影
                new_summary_obj = await self.persist_summary_segment(
                    session=session,
                    conv_id=conv_id,
                    from_id=from_msg_id,
                    to_id=upto_msg_id,
                    content=new_summary_content,
                )

                cost = time.time() - start_time
                logger.info(
                    f"[summary done] conv_id={conv_id} 第{new_seq}段完成, "
                    f"耗时={cost:.2f}s, 覆盖至 msg_id={upto_msg_id}"
                )
                return new_summary_obj

        except Exception as e:
            logger.error(f"[summary fail] conv_id={conv_id} 异常={e}", exc_info=True)
            return None
        finally:
            self._active_tasks.discard(conv_id)


default_summary_service = SummaryService()


def trigger_async_summary(
    conv_id: int,
    from_msg_id: int,
    upto_msg_id: int,
    model: Optional[Any] = None,
    layer2_tokens: Optional[int] = None,
    layer2_budget: Optional[int] = None,
) -> Optional[asyncio.Task]:
    return default_summary_service.trigger_async_summary(
        conv_id=conv_id,
        from_msg_id=from_msg_id,
        upto_msg_id=upto_msg_id,
        model=model,
        layer2_tokens=layer2_tokens,
        layer2_budget=layer2_budget,
    )


__all__ = [
    "SummaryService",
    "should_trigger_summary",
    "format_dialogue_for_summary",
    "summarize_dialogue",
    "summarize_dialogue_sync",
    "persist_summary_segment",
    "trigger_async_summary",
    "default_summary_service",
]
