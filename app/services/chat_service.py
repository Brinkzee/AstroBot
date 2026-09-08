import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

from app.models import Conversation, Message
from app.tools.registry import ToolRegistry, default_tool_registry
try:
    from app.tools.executor import ToolExecutor, default_tool_executor
except ImportError:
    from app.tools.executor import ToolExecutor
    default_tool_executor = ToolExecutor(registry=default_tool_registry)
from app.llm import get_chat_model
from app.prompts.customer_service import customer_service_prompt

logger = logging.getLogger(__name__)

# 五大业务工具中文名称映射表
TOOL_LABELS: Dict[str, str] = {
    "query_order": "查询订单",
    "query_product": "查询商品",
    "query_logistics": "查询物流",
    "query_faq": "查询常见问题",
    "create_ticket": "创建人工工单",
}


class ChatService:
    """具备工具编排、严格单轮收敛与数据落盘的客服聊天服务"""

    TOOL_LABELS = TOOL_LABELS

    def __init__(
        self,
        registry: ToolRegistry = default_tool_registry,
        executor: ToolExecutor = default_tool_executor,
        model: Optional[Any] = None,
        stream_model: Optional[Any] = None,
    ) -> None:
        self.registry = registry
        self.executor = executor
        self.model = model
        self.stream_model = stream_model

    async def get_or_create_conversation(
        self,
        db: AsyncSession,
        conversation_id: Optional[int] = None,
        user_id: str = "default_user",
    ) -> Conversation:
        """获取已有会话或创建新会话"""
        if conversation_id is not None:
            if hasattr(db, "get"):
                conv = await db.get(Conversation, conversation_id)
            else:
                stmt = select(Conversation).where(Conversation.id == conversation_id)
                result = await db.execute(stmt)
                conv = result.scalar_one_or_none()

            if conv is not None:
                return conv

        conv = Conversation(user_id=user_id, status="进行中")
        db.add(conv)
        await db.commit()
        await db.refresh(conv)
        return conv

    async def save_message(
        self,
        db: AsyncSession,
        conversation_id: int,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
    ) -> Message:
        """持久化单条会话消息流水"""
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        return msg

    async def load_conversation_messages(
        self,
        db: AsyncSession,
        conversation_id: int,
    ) -> List[BaseMessage]:
        """按时间顺序加载指定会话的历史消息并转换为 LangChain BaseMessage 格式"""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
        result = await db.execute(stmt)
        db_messages = result.scalars().all()

        messages: List[BaseMessage] = []
        for msg in db_messages:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content or ""))
            elif msg.role == "assistant":
                tc = msg.tool_calls
                if isinstance(tc, str):
                    try:
                        tc = json.loads(tc)
                    except Exception:
                        pass
                if tc:
                    messages.append(AIMessage(content=msg.content or "", tool_calls=tc))
                else:
                    messages.append(AIMessage(content=msg.content or ""))
            elif msg.role == "tool":
                messages.append(
                    ToolMessage(
                        content=msg.content or "",
                        tool_call_id=msg.tool_call_id or "",
                    )
                )
            elif msg.role == "system":
                messages.append(SystemMessage(content=msg.content or ""))

        return messages

    async def stream_chat(
        self,
        db: AsyncSession,
        conversation_id: Optional[int],
        message: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """单轮流式会话编排状态机
        
        Yields:
            {"event_type": "tool_start", "conversation_id": int, "tool_name": str, "tool_label": str, "args": dict}
            {"event_type": "tool_end", "conversation_id": int, "tool_name": str, "success": bool}
            {"event_type": "text", "conversation_id": int, "content": str}
            {"event_type": "error", "conversation_id": Optional[int], "error": str}
        """
        conv_id: Optional[int] = conversation_id
        try:
            # 1. 会话初始化与历史回溯
            conv = await self.get_or_create_conversation(db, conversation_id)
            conv_id = conv.id

            history = await self.load_conversation_messages(db, conv_id)

            # 2. 用户消息持久化落盘
            await self.save_message(
                db,
                conversation_id=conv_id,
                role="user",
                content=message,
            )

            # 3. 构造提示词上下文
            prompt_value = await customer_service_prompt.ainvoke(
                {"history": history, "input": message}
            )
            prompt_messages = prompt_value.to_messages()

            # 4. 获取决策模型并绑定工具集
            tools = self.registry.get_all_tools()
            llm = self.model if self.model is not None else get_chat_model(streaming=False)
            if tools and hasattr(llm, "bind_tools"):
                bound_llm = llm.bind_tools(tools)
            else:
                bound_llm = llm

            # 5. 执行首轮意图决策
            if hasattr(bound_llm, "ainvoke"):
                resp = await bound_llm.ainvoke(prompt_messages)
            elif hasattr(bound_llm, "invoke"):
                resp = bound_llm.invoke(prompt_messages)
            else:
                resp = await bound_llm(prompt_messages)

            tool_calls = getattr(resp, "tool_calls", None) or []

            # 6. 分支 A: 无需工具调用的纯对话分支
            if not tool_calls:
                content = resp.content if isinstance(resp.content, str) else str(resp.content or "")
                if content:
                    yield {
                        "event_type": "text",
                        "conversation_id": conv_id,
                        "content": content,
                    }
                await self.save_message(
                    db,
                    conversation_id=conv_id,
                    role="assistant",
                    content=content,
                )
                return

            # 7. 分支 B: 触发工具调用 (严格单轮收敛，仅取第一个工具执行)
            tool_call = tool_calls[0]
            tool_name = str(tool_call.get("name") or "")
            tool_label = self.TOOL_LABELS.get(tool_name, tool_name)
            raw_args = tool_call.get("args") or {}

            if isinstance(raw_args, str):
                try:
                    args_dict = json.loads(raw_args)
                except Exception:
                    args_dict = {}
            elif isinstance(raw_args, dict):
                args_dict = raw_args
            else:
                args_dict = {}

            # 为 create_ticket 自动补全 conversation_id（若模型遗漏）
            if tool_name == "create_ticket" and not args_dict.get("conversation_id"):
                args_dict["conversation_id"] = conv_id
                tool_call["args"] = args_dict

            # 7.1 记录 assistant 消息 (带 tool_calls) 到数据库
            await self.save_message(
                db,
                conversation_id=conv_id,
                role="assistant",
                content=resp.content if resp.content else None,
                tool_calls=[tool_call],
            )

            # 7.2 发射 tool_start 事件
            yield {
                "event_type": "tool_start",
                "conversation_id": conv_id,
                "tool_name": tool_name,
                "tool_label": tool_label,
                "args": args_dict,
            }

            # 7.3 执行工具调用
            tool_result = await self.executor.execute(tool_call, db=db)
            tool_output = str(tool_result.get("output") or "")
            tool_success = bool(tool_result.get("success", False))
            call_id = str(tool_call.get("id") or tool_result.get("tool_call_id") or "")

            # 7.4 发射 tool_end 事件
            yield {
                "event_type": "tool_end",
                "conversation_id": conv_id,
                "tool_name": tool_name,
                "success": tool_success,
            }

            # 7.5 记录 tool 结果消息到数据库
            await self.save_message(
                db,
                conversation_id=conv_id,
                role="tool",
                content=tool_output,
                tool_call_id=call_id,
            )

            # 7.6 回灌模型上下文，执行流式输出
            feedback_messages = list(prompt_messages) + [
                AIMessage(content=resp.content or "", tool_calls=[tool_call]),
                ToolMessage(content=tool_output, tool_call_id=call_id),
            ]

            stream_llm = self.stream_model
            if stream_llm is None:
                if self.model is not None and hasattr(self.model, "astream"):
                    stream_llm = self.model
                else:
                    stream_llm = get_chat_model(streaming=True)

            final_content = ""
            async for chunk in stream_llm.astream(feedback_messages):
                chunk_text = chunk.content if isinstance(chunk.content, str) else str(chunk.content or "")
                if chunk_text:
                    final_content += chunk_text
                    yield {
                        "event_type": "text",
                        "conversation_id": conv_id,
                        "content": chunk_text,
                    }

            # 7.7 记录最终 assistant 消息到数据库 (严格单轮收敛完成)
            await self.save_message(
                db,
                conversation_id=conv_id,
                role="assistant",
                content=final_content,
            )

        except Exception as e:
            logger.exception(f"Fatal error in stream_chat: {e}")
            yield {
                "event_type": "error",
                "conversation_id": conv_id,
                "error": str(e),
            }
