import inspect
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
from app.services.context.budget import calculate_context_budget, estimate_tokens
from app.services.context.manager import ContextManager
from app.services.context.summary_service import SummaryService
from app.services.workflow.engine import WorkflowEngine

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
        retriever: Optional[Any] = None,
        rag_generator: Optional[Any] = None,
        workflow_engine: Optional[Any] = None,
        use_workflow: Optional[bool] = None,
        summary_service: Optional[SummaryService] = None,
    ) -> None:
        self.registry = registry
        self.executor = executor
        self.model = model
        self.stream_model = stream_model
        self.retriever = retriever
        self.rag_generator = rag_generator
        self.workflow_engine = workflow_engine
        if use_workflow is not None:
            self.use_workflow = use_workflow
        elif workflow_engine is not None:
            self.use_workflow = True
        else:
            self.use_workflow = False
        self.summary_service = summary_service or SummaryService()


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
            {"event_type": "order_selector", "conversation_id": int, "orders": list}
            {"event_type": "text", "conversation_id": int, "content": str}
            {"event_type": "actions", "conversation_id": int, "actions": list}
            {"event_type": "error", "conversation_id": Optional[int], "error": str}
        """
        conv_id: Optional[int] = conversation_id
        try:
            if self.use_workflow:
                # 1. 会话初始化
                conv = await self.get_or_create_conversation(db, conversation_id)
                conv_id = conv.id

                # 2. 用户消息持久化
                await self.save_message(db, conversation_id=conv_id, role="user", content=message)

                # 3. 读取该会话历史消息流水，调用 calculate_context_budget() 动态计算当前预算
                stmt = (
                    select(Message)
                    .where(Message.conversation_id == conv_id)
                    .order_by(Message.created_at.asc(), Message.id.asc())
                )
                result = await db.execute(stmt)
                db_messages = list(result.scalars().all())

                budget = calculate_context_budget()

                # 4. 执行层 1 降级检查
                old_l1 = conv.layer1_from_msg_id or 0
                new_l1 = ContextManager.apply_layer1_degradation(conv, db_messages, budget.layer1_budget)
                if new_l1 is not None and new_l1 != old_l1:
                    logger.info(f"层1 降级 {old_l1}→{new_l1}")
                    await db.commit()
                    await db.refresh(conv)

                # 5. 通过 ContextManager.partition_messages 切分三层，并生成 layer2_messages 与 layer1_messages
                l3, l2, l1 = ContextManager.partition_messages(
                    db_messages,
                    summary_upto_msg_id=conv.summary_upto_msg_id,
                    layer1_from_msg_id=conv.layer1_from_msg_id,
                )
                layer2_messages = ContextManager.format_layer2_messages(l2)
                layer1_messages = ContextManager.format_layer1_messages(l1)

                # 6. 将 summary=conv.summary, layer2_messages, layer1_messages 注入工作流初始 State 执行
                if self.workflow_engine is None:
                    self.workflow_engine = WorkflowEngine()
                engine = self.workflow_engine
                final_state = await engine.run(
                    conversation_id=conv_id,
                    query=message,
                    db=db,
                    summary=conv.summary,
                    layer2_messages=layer2_messages,
                    layer1_messages=layer1_messages,
                )

                # 6.1 检测工作流是否挂起 (LangGraph interrupt)
                interrupts = final_state.get("__interrupt__")
                if interrupts:
                    interrupt_items = interrupts if isinstance(interrupts, (list, tuple)) else [interrupts]
                    for item in interrupt_items:
                        val = getattr(item, "value", item)
                        if isinstance(val, dict) and "value" in val:
                            val = val["value"]
                        if isinstance(val, dict):
                            if val.get("event_type") == "ticket_preview":
                                yield {
                                    "event_type": "ticket_preview",
                                    "conversation_id": val.get("conversation_id", conv_id),
                                    "ticket_type": val.get("ticket_type", "售后"),
                                    "description": val.get("description", ""),
                                    "tool_call_id": val.get("tool_call_id", ""),
                                }
                                return
                            elif "event_type" in val:
                                yield val
                                return
                    return

                # 7. 如果有工具调用历史（优先提取本轮新产生的工具消息，杜绝扫描历史消息导致重复回显与入库）
                msgs = final_state.get("current_turn_tool_messages")
                if msgs is None:
                    # 兼容性兜底：若未提供 current_turn_tool_messages 则回落至 final_state["messages"]
                    msgs = final_state.get("messages") or []
                tool_call_map: Dict[str, str] = {}
                for i, m in enumerate(msgs):
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        for tc in m.tool_calls:
                            t_name = tc.get("name", "")
                            t_id = str(tc.get("id") or "")
                            if t_id:
                                tool_call_map[t_id] = t_name
                            t_label = self.TOOL_LABELS.get(t_name, t_name)
                            t_args = tc.get("args") or {}
                            yield {
                                "event_type": "tool_start",
                                "conversation_id": conv_id,
                                "tool_name": t_name,
                                "tool_label": t_label,
                                "args": t_args,
                            }
                            await self.save_message(
                                db,
                                conversation_id=conv_id,
                                role="assistant",
                                content=m.content if m.content else None,
                                tool_calls=[tc],
                            )
                    elif isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool":
                        t_content = str(m.content or "")
                        t_call_id = str(getattr(m, "tool_call_id", "") or "")
                        t_name = getattr(m, "name", "") or tool_call_map.get(t_call_id, "")
                        yield {
                            "event_type": "tool_end",
                            "conversation_id": conv_id,
                            "tool_name": t_name,
                            "success": not t_content.startswith("执行异常"),
                        }
                        await self.save_message(
                            db,
                            conversation_id=conv_id,
                            role="tool",
                            content=t_content,
                            tool_call_id=t_call_id,
                        )

                # 8. 若状态为 need_order_selection 或存在 suggested_orders，发射 order_selector 卡片选择事件
                if final_state.get("status") == "need_order_selection" or final_state.get("suggested_orders"):
                    yield {
                        "event_type": "order_selector",
                        "conversation_id": conv_id,
                        "orders": final_state.get("suggested_orders") or [],
                    }

                # 9. 输出 text 文本事件
                resp_text = str(final_state.get("response_text") or "")
                if resp_text:
                    yield {
                        "event_type": "text",
                        "conversation_id": conv_id,
                        "content": resp_text,
                    }
                    await self.save_message(
                        db,
                        conversation_id=conv_id,
                        role="assistant",
                        content=resp_text,
                    )

                # 10. 如果有建议操作 suggested_actions，发射 actions 事件
                actions = final_state.get("suggested_actions") or []
                if actions:
                    yield {
                        "event_type": "actions",
                        "conversation_id": conv_id,
                        "actions": actions,
                    }

                # 11. 流式回复输出并落库完成后：检查层 2 Token 用量，若超标触发后台摘要
                l2_tokens = estimate_tokens(layer2_messages)
                start_from_id = (
                    (conv.summary_upto_msg_id + 1)
                    if (conv.summary_upto_msg_id and conv.summary_upto_msg_id > 0)
                    else 0
                )
                if self.summary_service.should_trigger_summary(l2_tokens, budget.layer2_budget):
                    self.summary_service.trigger_async_summary(
                        conv.id,
                        from_msg_id=start_from_id,
                        upto_msg_id=conv.layer1_from_msg_id or 0,
                        layer2_tokens=l2_tokens,
                        layer2_budget=budget.layer2_budget,
                    )
                return

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
            tools_res = self.registry.get_all_tools()
            if inspect.isawaitable(tools_res):
                tools = await tools_res
            else:
                tools = tools_res
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

            # 为 create_ticket 强制绑定当前活跃会话 ID（防止大模型误填订单号或幻觉 ID 导致外键违规）
            if tool_name == "create_ticket":
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

            # 若为知识库问答 (query_faq)，执行两阶段自评与受控流式生成
            if tool_name == "query_faq":
                retriever = self.retriever
                if retriever is None:
                    try:
                        from app.tools.business_tools import get_retriever
                        retriever = get_retriever()
                    except Exception as e:
                        logger.warning(f"获取知识库检索器失败: {e}")

                citations: List[Dict[str, Any]] = []
                query_kw = str(args_dict.get("keyword") or message or "").strip()
                last_res = getattr(retriever, "last_result", None) if retriever else None
                if (
                    last_res is not None
                    and hasattr(last_res, "citations")
                    and isinstance(getattr(last_res, "citations"), list)
                    and last_res.citations
                ):
                    citations = last_res.citations
                elif retriever is not None and hasattr(retriever, "retrieve_with_strategy"):
                    try:
                        retrieval_res = await retriever.retrieve_with_strategy(query=query_kw, min_score=0.25)
                        raw_cits = getattr(retrieval_res, "citations", [])
                        citations = raw_cits if isinstance(raw_cits, list) else []
                    except Exception as e:
                        logger.warning(f"调用进阶检索器获取 citations 失败: {e}")

                rag_gen = self.rag_generator
                if rag_gen is None:
                    from app.services.rag.generator import RAGControlledGenerator
                    rag_gen = RAGControlledGenerator(model=self.model, stream_model=self.stream_model)

                # Phase 1: 知识充分度自检
                check_res = await rag_gen.check_sufficiency(
                    query=message,
                    citations=citations,
                    db=db,
                    conversation_id=conv_id,
                )

                if not check_res.useful:
                    # 自评不足或超纲，下发标准拒答语，截断后续生成
                    refusal_text = getattr(
                        rag_gen,
                        "refusal_text",
                        "非常抱歉，当前知识库中暂未收录相关信息，已为您登记至后台人工处理，我们的客服专员将尽快为您核实解答。",
                    )
                    yield {
                        "event_type": "text",
                        "conversation_id": conv_id,
                        "content": refusal_text,
                    }
                    await self.save_message(
                        db,
                        conversation_id=conv_id,
                        role="assistant",
                        content=refusal_text,
                    )
                    return

                # Phase 2: 知识证据充分，首帧下发 citations 事件
                yield {
                    "event_type": "citations",
                    "conversation_id": conv_id,
                    "citations": citations,
                }

                # 流式下发受控生成的带角标文本
                final_content = ""
                async for chunk_text in rag_gen.astream_generate(
                    query=message, citations=citations, history=history
                ):
                    if chunk_text:
                        final_content += chunk_text
                        yield {
                            "event_type": "text",
                            "conversation_id": conv_id,
                            "content": chunk_text,
                        }

                await self.save_message(
                    db,
                    conversation_id=conv_id,
                    role="assistant",
                    content=final_content,
                )
                return

            # 7.6 回灌模型上下文，执行流式输出 (非 FAQ 业务工具)
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

    async def resume_chat(
        self,
        db: AsyncSession,
        conversation_id: int,
        action: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """恢复挂起工作流的流式接口

        Yields:
            {"event_type": "tool_start", "conversation_id": int, "tool_name": str, "tool_label": str, "args": dict}
            {"event_type": "tool_end", "conversation_id": int, "tool_name": str, "success": bool}
            {"event_type": "text", "conversation_id": int, "content": str}
            {"event_type": "actions", "conversation_id": int, "actions": list}
            {"event_type": "error", "conversation_id": int, "error": str}
        """
        try:
            if self.workflow_engine is None:
                self.workflow_engine = WorkflowEngine()
            engine = self.workflow_engine

            final_state = await engine.resume(conversation_id=conversation_id, action=action)

            # 1. 发射工具执行事件（若有）
            msgs = final_state.get("current_turn_tool_messages") or []
            tool_call_map: Dict[str, str] = {}
            for m in msgs:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        t_name = tc.get("name", "")
                        t_id = str(tc.get("id") or "")
                        if t_id:
                            tool_call_map[t_id] = t_name
                        t_label = self.TOOL_LABELS.get(t_name, t_name)
                        t_args = tc.get("args") or {}
                        yield {
                            "event_type": "tool_start",
                            "conversation_id": conversation_id,
                            "tool_name": t_name,
                            "tool_label": t_label,
                            "args": t_args,
                        }
                        await self.save_message(
                            db,
                            conversation_id=conversation_id,
                            role="assistant",
                            content=m.content if m.content else None,
                            tool_calls=[tc],
                        )
                elif isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool":
                    t_content = str(m.content or "")
                    t_call_id = str(getattr(m, "tool_call_id", "") or "")
                    t_name = getattr(m, "name", "") or tool_call_map.get(t_call_id, "")
                    yield {
                        "event_type": "tool_end",
                        "conversation_id": conversation_id,
                        "tool_name": t_name,
                        "success": not t_content.startswith("执行异常"),
                    }
                    await self.save_message(
                        db,
                        conversation_id=conversation_id,
                        role="tool",
                        content=t_content,
                        tool_call_id=t_call_id,
                    )

            # 2. 输出 text 文本事件并落库
            resp_text = str(final_state.get("response_text") or "")
            if resp_text:
                yield {
                    "event_type": "text",
                    "conversation_id": conversation_id,
                    "content": resp_text,
                }
                await self.save_message(
                    db,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=resp_text,
                )

            # 3. 输出建议操作
            actions = final_state.get("suggested_actions") or []
            if actions:
                yield {
                    "event_type": "actions",
                    "conversation_id": conversation_id,
                    "actions": actions,
                }

        except Exception as e:
            logger.exception(f"Fatal error in resume_chat: {e}")
            yield {
                "event_type": "error",
                "conversation_id": conversation_id,
                "error": str(e),
            }

