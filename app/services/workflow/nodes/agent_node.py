import inspect
import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from app.llm import get_chat_model
from app.services.workflow.state import AgentWorkflowState
from app.tools.registry import default_tool_registry

logger = logging.getLogger(__name__)

AGENT_SYSTEM_BASE = """你是电商平台的生产级主力客服 Agent。
你拥有自主多步决策与调用工具的能力。
规则：
1. 遇到需要查询的数据，主动调用对应工具；
2. 如果用户提供的信息不足以调用工具（例如查物流却未给订单号），请友好向用户追问缺失信息，不要凭空猜测；
3. 工具返回结果后，结合已有知识与上下文进行有条理、亲切专业的回答。
"""

MAX_STEPS = 5

async def main_agent_node(
    state: AgentWorkflowState,
    model: Optional[Any] = None,
    tools: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """主力 ReAct Agent 节点：支持工具自适应循环、多步推理、知识上下文注入与 5 步硬截断"""
    llm = model or get_chat_model(streaming=False)
    available_tools = tools if tools is not None else default_tool_registry.get_all_tools()
    tools_map = {t.name: t for t in available_tools}

    # 1. 构造上下文提示词（融合上游知识库证据）
    sys_content = AGENT_SYSTEM_BASE
    docs = state.get("retrieved_docs") or []
    if docs:
        knowledge_texts = []
        for i, doc in enumerate(docs[:3], 1):
            text = doc.get("text") or doc.get("content") or str(doc)
            knowledge_texts.append(f"【参考知识 {i}】{text}")
        sys_content += "\n以下为检索到的官方政策与业务知识，请参考并用于回答：\n" + "\n".join(knowledge_texts)

    # 2. 准备消息列表（以状态中原有历史为基础）
    messages: List[BaseMessage] = [SystemMessage(content=sys_content)]
    existing_messages = state.get("messages") or []
    for m in existing_messages:
        if not isinstance(m, SystemMessage):
            messages.append(m)

    bound_llm = llm.bind_tools(available_tools) if (available_tools and hasattr(llm, "bind_tools")) else llm

    steps = 0
    init_tokens = state.get("token_usage") or {}
    total_tokens = {
        "prompt_tokens": int(init_tokens.get("prompt_tokens", 0)),
        "completion_tokens": int(init_tokens.get("completion_tokens", 0)),
        "total_tokens": int(init_tokens.get("total_tokens", 0)),
    }
    final_text = ""

    while steps < MAX_STEPS:
        steps += 1
        resp = await bound_llm.ainvoke(messages)
        messages.append(resp)

        # 统计 token 消耗
        meta = getattr(resp, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or meta.get("usage") or getattr(resp, "usage_metadata", None) or {}
        if usage:
            prompt_t = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
            comp_t = int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
            tot_t = int(usage.get("total_tokens", 0)) or (prompt_t + comp_t)
            total_tokens["prompt_tokens"] += prompt_t
            total_tokens["completion_tokens"] += comp_t
            total_tokens["total_tokens"] += tot_t

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            # 自然收敛，完成推演
            final_text = str(resp.content or "")
            break

        # 执行工具并将结果回填
        for tool_call in tool_calls:
            name = tool_call.get("name", "")
            raw_args = tool_call.get("args") or {}
            call_id = str(tool_call.get("id") or "")

            target_tool = tools_map.get(name)
            if target_tool is None:
                output = f"工具 {name} 未注册"
            else:
                try:
                    if hasattr(target_tool, "ainvoke"):
                        output = await target_tool.ainvoke(raw_args)
                    elif inspect.iscoroutinefunction(target_tool):
                        output = await target_tool(**raw_args)
                    elif callable(target_tool):
                        output = target_tool(**raw_args)
                    else:
                        output = str(target_tool)
                except Exception as e:
                    output = f"执行异常: {str(e)}"

            messages.append(ToolMessage(content=str(output), tool_call_id=call_id))

    if not final_text:
        # 步数超限，执行总结
        final_summary = await llm.ainvoke(
            messages + [HumanMessage(content="请根据已有工具查询的信息，立即给出最终结论。")]
        )
        final_text = str(final_summary.content or "")
        meta = getattr(final_summary, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or meta.get("usage") or getattr(final_summary, "usage_metadata", None) or {}
        if usage:
            prompt_t = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
            comp_t = int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
            tot_t = int(usage.get("total_tokens", 0)) or (prompt_t + comp_t)
            total_tokens["prompt_tokens"] += prompt_t
            total_tokens["completion_tokens"] += comp_t
            total_tokens["total_tokens"] += tot_t

    return {
        "response_text": final_text,
        "messages": messages,
        "steps_taken": steps,
        "token_usage": total_tokens,
        "status": "success",
    }
