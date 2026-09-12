import inspect
from typing import Any, Callable, Dict, List
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

async def run_bare_agent_loop(
    llm: Any,
    tools: Dict[str, Callable],
    query: str,
    max_steps: int = 5,
) -> Dict[str, Any]:
    """手写裸 Agent 循环：不依赖任何 Agent 框架，纯靠 LLM 接口与 while 循环实现自适应工具调用"""
    messages: List[BaseMessage] = [
        SystemMessage(content="你是电商智能客服助手，请根据可用工具查询必要信息并回答用户问题。"),
        HumanMessage(content=query),
    ]

    tool_objs = list(tools.values())
    bound_llm = llm.bind_tools(tool_objs) if hasattr(llm, "bind_tools") else llm
    steps = 0

    while steps < max_steps:
        steps += 1
        resp = await bound_llm.ainvoke(messages)
        messages.append(resp)

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            # 收敛出自然语言答案，跳出循环
            return {
                "answer": str(resp.content or ""),
                "steps": steps,
                "messages": messages,
            }

        # 遍历执行工具调用并喂回 ToolMessage
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args") or {}
            call_id = str(tool_call.get("id") or "")

            tool_func = tools.get(tool_name)
            if tool_func is None:
                output = f"Error: Tool {tool_name} not found."
            else:
                if inspect.iscoroutinefunction(tool_func):
                    output = await tool_func(**tool_args)
                else:
                    output = tool_func(**tool_args)

            messages.append(ToolMessage(content=str(output), tool_call_id=call_id))

    # 步数超限防御：请求大模型基于已有中间上下文立即总结
    summary_messages = messages + [
        HumanMessage(content="已达到最大步数限制，请根据上述已有工具结果立即给出最终结论。")
    ]
    if hasattr(llm, "ainvoke"):
        final_resp = await llm.ainvoke(summary_messages)
    else:
        final_resp = await bound_llm.ainvoke(summary_messages)

    return {
        "answer": str(final_resp.content or ""),
        "steps": steps,
        "messages": messages,
    }
