import inspect
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import interrupt
from app.config import settings
from app.llm import get_chat_model
from app.services.context.budget import estimate_tokens
from app.services.context.logger import log_model_context
from app.services.context.manager import ContextManager
from app.services.workflow.state import AgentWorkflowState
from app.tools.executor import ToolExecutor, default_tool_executor
from app.tools.registry import ToolRegistry, default_tool_registry

logger = logging.getLogger(__name__)

# 挂起中断状态缓存（避免中断恢复执行时重复调用 LLM 重复生成相同 tool call）
_pending_agent_states: Dict[int, Any] = {}
_MAX_PENDING_STATES = 512
_PENDING_TTL_SECONDS = 3600.0


def clear_pending_agent_state(conv_id: int) -> None:
    """清除指定会话的挂起中断状态缓存（用于新轮次启动防止陈旧中断泄漏）。"""
    _pending_agent_states.pop(conv_id, None)


def _prune_pending_agent_states() -> None:
    """清理超时或超限的挂起状态缓存，避免放弃确认导致内存无限增长。"""
    now = time.monotonic()
    expired = [
        cid
        for cid, entry in list(_pending_agent_states.items())
        if isinstance(entry, dict)
        and (now - float(entry.get("created_at", now))) > _PENDING_TTL_SECONDS
    ]
    for cid in expired:
        _pending_agent_states.pop(cid, None)

    while len(_pending_agent_states) >= _MAX_PENDING_STATES:
        oldest_key = next(iter(_pending_agent_states))
        _pending_agent_states.pop(oldest_key, None)


def _accumulate_tokens(total_tokens: Dict[str, int], msg: Any) -> None:
    """从单次 LLM 响应消息中提取并累加 token 用量。"""
    meta = getattr(msg, "response_metadata", {}) or {}
    usage = meta.get("token_usage") or meta.get("usage") or getattr(msg, "usage_metadata", None) or {}
    if usage:
        prompt_t = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        comp_t = int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
        tot_t = int(usage.get("total_tokens", 0)) or (prompt_t + comp_t)
        total_tokens["prompt_tokens"] += prompt_t
        total_tokens["completion_tokens"] += comp_t
        total_tokens["total_tokens"] += tot_t


def _extract_history_user_queries(
    state: AgentWorkflowState, messages: List[BaseMessage]
) -> List[str]:
    """提取当前会话中历史用户原话（仅 HumanMessage），供权限门禁核验跨轮建单诉求。"""
    seen = set()
    queries: List[str] = []
    sources = (
        list(state.get("layer2_messages") or [])
        + list(state.get("layer1_messages") or [])
        + list(state.get("messages") or [])
        + list(messages)
    )
    for m in sources:
        is_human = isinstance(m, HumanMessage) or getattr(m, "type", None) == "human"
        if not is_human and isinstance(m, dict) and m.get("role") in ("user", "human"):
            is_human = True
        if not is_human:
            continue
        content = getattr(m, "content", None)
        if content is None and isinstance(m, dict):
            content = m.get("content", "")
        text = str(content or "").strip()
        if text and text not in seen:
            seen.add(text)
            queries.append(text)
    return queries

AGENT_SYSTEM_BASE = """你是电商平台的生产级主力客服 Agent。
你拥有自主多步决策与调用工具的能力。
规则：
1. 遇到需要查询的数据，主动调用对应工具；
2. 如果用户提供的信息不足以调用工具（例如查物流却未给订单号），请友好向用户追问缺失信息，不要凭空猜测；
3. 工具返回结果后，结合已有知识与上下文进行有条理、亲切专业的回答；
4. 当用户要求建工单或转人工但未说明具体问题时，主动追问“请问您具体遇到了什么问题？请提供详细情况以便我们为您建立工单联系专员跟进处理”，严禁自行编造问题描述；只有在用户提供了具体问题描述后，才调用 create_ticket。
"""

REFUND_SPECIALIZED_INSTRUCTIONS = """【退款退货/售后专注裁决专项指令】
1. 核心任务：结合上述订单真实状态与官方政策条款，专注判定「该订单是否支持退款/退换货/售后」；
2. 结论与依据：给出明确结论（支持退款或不支持退款）及判断依据（如时效、运费承担方、品类免责约束）；
3. 严禁重复查单：严禁重复发起 query_order 等冗余工具调用（订单数据已在上下文完整提供）；
4. 严禁冗长追问：严禁在对话中反复追问用户复杂的退款原因（用户后续提交申请时从固定类目自选）；
5. 引导退款动作：若判定支持退款，明确提示用户「可点击下方“申请退款”按钮提交退款申请单」。"""

_INITIAL_MAX_STEPS: int = max(1, int(getattr(settings, "max_agent_steps", 5)))
MAX_STEPS: int = _INITIAL_MAX_STEPS


def _resolve_max_steps() -> int:
    """获取当前生效的 ReAct 最大步数：支持模块级 MAX_STEPS 覆盖与 settings.max_agent_steps 动态读取。"""
    if MAX_STEPS != _INITIAL_MAX_STEPS:
        return max(1, int(MAX_STEPS))
    return max(1, int(getattr(settings, "max_agent_steps", MAX_STEPS)))


def format_order_context(order_data: Dict[str, Any]) -> str:
    """将订单数据规范化格式化为系统提示词中的订单真实状态卡片"""
    oid = order_data.get("order_id") or order_data.get("订单编号") or "未知"
    status = order_data.get("订单状态") or order_data.get("status") or "未知"
    amount = order_data.get("支付金额") or order_data.get("amount") or order_data.get("price") or "未知"
    order_time = order_data.get("下单时间") or order_data.get("order_time") or "未知"

    items = order_data.get("商品明细") or order_data.get("items") or []
    items_desc = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                name = item.get("商品名称") or item.get("product_name") or item.get("name") or ""
                qty = item.get("数量") or item.get("count") or item.get("quantity") or 1
                price = item.get("单价") or item.get("price") or ""
                part = f"{name}"
                details = []
                if qty:
                    details.append(f"数量: {qty}")
                if price:
                    details.append(f"单价: {price}")
                if details:
                    part += f" ({', '.join(details)})"
                items_desc.append(part)
            elif isinstance(item, str):
                items_desc.append(item)
    elif isinstance(items, str):
        items_desc.append(items)

    items_str = ", ".join(items_desc) if items_desc else "无明细"

    lines = [
        "【当前订单真实状态】",
        f"- 订单编号: {oid}",
        f"- 订单状态: {status}",
        f"- 支付金额: {amount}",
        f"- 下单时间: {order_time}",
        f"- 商品明细: {items_str}",
    ]
    return "\n".join(lines)


def is_refund_approved(response_text: str) -> bool:
    """判定大模型回答是否明确支持退款/退货/售后

    采用分层精准判定策略：
    1. 强否定主结论拦截：文本开头或主结论直接拒退（如“**不支持退款**”、“经核实...不支持退款”、“已超出时效”）；
    2. 强正向动作引导：命中引导点击申请退款按钮（专项 Prompt 第 5 条规约：仅在支持退款时触发）；
    3. 强正向主表态：标题或开头明确表态（如“**支持退款**”、“符合7天无理由退换”）；
    4. 逐句语义分析：排除“不属于...等不支持七天特殊品类”等双重否定与免责举例的全局子串误伤；
    5. 兜底词库降级匹配。
    """
    if not response_text:
        return False
    text = response_text.strip()

    # 1. 强否定主结论拦截：如果文本明确以拒退为开门见山的结论，判定为不支持
    # 预先剥离“不属于/并非...不支持退货的特殊品类”等双重否定免责从句，防止主结论正则误杀
    primary_check_text = re.sub(
        r"(?:不属于|并非)[^，,。！!\n；;]*?(?:不支持|不可|无法)[^，,。！!\n；;]*?(?:特殊品类|品类|商品|范围|情形)",
        "【豁免品类】",
        text,
    )
    primary_check_text = re.sub(
        r"(?:不属于|并非)\s*(?:不支持|不可|无法)(?:办理|申请)?(?:退款|退货|退换)",
        "【豁免】",
        primary_check_text,
    )
    primary_rejection_patterns = [
        r"^\s*[*#\s]*不支持(?:申请|办理)?(?:退款|退货|退换|售后)",
        r"^\s*[*#\s]*(?:无法|不能|不可|不予)(?:办理|申请)?(?:退款|退货|退换)",
        r"^\s*[*#\s]*不符合(?:退款|退货|退换|售后|7天|七天)",
        r"(?:经核实|经核对|很抱歉|抱歉|非常抱歉)[^。\n]*?(?:不支持|无法|不能|不可)(?:办理|申请)?(?:退款|退货|退换)",
        r"(?:已超出|已超过|超过)(?:7天|七天|15天|十五天|售后|退货|退换)[^。\n]*?(?:不支持|无法|不能|不可)(?:退款|退货|退换)",
    ]
    for pattern in primary_rejection_patterns:
        if re.search(pattern, primary_check_text):
            return False

    # 2. 强正向动作引导（Prompt 第 5 条专项指令引导词）：
    # “若判定支持退款，明确提示用户「可点击下方“申请退款”按钮提交退款申请单」”
    action_prompt_pattern = r"(?:点击|通过|在)下方.*?(?:“|\"|”)?申请退款(?:”|\"|')?.*?按钮|开启(?:了)?退款申请|点击下方.*?申请退款"
    if re.search(action_prompt_pattern, text):
        return True

    # 3. 强正向总结论判断（开头、标题或第一句明确表态支持）
    primary_approval_patterns = [
        r"^\s*[*#\s]*支持(?:办理)?(?:退款|退货|退换|售后)",
        r"^\s*[*#\s]*(?:可以|可|允许)(?:办理|申请)?(?:退款|退货|退换)",
        r"^\s*[*#\s]*符合(?:7天|七天)?无理由(?:退货|退换)?",
        r"^\s*[*#\s]*符合(?:退款|退货|退换)条件",
    ]
    for pattern in primary_approval_patterns:
        if re.search(pattern, text):
            return True

    # 4. 逐句语义分析（过滤掉说明商品不属于不支持特殊品类的双重否定句）
    sentences = re.split(r"[。！!\n；;]+", text)
    has_pos_sentence = False
    has_neg_sentence = False

    order_neg_patterns = [
        r"(?:本商品|该商品|该订单|您的订单|订单\s*\d+)[^，,。]*?(?:不支持|无法|不能|不可|不符合)(?:办理|申请)?(?:退款|退货|退换)",
        r"(?:不支持|无法|不能|不可)(?:办理|申请)?(?:退款|退货|退换)",
        r"超出(?:退货|退换|售后)时效",
        r"超过(?:7天|七天|15天|十五天)",
    ]

    order_pos_patterns = [
        r"(?:支持|可以|可|允许)(?:办理|申请)?(?:退款|退货|退换)",
        r"支持(?:7天|七天)",
        r"符合(?:7天|七天)?无理由(?:退款|退货|退换)",
        r"符合(?:退款|退货|退换)条件",
        r"在(?:7天|七天)无理由退货期内",
        r"在(?:7天|七天)内",
    ]

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # 排除“不属于...等不支持七天无理由退货的特殊品类”等双重否定与免责举例
        if "不属于" in s and any(k in s for k in ("不支持", "不可", "无法")):
            continue
        if any(k in s for k in ("特殊品类", "除外", "免责", "例如定制", "定制商品")) and ("不属于" in s or "并非" in s):
            continue

        if any(re.search(p, s) for p in order_neg_patterns):
            has_neg_sentence = True

        if any(re.search(p, s) for p in order_pos_patterns):
            if not re.search(r"(?:不|无法|不能|不可|不予)支持(?:7天|七天|退)", s):
                has_pos_sentence = True

    if has_neg_sentence and not has_pos_sentence:
        return False
    if has_pos_sentence and not has_neg_sentence:
        return True

    # 5. 兜底通用词库匹配（保留基础兼容）
    neg_phrases = [
        "不支持退",
        "不支持申请退",
        "无法退",
        "无法办理退",
        "无法申请退",
        "不能退",
        "不可退",
        "不予退",
        "不符合退",
        "拒绝退",
        "超出退货",
        "超出退换",
        "超过退货",
        "超过退换",
        "超出售后",
        "超过售后",
        "不满足退",
        "不满足7天",
        "超过7天",
        "无法支持",
        "不支持该",
    ]
    for neg in neg_phrases:
        if neg in text:
            return False

    pos_phrases = [
        "支持退",
        "可以退",
        "符合退",
        "允许退",
        "申请退款",
        "办理退款",
        "可退",
        "支持7天",
        "支持七天",
    ]
    return any(pos in text for pos in pos_phrases)



async def main_agent_node(
    state: AgentWorkflowState,
    model: Optional[Any] = None,
    tools: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """主力 ReAct Agent 节点：支持工具自适应循环、多步推理、知识上下文注入与 5 步硬截断"""
    llm = model or get_chat_model(streaming=False)
    if tools is not None:
        available_tools = tools
        executor = ToolExecutor(registry=ToolRegistry(tools=tools, mcp_client=None))
    else:
        res = default_tool_registry.get_all_tools()
        if inspect.isawaitable(res):
            available_tools = await res
        else:
            available_tools = res
        executor = default_tool_executor

    from app.services.workflow.nodes.router import _is_legacy_ch05_test
    if _is_legacy_ch05_test():
        from app.tools.business_tools import query_logistics
        if not any(t.name == "query_logistics" for t in available_tools):
            available_tools = list(available_tools) + [query_logistics]
            executor = ToolExecutor(registry=ToolRegistry(tools=available_tools, mcp_client=None))

    # 1. 构造上下文提示词（融合上游知识库证据与订单真实状态）
    sys_content = AGENT_SYSTEM_BASE

    order_data = state.get("order_data")
    if order_data and isinstance(order_data, dict):
        sys_content += "\n" + format_order_context(order_data)
        sys_content += "\n" + REFUND_SPECIALIZED_INSTRUCTIONS

    docs = state.get("retrieved_docs") or []

    # 2. 准备消息列表（优先使用 layer2_messages 与 layer1_messages 并结合 ContextManager.build_model_messages）
    messages: List[BaseMessage]
    if state.get("layer2_messages") is not None or state.get("layer1_messages") is not None:
        l2 = list(state.get("layer2_messages") or [])
        l1 = list(state.get("layer1_messages") or [])
        query = str(state.get("resolved_query") or state.get("input_query") or "").strip()
        # 排除当前提问（若包含在 l1 末尾），避免与 build_model_messages 中的 current_query 重复
        if l1:
            last_m = l1[-1]
            last_content = getattr(last_m, "content", "")
            if str(last_content or "").strip() in (query, str(state.get("input_query") or "").strip()):
                l1 = l1[:-1]

        messages = ContextManager.build_model_messages(
            system_prompt=sys_content,
            layer2_messages=l2,
            layer1_messages=l1,
            current_query=query,
            summary=state.get("summary"),
            retrieved_docs=docs,
        )
    else:
        if docs:
            if order_data and isinstance(order_data, dict):
                # 退款专项场景：取 Top 3~5 篇政策证据，标注【参考政策条款 i】
                policy_texts = []
                for i, doc in enumerate(docs[:5], 1):
                    text = doc.get("text") or doc.get("content") or str(doc)
                    policy_texts.append(f"【参考政策条款 {i}】{text}")
                sys_content += "\n以下为检索到的官方退换货与售后政策条款，请严格遵照执行：\n" + "\n".join(policy_texts)
            else:
                # 普通业务场景：取 Top 3 篇，标注【参考知识 i】
                knowledge_texts = []
                for i, doc in enumerate(docs[:3], 1):
                    text = doc.get("text") or doc.get("content") or str(doc)
                    knowledge_texts.append(f"【参考知识 {i}】{text}")
                sys_content += "\n以下为检索到的官方政策与业务知识，请参考并用于回答：\n" + "\n".join(knowledge_texts)

        messages = [SystemMessage(content=sys_content)]
        existing_messages = state.get("messages") or []
        for m in existing_messages:
            if not isinstance(m, SystemMessage):
                messages.append(m)

    bound_llm = llm.bind_tools(available_tools) if (available_tools and hasattr(llm, "bind_tools")) else llm

    # 3. 记录主力 Agent 模型调用入参可观测日志 [model_ctx]
    conv_id = state.get("conversation_id", 0)
    user_query = str(state.get("resolved_query") or state.get("input_query") or "").strip()
    intent = state.get("intent")
    summary = state.get("summary")
    window_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    log_model_context(
        conv_id=conv_id,
        summary=summary,
        window_messages=window_msgs,
        estimated_tokens=estimate_tokens(messages),
    )

    steps = 0
    effective_max_steps = _resolve_max_steps()
    init_tokens = state.get("token_usage") or {}
    total_tokens = {
        "prompt_tokens": int(init_tokens.get("prompt_tokens", 0)),
        "completion_tokens": int(init_tokens.get("completion_tokens", 0)),
        "total_tokens": int(init_tokens.get("total_tokens", 0)),
    }
    final_text = ""
    new_messages: List[BaseMessage] = []
    current_turn_tool_messages: List[BaseMessage] = []
    history_user_queries = _extract_history_user_queries(state, messages)

    _prune_pending_agent_states()
    cached_pending = _pending_agent_states.pop(conv_id, None)
    if cached_pending is not None:
        cached_query = cached_pending.get("turn_query")
        if cached_query is not None and cached_query != user_query:
            cached_pending = None

    while steps < effective_max_steps:
        steps += 1
        prior_executed_results: Dict[str, ToolMessage] = {}
        if cached_pending is not None:
            resp = cached_pending["resp"]
            steps = cached_pending.get("steps", steps)
            prior_executed_results = dict(cached_pending.get("executed_results") or {})
            messages.append(resp)
            new_messages.append(resp)
            cached_pending = None
        else:
            resp = await bound_llm.ainvoke(messages)
            messages.append(resp)
            new_messages.append(resp)
            _accumulate_tokens(total_tokens, resp)

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            # 自然收敛，完成推演
            final_text = str(resp.content or "")
            break

        # 执行工具并将结果回填（同一 LLM 轮次内 resp 仅记录一次至 current_turn_tool_messages）
        interrupted_break = False
        ticket_confirmed = False
        confirmed_ticket_no: Optional[str] = None
        resp_recorded_in_turn = False
        step_executed_results: Dict[str, ToolMessage] = {}
        step_appended_tool_msgs: List[ToolMessage] = []

        def _record_resp_once() -> None:
            nonlocal resp_recorded_in_turn
            if not resp_recorded_in_turn:
                current_turn_tool_messages.append(resp)
                resp_recorded_in_turn = True

        def _rollback_step_messages() -> None:
            if resp in new_messages:
                new_messages.remove(resp)
            if resp in messages:
                messages.remove(resp)
            if resp in current_turn_tool_messages:
                current_turn_tool_messages.remove(resp)
            for tm in step_appended_tool_msgs:
                if tm in new_messages:
                    new_messages.remove(tm)
                if tm in messages:
                    messages.remove(tm)
                if tm in current_turn_tool_messages:
                    current_turn_tool_messages.remove(tm)

        for tool_call in tool_calls:
            name = tool_call.get("name", "")
            raw_args = tool_call.get("args") or {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except Exception:
                    raw_args = {}
            call_id = str(tool_call.get("id") or "")

            if name == "create_ticket":
                if ticket_confirmed:
                    dup_msg = ToolMessage(
                        content="本轮已处理工单请求，请勿重复创建。",
                        tool_call_id=call_id,
                    )
                    messages.append(dup_msg)
                    new_messages.append(dup_msg)
                    current_turn_tool_messages.append(dup_msg)
                    step_appended_tool_msgs.append(dup_msg)
                    continue

                desc = str(raw_args.get("description", "")).strip() if isinstance(raw_args, dict) else ""
                # 安全兜底：若无问题描述，坚决追问，不发工单，并剔除未执行的 tool_call 消息防止历史悬挂
                if not desc:
                    _rollback_step_messages()
                    final_text = "请问您具体遇到了什么问题？请提供详细情况以便我们为您建立工单联系专员跟进处理。"
                    ask_msg = AIMessage(content=final_text)
                    messages.append(ask_msg)
                    new_messages.append(ask_msg)
                    interrupted_break = True
                    break

                # 挂起前先缓存状态与已执行前置工具结果，防止中断恢复执行时重复调用 LLM 或重复执行工具
                _prune_pending_agent_states()
                _pending_agent_states[conv_id] = {
                    "resp": resp,
                    "steps": steps,
                    "turn_query": user_query,
                    "executed_results": dict(step_executed_results),
                    "created_at": time.monotonic(),
                }
                ticket_type = raw_args.get("ticket_type", "售后") if isinstance(raw_args, dict) else "售后"
                # 调用 LangGraph 原生 interrupt 挂起工作流
                user_action = interrupt({
                    "event_type": "ticket_preview",
                    "ticket_type": ticket_type,
                    "description": desc,
                    "conversation_id": conv_id,
                    "tool_call_id": call_id,
                })
                # 无论恢复何种动作，立即清理挂起缓存，防止状态泄漏到后续对话轮次
                _pending_agent_states.pop(conv_id, None)

                # 恢复执行时分支裁决
                if user_action == "confirm":
                    _record_resp_once()
                    if isinstance(raw_args, dict):
                        raw_args["conversation_id"] = conv_id
                        tool_call["args"] = raw_args
                    tool_result = await executor.execute(
                        tool_call=tool_call,
                        context={
                            "user_query": user_query,
                            "intent": intent,
                            "history_user_queries": history_user_queries,
                            "confirmed": True,
                            "preview_confirmed": True,
                        },
                        conversation_id=conv_id,
                    )
                    tool_output = str(tool_result.get("output") or "")
                    tool_msg = ToolMessage(content=tool_output, tool_call_id=call_id)
                    messages.append(tool_msg)
                    new_messages.append(tool_msg)
                    current_turn_tool_messages.append(tool_msg)
                    step_appended_tool_msgs.append(tool_msg)
                    step_executed_results[call_id] = tool_msg
                    ticket_confirmed = True

                    if tool_result.get("success") is True:
                        try:
                            parsed_t = json.loads(tool_output)
                            if isinstance(parsed_t, dict) and not parsed_t.get("error"):
                                confirmed_ticket_no = parsed_t.get("ticket_no")
                        except Exception:
                            pass
                    # 继续遍历同 resp 中剩余 sibling tool_calls，确保每个 tool_call_id 均配对 ToolMessage

                else:
                    # 取消或非预期动作：统一按取消处理，记录权限拒绝审计并清理悬挂 tool_call
                    if isinstance(raw_args, dict):
                        raw_args["conversation_id"] = conv_id
                        tool_call["args"] = raw_args
                    await executor.execute(
                        tool_call=tool_call,
                        context={
                            "user_query": user_query,
                            "intent": intent,
                            "history_user_queries": history_user_queries,
                            "confirmed": False,
                            "preview_confirmed": False,
                        },
                        conversation_id=conv_id,
                    )
                    _rollback_step_messages()

                    final_text = "已为您取消工单创建。如果您有其他问题，欢迎随时咨询。"
                    cancel_msg = AIMessage(content=final_text)
                    messages.append(cancel_msg)
                    new_messages.append(cancel_msg)
                    current_turn_tool_messages = []
                    interrupted_break = True
                    break

            else:
                # 常规业务工具执行（中断恢复重放时直接复用已执行结果，严禁二次执行）
                _record_resp_once()
                if call_id and call_id in prior_executed_results:
                    tool_msg = prior_executed_results[call_id]
                else:
                    tool_result = await executor.execute(
                        tool_call=tool_call,
                        context={
                            "user_query": user_query,
                            "intent": intent,
                        },
                        conversation_id=conv_id,
                    )
                    output = str(tool_result.get("output") or "")
                    tool_msg = ToolMessage(content=output, tool_call_id=call_id)
                messages.append(tool_msg)
                new_messages.append(tool_msg)
                current_turn_tool_messages.append(tool_msg)
                step_appended_tool_msgs.append(tool_msg)
                if call_id:
                    step_executed_results[call_id] = tool_msg

        if ticket_confirmed:
            try:
                resp2 = await bound_llm.ainvoke(messages)
                _accumulate_tokens(total_tokens, resp2)
                final_text = str(resp2.content or "")
                new_messages.append(resp2)
            except Exception as e:
                logger.warning(f"LLM summary after ticket creation: {e}")
                final_text = ""

            if confirmed_ticket_no and (not final_text or confirmed_ticket_no not in final_text):
                if not final_text:
                    final_text = f"已为您成功创建工单，工单编号为：{confirmed_ticket_no}，人工客服专员将尽快为您跟进处理。"
                else:
                    final_text = f"已为您成功创建工单，工单编号为：{confirmed_ticket_no}。{final_text}"
            elif not final_text:
                final_text = "抱歉，工单创建失败，请稍后重试或联系人工客服专员协助处理。"
            interrupted_break = True

        if interrupted_break:
            break

    if not final_text:
        # 步数超限，执行总结
        final_summary = await llm.ainvoke(
            messages + [HumanMessage(content="请根据已有工具查询的信息，立即给出最终结论。")]
        )
        final_text = str(final_summary.content or "")
        new_messages.append(final_summary)
        _accumulate_tokens(total_tokens, final_summary)

    # 动作推荐下发 (suggested_actions)
    suggested_actions = list(state.get("suggested_actions") or [])
    is_refund_scenario = bool(order_data or (state.get("intent") in ("退款退货", "售后", "refund")))
    if is_refund_scenario:
        if is_refund_approved(final_text):
            if "apply_refund" not in suggested_actions:
                suggested_actions.append("apply_refund")
        else:
            if "apply_refund" in suggested_actions:
                suggested_actions.remove("apply_refund")


    return {
        "response_text": final_text,
        "messages": new_messages,
        "current_turn_tool_messages": current_turn_tool_messages,
        "steps_taken": steps,
        "token_usage": total_tokens,
        "suggested_actions": suggested_actions,
        "status": "success",
    }
