import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from app.llm import get_chat_model
from app.services.context.logger import log_history_context
from app.services.context.manager import ContextManager
from app.services.workflow.state import AgentWorkflowState

logger = logging.getLogger(__name__)

COREFERENCE_REWRITE_SYSTEM_PROMPT = """你是一名电商客服语义改写与指代消解专家。
请根据提供的对话历史（若有）和当前用户提问，对提问进行规范化改写或原样返回。

你的职责：
1. 指代消解：如果用户的提问包含代词（如「它」、「这个」、「那件商品」）或省略主语（如「能退吗」、「到哪了」），必须结合对话历史补全为语义完整、独立可理解的提问（例如将「它能退吗」结合前文补全为「订单1001极简保暖羽绒服支持退货吗」）；
2. 口语归一化：将口语化、含混或方言问法（如「货走到哪了老铁」）归一化为标准的业务提问（如「查询订单物流最新轨迹」）；
3. 原样透传保护 (Hard Gate)：如果用户的提问本身已经语义完整、指代明确、无歧义（例如「查询订单1001」、「什么是七天无理由退货」），或者仅是打招呼/寒暄（如「你好」、「在吗」），必须直接原样返回，严禁过度改写或擅自添加多余修饰！

输出格式要求：
直接返回补全/改写后的单行问题文本，严禁包含任何多余解释、前后缀、引号或标记。"""

GREETINGS = {
    "你好", "您好", "在吗", "在不在", "有人吗", "哈喽", "hello", "hi", "早",
    "早上好", "下午好", "晚上好", "谢谢", "多谢", "感谢", "再见", "拜拜"
}

COLLOQUIAL_OR_INCOMPLETE_PATTERNS = [
    "老铁", "咋样", "走哪了", "到哪了", "算求", "这玩意", "这东西", "咋回事", "咋办",
    "能退吗", "发货没", "发了吗", "它", "这个", "那件", "前一个", "刚才那个", "那单"
]


def _is_pure_greeting(query: str) -> bool:
    """判断是否为纯寒暄、打招呼或礼貌用语"""
    q = query.strip().rstrip("!！?？~～呀啊吧呢哦")
    if q in GREETINGS:
        return True
    return any(query.startswith(g) and len(query) <= len(g) + 2 for g in ["你好", "您好", "早安", "晚安", "哈喽", "嗨"])


def _needs_rewrite_without_history(query: str) -> bool:
    """在无历史对话时，判断是否包含口语化或明显缺失主语的表达需要模型介入归一化"""
    if _is_pure_greeting(query):
        return False
    return any(pattern in query for pattern in COLLOQUIAL_OR_INCOMPLETE_PATTERNS)


def _format_conversation_history(messages: List[Any], current_query: str) -> str:
    """提取会话历史，排除当前轮次并格式化为多轮人机对话文本"""
    if not messages:
        return ""

    valid_msgs = list(messages)
    # 如果最后一条消息是当前用户的提问，排除之（只保留前序历史轮次）
    if valid_msgs:
        last_m = valid_msgs[-1]
        last_content = getattr(last_m, "content", None)
        if last_content is None and isinstance(last_m, dict):
            last_content = last_m.get("content", "")
        if str(last_content or "").strip() == current_query.strip():
            valid_msgs = valid_msgs[:-1]

    # 保留最近 8 条消息 (约 4 轮对话)
    valid_msgs = valid_msgs[-8:]

    lines = []
    for m in valid_msgs:
        content = getattr(m, "content", None)
        if content is None and isinstance(m, dict):
            content = m.get("content", "")
        content = str(content or "").strip()
        if not content:
            continue

        role = "用户"
        if isinstance(m, AIMessage) or (isinstance(m, dict) and m.get("role") in ("assistant", "ai")):
            role = "客服"
        elif isinstance(m, HumanMessage) or (isinstance(m, dict) and m.get("role") in ("user", "human")):
            role = "用户"
        elif isinstance(m, SystemMessage) or (isinstance(m, dict) and m.get("role") == "system"):
            continue
        elif hasattr(m, "type"):
            role = "客服" if m.type == "ai" else "用户"

        lines.append(f"{role}: {content}")

    return "\n".join(lines).strip()


def _clean_resolved_query(text: str) -> str:
    """清洗大模型改写输出，剥离多余引号、markdown 与前缀"""
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    if (cleaned.startswith('“') and cleaned.endswith('”')) or (cleaned.startswith('‘') and cleaned.endswith('’')):
        cleaned = cleaned[1:-1].strip()
    for prefix in ["改写后的提问：", "改写后的问题：", "改写后：", "改写结果：", "规范化提问：", "输出：", "改写："]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned.strip()


class _AwaitableDict(dict):
    """同时兼容同步字典访问与异步 await 执行的结果字典容器"""
    def __init__(self, data: Dict[str, Any], async_runner=None):
        super().__init__(data)
        self._async_runner = async_runner

    def __await__(self):
        if self._async_runner:
            async def _wrap():
                res = await self._async_runner()
                self.update(res)
                return self
            return _wrap().__await__()
        async def _wrap():
            return self
        return _wrap().__await__()


def coreference_resolution_node(
    state: AgentWorkflowState,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """指代消解与口语归一化节点：结合多轮历史进行代词消除与口语标准化，提供透传保护与异常兜底"""
    query = str(state.get("input_query") or "").strip()
    conv_id = state.get("conversation_id", 0)
    summary = state.get("summary")
    messages = state.get("messages") or []

    # 提取历史滑窗用于可观测日志记录与语义改写
    if state.get("layer2_messages") is not None or state.get("layer1_messages") is not None:
        l2 = list(state.get("layer2_messages") or [])
        l1 = list(state.get("layer1_messages") or [])
        if l1:
            last_m = l1[-1]
            last_content = getattr(last_m, "content", "")
            if str(last_content or "").strip() == query:
                l1 = l1[:-1]
        summary_line, sliding_window_text = ContextManager.build_history_context_text(
            layer2_messages=l2,
            layer1_messages=l1,
            summary=summary,
        )
        valid_history_msgs = l2 + l1
        history_text = sliding_window_text
    else:
        valid_history_msgs = list(messages)
        if valid_history_msgs:
            last_m = valid_history_msgs[-1]
            last_content = getattr(last_m, "content", None)
            if last_content is None and isinstance(last_m, dict):
                last_content = last_m.get("content", "")
            if str(last_content or "").strip() == query:
                valid_history_msgs = valid_history_msgs[:-1]
        valid_history_msgs = valid_history_msgs[-8:]
        summary_line = summary.strip().splitlines()[0].strip() if (summary and summary.strip()) else ""
        history_text = _format_conversation_history(state.get("messages") or [], query)

    # 每轮必打：在任何分流和透传分支之前输出 [history_ctx] 到 log/app.log
    log_history_context(
        conv_id=conv_id,
        summary_line=summary_line,
        window_messages=valid_history_msgs,
    )

    if not query:
        return _AwaitableDict({"resolved_query": ""})

    # 1. 寒暄打招呼且无历史，直接原样透传，无需调用大模型 (Hard Gate)
    if not history_text and _is_pure_greeting(query):
        return _AwaitableDict({"resolved_query": query})

    # 2. 如果未显式传入模型（工作流默认运行模式），且无历史上下文、输入已完整，直接原样透传 (Hard Gate)
    if model is None and not history_text and not _needs_rewrite_without_history(query):
        return _AwaitableDict({"resolved_query": query})

    async def _async_runner() -> Dict[str, Any]:
        llm = model or get_chat_model(streaming=False)
        if history_text:
            prompt_content = f"【对话历史】\n{history_text}\n\n【当前用户提问】\n{query}\n\n请直接输出指代消解或规范化后的提问（如无需修改则原样返回）："
        else:
            prompt_content = f"【当前用户提问】\n{query}\n\n请直接输出指代消解或规范化后的提问（如无需修改则原样返回）："

        messages = [
            SystemMessage(content=COREFERENCE_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content),
        ]
        try:
            resp = await llm.ainvoke(messages)
            resolved = _clean_resolved_query(str(resp.content or ""))
            if not resolved:
                resolved = query
            return {"resolved_query": resolved}
        except Exception as e:
            logger.warning(f"指代消解节点执行异常，降级为原样透传 input_query: {e}")
            return {"resolved_query": query}

    return _AwaitableDict({"resolved_query": query}, async_runner=_async_runner)


coreference_rewrite_node = coreference_resolution_node

VALID_INTENTS = {"物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "人工"}
ALL_INTENTS = VALID_INTENTS | {"其他"}

INTENT_FOUR_ELEMENTS_PROMPT = """你是一个电商客服意图识别专家。
请将用户的输入严格分类为以下 9 种意图之一，并评估你的判定置信度（0.0 ~ 1.0 之间的浮点数）。

【要素 1: 9 分类枚举说明】
1. 物流：包裹轨迹、发货时效、快递单号、派送进度等
2. 订单：订单详情、金额、明细、下单时间、订单修改等
3. 商品咨询：商品价格、规格、尺码、库存、功能等咨询
4. 退款退货：退货政策、退款流程、运费承担、是否能退等
5. 售后：质保维修、商品损坏、少件漏发补寄等售后技术服务
6. 投诉：对服务态度不满、虚假宣传、强烈抗议、明确表示要投诉等
7. 人工：明确要求转人工、建工单或者找客服专员跟进的（无论是否有具体问题，只要提到[转人工/建工单]诉求就归为此类意图）
8. 闲聊：打招呼、问候、感谢、非业务日常寒暄等
9. 其他：非电商商城业务问题（例如询问天气、算算术、讲故事、写诗）、语义含混无法辨识的怪问题

【要素 2: 强制 JSON 输出格式】
请严格返回如下 JSON 格式，不要返回任何额外文字、解释或代码标记：
{"intent": "分类名称", "confidence": 0.95, "reason": "判定依据简述"}

【要素 3: 边界 Few-shot 样例】
- 用户："羽绒服拉链坏了怎么修" -> {"intent": "售后", "confidence": 0.95, "reason": "商品拉链损坏维修质保属于售后"}
- 用户："衣服收到了码数小了想退掉" -> {"intent": "退款退货", "confidence": 0.98, "reason": "尺码不合申请退货退款"}
- 用户："我的快递到哪了赶紧催催" -> {"intent": "物流", "confidence": 0.95, "reason": "催促快递进度属于物流轨迹"}
- 用户："退款怎么还没到账" -> {"intent": "退款退货", "confidence": 0.96, "reason": "催退款进度属于退款流程"}
- 用户："帮我转人工客服" -> {"intent": "人工", "confidence": 0.99, "reason": "明确要求转接人工客服"}
- 用户："羽绒服拉链坏了，帮我建个工单" -> {"intent": "人工", "confidence": 0.98, "reason": "明确要求建立工单"}
- 用户："我要建个工单让专员联系我" -> {"intent": "人工", "confidence": 0.95, "reason": "要求建工单并由专员跟进"}
- 用户："今天北京天气怎么样" -> {"intent": "其他", "confidence": 0.99, "reason": "天气等非电商商城业务问题"}
- 用户："你们什么态度，我要投诉你们店铺" -> {"intent": "投诉", "confidence": 0.98, "reason": "对服务态度强烈抗议明确要求投诉"}

【要素 4: 「其他」兜底保护规则】
拿不准、信息缺失无法判断、或超出上述 8 类电商业务领域的怪问题，一律判定为「其他」，置信度依语义评估，绝不硬塞进业务意图！"""

INTENT_SYSTEM_PROMPT = INTENT_FOUR_ELEMENTS_PROMPT

CHITCHAT_FIXED_TEXT = "您好！我是您的智能客服助手，请问有什么可以帮您？如果您需要查询订单、追踪物流或咨询售后退换货政策，随时发给我哦~"

COMPLAINT_SOOTHING_TEXT = "非常抱歉给您带来了不好的体验，请您消消气。我们非常重视您的反馈与诉求！您可以点击下方选项转接人工客服实时沟通，或直接提交人工客服工单，我们将由专人第一时间跟进为您处理。"

OTHER_FALLBACK_TEXT = "您好，我是电商智能客服助手，专注于为您解答商品咨询、查询订单物流、办理售后与退换货等商城业务问题。关于您刚才提到的问题已超出我的业务范围，如需其他协助可选择转接人工客服。"


def _clean_json_markdown(content: str) -> str:
    """清除 LLM 输出中可能包含的 markdown 代码块标记"""
    text = content.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_intent_payload(content: str) -> Dict[str, Any]:
    """解析大模型返回的 JSON 意图及置信度 payload，具备异常兜底能力"""
    text = _clean_json_markdown(content)
    data = json.loads(text)
    raw_intent = str(data.get("intent", "")).strip()
    if raw_intent in ALL_INTENTS:
        intent = raw_intent
    else:
        intent = "其他"

    raw_conf = data.get("confidence")
    if raw_conf is None:
        confidence = 1.0
    else:
        confidence = float(raw_conf)

    reason = str(data.get("reason") or data.get("intent_reason") or "")
    return {
        "intent": intent,
        "confidence": confidence,
        "intent_reason": reason,
    }


async def _invoke_intent_model(target_model: Any, query: str) -> Dict[str, Any]:
    """调用单个意图识别模型并返回标准化意图字典，解析失败优雅降级为「其他」"""
    messages = [
        SystemMessage(content=INTENT_FOUR_ELEMENTS_PROMPT),
        HumanMessage(content=f"用户提问：{query}"),
    ]
    try:
        resp = await target_model.ainvoke(messages)
        content = str(resp.content or "").strip()
        return _parse_intent_payload(content)
    except Exception as e:
        logger.warning(f"意图识别模型调用或 JSON 解析异常，自动降级为「其他」: {e}")
        return {
            "intent": "其他",
            "confidence": 0.0,
            "intent_reason": "解析异常降级",
        }


async def intent_recognition_node(
    state: AgentWorkflowState,
    model: Optional[Any] = None,
    small_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """意图识别节点：严格执行 Prompt 四件套与置信度双层模型级联降级策略"""
    query = state.get("resolved_query") or state.get("input_query", "")

    # 策略 1: 若未传入小模型，直接使用主力大模型判定
    if small_model is None:
        primary_model = model or get_chat_model(streaming=False)
        return await _invoke_intent_model(primary_model, query)

    # 策略 2: 传入小模型时，先由小模型轻量判定
    small_result = await _invoke_intent_model(small_model, query)

    # 若置信度 >= 0.85 且意图属于 8 类常规业务之一（非「其他」），直接采纳
    if small_result["confidence"] >= 0.85 and small_result["intent"] in VALID_INTENTS:
        return small_result

    # 若置信度 < 0.85 或分类落入「其他」，触发级联重评：调用主力大模型重新判定
    primary_model = model or get_chat_model(streaming=False)
    return await _invoke_intent_model(primary_model, query)


def chitchat_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """闲聊节点：直接返回预设固定亲和话术，不消耗模型调用"""
    return {
        "response_text": CHITCHAT_FIXED_TEXT,
        "suggested_actions": [],
        "status": "chitchat",
    }


def complaint_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """投诉节点：安抚话术 + 推荐双按钮，不进 Agent，后端不自动执行动作"""
    return {
        "response_text": COMPLAINT_SOOTHING_TEXT,
        "suggested_actions": ["transfer_agent", "create_ticket"],
        "status": "complaint",
    }


def other_fallback_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """兜底节点：超出电商业务领域或怪问题的统一兜底话术与转人工建议动作"""
    return {
        "response_text": OTHER_FALLBACK_TEXT,
        "suggested_actions": ["transfer_agent"],
        "status": "other_fallback",
    }
