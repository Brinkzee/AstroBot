import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from app.llm import get_chat_model
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
    if not query:
        return _AwaitableDict({"resolved_query": ""})

    history_text = _format_conversation_history(state.get("messages") or [], query)

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

VALID_INTENTS = {"物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"}

INTENT_SYSTEM_PROMPT = """你是一个电商客服意图识别专家。
请将用户的输入严格分类为以下七类之一：
1. 物流：询问包裹轨迹、发货进度、快递单号等
2. 订单：询问订单详情、金额、明细、下单时间等
3. 商品咨询：询问商品价格、规格、尺码、库存等
4. 退款退货：询问退款政策、退换货流程、运费规则等
5. 售后：质保维修、商品损坏等售后支持
6. 投诉：对服务态度不满、虚假宣传、强烈抗议、明确表示要投诉等
7. 闲聊：打招呼、问候、感谢、非业务寒暄等

请严格返回如下 JSON 格式，不要返回任何额外文字：
{"intent": "分类名称", "reason": "判定依据简述"}
"""

CHITCHAT_FIXED_TEXT = "您好！我是您的智能客服助手，请问有什么可以帮您？如果您需要查询订单、追踪物流或咨询售后退换货政策，随时发给我哦~"

COMPLAINT_SOOTHING_TEXT = "非常抱歉给您带来了不好的体验，请您消消气。我们非常重视您的反馈与诉求！您可以点击下方选项转接人工客服实时沟通，或直接提交人工客服工单，我们将由专人第一时间跟进为您处理。"

async def intent_recognition_node(
    state: AgentWorkflowState,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """意图识别节点：单次轻量 Prompt 判定 7 分类，输出 JSON"""
    query = state.get("resolved_query") or state.get("input_query", "")
    llm = model or get_chat_model(streaming=False)
    
    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=f"用户提问：{query}"),
    ]
    try:
        resp = await llm.ainvoke(messages)
        content = str(resp.content or "").strip()
        # 清除可能的 markdown 代码块标识
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content.strip())
        raw_intent = str(data.get("intent", "")).strip()
        intent = raw_intent if raw_intent in VALID_INTENTS else "售后"
        reason = str(data.get("reason", ""))
    except Exception as e:
        logger.warning(f"意图识别 JSON 解析异常，自动降级为业务数据类-售后: {e}")
        intent = "售后"
        reason = f"解析兜底: {str(e)}"

    return {
        "intent": intent,
        "intent_reason": reason,
    }

def chitchat_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """闲聊节点：直接返回预设固定亲和话术，不花大模型调用"""
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
