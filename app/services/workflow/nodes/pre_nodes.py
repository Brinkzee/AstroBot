import json
import logging
from typing import Any, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from app.llm import get_chat_model
from app.services.workflow.state import AgentWorkflowState

logger = logging.getLogger(__name__)

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

def coreference_resolution_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """指代消解节点：本章最简版，原样透传用户输入"""
    return {
        "resolved_query": state.get("input_query", ""),
    }

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
