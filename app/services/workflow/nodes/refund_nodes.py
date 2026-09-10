import json
import logging
import re
from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.services.workflow.state import AgentWorkflowState
from app.tools.business_tools import MOCK_ORDERS, query_order

logger = logging.getLogger(__name__)

REFUND_SELECT_ORDER_PROMPT = "为您查询退款政策前，请先选择您需要咨询的订单："

ORDER_PATTERNS = [
    re.compile(r'(?:退款)?(?:订单|单号)(?:编号|号)?(?:是|为)?[:：\s]*([A-Za-z0-9_-]+)'),
    re.compile(r'(?i)(?<![0-9a-zA-Z])(TB\d+)(?![0-9a-zA-Z])'),
    re.compile(r'(?<!\d)(100[1-3])(?!\d)'),
    re.compile(r'(?<!\d)(\d{4,})(?!\d|[年月号天分秒点时元块角毛件个支本双条台斤两克米折倍])'),
]


class _AwaitableDict(dict):
    """支持直接同步作为字典读取，也可直接 await 返回字典自身，保证双重调用兼容性"""

    def __await__(self):
        async def _wrap():
            return self

        return _wrap().__await__()


def _extract_from_text(text: str) -> Optional[str]:
    """从单段文本中按正则提取订单号"""
    if not text or not isinstance(text, str):
        return None
    clean_text = text.strip()
    for pat in ORDER_PATTERNS:
        m = pat.search(clean_text)
        if m:
            val = m.group(1).strip()
            if val:
                return val
    return None


def extract_order_id(
    query_or_state: Optional[Any] = None,
    *,
    query: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """从状态、提问或消息历史中提取订单号。

    提取优先级策略：
    1. state.get("order_id") (若已存在且有效，直接使用)
    2. state.get("resolved_query") / state.get("input_query") / query (正则提取)
    3. state.get("messages") 倒序追溯用户历史提问
    """
    actual_state: Optional[Dict[str, Any]] = state
    actual_query: Optional[str] = query

    if isinstance(query_or_state, dict):
        if actual_state is None:
            actual_state = query_or_state
    elif isinstance(query_or_state, str):
        if actual_query is None:
            actual_query = query_or_state

    # 优先级 1: 如果 state.get("order_id") 已经存在，直接使用
    if actual_state and actual_state.get("order_id"):
        oid = str(actual_state["order_id"]).strip()
        if oid and oid.lower() != "none":
            return oid

    # 优先级 2: 从 resolved_query 以及 input_query / query 中使用正则提取
    candidate_texts: List[str] = []
    if actual_state:
        rq = actual_state.get("resolved_query")
        if rq:
            candidate_texts.append(str(rq).strip())
        iq = actual_state.get("input_query")
        if iq and str(iq).strip() not in candidate_texts:
            candidate_texts.append(str(iq).strip())
    if actual_query and str(actual_query).strip() not in candidate_texts:
        candidate_texts.append(str(actual_query).strip())

    for text in candidate_texts:
        oid = _extract_from_text(text)
        if oid:
            return oid

    # 优先级 3: 从 messages 倒序检查是否有用户历史中明确提供的订单号
    if actual_state and actual_state.get("messages"):
        messages = actual_state.get("messages") or []
        for msg in reversed(messages):
            # 仅检查来自用户的消息，跳过 AI/Assistant 提示消息
            is_ai = False
            if isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai":
                is_ai = True
            elif isinstance(msg, dict) and msg.get("role") in ("assistant", "ai"):
                is_ai = True

            if is_ai:
                continue

            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content", "")
            oid = _extract_from_text(str(content or ""))
            if oid:
                return oid

    return None


def fetch_order_data(order_id: str) -> Dict[str, Any]:
    """获取订单详细数据，优先从 MOCK_ORDERS 读取，未命中则通过 query_order 获取或生成标注文档"""
    key = str(order_id).strip()
    if key in MOCK_ORDERS:
        return dict(MOCK_ORDERS[key])

    try:
        raw = query_order.invoke({"order_id": key})
        if isinstance(raw, str):
            return json.loads(raw)
        elif isinstance(raw, dict):
            return raw
    except Exception as e:
        logger.warning(f"调用 query_order 获取订单 {key} 失败，使用兜底结构: {e}")

    return {
        "order_id": key,
        "订单状态": "已发货",
        "支付金额": "199.00元",
        "商品明细": [
            {"商品名称": "热销精选商品", "数量": 1, "单价": "199.00元"}
        ],
        "下单时间": "2026-09-05 12:00:00",
    }


def get_suggested_orders() -> List[Dict[str, Any]]:
    """装配 Mock 候选订单卡片数据列表（包含 1001、1002、1003 的订单号、商品名称、金额、状态等展示字段）"""
    candidate_orders = []
    for oid, item in MOCK_ORDERS.items():
        first_item = item.get("商品明细", [{}])[0] if item.get("商品明细") else {}
        prod_name = first_item.get("商品名称", "")
        amount = item.get("支付金额", "")
        status = item.get("订单状态", "")
        order_time = item.get("下单时间", "")
        candidate_orders.append({
            "order_id": oid,
            "product_name": prod_name,
            "商品名称": prod_name,
            "amount": amount,
            "price": amount,
            "支付金额": amount,
            "status": status,
            "订单状态": status,
            "order_time": order_time,
            "下单时间": order_time,
            "items": list(item.get("商品明细", [])),
            "商品明细": list(item.get("商品明细", [])),
        })
    return candidate_orders


def emit_order_selector_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """缺单时订单卡片装配与挂起引导节点"""
    candidate_orders = get_suggested_orders()
    return _AwaitableDict({
        "order_id": None,
        "order_data": None,
        "suggested_orders": candidate_orders,
        "response_text": REFUND_SELECT_ORDER_PROMPT,
        "status": "need_order_selection",
    })


def refund_order_check_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """确定性退款子流程订单检查节点：
    1. 提取订单号；
    2. 命中有效订单号时预取订单数据，返回 status="order_fetched"；
    3. 缺失订单号时挂起拦截，装配 suggested_orders 候选卡片并引导用户选择，返回 status="need_order_selection"。
    """
    order_id = extract_order_id(state=state)
    if order_id:
        order_info = fetch_order_data(order_id)
        return _AwaitableDict({
            "order_id": order_id,
            "order_data": order_info,
            "status": "order_fetched",
        })
    return emit_order_selector_node(state)
