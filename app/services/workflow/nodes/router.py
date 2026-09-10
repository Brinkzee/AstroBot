from app.services.workflow.state import AgentWorkflowState

def route_by_intent(state: AgentWorkflowState) -> str:
    """分流路由条件边：将 7 类意图硬编码映射至 4 大出口"""
    intent = state.get("intent")
    if intent == "闲聊":
        return "chitchat"
    elif intent == "投诉":
        return "complaint"
    elif intent in ("商品咨询", "退款退货"):
        return "knowledge"
    elif intent in ("物流", "订单", "售后"):
        return "business_data"
    else:
        # 未知意图默认进业务数据类交主力 Agent 自由决策
        return "business_data"
