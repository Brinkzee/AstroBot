from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

class AgentWorkflowState(TypedDict):
    conversation_id: int
    user_id: str
    input_query: str
    resolved_query: str
    intent: Optional[str]
    confidence: Optional[float]
    intent_reason: Optional[str]
    order_id: Optional[str]
    order_data: Optional[Dict[str, Any]]
    suggested_orders: Optional[List[Dict[str, Any]]]
    retrieved_docs: List[Dict[str, Any]]
    confidence_passed: Optional[bool]
    messages: Annotated[List[BaseMessage], add_messages]
    response_text: str
    suggested_actions: List[str]
    token_usage: Dict[str, int]
    steps_taken: int
    status: str
    summary: Optional[str]
    layer2_messages: Optional[List[BaseMessage]]
    layer1_messages: Optional[List[BaseMessage]]
    current_turn_tool_messages: Optional[List[BaseMessage]]

def create_initial_state(
    conversation_id: int,
    query: str,
    user_id: str = "default_user",
    summary: Optional[str] = None,
    layer2_messages: Optional[List[BaseMessage]] = None,
    layer1_messages: Optional[List[BaseMessage]] = None,
) -> AgentWorkflowState:
    """构建单轮工作流启动初始 State"""
    clean_query = str(query or "").strip()
    return {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "input_query": clean_query,
        "resolved_query": clean_query,
        "intent": None,
        "confidence": None,
        "intent_reason": None,
        "order_id": None,
        "order_data": None,
        "suggested_orders": None,
        "retrieved_docs": [],
        "confidence_passed": None,
        "messages": [HumanMessage(content=clean_query)],
        "response_text": "",
        "suggested_actions": [],
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "steps_taken": 0,
        "status": "initialized",
        "summary": summary,
        "layer2_messages": layer2_messages,
        "layer1_messages": layer1_messages,
        "current_turn_tool_messages": [],
    }
