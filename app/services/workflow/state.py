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
    intent_reason: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    confidence_passed: Optional[bool]
    messages: Annotated[List[BaseMessage], add_messages]
    response_text: str
    suggested_actions: List[str]
    token_usage: Dict[str, int]
    steps_taken: int
    status: str

def create_initial_state(
    conversation_id: int,
    query: str,
    user_id: str = "default_user",
) -> AgentWorkflowState:
    """构建单轮工作流启动初始 State"""
    clean_query = str(query or "").strip()
    return {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "input_query": clean_query,
        "resolved_query": clean_query,
        "intent": None,
        "intent_reason": None,
        "retrieved_docs": [],
        "confidence_passed": None,
        "messages": [HumanMessage(content=clean_query)],
        "response_text": "",
        "suggested_actions": [],
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "steps_taken": 0,
        "status": "initialized",
    }
