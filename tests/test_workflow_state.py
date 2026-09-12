import pytest
from typing import get_type_hints
from langchain_core.messages import HumanMessage
from app.services.workflow.state import AgentWorkflowState, create_initial_state

def test_create_initial_state_defaults():
    state = create_initial_state(conversation_id=101, query="我要查物流", user_id="user_1")
    assert state["conversation_id"] == 101
    assert state["user_id"] == "user_1"
    assert state["input_query"] == "我要查物流"
    assert state["resolved_query"] == "我要查物流"
    assert state["intent"] is None
    assert state["confidence"] is None
    assert state["order_id"] is None
    assert state["order_data"] is None
    assert state["suggested_orders"] is None
    assert state["retrieved_docs"] == []
    assert state["confidence_passed"] is None
    assert state["suggested_actions"] == []
    assert state["token_usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert state["steps_taken"] == 0
    assert state["status"] == "initialized"
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)
    assert state["messages"][0].content == "我要查物流"

def test_create_initial_state_default_user_and_strip():
    state = create_initial_state(conversation_id=202, query="  退款申请  ")
    assert state["conversation_id"] == 202
    assert state["user_id"] == "default_user"
    assert state["input_query"] == "退款申请"
    assert state["resolved_query"] == "退款申请"
    assert state["messages"][0].content == "退款申请"

def test_agent_workflow_state_keys():
    expected_keys = {
        "conversation_id",
        "user_id",
        "input_query",
        "resolved_query",
        "intent",
        "confidence",
        "intent_reason",
        "order_id",
        "order_data",
        "suggested_orders",
        "retrieved_docs",
        "confidence_passed",
        "messages",
        "response_text",
        "suggested_actions",
        "token_usage",
        "steps_taken",
        "status",
        "summary",
        "layer2_messages",
        "layer1_messages",
    }
    assert set(AgentWorkflowState.__annotations__.keys()) == expected_keys
