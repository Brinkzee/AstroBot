import inspect
import logging
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.services.workflow.state import AgentWorkflowState, create_initial_state
from app.services.workflow.nodes import pre_nodes
from app.services.workflow.nodes import router
from app.services.workflow.nodes import knowledge_node
from app.services.workflow.nodes import gate
from app.services.workflow.nodes import agent_node

# 保留符号导入以便外部模块引用
from app.services.workflow.nodes.pre_nodes import (
    coreference_resolution_node,
    intent_recognition_node,
    chitchat_node,
    complaint_node,
)
from app.services.workflow.nodes.router import route_by_intent
from app.services.workflow.nodes.knowledge_node import knowledge_retrieval_node, knowledge_fallback_node
from app.services.workflow.nodes.gate import confidence_gate
from app.services.workflow.nodes.agent_node import main_agent_node

logger = logging.getLogger(__name__)


def logging_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """日志与审计节点"""
    logger.info(
        f"[Workflow Audit] ConvID={state.get('conversation_id')} "
        f"Intent={state.get('intent')} Status={state.get('status')} "
        f"Steps={state.get('steps_taken')} Actions={state.get('suggested_actions')}"
    )
    return {}


async def _coreference_node(state: AgentWorkflowState) -> Dict[str, Any]:
    res = pre_nodes.coreference_resolution_node(state)
    if inspect.isawaitable(res):
        return await res
    return res


async def _intent_node(state: AgentWorkflowState) -> Dict[str, Any]:
    res = pre_nodes.intent_recognition_node(state)
    if inspect.isawaitable(res):
        return await res
    return res


async def _chitchat_node(state: AgentWorkflowState) -> Dict[str, Any]:
    res = pre_nodes.chitchat_node(state)
    if inspect.isawaitable(res):
        return await res
    return res


async def _complaint_node(state: AgentWorkflowState) -> Dict[str, Any]:
    res = pre_nodes.complaint_node(state)
    if inspect.isawaitable(res):
        return await res
    return res


async def _knowledge_retrieval_node(state: AgentWorkflowState) -> Dict[str, Any]:
    res = knowledge_node.knowledge_retrieval_node(state)
    if inspect.isawaitable(res):
        return await res
    return res


async def _knowledge_fallback_node(state: AgentWorkflowState) -> Dict[str, Any]:
    res = knowledge_node.knowledge_fallback_node(state)
    if inspect.isawaitable(res):
        return await res
    return res


async def _main_agent_node(state: AgentWorkflowState) -> Dict[str, Any]:
    res = agent_node.main_agent_node(state)
    if inspect.isawaitable(res):
        return await res
    return res


def _route_by_intent(state: AgentWorkflowState) -> str:
    return router.route_by_intent(state)


def _confidence_gate(state: AgentWorkflowState) -> str:
    return gate.confidence_gate(state)


def build_workflow_graph(checkpointer: Optional[Any] = None):
    """装配 LangGraph 确定性工作流图"""
    builder = StateGraph(AgentWorkflowState)

    # 1. 注册所有节点（采用动态转发包装，支持单元测试与热修 Mock）
    builder.add_node("coreference", _coreference_node)
    builder.add_node("intent", _intent_node)
    builder.add_node("chitchat", _chitchat_node)
    builder.add_node("complaint", _complaint_node)
    builder.add_node("knowledge_retrieval", _knowledge_retrieval_node)
    builder.add_node("knowledge_fallback", _knowledge_fallback_node)
    builder.add_node("main_agent", _main_agent_node)
    builder.add_node("logging", logging_node)

    # 2. 基础顺序流
    builder.add_edge(START, "coreference")
    builder.add_edge("coreference", "intent")

    # 3. 四出口意图分流
    builder.add_conditional_edges(
        "intent",
        _route_by_intent,
        {
            "chitchat": "chitchat",
            "complaint": "complaint",
            "knowledge": "knowledge_retrieval",
            "business_data": "main_agent",
        },
    )

    # 4. 知识类置信度闸门分支
    builder.add_conditional_edges(
        "knowledge_retrieval",
        _confidence_gate,
        {
            "pass": "main_agent",
            "fallback": "knowledge_fallback",
        },
    )

    # 5. 各分支汇聚至 logging 节点
    builder.add_edge("chitchat", "logging")
    builder.add_edge("complaint", "logging")
    builder.add_edge("knowledge_fallback", "logging")
    builder.add_edge("main_agent", "logging")
    builder.add_edge("logging", END)

    memory = checkpointer or MemorySaver()
    return builder.compile(checkpointer=memory)


class WorkflowEngine:
    """生产级智能客服工作流引擎"""

    def __init__(self, checkpointer: Optional[Any] = None) -> None:
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = build_workflow_graph(checkpointer=self.checkpointer)

    async def run(
        self,
        conversation_id: int,
        query: str,
        user_id: str = "default_user",
        db: Optional[Any] = None,
    ) -> AgentWorkflowState:
        """执行单轮工作流"""
        initial_state = create_initial_state(
            conversation_id=conversation_id,
            query=query,
            user_id=user_id,
        )
        config = {"configurable": {"thread_id": str(conversation_id)}}
        final_state = await self.graph.ainvoke(initial_state, config=config)
        return final_state
