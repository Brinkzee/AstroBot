import logging
from typing import Any, Dict, List, Optional
from app.services.workflow.state import AgentWorkflowState
from app.tools.business_tools import get_retriever

logger = logging.getLogger(__name__)

KNOWLEDGE_FALLBACK_TEXT = "抱歉，关于您咨询的问题，当前知识库中暂未收录确切规范或相关信息不足。建议您尝试更换问法，或选择转接人工客服/提交人工工单获取专人协助。"

CONFIDENCE_SCORE_THRESHOLD = 0.35


async def knowledge_retrieval_node(
    state: AgentWorkflowState,
    retriever: Optional[Any] = None,
) -> Dict[str, Any]:
    """知识检索节点：强制预检索 RAG 知识库"""
    query = state.get("resolved_query") or state.get("input_query", "")
    active_retriever = retriever or get_retriever()

    retrieved_docs: List[Dict[str, Any]] = []
    try:
        if hasattr(active_retriever, "retrieve_with_strategy"):
            res = await active_retriever.retrieve_with_strategy(query=query, min_score=0.1)
            hits = getattr(res, "hits", []) or []
            for hit in hits:
                if hasattr(hit, "to_dict"):
                    retrieved_docs.append(hit.to_dict())
                else:
                    retrieved_docs.append({
                        "text": getattr(hit, "text", str(hit)),
                        "score": getattr(hit, "score", 0.0),
                    })
        elif hasattr(active_retriever, "retrieve"):
            hits = await active_retriever.retrieve(query=query, top_k=5)
            for hit in hits:
                retrieved_docs.append({
                    "text": getattr(hit, "text", str(hit)),
                    "score": getattr(hit, "score", 0.5),
                })
    except Exception as e:
        logger.warning(f"知识检索节点异常: {e}")

    return {
        "retrieved_docs": retrieved_docs,
    }


async def knowledge_fallback_node(
    state: AgentWorkflowState,
    db: Optional[Any] = None,
) -> Dict[str, Any]:
    """兜底话术节点：弱证据拦截，记录至 low_confidence_questions 表，不进 Agent"""
    query = state.get("input_query", "")
    conv_id = state.get("conversation_id", 0)
    target_conv_id = conv_id if conv_id else None

    # 异步沉淀至 low_confidence_questions 表
    if db is not None:
        try:
            from app.models.low_confidence import LowConfidenceQuestion
            try:
                low_q = LowConfidenceQuestion(
                    raw_question=query,
                    source="retrieval_low_conf",
                    conversation_id=target_conv_id,
                    reason="retrieval_score_below_threshold",
                )
            except Exception:
                low_q = LowConfidenceQuestion(
                    query=query,
                    conversation_id=conv_id,
                    entrance="workflow_confidence_gate",
                    reason="retrieval_score_below_threshold",
                )
            db.add(low_q)
            await db.commit()
        except Exception as e:
            logger.warning(f"沉淀低置信度问题失败: {e}")

    return {
        "response_text": KNOWLEDGE_FALLBACK_TEXT,
        "confidence_passed": False,
        "suggested_actions": ["transfer_agent", "create_ticket"],
        "status": "fallback",
    }
