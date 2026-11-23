import logging
from typing import Any, Dict, List, Optional
from app.config import settings
from app.services.workflow.state import AgentWorkflowState
from app.services.workflow.nodes.gate import compute_evidence_confidence, extract_doc_score
from app.tools.business_tools import get_retriever

logger = logging.getLogger(__name__)

KNOWLEDGE_FALLBACK_TEXT = "抱歉，关于您咨询的问题，当前知识库中暂未收录确切规范或相关信息不足。建议您尝试更换问法，或选择转接人工客服/提交人工工单获取专人协助。"

CONFIDENCE_SCORE_THRESHOLD = 0.35


def _extract_retrieval_items(res: Any) -> List[Any]:
    """从检索返回对象中兼容提取文档列表（优先非空 docs，兼容旧版 hits 或直接 list）。"""
    if isinstance(res, (list, tuple)):
        return list(res)
    raw_docs = getattr(res, "docs", None)
    if isinstance(raw_docs, (list, tuple)) and raw_docs:
        return list(raw_docs)
    raw_hits = getattr(res, "hits", None)
    if isinstance(raw_hits, (list, tuple)) and raw_hits:
        return list(raw_hits)
    if isinstance(raw_docs, (list, tuple)):
        return list(raw_docs)
    if isinstance(raw_hits, (list, tuple)):
        return list(raw_hits)
    return []


def _extract_doc_text(doc: Dict[str, Any], fallback_obj: Any = None) -> str:
    for key in ("text", "content", "chunk_text", "answer"):
        val = doc.get(key)
        if val:
            return str(val)
    if fallback_obj is not None and not isinstance(fallback_obj, dict):
        for attr in ("text", "content", "chunk_text", "answer"):
            val = getattr(fallback_obj, attr, None)
            if val:
                return str(val)
        return str(fallback_obj)
    return ""


def _normalize_retrieved_doc(item: Any, default_score: float = 0.0) -> Dict[str, Any]:
    """将检索命中项标准化为下游闸门、上下文组装与落池快照统一可消费的字典。"""
    if isinstance(item, dict):
        doc = dict(item)
    elif hasattr(item, "to_dict") and callable(item.to_dict):
        raw = item.to_dict()
        doc = dict(raw) if isinstance(raw, dict) else {"text": str(raw)}
    else:
        doc = {
            "text": getattr(item, "text", str(item)),
            "section_path": getattr(item, "section_path", ""),
        }

    # 移除 Milvus 原始 entity（包含 1024 维向量，避免污染工作流状态与 Prompt）
    doc.pop("entity", None)

    score = extract_doc_score(doc, default=extract_doc_score(item, default=default_score))
    doc["score"] = score

    text = _extract_doc_text(doc, fallback_obj=item)
    if text:
        doc.setdefault("text", text)
        doc.setdefault("content", text)

    return doc


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
            items = _extract_retrieval_items(res)
            for item in items:
                retrieved_docs.append(_normalize_retrieved_doc(item, default_score=0.0))
        elif hasattr(active_retriever, "retrieve"):
            hits = await active_retriever.retrieve(query=query, top_k=5)
            for hit in _extract_retrieval_items(hits):
                retrieved_docs.append(_normalize_retrieved_doc(hit, default_score=0.5))
    except Exception as e:
        logger.warning(f"知识检索节点异常: {e}")

    score = compute_evidence_confidence(retrieved_docs)
    threshold = getattr(settings, "evidence_confidence_threshold", 0.40)

    return {
        "retrieved_docs": retrieved_docs,
        "evidence_confidence": score,
        "confidence_passed": score >= threshold,
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
            from app.models.low_confidence import record_low_confidence
            retrieved_docs = state.get("retrieved_docs") or []
            retrieved_chunks = [
                {
                    "text": _extract_doc_text(doc)[:300] if isinstance(doc, dict) else str(doc)[:300],
                    "score": extract_doc_score(doc),
                    "section": str(doc.get("section_path") or doc.get("section") or "") if isinstance(doc, dict) else ""
                }
                for doc in retrieved_docs[:5]
            ] if retrieved_docs else None

            await record_low_confidence(
                db=db,
                raw_question=query,
                source="retrieval_low_conf",
                conversation_id=target_conv_id,
                reason="retrieval_score_below_threshold",
                retrieved_chunks=retrieved_chunks
            )
        except Exception as e:
            logger.warning(f"沉淀低置信度问题失败: {e}")

    return {
        "response_text": KNOWLEDGE_FALLBACK_TEXT,
        "confidence_passed": False,
        "suggested_actions": ["transfer_agent", "create_ticket"],
        "status": "fallback",
    }
