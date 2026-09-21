from typing import List, Dict, Any
from app.services.workflow.state import AgentWorkflowState
from app.config import settings

def compute_evidence_confidence(docs: List[Dict[str, Any]], k: int = 3) -> float:
    if not docs:
        return 0.0

    scores = sorted([float(d.get("score") or 0.0) for d in docs], reverse=True)
    
    top1_score = scores[0]
    valid_count = sum(1 for s in scores if s >= 0.25)
    score_margin = (scores[0] - scores[1]) if len(scores) > 1 else (scores[0] * 0.5)
    
    evidence_confidence = 0.60 * top1_score + 0.25 * min(valid_count / k, 1.0) + 0.15 * max(score_margin, 0.0)
    
    return max(0.0, min(evidence_confidence, 1.0))

def confidence_gate(state: AgentWorkflowState) -> str:
    """置信度闸门条件判定：计算多信号置信度分数"""
    docs = state.get("retrieved_docs") or []
    
    score = compute_evidence_confidence(docs)
    
    if hasattr(state, '__setitem__'):
        state["evidence_confidence"] = score
        
    threshold = getattr(settings, "evidence_confidence_threshold", 0.40)
    
    if score >= threshold:
        return "pass"
    return "fallback"
