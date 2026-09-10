from app.services.workflow.state import AgentWorkflowState

CONFIDENCE_SCORE_THRESHOLD = 0.35


def confidence_gate(state: AgentWorkflowState) -> str:
    """置信度闸门条件判定：最高分 >= 0.35 且有命中方可通过"""
    docs = state.get("retrieved_docs") or []
    if not docs:
        return "fallback"

    max_score = 0.0
    for doc in docs:
        score = float(doc.get("score") or 0.0)
        if score > max_score:
            max_score = score

    if max_score >= CONFIDENCE_SCORE_THRESHOLD:
        return "pass"
    return "fallback"
