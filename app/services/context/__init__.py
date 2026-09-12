"""
会话上下文管理模块 (Context Management Module)
提供动态 Token 预算倒推、三层上下文管理器与分段事实摘要引擎。
"""

from app.services.context.budget import (
    ContextBudgetResult,
    calculate_context_budget,
    check_budget_on_startup,
    estimate_tokens,
)
from app.services.context.manager import (
    ContextManager,
    apply_layer1_degradation,
    build_history_context_text,
    build_model_messages,
    format_layer1_messages,
    format_layer2_messages,
    partition_messages,
)

from app.services.context.summary_service import (
    SummaryService,
    default_summary_service,
    format_dialogue_for_summary,
    persist_summary_segment,
    should_trigger_summary,
    summarize_dialogue,
    summarize_dialogue_sync,
    trigger_async_summary,
)

__all__ = [
    "ContextBudgetResult",
    "calculate_context_budget",
    "check_budget_on_startup",
    "estimate_tokens",
    "ContextManager",
    "apply_layer1_degradation",
    "build_history_context_text",
    "build_model_messages",
    "format_layer1_messages",
    "format_layer2_messages",
    "partition_messages",
    "SummaryService",
    "default_summary_service",
    "format_dialogue_for_summary",
    "persist_summary_segment",
    "should_trigger_summary",
    "summarize_dialogue",
    "summarize_dialogue_sync",
    "trigger_async_summary",
]

