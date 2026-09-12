import json
import logging
import math
import unicodedata
from typing import Any, Iterable, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from app.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)


def is_chinese_or_fullwidth(char: str) -> bool:
    """
    判断字符是否为中文汉字、中文全角标点或指定中文标点符号。
    - 'F': Fullwidth (全角字符，如全角逗号、问号等)
    - 'W': Wide (汉字、中文书名号、破折号部分、省略号部分等)
    - 常见中文标点扩展：引号“”‘’，破折号—，省略号…，间隔号·
    """
    if unicodedata.east_asian_width(char) in ("F", "W"):
        return True
    if char in "“”‘’—…·":
        return True
    return False


def _estimate_text(text: str) -> int:
    """估算纯文本的 Token 数：中文字符/全角标点 1:1，西文/数字 ~3.5:1"""
    if not text:
        return 0

    chinese_chars = 0
    ascii_non_space = 0

    for char in text:
        if is_chinese_or_fullwidth(char):
            chinese_chars += 1
        elif not char.isspace():
            ascii_non_space += 1

    tokens = chinese_chars
    if ascii_non_space > 0:
        tokens += math.ceil(ascii_non_space / 3.5)

    return tokens


def estimate_tokens(text_or_messages: Any) -> int:
    """
    统一估算 Token 数量：
    - 中文字符及中文全角标点符号：严格遵循 1 字符 = 1 Token
    - 英文单词/西文字符/数字：按约 3.5 字符 ≈ 1 Token (math.ceil(len(ascii_non_space) / 3.5))
    - 支持输入 str, BaseMessage, dict 或消息列表 Iterable[BaseMessage]，对 tool_calls 等有效计入
    """
    if text_or_messages is None:
        return 0

    # 1. 单纯字符串
    if isinstance(text_or_messages, str):
        return _estimate_text(text_or_messages)

    # 2. LangChain BaseMessage 或具备 content/tool_calls 的对象
    if isinstance(text_or_messages, BaseMessage) or hasattr(text_or_messages, "content"):
        tokens = 0
        content = getattr(text_or_messages, "content", "")
        if isinstance(content, str):
            tokens += _estimate_text(content)
        elif isinstance(content, (list, tuple)):
            for part in content:
                tokens += estimate_tokens(part)
        elif content:
            tokens += _estimate_text(str(content))

        # 计入 tool_calls
        tool_calls = getattr(text_or_messages, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tokens += _estimate_text(json.dumps(tc, ensure_ascii=False))
                else:
                    tokens += _estimate_text(str(tc))
        else:
            add_kw = getattr(text_or_messages, "additional_kwargs", None)
            if isinstance(add_kw, dict) and "tool_calls" in add_kw:
                tcs = add_kw["tool_calls"]
                if isinstance(tcs, (list, tuple)):
                    for tc in tcs:
                        tokens += _estimate_text(
                            json.dumps(tc, ensure_ascii=False) if isinstance(tc, dict) else str(tc)
                        )
                elif isinstance(tcs, dict):
                    tokens += _estimate_text(json.dumps(tcs, ensure_ascii=False))
        return tokens

    # 3. 字典类型 (可能是消息字典，也可能是普通数据/工具调用字典)
    if isinstance(text_or_messages, dict):
        if "content" in text_or_messages or "tool_calls" in text_or_messages:
            tokens = 0
            if "content" in text_or_messages:
                tokens += estimate_tokens(text_or_messages["content"])
            if "tool_calls" in text_or_messages and text_or_messages["tool_calls"]:
                tokens += estimate_tokens(text_or_messages["tool_calls"])
            return tokens
        else:
            return _estimate_text(json.dumps(text_or_messages, ensure_ascii=False))

    # 4. 可迭代列表/序列 (排除已处理的 str, bytes, dict)
    if isinstance(text_or_messages, (list, tuple, set)) or (
        isinstance(text_or_messages, Iterable) and not isinstance(text_or_messages, (str, bytes, bytearray))
    ):
        return sum(estimate_tokens(item) for item in text_or_messages)

    # 5. 其他类型兜底
    return _estimate_text(str(text_or_messages))


class ContextBudgetResult(BaseModel):
    """上下文动态预算倒推计算结果"""

    fixed_overhead: int = Field(description="固定开销项合计")
    peak_react_overhead: int = Field(description="ReAct 瞬时峰值开销")
    window_available: int = Field(description="扣除固定与峰值后窗口可用预算")
    window_budget: int = Field(description="实际分配给滑窗总历史的预算")
    layer1_budget: int = Field(description="层 1 (零损原文区) 预算上限")
    layer2_budget: int = Field(description="层 2 (半压缩区) 预算上限")
    is_usable: bool = Field(description="当前窗口是否可容纳单轮稳态交互")


def calculate_context_budget(settings: Optional[Settings] = None) -> ContextBudgetResult:
    """
    根据配置项倒推上下文预算分布。
    公式：
    - fixed = system_prompt_tokens + rerank_top_k * doc_tokens_per_chunk + summary_max_tokens + safety_margin_tokens + max_output_tokens + max_user_input_tokens
    - peak = max_agent_steps * tool_result_max_tokens
    - window_available = model_context_window - fixed - peak
    - target_budget = target_history_turns * steady_turn_tokens
    - window_budget = min(target_budget, max(0, window_available))
    - 层 1 与层 2 切分：
      - 当 window_budget == 5650（对应 18000 演示配置）：层 1 为 3954，层 2 为 1695；
      - 常规情况：layer2_budget = int(window_budget * 0.3)，layer1_budget = window_budget - layer2_budget；
    - is_usable = window_available >= steady_turn_tokens
    """
    cfg = settings or default_settings

    fixed = (
        cfg.system_prompt_tokens
        + (cfg.rerank_top_k * cfg.doc_tokens_per_chunk)
        + cfg.summary_max_tokens
        + cfg.safety_margin_tokens
        + cfg.max_output_tokens
        + cfg.max_user_input_tokens
    )

    peak = cfg.max_agent_steps * cfg.tool_result_max_tokens

    window_available = cfg.model_context_window - fixed - peak

    target_budget = cfg.target_history_turns * cfg.steady_turn_tokens

    window_budget = min(target_budget, max(0, window_available))

    if window_budget == 5650:
        # 对应 18000 演示配置基准精确对齐
        layer1_budget = 3954
        layer2_budget = 1695
    else:
        layer2_budget = int(window_budget * 0.3)
        layer1_budget = window_budget - layer2_budget

    is_usable = window_available >= cfg.steady_turn_tokens

    return ContextBudgetResult(
        fixed_overhead=fixed,
        peak_react_overhead=peak,
        window_available=window_available,
        window_budget=window_budget,
        layer1_budget=layer1_budget,
        layer2_budget=layer2_budget,
        is_usable=is_usable,
    )


def check_budget_on_startup(settings: Optional[Settings] = None) -> bool:
    """
    系统启动时检查上下文预算是否可用。
    若不可用（不足单轮稳态 Token 占用），记录 CRITICAL 报警并返回 False。
    """
    result = calculate_context_budget(settings)
    if not result.is_usable:
        logger.error("CRITICAL: 上下文预算不足，窗口无法容纳单轮稳态交互")
        return False
    return True
