import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# 定位项目根目录及 log/app.log
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "log"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "app.log"


class LineBufferedFileHandler(logging.FileHandler):
    """实时刷新行缓冲的文件日志处理器，确保每行日志写入后立刻落盘"""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def get_context_logger(log_file: Optional[Path] = None) -> logging.Logger:
    """获取并初始化上下文可观测性专用 Logger（写入 log/app.log）"""
    target_file = (log_file or DEFAULT_LOG_FILE).resolve()
    target_file.parent.mkdir(parents=True, exist_ok=True)

    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)

    file_path_str = str(target_file)
    handler_exists = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == file_path_str
        for h in app_logger.handlers
    )

    if not handler_exists:
        handler = LineBufferedFileHandler(file_path_str, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)
        app_logger.addHandler(handler)

    context_logger = logging.getLogger("app.context.observability")
    context_logger.setLevel(logging.INFO)
    context_logger.propagate = True
    return context_logger


def _format_single_message(msg: Any) -> Tuple[str, str]:
    """提取单条消息的角色与文本展示内容"""
    if isinstance(msg, HumanMessage):
        return "user", str(msg.content or "")
    if isinstance(msg, AIMessage):
        content = str(msg.content or "")
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls and not content:
            content = f"[tool_calls: {len(tool_calls)}]"
        return "assistant", content
    if isinstance(msg, ToolMessage):
        return "tool", str(msg.content or "")
    if isinstance(msg, SystemMessage):
        return "system", str(msg.content or "")

    role = "user"
    content = ""
    if isinstance(msg, dict):
        raw_role = str(msg.get("role", "user")).lower()
        if raw_role in ("ai", "assistant"):
            role = "assistant"
        elif raw_role in ("human", "user"):
            role = "user"
        elif raw_role in ("tool", "system"):
            role = raw_role
        else:
            role = raw_role
        content = str(msg.get("content", "") or "")
    elif hasattr(msg, "content"):
        raw_role = str(getattr(msg, "role", getattr(msg, "type", "user"))).lower()
        if raw_role in ("ai", "assistant"):
            role = "assistant"
        elif raw_role in ("human", "user"):
            role = "user"
        else:
            role = raw_role
        content = str(getattr(msg, "content", "") or "")
    else:
        content = str(msg or "")

    return role, content


def log_model_context(
    conv_id: int,
    summary: Optional[str],
    window_messages: Sequence[Any],
    estimated_tokens: Optional[int] = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    格式化输出 [model_ctx] 到 log/app.log：
    格式规范：
    [model_ctx] conv_id={id} summary_len={len} window_msgs={count} estimated_tokens={tokens}
    --- SUMMARY ---
    {summary or "(none)"}
    --- SLIDING WINDOW ---
    [1] (user) ...
    [2] (assistant) ...
    """
    summary_clean = str(summary or "").strip()
    summary_len = len(summary_clean)
    window_msgs_count = len(window_messages)

    if estimated_tokens is None:
        from app.services.context.budget import estimate_tokens
        tokens = estimate_tokens(window_messages)
        if summary_clean:
            tokens += estimate_tokens(summary_clean)
    else:
        tokens = int(estimated_tokens)

    header = f"[model_ctx] conv_id={conv_id} summary_len={summary_len} window_msgs={window_msgs_count} estimated_tokens={tokens}"
    summary_sec = summary_clean if summary_clean else "(none)"

    window_lines = []
    for idx, msg in enumerate(window_messages, 1):
        role, content = _format_single_message(msg)
        window_lines.append(f"[{idx}] ({role}) {content}")

    window_sec = "\n".join(window_lines) if window_lines else "(none)"

    text = (
        f"{header}\n"
        f"--- SUMMARY ---\n"
        f"{summary_sec}\n"
        f"--- SLIDING WINDOW ---\n"
        f"{window_sec}"
    )

    target_logger = logger or get_context_logger()
    target_logger.info(text)
    return text


def log_history_context(
    conv_id: int,
    summary_line: Optional[str],
    window_messages: Union[Sequence[Any], str],
    *,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    格式化输出 [history_ctx] 到 log/app.log：
    格式规范：
    [history_ctx] conv_id={id} summary_line="{summary_first_line}" window_msgs={count}
    --- HISTORY WINDOW ---
    {formatted_history}
    """
    first_line = ""
    if summary_line:
        lines = str(summary_line).strip().splitlines()
        if lines:
            first_line = lines[0].strip()

    if isinstance(window_messages, str):
        formatted_history = window_messages.strip()
        non_empty = [l for l in formatted_history.splitlines() if l.strip()]
        count = len(non_empty)
    else:
        count = len(window_messages)
        formatted_lines = []
        for msg in window_messages:
            if isinstance(msg, str):
                formatted_lines.append(msg)
                continue
            role, content = _format_single_message(msg)
            display_role = "用户" if role == "user" else ("客服" if role == "assistant" else ("工具" if role == "tool" else role))
            formatted_lines.append(f"{display_role}: {content}")
        formatted_history = "\n".join(formatted_lines).strip()

    header = f'[history_ctx] conv_id={conv_id} summary_line="{first_line}" window_msgs={count}'
    history_sec = formatted_history if formatted_history else "(none)"

    text = (
        f"{header}\n"
        f"--- HISTORY WINDOW ---\n"
        f"{history_sec}"
    )

    target_logger = logger or get_context_logger()
    target_logger.info(text)
    return text


def log_summary_lifecycle(
    stage: str,
    conv_id: int,
    detail: Optional[str] = None,
    *,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> str:
    """
    记录后台摘要全生命周期结构化日志：
    [summary {stage}] conv_id={conv_id} ...
    """
    detail_str = f" {detail}" if detail else ""
    if kwargs:
        kw_str = " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
    else:
        kw_str = ""

    text = f"[summary {stage}] conv_id={conv_id}{detail_str}{kw_str}"
    target_logger = logger or get_context_logger()
    target_logger.info(text)
    return text
