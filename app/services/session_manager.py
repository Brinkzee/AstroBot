import uuid
from typing import Optional, List, Sequence, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, trim_messages
import tiktoken

def count_tokens_fallback(messages: Union[List[BaseMessage], Sequence[BaseMessage], BaseMessage]) -> int:
    """计算消息列表或单条消息的 Token 数量"""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

    if isinstance(messages, BaseMessage):
        msg_list = [messages]
    else:
        msg_list = list(messages)

    total = 0
    for m in msg_list:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if enc:
            total += len(enc.encode(content)) + 4
        else:
            total += len(content) // 2 + 4
    return total

class SessionManager:
    """服务端内存会话管理器，负责会话生命周期维护与滑动窗口 Token 预算裁剪"""

    def __init__(self):
        self._sessions: dict[str, List[BaseMessage]] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        if not session_id or not session_id.strip():
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return session_id

    def get_history(self, session_id: str) -> List[BaseMessage]:
        return list(self._sessions.get(session_id, []))

    def add_user_message(self, session_id: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(HumanMessage(content=content))

    def add_ai_message(self, session_id: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(AIMessage(content=content))

    def get_trimmed_history(self, session_id: str, max_tokens: int) -> List[BaseMessage]:
        raw_history = self.get_history(session_id)
        if not raw_history:
            return []

        # 调用 LangChain 官方 trim_messages
        trimmed = trim_messages(
            raw_history,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=count_tokens_fallback,
            start_on="human",
            include_system=True
        )
        return list(trimmed)

    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

session_manager = SessionManager()
