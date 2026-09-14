from typing import Any, Optional
from pydantic import BaseModel


class ConversationItem(BaseModel):
    id: int
    title: str  # 首问预览（截取前25字符，无提问时为"新会话"）
    has_summary: bool  # 是否已有摘要标记
    status: str
    message_count: int
    created_at: str
    updated_at: str


class ConversationMessageItem(BaseModel):
    id: int
    role: str
    content: Optional[str] = None
    tool_calls: Optional[Any] = None
    tool_call_id: Optional[str] = None
    created_at: str


# 便于外部不同语义命名的别名兼容
ConversationListItem = ConversationItem
