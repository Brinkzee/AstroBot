from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    """对话流式请求参数"""
    conversation_id: Optional[int] = Field(default=None, description="会话ID（自增主键）")
    session_id: Optional[str] = Field(default=None, description="会话ID，兼容旧版参数别名")
    message: str = Field(..., min_length=1, description="用户本次输入的内容")

    @property
    def effective_conversation_id(self) -> Optional[int]:
        """提取有效的会话ID整型数字，优先取 conversation_id，其次若 session_id 为纯数字则转换"""
        if self.conversation_id is not None:
            return self.conversation_id
        if self.session_id is not None:
            s = str(self.session_id).strip()
            if s.isdigit():
                return int(s)
        return None


class ChatResumeRequest(BaseModel):
    """恢复挂起工作流的请求参数"""
    conversation_id: int = Field(..., description="会话ID")
    action: Literal["confirm", "cancel"] = Field(..., description="用户裁决动作: 'confirm' 或 'cancel'")
