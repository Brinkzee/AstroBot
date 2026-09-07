from typing import Optional
from pydantic import BaseModel, Field

class ChatStreamRequest(BaseModel):
    """对话流式请求参数"""
    session_id: Optional[str] = Field(default=None, description="会话ID，若为空则由服务端自动生成")
    message: str = Field(..., min_length=1, description="用户本次输入的内容")
