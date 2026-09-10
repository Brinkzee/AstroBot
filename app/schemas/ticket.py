from typing import Optional
from pydantic import BaseModel, Field


class TicketCreateRequest(BaseModel):
    conversation_id: Optional[int] = Field(None, description="关联会话ID")
    description: str = Field("用户自主申请客服工单", description="工单详细描述")
    ticket_type: str = Field("投诉", description="工单类型: 投诉/售后/咨询")


class TicketCreateResponse(BaseModel):
    ticket_no: str
    conversation_id: int
    ticket_type: str
    status: str
