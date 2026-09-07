from typing import Optional
from pydantic import BaseModel, Field

class AfterSaleTicket(BaseModel):
    """电商售后工单提取结构化对象"""
    order_id: Optional[str] = Field(
        default=None,
        description="从用户描述中提取到的订单号或交易单号（如 TB123456、20240901xxxx 等）。如果用户完全未提及订单号，则为 null。"
    )
    issue_type: str = Field(
        description="售后问题的类型。例如：商品质量问题/破损、少件漏发、错发商品、物流延迟/未送达、七天无理由退货、退差价/保价、商品咨询/使用问题等。"
    )
    expected_solution: str = Field(
        description="用户期望的解决方式。例如：仅退款、退货退款、换货、补发商品、维修、催促物流、赔偿补偿等。若未明确提及，请根据上下文合理归纳最为恰当的诉求。"
    )
    raw_description: str = Field(
        description="用户提供的原始售后问题描述全文。"
    )

class AfterSaleExtractRequest(BaseModel):
    """售后提取接口请求参数"""
    description: str = Field(..., min_length=1, description="用户售后问题口述或文字描述")
