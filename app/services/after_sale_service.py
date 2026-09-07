from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from app.schemas.after_sale import AfterSaleTicket
from app.llm import get_chat_model

AFTER_SALE_SYSTEM_PROMPT = """你是一位专业的电商售后工单分析专员。
你的任务是从用户口述或输入的售后投诉描述中，精确提取关键结构化信息：
1. order_id: 提取用户的订单号（如有英文字母+数字、纯长数字串等）。若用户未提供订单号，必须设为 null。
2. issue_type: 提炼售后的主要问题类型（如商品质量问题/破损、错发/漏发、物流延迟/丢件、七天无理由退货、价保退差等）。
3. expected_solution: 提炼用户期望的解决手段（如仅退款、退货退款、换货、补寄零件、催件派送等）。
4. raw_description: 原样保留用户的输入描述文本。
请基于客观事实进行提炼，不得臆造不存在的订单号。
"""

EVALUATION_SAMPLES: List[Dict[str, Any]] = [
    {
        "id": "sample-1",
        "description": "我上周买的羽绒服订单号是 TB20240901，拉链卡住了拉不上，我想直接换一件新的。",
        "expected_order_id": "TB20240901",
        "expected_issue_type": "质量问题/破损",
        "expected_solution": "换货"
    },
    {
        "id": "sample-2",
        "description": "买的洗发水收到包装全漏了，箱子都湿透了，单号 8839210492，麻烦直接退款给我吧。",
        "expected_order_id": "8839210492",
        "expected_issue_type": "破损/漏液",
        "expected_solution": "仅退款/退款"
    },
    {
        "id": "sample-3",
        "description": "买了两件衣服只收到了一件，里面缺了一条牛仔裤，赶紧给我补发过来！",
        "expected_order_id": None,
        "expected_issue_type": "少件漏发",
        "expected_solution": "补发"
    },
    {
        "id": "sample-4",
        "description": "订单 100862214 发货都四天了物流还停在杭州转运中心不动，能不能帮我加急催一下快递？",
        "expected_order_id": "100862214",
        "expected_issue_type": "物流延迟/停滞",
        "expected_solution": "催促物流/加急催件"
    },
    {
        "id": "sample-5",
        "description": "衣服试穿了一下号太小了，标签都还在，想退掉重新拍一件。",
        "expected_order_id": None,
        "expected_issue_type": "尺码不合/七天无理由",
        "expected_solution": "退货退款/退换"
    }
]

def extract_after_sale_ticket(
    description: str,
    model: Optional[ChatOpenAI] = None
) -> AfterSaleTicket:
    """使用 with_structured_output 提取电商售后工单"""
    if model is None:
        model = get_chat_model(streaming=False, temperature=0.0)

    structured_model = model.with_structured_output(AfterSaleTicket)
    messages = [
        ("system", AFTER_SALE_SYSTEM_PROMPT),
        ("human", description)
    ]
    result = structured_model.invoke(messages)

    if isinstance(result, dict):
        result = AfterSaleTicket(**result)

    result.raw_description = description
    return result
