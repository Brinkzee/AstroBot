import json
import random
from datetime import datetime
from typing import Any, Dict, List
from langchain_core.tools import tool
from sqlalchemy import select, or_
from app.db.session import AsyncSessionLocal
from app.models import FAQ, Ticket


# 模拟订单数据库缓存
MOCK_ORDERS: Dict[str, Dict[str, Any]] = {
    "1001": {
        "order_id": "1001",
        "订单状态": "已发货",
        "支付金额": "299.00元",
        "商品明细": [
            {"商品名称": "极简保暖羽绒服", "数量": 1, "单价": "299.00元"}
        ],
        "下单时间": "2026-09-05 14:20:00",
    },
    "1002": {
        "order_id": "1002",
        "订单状态": "待支付",
        "支付金额": "159.00元",
        "商品明细": [
            {"商品名称": "潮流纯棉连帽卫衣", "数量": 1, "单价": "159.00元"}
        ],
        "下单时间": "2026-09-06 10:15:00",
    },
    "1003": {
        "order_id": "1003",
        "订单状态": "已完成",
        "支付金额": "89.00元",
        "商品明细": [
            {"商品名称": "经典修身牛仔裤", "数量": 1, "单价": "89.00元"}
        ],
        "下单时间": "2026-09-01 09:30:00",
    },
}

# 模拟商品数据库缓存
MOCK_PRODUCTS: List[Dict[str, Any]] = [
    {
        "product_id": "P1001",
        "商品名称": "极简保暖羽绒服",
        "价格": "599.00元",
        "库存": "现货充裕 (剩余 120 件)",
        "规格属性": "黑色/白色/灰色; M/L/XL/XXL; 90%白鸭绒填充",
    },
    {
        "product_id": "P1002",
        "商品名称": "潮流纯棉连帽卫衣",
        "价格": "199.00元",
        "库存": "现货充裕 (剩余 350 件)",
        "规格属性": "藏青/米白/墨绿; S/M/L/XL; 100%纯棉重磅",
    },
    {
        "product_id": "P1003",
        "商品名称": "经典修身牛仔裤",
        "价格": "239.00元",
        "库存": "库存紧张 (剩余 15 件)",
        "规格属性": "复古蓝/原色黑; 28-34码; 微弹水洗牛仔面料",
    },
]

# 模拟物流数据库缓存
MOCK_LOGISTICS: Dict[str, Dict[str, Any]] = {
    "1001": {
        "order_id": "1001",
        "快递公司": "顺丰速运",
        "运单编号": "SF14285700199",
        "物流状态": "派送中",
        "最新轨迹": "正在派件中，快递员已出发，请保持电话畅通",
        "轨迹明细": [
            {"时间": "2026-09-06 08:30:00", "节点": "快件已到达【北京海淀中关村营业点】"},
            {"时间": "2026-09-06 09:15:00", "节点": "正在派件中，快递员已出发，请保持电话畅通"},
        ],
    },
    "1002": {
        "order_id": "1002",
        "快递公司": "中通快递",
        "运单编号": "ZT7788991122",
        "物流状态": "运输中",
        "最新轨迹": "快件由【杭州转运中心】发往【北京中转场】",
        "轨迹明细": [
            {"时间": "2026-09-06 14:00:00", "节点": "快件由【杭州转运中心】发往【北京中转场】"},
        ],
    },
    "1003": {
        "order_id": "1003",
        "快递公司": "顺丰速运",
        "运单编号": "SF9988776655",
        "物流状态": "已签收",
        "最新轨迹": "快件已签收，签收人：本人签收",
        "轨迹明细": [
            {"时间": "2026-09-03 16:45:00", "节点": "快件已签收，签收人：本人签收"},
        ],
    },
}


@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单详细信息。当用户询问订单状态、支付金额、购买商品明细或下单时间时调用此工具。

    Args:
        order_id: 订单编号，例如 '1001'、'1002'
    """
    key = str(order_id).strip()
    if not key:
        return json.dumps({"error": "订单号不能为空"}, ensure_ascii=False)

    if key in MOCK_ORDERS:
        order_info = MOCK_ORDERS[key]
    else:
        # 针对未预置订单号生成真实结构的模拟数据
        order_info = {
            "order_id": key,
            "订单状态": "已发货",
            "支付金额": "199.00元",
            "商品明细": [
                {"商品名称": "热销精选商品", "数量": 1, "单价": "199.00元"}
            ],
            "下单时间": "2026-09-05 12:00:00",
        }

    return json.dumps(order_info, ensure_ascii=False, indent=2)


@tool
def query_product(product_id_or_name: str) -> str:
    """根据商品编号或商品名称查询商品详情。当用户咨询商品价格、库存情况、商品规格与尺码等信息时调用此工具。

    Args:
        product_id_or_name: 商品编号或商品名称关键字，例如 'P1001'、'羽绒服'、'卫衣'
    """
    target = str(product_id_or_name).strip()
    if not target:
        return json.dumps({"error": "商品编号或名称不能为空"}, ensure_ascii=False)

    matched = None
    for item in MOCK_PRODUCTS:
        if (
            target.lower() == item["product_id"].lower()
            or target in item["商品名称"]
            or item["商品名称"] in target
        ):
            matched = item
            break

    if matched:
        return json.dumps(matched, ensure_ascii=False, indent=2)

    # 动态模拟未知商品信息
    simulated = {
        "product_id": f"P{abs(hash(target)) % 9000 + 1000}",
        "商品名称": target,
        "价格": "299.00元",
        "库存": "现货充裕 (剩余 88 件)",
        "规格属性": "标准规格; 均码; 现货正品",
    }
    return json.dumps(simulated, ensure_ascii=False, indent=2)


@tool
def query_logistics(order_id: str) -> str:
    """根据订单号查询物流配送轨迹及状态。当用户询问包裹发货进度、快递公司、运单编号、快递当前位置等配送进度时调用此工具。

    Args:
        order_id: 订单编号，例如 '1001'、'1002'
    """
    key = str(order_id).strip()
    if not key:
        return json.dumps({"error": "订单号不能为空"}, ensure_ascii=False)

    if key in MOCK_LOGISTICS:
        logistics_info = MOCK_LOGISTICS[key]
    else:
        # 针对未预置订单号生成真实结构的物流模拟数据
        logistics_info = {
            "order_id": key,
            "快递公司": "顺丰速运",
            "运单编号": f"SF{abs(hash(key)) % 9000000000 + 1000000000}",
            "物流状态": "运输中",
            "最新轨迹": "包裹已由顺丰速运揽收，正在运往转运中心",
            "轨迹明细": [
                {"时间": "2026-09-06 10:00:00", "节点": "包裹已由顺丰速运揽收，正在运往转运中心"},
                {"时间": "2026-09-06 18:20:00", "节点": "快件在干线运输途中，预计次日送达"},
            ],
        }

    return json.dumps(logistics_info, ensure_ascii=False, indent=2)


@tool
async def query_faq(keyword: str) -> str:
    """查询常见问题解答 (FAQ) 知识库。当用户咨询平台规则、退换货政策、发票开具、运费标准等常见业务规范时调用此工具。

    Args:
        keyword: 检索关键词，例如 '退货'、'运费'、'发票' 等
    """
    kw = str(keyword).strip()
    if not kw:
        return f"未找到与【{keyword}】相关的常见问题解答。"

    async with AsyncSessionLocal() as session:
        stmt = (
            select(FAQ)
            .where(or_(FAQ.question.like(f"%{kw}%"), FAQ.answer.like(f"%{kw}%")))
            .limit(3)
        )
        result = await session.execute(stmt)
        faqs = result.scalars().all()

    if not faqs:
        return f"未找到与【{keyword}】相关的常见问题解答。"

    lines = []
    for idx, faq in enumerate(faqs, 1):
        lines.append(f"{idx}. 问：{faq.question}\n   答：{faq.answer}")
    return "\n\n".join(lines)


@tool
async def create_ticket(
    conversation_id: int,
    description: str,
    ticket_type: str = "售后",
) -> str:
    """创建人工客服工单。当用户遇到复杂问题、投诉、争议或明确要求转人工客服处理时调用此工具。

    Args:
        conversation_id: 关联的当前会话主键 ID (int)
        description: 工单问题详细描述
        ticket_type: 工单类型，可选值为 '售后'、'投诉' 或 '咨询'，默认为 '售后'
    """
    valid_types = ["售后", "投诉", "咨询"]
    if ticket_type not in valid_types:
        ticket_type = "售后"

    ticket_no = f"T{datetime.now().strftime('%Y%m%d%H%M%S')}{random.randint(100, 999)}"

    ticket = Ticket(
        ticket_no=ticket_no,
        conversation_id=conversation_id,
        description=description,
        ticket_type=ticket_type,
        status="待处理",
    )

    async with AsyncSessionLocal() as session:
        session.add(ticket)
        await session.commit()

    return json.dumps(
        {
            "ticket_no": ticket_no,
            "conversation_id": conversation_id,
            "ticket_type": ticket_type,
            "status": "工单已创建，人工客服将在24小时内跟进处理",
        },
        ensure_ascii=False,
        indent=2,
    )
