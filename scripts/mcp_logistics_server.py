"""物流独立进程业务 MCP Server (LogisticsServer)

提供基于 Streamable HTTP 协议的物流状态及轨迹查询工具。
"""

import json
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP

# 预置模拟物流数据（与内置业务数据保持一致）
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


def query_logistics(order_id: str) -> str:
    """根据订单号查询物流配送轨迹及状态。

    当用户询问包裹发货进度、快递公司、运单编号、快递当前位置等配送进度时调用此工具。

    Args:
        order_id: 订单编号，例如 '1001'、'1002'
    """
    key = str(order_id).strip()
    if not key:
        return json.dumps({"error": "订单号不能为空"}, ensure_ascii=False)

    if key in MOCK_LOGISTICS:
        logistics_info = MOCK_LOGISTICS[key]
    else:
        # 针对未预置订单号动态生成标准结构的物流模拟数据
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


def create_logistics_server(
    host: str = "127.0.0.1", port: int = 8001
) -> FastMCP:
    """创建并配置物流 FastMCP Server 实例。"""
    server = FastMCP(
        "LogisticsServer",
        host=host,
        port=port,
        streamable_http_path="/mcp",
    )
    server.add_tool(query_logistics)
    return server


if __name__ == "__main__":
    server = create_logistics_server()
    server.run(transport="streamable-http")
