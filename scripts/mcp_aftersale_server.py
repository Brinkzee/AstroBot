"""售后独立进程业务 MCP Server (AftersaleServer)

提供基于 Streamable HTTP 协议的质保状态查询与退货进度查询工具，
并支持动态工具注册与热插拔扩展。
"""

import json
from typing import Any, Callable, Dict
from mcp.server.fastmcp import FastMCP

# 预置模拟质保数据
MOCK_WARRANTY: Dict[str, Dict[str, Any]] = {
    "1001": {
        "order_id": "1001",
        "商品名称": "极简保暖羽绒服",
        "质保状态": "在保",
        "质保截止日期": "2027-09-05",
        "保修条款": "整机包修1年，主要部件包修2年，支持全国联保",
    },
    "1002": {
        "order_id": "1002",
        "商品名称": "潮流纯棉连帽卫衣",
        "质保状态": "在保",
        "质保截止日期": "2027-09-06",
        "保修条款": "整机包修1年，支持7天无理由退换及全国联保",
    },
    "1003": {
        "order_id": "1003",
        "商品名称": "经典修身牛仔裤",
        "质保状态": "已过保",
        "质保截止日期": "2025-09-01",
        "保修条款": "整机包修1年（已过保，可享受付费维修服务）",
    },
}

# 预置模拟退货退款数据
MOCK_RETURNS: Dict[str, Dict[str, Any]] = {
    "1001": {
        "return_id": "RET1001",
        "order_id": "1001",
        "退货单号": "RET1001",
        "对应订单号": "1001",
        "状态": "商家已同意退货申请，请尽快寄回商品",
        "更新时间": "2026-09-06 15:30:00",
        "说明": "请使用顺丰或中通等正规快递寄回，并上传寄回运单号",
    },
    "1002": {
        "return_id": "RET1002",
        "order_id": "1002",
        "退货单号": "RET1002",
        "对应订单号": "1002",
        "状态": "仓库已验货入库，退款处理中",
        "更新时间": "2026-09-06 16:45:00",
        "说明": "仓库质检通过，预计1-3个工作日原路退款",
    },
    "1003": {
        "return_id": "RET1003",
        "order_id": "1003",
        "退货单号": "RET1003",
        "对应订单号": "1003",
        "状态": "退款已完成",
        "更新时间": "2026-09-04 11:20:00",
        "说明": "退款已原路退回至支付账户",
    },
}


def check_warranty(order_id: str, product_name: str = "") -> str:
    """根据订单号和商品名称查询质保期限、保修条款与在保状态。

    当用户咨询商品是否在保修期内、质保截止时间、保修范围条款或售后维修权益时调用此工具。

    Args:
        order_id: 订单编号，例如 '1001'
        product_name: 可选商品名称，例如 '极简保暖羽绒服'
    """
    key = str(order_id).strip()
    if not key:
        return json.dumps({"error": "订单号不能为空"}, ensure_ascii=False)

    if key in MOCK_WARRANTY:
        warranty_info = dict(MOCK_WARRANTY[key])
        if product_name.strip():
            warranty_info["商品名称"] = product_name.strip()
    else:
        # 针对未预置订单动态模拟质保数据
        p_name = product_name.strip() if product_name.strip() else "智能精选商品"
        warranty_info = {
            "order_id": key,
            "商品名称": p_name,
            "质保状态": "在保",
            "质保截止日期": "2027-09-01",
            "保修条款": "整机包修1年，主要部件包修2年，支持全国联保",
        }

    return json.dumps(warranty_info, ensure_ascii=False, indent=2)


def query_return_progress(return_id_or_order_id: str) -> str:
    """根据退货单号或订单号查询退货退款审核状态与处理进度。

    当用户询问退货申请是否通过、仓库是否收到寄回件、退款进度何时到账时调用此工具。

    Args:
        return_id_or_order_id: 退货单号或关联订单编号，例如 'RET1001' 或 '1001'
    """
    key = str(return_id_or_order_id).strip()
    if not key:
        return json.dumps({"error": "退货单号或订单号不能为空"}, ensure_ascii=False)

    # 匹配预置数据（支持按订单号或 RET 编号检索）
    matched = None
    for ret_key, data in MOCK_RETURNS.items():
        if key.lower() == ret_key.lower() or key.lower() == data["return_id"].lower():
            matched = data
            break

    if matched:
        return json.dumps(matched, ensure_ascii=False, indent=2)

    # 动态模拟未知退货记录
    simulated = {
        "return_id": f"RET{abs(hash(key)) % 900000 + 100000}",
        "order_id": key,
        "退货单号": f"RET{abs(hash(key)) % 900000 + 100000}",
        "对应订单号": key,
        "状态": "商家已同意退货申请，请尽快寄回商品",
        "更新时间": "2026-09-06 10:00:00",
        "说明": "退货申请审核通过，请妥善包装后寄回",
    }
    return json.dumps(simulated, ensure_ascii=False, indent=2)


def register_dynamic_tool(server: FastMCP, tool_func: Callable[..., Any]) -> None:
    """动态挂载新工具至 FastMCP Server，支持热插拔探测与扩展。

    Args:
        server: FastMCP 实例
        tool_func: 待注册的工具函数
    """
    server.add_tool(tool_func)


def create_aftersale_server(
    host: str = "127.0.0.1", port: int = 8002
) -> FastMCP:
    """创建并配置售后 FastMCP Server 实例。"""
    server = FastMCP(
        "AftersaleServer",
        host=host,
        port=port,
        streamable_http_path="/mcp",
    )
    server.add_tool(check_warranty)
    server.add_tool(query_return_progress)
    return server


if __name__ == "__main__":
    server = create_aftersale_server()
    server.run(transport="streamable-http")
