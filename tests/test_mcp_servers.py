import json
import pytest
from scripts.mcp_logistics_server import create_logistics_server, query_logistics
from scripts.mcp_aftersale_server import (
    create_aftersale_server,
    check_warranty,
    query_return_progress,
    register_dynamic_tool,
)


@pytest.mark.asyncio
async def test_logistics_mcp_server_properties_and_tools():
    """验证物流 Server 基础属性与工具导出列表"""
    server = create_logistics_server()
    assert server.name == "LogisticsServer"
    assert server.settings.port == 8001
    assert server.settings.host == "127.0.0.1"

    app = server.streamable_http_app()
    route_paths = [r.path for r in app.routes]
    assert "/mcp" in route_paths

    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    assert "query_logistics" in tool_names


@pytest.mark.asyncio
async def test_logistics_query_logistics_tool_logic():
    """验证物流查询工具业务逻辑：预置数据、动态生成及异常输入"""
    # 1. 预置订单 1001 (顺丰 派送中)
    res_1001 = json.loads(query_logistics("1001"))
    assert res_1001["order_id"] == "1001"
    assert res_1001["快递公司"] == "顺丰速运"
    assert res_1001["物流状态"] == "派送中"
    assert "SF" in res_1001["运单编号"]
    assert len(res_1001["轨迹明细"]) >= 1

    # 2. 预置订单 1002 (中通 运输中)
    res_1002 = json.loads(query_logistics("1002"))
    assert res_1002["order_id"] == "1002"
    assert res_1002["快递公司"] == "中通快递"
    assert res_1002["物流状态"] == "运输中"

    # 3. 预置订单 1003 (顺丰 已签收)
    res_1003 = json.loads(query_logistics("1003"))
    assert res_1003["order_id"] == "1003"
    assert res_1003["快递公司"] == "顺丰速运"
    assert res_1003["物流状态"] == "已签收"

    # 4. 未预置订单号动态生成
    res_random = json.loads(query_logistics("8888"))
    assert res_random["order_id"] == "8888"
    assert "快递公司" in res_random
    assert "运单编号" in res_random
    assert "物流状态" in res_random
    assert "轨迹明细" in res_random
    assert len(res_random["轨迹明细"]) >= 1

    # 5. 空入参处理
    res_empty = json.loads(query_logistics("   "))
    assert "error" in res_empty


@pytest.mark.asyncio
async def test_logistics_call_tool_mcp_interface():
    """验证物流 Server 通过 FastMCP 协议接口 call_tool 调用"""
    server = create_logistics_server()
    res = await server.call_tool("query_logistics", {"order_id": "1001"})
    # FastMCP call_tool 返回 tuple (Sequence[ContentBlock], dict)
    content_blocks, output_dict = res
    text_content = content_blocks[0].text
    data = json.loads(text_content)
    assert data["order_id"] == "1001"
    assert data["物流状态"] == "派送中"


@pytest.mark.asyncio
async def test_aftersale_mcp_server_properties_and_tools():
    """验证售后 Server 基础属性与工具导出列表"""
    server = create_aftersale_server()
    assert server.name == "AftersaleServer"
    assert server.settings.port == 8002
    assert server.settings.host == "127.0.0.1"

    app = server.streamable_http_app()
    route_paths = [r.path for r in app.routes]
    assert "/mcp" in route_paths

    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    assert "check_warranty" in tool_names
    assert "query_return_progress" in tool_names


@pytest.mark.asyncio
async def test_aftersale_check_warranty_tool_logic():
    """验证售后在保查询逻辑：预置订单与动态数据"""
    # 1. 在保订单 1001
    res_1001 = json.loads(check_warranty("1001"))
    assert res_1001["order_id"] == "1001"
    assert res_1001["质保状态"] == "在保"
    assert "质保截止日期" in res_1001
    assert "保修条款" in res_1001

    # 2. 过保订单 1003
    res_1003 = json.loads(check_warranty("1003"))
    assert res_1003["order_id"] == "1003"
    assert res_1003["质保状态"] == "已过保"

    # 3. 动态入参
    res_dynamic = json.loads(check_warranty("9999", product_name="智能降噪耳机"))
    assert res_dynamic["order_id"] == "9999"
    assert "智能降噪耳机" in res_dynamic["商品名称"]
    assert "质保状态" in res_dynamic

    # 4. 空入参
    res_empty = json.loads(check_warranty("  "))
    assert "error" in res_empty


@pytest.mark.asyncio
async def test_aftersale_query_return_progress_tool_logic():
    """验证售后退货进度查询逻辑"""
    # 1. 预置退货单/订单 1001
    res_1001 = json.loads(query_return_progress("1001"))
    assert res_1001["order_id"] == "1001"
    assert "退货申请" in res_1001["状态"]
    assert "更新时间" in res_1001

    # 2. 预置退货单/订单 1002
    res_1002 = json.loads(query_return_progress("1002"))
    assert res_1002["order_id"] == "1002"
    assert "验货入库" in res_1002["状态"]

    # 3. 动态退货单
    res_dynamic = json.loads(query_return_progress("RET20269999"))
    assert "状态" in res_dynamic
    assert "更新时间" in res_dynamic

    # 4. 空入参
    res_empty = json.loads(query_return_progress(""))
    assert "error" in res_empty


@pytest.mark.asyncio
async def test_aftersale_dynamic_tool_registration():
    """验证售后 Server 动态挂载新工具 (验收标准 3)"""
    server = create_aftersale_server()

    # 动态定义一个回收估价工具
    def evaluate_recycling_value(product_model: str, condition: str = "good") -> str:
        """评估二手商品以旧换新回收价格"""
        return json.dumps(
            {
                "product_model": product_model,
                "condition": condition,
                "estimated_value": "1200.00元",
            },
            ensure_ascii=False,
        )

    register_dynamic_tool(server, evaluate_recycling_value)

    # 验证 list_tools 能够探测到动态注册的工具
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    assert "evaluate_recycling_value" in tool_names

    # 验证 call_tool 可调用该动态工具
    res = await server.call_tool(
        "evaluate_recycling_value",
        {"product_model": "Phone 16", "condition": "like_new"},
    )
    content_blocks, _ = res
    ret_data = json.loads(content_blocks[0].text)
    assert ret_data["product_model"] == "Phone 16"
    assert ret_data["estimated_value"] == "1200.00元"
