import pytest
from langchain_core.messages import HumanMessage, AIMessage
from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.refund_nodes import (
    extract_order_id,
    refund_order_check_node,
    emit_order_selector_node,
)


def test_order_id_extraction_from_query_prefix():
    """提问中包含前缀形式订单号，如「退款订单: 1001」或「订单1001可以退款吗」"""
    # 场景 1: 退款订单: 1001
    state1 = create_initial_state(1, "退款订单: 1001")
    res1 = refund_order_check_node(state1)
    assert res1["status"] == "order_fetched"
    assert res1["order_id"] == "1001"
    assert res1["order_data"] is not None
    assert res1["order_data"]["order_id"] == "1001"
    assert res1["order_data"]["支付金额"] == "299.00元"
    assert res1["order_data"]["商品明细"][0]["商品名称"] == "极简保暖羽绒服"

    # 场景 2: 订单1001可以退款吗
    state2 = create_initial_state(2, "订单1001可以退款吗")
    res2 = refund_order_check_node(state2)
    assert res2["status"] == "order_fetched"
    assert res2["order_id"] == "1001"
    assert res2["order_data"]["order_id"] == "1001"


def test_order_id_extraction_from_resolved_query():
    """当前 input_query 无订单号，但前置指代消解节点输出的 resolved_query 包含订单号"""
    state = create_initial_state(1, "这件衣服可以申请退货退款吗")
    state["resolved_query"] = "订单1002潮流纯棉连帽卫衣支持退货退款吗"
    res = refund_order_check_node(state)
    assert res["status"] == "order_fetched"
    assert res["order_id"] == "1002"
    assert res["order_data"] is not None
    assert res["order_data"]["order_id"] == "1002"
    assert res["order_data"]["支付金额"] == "159.00元"
    assert res["order_data"]["商品明细"][0]["商品名称"] == "潮流纯棉连帽卫衣"


def test_missing_order_id_triggers_need_order_selection():
    """用户仅提「我想退货」无订单号，拦截挂起并装配推荐候选订单卡片数据"""
    state = create_initial_state(1, "我想退货")
    res = refund_order_check_node(state)

    assert res["status"] == "need_order_selection"
    assert res["order_id"] is None
    assert res["order_data"] is None
    assert res["response_text"] == "为您查询退款政策前，请先选择您需要咨询的订单："
    assert isinstance(res["suggested_orders"], list)
    assert len(res["suggested_orders"]) == 3

    # 验证候选卡片字段
    order_ids = [item["order_id"] for item in res["suggested_orders"]]
    assert "1001" in order_ids
    assert "1002" in order_ids
    assert "1003" in order_ids
    first_item = res["suggested_orders"][0]
    assert "商品名称" in first_item or "product_name" in first_item
    assert "支付金额" in first_item or "amount" in first_item or "price" in first_item
    assert "订单状态" in first_item or "status" in first_item


def test_emit_order_selector_node_directly():
    """独立调用 emit_order_selector_node 节点输出规范格式"""
    state = create_initial_state(1, "申请售后退款")
    res = emit_order_selector_node(state)
    assert res["status"] == "need_order_selection"
    assert res["order_id"] is None
    assert res["order_data"] is None
    assert res["response_text"] == "为您查询退款政策前，请先选择您需要咨询的订单："
    assert len(res["suggested_orders"]) == 3


def test_order_id_extraction_from_history_messages():
    """订单号存在于前序历史消息中（多轮对话场景），倒序精准提取"""
    state = create_initial_state(1, "能退吗")
    state["resolved_query"] = "能退吗"
    state["messages"] = [
        HumanMessage(content="你好，我前几天买的订单1003"),
        AIMessage(content="您好！已为您查到订单1003【经典修身牛仔裤】，请问需要什么帮助？"),
        HumanMessage(content="能退吗"),
    ]
    res = refund_order_check_node(state)
    assert res["status"] == "order_fetched"
    assert res["order_id"] == "1003"
    assert res["order_data"]["order_id"] == "1003"
    assert res["order_data"]["商品明细"][0]["商品名称"] == "经典修身牛仔裤"


def test_history_messages_ai_message_not_mistakenly_extracted():
    """历史消息中 AI 提示候选订单列表（如包含 1001、1002），不应被误判为用户选中的订单号"""
    state = create_initial_state(1, "随便")
    state["resolved_query"] = "随便"
    state["messages"] = [
        HumanMessage(content="我想退货"),
        AIMessage(content="请先选择订单：1001、1002 或 1003"),
        HumanMessage(content="随便"),
    ]
    res = refund_order_check_node(state)
    # 因为用户没有明确指定订单号，AI 消息中的 1001 不得误判为用户订单
    assert res["status"] == "need_order_selection"
    assert res["order_id"] is None


def test_extract_order_id_pure_function():
    """extract_order_id 纯函数的各种边界提取规则单元测试"""
    # 字符串入参
    assert extract_order_id("退款订单: 1001") == "1001"
    assert extract_order_id("退款订单:1001") == "1001"
    assert extract_order_id("订单1001可以退款吗") == "1001"
    assert extract_order_id("订单号: 1002") == "1002"
    assert extract_order_id("订单号是1003") == "1003"
    assert extract_order_id("我的单号: TB987654") == "TB987654"
    assert extract_order_id("TB1001能退吗") == "TB1001"
    assert extract_order_id("1002") == "1002"
    assert extract_order_id("我想退1003") == "1003"

    # 排除干扰数字（金额、年份等）
    assert extract_order_id("退款3000元") is None
    assert extract_order_id("2026年9月5日购买") is None
    assert extract_order_id("我想退货") is None
    assert extract_order_id("") is None
    assert extract_order_id(None) is None

    # 关键字入参 query
    assert extract_order_id(query="订单1001") == "1001"

    # 优先级 1: state["order_id"] 显式存在优先
    assert extract_order_id(state={"order_id": "1001", "resolved_query": "订单1002"}) == "1001"

    # 优先级 2: resolved_query 优于 input_query
    assert extract_order_id(state={"order_id": None, "resolved_query": "订单1002", "input_query": "订单1001"}) == "1002"

    # 优先级 3: messages 倒序提取用户消息
    hist_state = {
        "order_id": None,
        "resolved_query": "这个能退吗",
        "input_query": "这个能退吗",
        "messages": [
            HumanMessage(content="查询订单1001"),
            AIMessage(content="订单1001羽绒服已发货"),
            HumanMessage(content="查询订单1003"),
            AIMessage(content="订单1003牛仔裤已完成"),
        ],
    }
    # 倒序检查应该匹配到最新的 1003
    assert extract_order_id(state=hist_state) == "1003"


def test_non_mock_order_id_fallback():
    """未在预设缓存的真实格式订单号（如 9999），通过工具 fallback 返回模拟结构"""
    state = create_initial_state(1, "订单9999可以退款吗")
    res = refund_order_check_node(state)
    assert res["status"] == "order_fetched"
    assert res["order_id"] == "9999"
    assert res["order_data"] is not None
    assert res["order_data"]["order_id"] == "9999"
    assert "订单状态" in res["order_data"]
    assert "商品明细" in res["order_data"]


@pytest.mark.asyncio
async def test_async_await_compatibility():
    """验证 refund_order_check_node 与 emit_order_selector_node 支持直接 await"""
    state = create_initial_state(1, "订单1001可以退款吗")
    res = await refund_order_check_node(state)
    assert res["status"] == "order_fetched"
    assert res["order_id"] == "1001"

    state_empty = create_initial_state(2, "我想退款")
    res_empty = await refund_order_check_node(state_empty)
    assert res_empty["status"] == "need_order_selection"

    res_emit = await emit_order_selector_node(state_empty)
    assert res_emit["status"] == "need_order_selection"
