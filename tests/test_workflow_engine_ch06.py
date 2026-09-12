import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

from app.services.workflow.engine import WorkflowEngine, build_workflow_graph
from app.services.workflow.nodes.router import route_by_intent, route_refund_slot


# ==========================================
# 1. 单元测试：8 分类意图分流路由器条件边
# ==========================================

def test_route_by_intent_eight_classes():
    """验证 route_by_intent 正式版 8 分类路由映射（含未收录意图兜底）"""
    assert route_by_intent({"intent": "闲聊"}) == "chitchat"
    assert route_by_intent({"intent": "投诉"}) == "complaint"
    assert route_by_intent({"intent": "其他"}) == "other_fallback"
    assert route_by_intent({"intent": "商品咨询"}) == "knowledge"
    assert route_by_intent({"intent": "退款退货"}) == "refund_subflow"
    assert route_by_intent({"intent": "售后"}) == "refund_subflow"
    assert route_by_intent({"intent": "物流"}) == "business_data"
    assert route_by_intent({"intent": "订单"}) == "business_data"
    # 缺省与未知分类统一走 other_fallback
    assert route_by_intent({"intent": "未知意图"}) == "other_fallback"
    assert route_by_intent({"intent": None}) == "other_fallback"
    assert route_by_intent({}) == "other_fallback"


# ==========================================
# 2. 单元测试：退款槽位状态条件边
# ==========================================

def test_route_refund_slot():
    """验证 route_refund_slot 在缺失订单号挂起 vs 命中订单继续流转的判定"""
    # 缺单挂起 -> need_order (跳转 logging 终止此轮)
    assert route_refund_slot({"status": "need_order_selection"}) == "need_order"
    # 已提取/预取订单 -> has_order (流向政策扩写检索与主力 Agent)
    assert route_refund_slot({"status": "order_fetched"}) == "has_order"
    assert route_refund_slot({"order_id": "1001", "status": "completed"}) == "has_order"
    assert route_refund_slot({}) == "has_order"


# ==========================================
# 3. 全图端到端：闲聊流直通 chitchat
# ==========================================

@pytest.mark.asyncio
async def test_workflow_engine_chitchat_e2e_ch06():
    """闲聊直通固定亲和话术，不调检索也不入 Agent"""
    engine = WorkflowEngine()
    fake_intent = {"intent": "闲聊", "confidence": 0.99, "intent_reason": "日常打招呼"}
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)):
        result = await engine.run(conversation_id=601, query="你好呀客服")
        assert "智能客服助手" in result["response_text"]
        assert result["status"] == "chitchat"
        assert result["suggested_actions"] == []


# ==========================================
# 4. 全图端到端：超出范围问题流向 other_fallback
# ==========================================

@pytest.mark.asyncio
async def test_workflow_engine_other_fallback_e2e_ch06():
    """怪问题/离群意图直通 other_fallback 友好解释并推荐转人工客服"""
    engine = WorkflowEngine()
    fake_intent = {"intent": "其他", "confidence": 0.98, "intent_reason": "询问非电商天气"}
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)):
        result = await engine.run(conversation_id=602, query="今天北京天气怎么样")
        assert "超出我的业务范围" in result["response_text"]
        assert result["status"] == "other_fallback"
        assert "transfer_agent" in result["suggested_actions"]


# ==========================================
# 5. 全图端到端：无单号退款诉求触发槽位中断挂起
# ==========================================

@pytest.mark.asyncio
async def test_workflow_engine_refund_without_order_slot_interruption():
    """无单号咨询退款退货，工作流在 refund_order_check 挂起，下发 need_order_selection 与卡片"""
    engine = WorkflowEngine()
    fake_intent = {"intent": "退款退货", "confidence": 0.96, "intent_reason": "申请退货退款"}
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)):
        result = await engine.run(conversation_id=603, query="这件衣服我想退掉")
        assert result["status"] == "need_order_selection"
        assert "请先选择您需要咨询的订单" in result["response_text"]
        assert result.get("suggested_orders") is not None
        assert len(result["suggested_orders"]) >= 1
        # 验证包含 1001 等 Mock 订单卡片字段
        order_ids = [o["order_id"] for o in result["suggested_orders"]]
        assert "1001" in order_ids


# ==========================================
# 6. 全图端到端：有单号退款诉求完整走通确定性退款子流程
# ==========================================

@pytest.mark.asyncio
async def test_workflow_engine_refund_with_order_deterministic_subflow():
    """有单号退款提问：refund_order_check 预取数据 -> refund_expansion_retrieval 检索政策 -> main_agent 裁决下发 apply_refund"""
    engine = WorkflowEngine()
    fake_intent = {"intent": "退款退货", "confidence": 0.98, "intent_reason": "订单退款咨询"}
    fake_expansion_docs = {
        "retrieved_docs": [
            {"text": "七天无理由退换货规则：签收后7日内商品完好未洗涤可申请退货", "score": 0.92}
        ]
    }
    fake_agent_result = {
        "messages": [AIMessage(content="您的订单1001在七天无理由退货期内，支持办理退货。")],
        "response_text": "您的订单1001在七天无理由退货期内，支持办理退货。",
        "suggested_actions": ["apply_refund"],
        "status": "completed",
        "steps_taken": 1,
    }

    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_intent)):
        with patch("app.services.workflow.nodes.refund_nodes.refund_expansion_retrieval_node", new=AsyncMock(return_value=fake_expansion_docs)):
            with patch("app.services.workflow.nodes.agent_node.main_agent_node", new=AsyncMock(return_value=fake_agent_result)):
                result = await engine.run(conversation_id=604, query="订单1001可以退吗")
                assert result["order_id"] == "1001"
                assert result["order_data"] is not None
                assert result["order_data"]["order_id"] == "1001"
                assert "支持办理退货" in result["response_text"]
                assert "apply_refund" in result["suggested_actions"]
                assert result["status"] == "completed"


# ==========================================
# 7. 全图端到端：商品咨询走知识库，物流走业务直通主力 Agent
# ==========================================

@pytest.mark.asyncio
async def test_workflow_engine_knowledge_and_logistics_flows():
    """验证知识咨询走 knowledge_retrieval，物流咨询直通 main_agent"""
    engine = WorkflowEngine()

    # 场景 A: 商品咨询 -> knowledge_retrieval -> gate pass -> main_agent
    fake_product_intent = {"intent": "商品咨询", "confidence": 0.95, "intent_reason": "尺码咨询"}
    fake_docs = [{"text": "该羽绒服版型标准，建议按平时尺码购买L码即可", "score": 0.85}]
    fake_agent_result_kb = {
        "messages": [AIMessage(content="建议您选购L码。")],
        "response_text": "建议您选购L码。",
        "steps_taken": 1,
        "status": "completed",
    }
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_product_intent)):
        with patch("app.services.workflow.nodes.knowledge_node.knowledge_retrieval_node", new=AsyncMock(return_value={"retrieved_docs": fake_docs})):
            with patch("app.services.workflow.nodes.agent_node.main_agent_node", new=AsyncMock(return_value=fake_agent_result_kb)):
                result_kb = await engine.run(conversation_id=605, query="羽绒服什么尺码合适")
                assert "L码" in result_kb["response_text"]
                assert result_kb["status"] == "completed"

    # 场景 B: 物流 -> business_data -> main_agent 直通
    fake_logistics_intent = {"intent": "物流", "confidence": 0.95, "intent_reason": "查询快递进度"}
    fake_agent_result_logistics = {
        "messages": [AIMessage(content="您的包裹已由顺丰速运揽收，正在运往北京转运中心。")],
        "response_text": "您的包裹已由顺丰速运揽收，正在运往北京转运中心。",
        "steps_taken": 2,
        "status": "completed",
    }
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value=fake_logistics_intent)):
        with patch("app.services.workflow.nodes.agent_node.main_agent_node", new=AsyncMock(return_value=fake_agent_result_logistics)):
            result_logistics = await engine.run(conversation_id=606, query="我的包裹发货了吗")
            assert "顺丰速运" in result_logistics["response_text"]
            assert result_logistics["status"] == "completed"
