import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from app.services.workflow.engine import WorkflowEngine, build_workflow_graph, logging_node

@pytest.mark.asyncio
async def test_workflow_engine_chitchat_e2e():
    """闲聊端到端工作流验证"""
    engine = WorkflowEngine()
    # 模拟意图为闲聊
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value={"intent": "闲聊", "intent_reason": "问候"})):
        result = await engine.run(conversation_id=201, query="你好呀")
        assert "智能客服助手" in result["response_text"]
        assert result["status"] == "chitchat"

@pytest.mark.asyncio
async def test_workflow_engine_complaint_e2e():
    """投诉端到端工作流验证"""
    engine = WorkflowEngine()
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value={"intent": "投诉", "intent_reason": "抗议"})):
        result = await engine.run(conversation_id=202, query="我要投诉客服")
        assert "非常抱歉" in result["response_text"]
        assert result["suggested_actions"] == ["transfer_agent", "create_ticket"]
        assert result["status"] == "complaint"

@pytest.mark.asyncio
async def test_workflow_engine_knowledge_fallback_e2e():
    """知识类检索未命中/低置信度时，置信度闸门分流至兜底话术节点"""
    engine = WorkflowEngine()
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value={"intent": "退款退货", "intent_reason": "询问退款"})):
        with patch("app.services.workflow.nodes.knowledge_node.knowledge_retrieval_node", new=AsyncMock(return_value={"retrieved_docs": []})):
            result = await engine.run(conversation_id=203, query="退货怎么算运费")
            assert "抱歉，关于您咨询的问题，当前知识库中暂未收录确切规范" in result["response_text"]
            assert result["confidence_passed"] is False
            assert result["status"] == "fallback"

@pytest.mark.asyncio
async def test_workflow_engine_knowledge_pass_to_main_agent_e2e():
    """知识检索命中且高置信度 (>=0.35) 时，置信度闸门放行至主力 Agent 节点"""
    engine = WorkflowEngine()
    fake_docs = [{"text": "7天无理由退货运费由买家承担，质量问题由商家承担", "score": 0.88}]
    fake_agent_result = {
        "messages": [AIMessage(content="质量问题退货运费由商家承担。")],
        "response_text": "质量问题退货运费由商家承担。",
        "steps_taken": 1,
        "status": "completed",
    }
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value={"intent": "退款退货", "intent_reason": "退货运费"})):
        with patch("app.services.workflow.nodes.knowledge_node.knowledge_retrieval_node", new=AsyncMock(return_value={"retrieved_docs": fake_docs})):
            with patch("app.services.workflow.nodes.agent_node.main_agent_node", new=AsyncMock(return_value=fake_agent_result)):
                result = await engine.run(conversation_id=205, query="退货谁出运费")
                assert "商家承担" in result["response_text"]
                assert result["status"] == "completed"

@pytest.mark.asyncio
async def test_workflow_engine_business_data_main_agent_e2e():
    """业务数据类意图分流至主力 Agent 节点并记录审计日志"""
    engine = WorkflowEngine()
    fake_agent_result = {
        "messages": [AIMessage(content="您的订单已发货，快递单号为 SF123456。")],
        "response_text": "您的订单已发货，快递单号为 SF123456。",
        "steps_taken": 2,
        "status": "completed",
    }
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value={"intent": "物流", "intent_reason": "查询物流"})):
        with patch("app.services.workflow.nodes.agent_node.main_agent_node", new=AsyncMock(return_value=fake_agent_result)):
            result = await engine.run(conversation_id=204, query="我的快递到哪了")
            assert "SF123456" in result["response_text"]
            assert result["status"] == "completed"
            assert result["steps_taken"] == 2

def test_logging_node_execution():
    """验证日志与审计节点输出为空字典且无异常"""
    state = {
        "conversation_id": 999,
        "intent": "闲聊",
        "status": "chitchat",
        "steps_taken": 0,
        "suggested_actions": [],
    }
    res = logging_node(state)
    assert res == {}

def test_build_workflow_graph_custom_checkpointer():
    """验证可注入自定义 Checkpointer 编译工作流图"""
    custom_saver = MemorySaver()
    graph = build_workflow_graph(checkpointer=custom_saver)
    assert graph is not None
