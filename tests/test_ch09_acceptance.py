"""
Chapter 09 Acceptance Tests: End-to-End Acceptance Suite
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
import os
import json
from unittest.mock import patch, MagicMock, AsyncMock
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from app.config import settings
from app.db.session import Base, get_db
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.eval_run import EvalRun, TriggeredBy
from app.models.review_queue import ReviewQueue, ReviewStatus
from app.models.low_confidence import LowConfidenceQuestion
from app.models.knowledge import KnowledgeChunk

from app.services.observability.langfuse_service import LangfuseManager
from app.services.observability.cost_analytics import CostAnalyticsService
from app.services.workflow.nodes.gate import compute_evidence_confidence, confidence_gate
from app.services.workflow.nodes.pre_nodes import intent_recognition_node
from app.services.workflow.nodes.knowledge_node import knowledge_fallback_node
from app.services.workflow.engine import build_workflow_graph
from app.services.rag.generator import RAGControlledGenerator
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.services.flywheel.pipeline import FlywheelPipeline
from app.services.eval.eval_pipeline import EvalPipelineService

@pytest_asyncio.fixture
async def memory_db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestingSessionLocal = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with TestingSessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
def test_app(memory_db):
    async def override_get_db():
        yield memory_db

    app.dependency_overrides[get_db] = override_get_db
    yield app
    app.dependency_overrides.clear()


# =========================================================================
# Acceptance 1: Langfuse 观测与宽容降级验证
# =========================================================================
@pytest.mark.asyncio
async def test_acceptance_1_langfuse():
    # 1. 未配置或禁用时，优雅降级返回 None，不抛出异常
    with patch.object(settings, "langfuse_enabled", False), \
         patch.object(settings, "langfuse_public_key", None), \
         patch.object(settings, "langfuse_secret_key", None):
        assert LangfuseManager.get_callback_handler() is None
        LangfuseManager.update_current_trace(metadata={"test": "val"})  # 安全无操作
    
    # 2. 启用模式在有效配置下可正常判断
    with patch.object(settings, "langfuse_enabled", True), \
         patch.object(settings, "langfuse_public_key", "pk_test"), \
         patch.object(settings, "langfuse_secret_key", "sk_test"), \
         patch.object(settings, "langfuse_host", "http://localhost:3000"):
        assert LangfuseManager.is_enabled() is True

    # 3. LangGraph 编译期单点注入验证
    graph = build_workflow_graph()
    assert graph is not None
    assert hasattr(graph, "ainvoke")


# =========================================================================
# Acceptance 2: 意图成本账 (Cost Control) 验证
# =========================================================================
@pytest.mark.asyncio
async def test_acceptance_2_cost_analytics():
    # 1. 意图节点 Trace 根元数据注入逻辑验证
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=MagicMock(content='{"intent": "售后", "confidence": 0.95}'))
    state = {"input_query": "东西坏了怎么修"}
    with patch("app.services.observability.langfuse_service.LangfuseManager.update_current_trace") as mock_update:
        res_state = await intent_recognition_node(state, model=mock_model)
        assert res_state["intent"] == "售后"
        mock_update.assert_called_once_with(metadata={"intent": "售后"}, tags=["售后"])

    # 2. 意图成本汇总与排序计算验证
    service = CostAnalyticsService()
    records = [
        {"intent": "qa", "prompt_tokens": 100, "completion_tokens": 20},
        {"intent": "refund", "prompt_tokens": 500, "completion_tokens": 100},
        {"intent": "refund", "prompt_tokens": 300, "completion_tokens": 50},
        {"intent": "chat", "prompt_tokens": 50, "completion_tokens": 10},
    ]
    report = service.aggregate_records(records)
    
    # 最贵意图排在第 1 位
    assert report["items"][0]["intent"] == "refund"
    assert report["items"][0]["total_calls"] == 2
    # 各项占比总和为 100%
    total_pct = sum(item["cost_percentage"] for item in report["items"])
    assert abs(total_pct - 100.0) < 0.1


# =========================================================================
# Acceptance 3: 证据置信度闸门与网格校准验证
# =========================================================================
@pytest.mark.asyncio
async def test_acceptance_3_evidence_confidence():
    # 1. 空证据置信度为 0
    assert compute_evidence_confidence([]) == 0.0

    # 2. 高分强证据 + 首位分差大 -> 高置信度
    high_docs = [
        {"score": 0.88, "text": "7天无理由退货条款"},
        {"score": 0.30, "text": "其他弱相关段落"},
    ]
    score_high = compute_evidence_confidence(high_docs)
    assert score_high >= 0.50
    assert confidence_gate({"retrieved_docs": high_docs}) == "pass"

    # 3. 弱证据 -> 低置信度 -> 触发 fallback 分支
    low_docs = [{"score": 0.15, "text": "弱匹配干扰段落"}]
    score_low = compute_evidence_confidence(low_docs)
    assert score_low < 0.40
    assert confidence_gate({"retrieved_docs": low_docs}) == "fallback"


# =========================================================================
# Acceptance 4: 低置信度三路落池与快照回捞闭环验证
# =========================================================================
@pytest.mark.asyncio
async def test_acceptance_4_low_confidence_three_entry_points(test_app, memory_db):
    # Entry 1: 闸门拦截落池 (retrieval_low_conf)
    state = {
        "input_query": "你们包邮吗？",
        "conversation_id": 101,
        "retrieved_docs": [{"text": "非包邮偏远地区说明", "score": 0.18, "section_path": "运费规则"}]
    }
    await knowledge_fallback_node(state, db=memory_db)
    
    q1 = (await memory_db.execute(
        select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == 101)
    )).scalar_one()
    assert q1.source == "retrieval_low_conf"
    assert len(q1.retrieved_chunks) == 1
    assert q1.retrieved_chunks[0]["score"] == 0.18

    # Entry 2: 生成端自评不足落池 (self_check)
    generator = RAGControlledGenerator(model=None, stream_model=None)
    citations = [{"text": "普通客服对话段落", "score": 0.7, "section_path": "通用规范"}]
    # 触发自评知识不足
    await generator.check_sufficiency(
        query="火星旅行退改签规则",
        citations=citations,
        db=memory_db,
        conversation_id=102
    )
    q2 = (await memory_db.execute(
        select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == 102)
    )).scalar_one()
    assert q2.source == "self_check"
    assert len(q2.retrieved_chunks) >= 1

    # Entry 3: 用户点踩反馈回捞落池 (user_feedback)
    conv = Conversation(id=103)
    msg = Message(id=203, conversation_id=103, role="user", content="买的衣服掉色严重怎么维权？")
    memory_db.add_all([conv, msg])
    await memory_db.commit()

    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        resp = await ac.post("/api/chat/feedback", json={
            "conversation_id": 103,
            "message_id": 203,
            "feedback_type": "down",
            "reason": "回答不满意"
        })
        assert resp.status_code == 200
        
    q3 = (await memory_db.execute(
        select(LowConfidenceQuestion).where(LowConfidenceQuestion.conversation_id == 103)
    )).scalar_one()
    assert q3.source == "user_feedback"
    assert q3.raw_question == "买的衣服掉色严重怎么维权？"


# =========================================================================
# Acceptance 5: 飞轮流水线标准化与两阶段查重归并验证
# =========================================================================
@pytest.mark.asyncio
async def test_acceptance_5_flywheel_pipeline_normalize_and_deduplicate(memory_db):
    pipeline = FlywheelPipeline()

    class MockFlywheelLLM:
        def __init__(self):
            self.calls = 0
        def invoke(self, prompt):
            p = str(prompt)
            if "FAQ" in p:
                self.calls += 1
                if self.calls == 1:
                    return type('O', (), {'content': "如何申请保价退差价？\n\n可在订单页点击申请保价。"})()
                else:
                    return type('O', (), {'content': "保价流程说明\n\n订单详情中申请退差价。"})()
            if "NONE" in p:
                return type('O', (), {'content': "YES"})()
            return type('O', (), {'content': ""})()

    mock_llm = MockFlywheelLLM()

    # 1. 插入第一道问题并处理
    q1 = LowConfidenceQuestion(raw_question="双十一买贵了怎么退差价啊", source="retrieval_low_conf")
    memory_db.add(q1)
    await memory_db.commit()

    res1 = await pipeline.process_pending_questions(memory_db, batch_size=10, llm=mock_llm)
    assert res1["processed_count"] == 1
    assert res1["new_created"] == 1
    assert res1["merged_count"] == 0

    await memory_db.refresh(q1)
    assert q1.matched_review_id is not None
    review_item = await memory_db.get(ReviewQueue, q1.matched_review_id)
    assert review_item.occurrence_count == 1
    assert "保价" in review_item.normalized_question

    # 2. 插入第二道同义问题并查重归并
    q2 = LowConfidenceQuestion(raw_question="保价退钱怎么操作", source="user_feedback")
    memory_db.add(q2)
    await memory_db.commit()

    res2 = await pipeline.process_pending_questions(memory_db, batch_size=10, llm=mock_llm)
    assert res2["processed_count"] == 1
    assert res2["new_created"] == 0
    assert res2["merged_count"] == 1

    await memory_db.refresh(q2)
    assert q2.matched_review_id == review_item.id
    await memory_db.refresh(review_item)
    assert review_item.occurrence_count == 2


# =========================================================================
# Acceptance 6: 审核通过与 Ch03 知识库双写闭环验证
# =========================================================================
@pytest.mark.asyncio
async def test_acceptance_6_review_approval_and_knowledge_dual_write(test_app, memory_db):
    review_item = ReviewQueue(
        normalized_question="双十一保价标准答复",
        ai_suggested_answer="请在订单页申请",
        occurrence_count=3,
        review_status=ReviewStatus.PENDING.value
    )
    memory_db.add(review_item)
    await memory_db.commit()
    await memory_db.refresh(review_item)

    # 审核通过，集成 Ch03 KnowledgeDualWriter 双写落库
    with patch("app.api.review_queue_routes.KnowledgeDualWriter") as MockWriter:
        mock_writer_instance = MockWriter.return_value
        mock_writer_instance.write_chunks = AsyncMock(return_value=[])

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
            resp = await ac.post(
                f"/api/review-queue/{review_item.id}/approve",
                json={"approved_answer": "订单详情页点击申请保价，系统自动返还原路退回。"}
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "approved"

        # 验证审核状态变为 APPROVED
        await memory_db.refresh(review_item)
        assert review_item.review_status == ReviewStatus.APPROVED.value
        assert "申请保价" in review_item.approved_answer

        # 验证 Ch03 KnowledgeDualWriter 被成功调用且参数正确
        mock_writer_instance.write_chunks.assert_called_once()
        chunks = mock_writer_instance.write_chunks.call_args[0][1]
        assert len(chunks) == 1
        assert chunks[0].questions == "双十一保价标准答复"
        assert chunks[0].category == "常见问题"
        assert chunks[0].content_type == "faq"


# =========================================================================
# Acceptance 7: 自动化评测时序趋势与只读看板 API 验证
# =========================================================================
@pytest.mark.asyncio
async def test_acceptance_7_eval_trend_and_overview_api(test_app, memory_db):
    eval_service = EvalPipelineService()
    # 连续执行 2 轮评测并持久化
    run1 = await eval_service.run_eval_round(db=memory_db, sample_limit=2)
    run2 = await eval_service.run_eval_round(db=memory_db, sample_limit=3)
    
    # 验证 Delta 趋势计算
    runs_with_deltas = await eval_service.get_recent_runs_with_deltas(db=memory_db, limit=10)
    assert len(runs_with_deltas) == 2
    latest_run = runs_with_deltas[0]
    assert "deltas" in latest_run
    assert "recall_at_3" in latest_run["deltas"]
    assert "recall_at_3_trend" in latest_run["deltas"]

    # 验证 /api/observability/overview 只读端点
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as ac:
        resp = await ac.get("/api/observability/overview")
        assert resp.status_code == 200
        overview = resp.json()
        assert "cost_block" in overview
        assert "eval_trend_block" in overview
        assert "calibration_block" in overview
        # 评测趋势块处于就绪状态
        assert overview["eval_trend_block"]["present"] is True
        assert len(overview["eval_trend_block"]["runs"]) == 2
