import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.session import Base
from app.models import Conversation
from app.models.low_confidence import LowConfidenceQuestion, LowConfidenceSource, record_low_confidence
from app.models.faith_case import FaithCase, FaithCaseStatus, upsert_faith_case
from scripts.init_ch04_db import split_sql_statements, parse_ddl_statements, init_ch04_db


def test_low_confidence_question_attributes_and_defaults():
    lcq = LowConfidenceQuestion(
        raw_question="怎么退款？",
        source="retrieval_low_conf",
        reason="未检索到相关退款知识",
    )
    assert lcq.raw_question == "怎么退款？"
    assert lcq.source == "retrieval_low_conf"
    assert lcq.reason == "未检索到相关退款知识"
    assert lcq.conversation_id is None
    assert lcq.id is None


def test_faith_case_attributes_and_defaults():
    citations_data = [
        {
            "n": 1,
            "chunk_id": 10,
            "section_path": "售后政策 > 退款指南",
            "question": "退款要多久？",
            "answer": "退款将在1-3个工作日内原路退回。",
        }
    ]
    case = FaithCase(
        eval_id="A43",
        bucket="A_policy",
        query="退款要多久？",
        answer="退款将在当天立即可用且赠送100元代金券。",
        reason="证据原文明确说明1-3个工作日，模型编造了立即可用与赠送100元代金券",
        citations=citations_data,
    )
    assert case.eval_id == "A43"
    assert case.bucket == "A_policy"
    assert case.query == "退款要多久？"
    assert case.strategy == "hybrid_rerank"
    assert case.answer == "退款将在当天立即可用且赠送100元代金券。"
    assert case.reason == "证据原文明确说明1-3个工作日，模型编造了立即可用与赠送100元代金券"
    assert case.citations == citations_data
    assert case.status == "未解决"
    assert case.seen_count == 1
    assert case.judge_model is None
    assert case.resolution is None
    assert case.resolved_at is None
    assert case.id is None


def test_models_table_metadata():
    assert LowConfidenceQuestion.__tablename__ == "low_confidence_questions"
    assert FaithCase.__tablename__ == "faith_cases"

    # low_confidence_questions columns
    lcq_cols = set(LowConfidenceQuestion.__table__.columns.keys())
    expected_lcq_cols = {
        "id",
        "conversation_id",
        "raw_question",
        "source",
        "reason",
        "created_at",
    }
    assert expected_lcq_cols.issubset(lcq_cols)

    # faith_cases columns
    fc_cols = set(FaithCase.__table__.columns.keys())
    expected_fc_cols = {
        "id",
        "eval_id",
        "bucket",
        "query",
        "strategy",
        "answer",
        "reason",
        "citations",
        "judge_model",
        "status",
        "seen_count",
        "first_seen_at",
        "last_seen_at",
        "resolution",
        "resolved_at",
    }
    assert expected_fc_cols.issubset(fc_cols)

    # Primary key assertions
    assert [c.name for c in LowConfidenceQuestion.__table__.primary_key.columns] == ["id"]
    assert [c.name for c in FaithCase.__table__.primary_key.columns] == ["id"]

    # Nullable assertions
    table_lcq = LowConfidenceQuestion.__table__
    assert table_lcq.c.raw_question.nullable is False
    assert table_lcq.c.source.nullable is False
    assert table_lcq.c.conversation_id.nullable is True
    assert table_lcq.c.reason.nullable is True

    table_fc = FaithCase.__table__
    assert table_fc.c.eval_id.nullable is False
    assert table_fc.c.bucket.nullable is False
    assert table_fc.c.query.nullable is False
    assert table_fc.c.strategy.nullable is False
    assert table_fc.c.answer.nullable is False
    assert table_fc.c.reason.nullable is False
    assert table_fc.c.citations.nullable is True
    assert table_fc.c.status.nullable is False
    assert table_fc.c.seen_count.nullable is False

    # Foreign key assertions on low_confidence_questions
    lcq_fk_targets = {fk.target_fullname for fk in table_lcq.c.conversation_id.foreign_keys}
    assert "conversations.id" in lcq_fk_targets

    # Unique constraint assertions on faith_cases (eval_id)
    unique_col_names = set()
    for uc in table_fc.constraints:
        if hasattr(uc, "columns") and (getattr(uc, "unique", False) or type(uc).__name__ == "UniqueConstraint"):
            for col in uc.columns:
                unique_col_names.add(col.name)
    assert "eval_id" in unique_col_names or table_fc.c.eval_id.unique is True

    # Index assertions
    lcq_indexed_cols = {c.name for idx in table_lcq.indexes for c in idx.columns}
    assert "source" in lcq_indexed_cols or table_lcq.c.source.index is True
    assert "created_at" in lcq_indexed_cols or table_lcq.c.created_at.index is True

    fc_indexed_cols = {c.name for idx in table_fc.indexes for c in idx.columns}
    assert "status" in fc_indexed_cols or table_fc.c.status.index is True
    assert "last_seen_at" in fc_indexed_cols or table_fc.c.last_seen_at.index is True


def test_sqlite_in_memory_crud():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # 1. Create a Conversation
        conv = Conversation(user_id="user_ch04_test", status="进行中")
        session.add(conv)
        session.commit()
        assert conv.id is not None
        conv_id = conv.id

        # 2. Create LowConfidenceQuestion linked to conversation
        lcq = LowConfidenceQuestion(
            conversation_id=conv_id,
            raw_question="能便宜一点吗？",
            source="retrieval_low_conf",
            reason="非标准议价问题，无匹配 FAQ 条款",
        )
        session.add(lcq)
        session.commit()
        assert lcq.id is not None
        assert lcq.conversation_id == conv_id

        # 3. Create FaithCase with JSON citations
        citations = [
            {
                "n": 1,
                "chunk_id": 42,
                "section_path": "服务协议 > 保修",
                "question": "保修范围",
                "answer": "非人为损坏提供一年保修。",
            }
        ]
        fc = FaithCase(
            eval_id="B12",
            bucket="B_model",
            query="保修期多久？",
            strategy="hybrid_rerank",
            answer="保修期为永久保修。",
            reason="证据表明保修一年，模型回答永久保修",
            citations=citations,
            judge_model="qwen-turbo",
            status="未解决",
            seen_count=1,
        )
        session.add(fc)
        session.commit()
        assert fc.id is not None

        # 4. Query and verify JSON serialization
        queried_fc = session.get(FaithCase, fc.id)
        assert queried_fc is not None
        assert queried_fc.eval_id == "B12"
        assert queried_fc.citations == citations
        assert queried_fc.status == "未解决"

        # 5. Verify eval_id uniqueness constraint (duplicate eval_id raises IntegrityError)
        duplicate_fc = FaithCase(
            eval_id="B12",
            bucket="B_model",
            query="保修期多久？（重复录入）",
            strategy="hybrid_rerank",
            answer="重复记录",
            reason="重复测试",
        )
        session.add(duplicate_fc)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


@pytest.mark.asyncio
async def test_record_low_confidence_helper():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    res = await record_low_confidence(
        db=mock_session,
        raw_question="请问有大促优惠吗？",
        source="self_check",
        conversation_id=88,
        reason="生成前置自评判为知识不足以回答",
    )

    assert isinstance(res, LowConfidenceQuestion)
    assert res.raw_question == "请问有大促优惠吗？"
    assert res.source == "self_check"
    assert res.conversation_id == 88
    assert res.reason == "生成前置自评判为知识不足以回答"

    mock_session.add.assert_called_once_with(res)
    mock_session.commit.assert_awaited_once()
    mock_session.refresh.assert_awaited_once_with(res)


@pytest.mark.asyncio
async def test_upsert_faith_case_new():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    # Mock execute returning None (new case)
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    citations = [{"n": 1, "chunk_id": 99, "section_path": "售后", "question": "退货", "answer": "7天"}]
    res = await upsert_faith_case(
        db=mock_session,
        eval_id="C05",
        bucket="C_colloquial",
        query="怎么退？",
        answer="直接扔了就行",
        reason="编造严重偏离退货流程",
        strategy="hybrid_rerank",
        citations=citations,
        judge_model="deepseek-flash",
    )

    assert isinstance(res, FaithCase)
    assert res.eval_id == "C05"
    assert res.bucket == "C_colloquial"
    assert res.seen_count == 1
    assert res.status == "未解决"
    assert res.resolution is None
    assert res.resolved_at is None
    assert res.citations == citations

    mock_session.add.assert_called_once_with(res)
    mock_session.commit.assert_awaited_once()
    mock_session.refresh.assert_awaited_once_with(res)


@pytest.mark.asyncio
async def test_upsert_faith_case_existing_unresolved():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    existing = FaithCase(
        eval_id="C05",
        bucket="C_colloquial",
        query="怎么退？",
        strategy="hybrid_rerank",
        answer="旧编造答案",
        reason="旧编造原因",
        citations=[{"n": 1, "chunk_id": 1}],
        judge_model="deepseek-flash",
        status="未解决",
        seen_count=1,
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing
    mock_session.execute.return_value = mock_result

    new_citations = [{"n": 1, "chunk_id": 2}]
    res = await upsert_faith_case(
        db=mock_session,
        eval_id="C05",
        bucket="C_colloquial",
        query="怎么退？",
        answer="新编造答案：寄顺丰到付",
        reason="新理由：未说明商家指定地址",
        strategy="hybrid_rerank",
        citations=new_citations,
        judge_model="qwen-plus",
    )

    assert res is existing
    assert res.seen_count == 2
    assert res.answer == "新编造答案：寄顺丰到付"
    assert res.reason == "新理由：未说明商家指定地址"
    assert res.citations == new_citations
    assert res.judge_model == "qwen-plus"
    assert res.status == "未解决"

    # For existing entity, db.add should not be called again
    mock_session.add.assert_not_called()
    mock_session.commit.assert_awaited_once()
    mock_session.refresh.assert_awaited_once_with(existing)


@pytest.mark.asyncio
async def test_upsert_faith_case_recurrence_rollback_resolved():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    prior_resolved_time = datetime(2026, 9, 1, 12, 0, 0)
    existing = FaithCase(
        eval_id="E08",
        bucket="E_multi",
        query="多轮退换货咨询",
        strategy="hybrid_rerank",
        answer="旧答案",
        reason="旧理由",
        citations=[],
        judge_model="gpt-4o",
        status="已解决",
        seen_count=2,
        resolution="已修改知识库条目，补充多轮问答",
        resolved_at=prior_resolved_time,
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing
    mock_session.execute.return_value = mock_result

    res = await upsert_faith_case(
        db=mock_session,
        eval_id="E08",
        bucket="E_multi",
        query="多轮退换货咨询",
        answer="再次复发的编造答案",
        reason="再次复发的裁判理由",
        strategy="hybrid_rerank",
        citations=[{"n": 1, "chunk_id": 55}],
    )

    assert res is existing
    assert res.seen_count == 3
    # 状态自动回退为「未解决」
    assert res.status == "未解决"
    # 处置说明清空
    assert res.resolution is None
    # 解决时间保留以留存复发痕迹
    assert res.resolved_at == prior_resolved_time

    mock_session.commit.assert_awaited_once()
    mock_session.refresh.assert_awaited_once_with(existing)


def test_split_sql_statements_with_quotes_and_semicolons():
    raw_sql = (
        "SET NAMES utf8mb4;\n"
        "CREATE TABLE test_tab (id INT COMMENT 'semicolon; inside; single; quote', name VARCHAR(32) COMMENT \"double; quote;\");\n"
        "SELECT 'escaped\\' quote; test' AS val;\n"
    )
    stmts = split_sql_statements(raw_sql)
    assert len(stmts) == 3
    assert "semicolon; inside; single; quote" in stmts[1]
    assert "double; quote;" in stmts[1]
    assert "escaped\\' quote; test" in stmts[2]


def test_parse_ddl_statements():
    ddl_path = Path(__file__).resolve().parent.parent / "sql" / "ch04_ddl.sql"
    assert ddl_path.exists()
    statements = parse_ddl_statements(ddl_path)
    # 严格断言精准等于 3 条语句 (SET NAMES, low_confidence_questions, faith_cases)
    assert len(statements) == 3

    assert statements[0].strip() == "SET NAMES utf8mb4"
    assert "low_confidence_questions" in statements[1]
    assert "faith_cases" in statements[2]

    create_tables = [s for s in statements if "CREATE TABLE" in s.upper()]
    assert len(create_tables) == 2
    for ct in create_tables:
        assert "IF NOT EXISTS" in ct.upper()

    # 验证 faith_cases 建表语句完整闭合，包含 PRIMARY KEY 与 UNIQUE KEY
    fc_ddl = statements[2]
    assert "PRIMARY KEY (id)" in fc_ddl
    assert "UNIQUE KEY uk_eval_id (eval_id)" in fc_ddl
    assert fc_ddl.strip().endswith("COMMENT='ch04 忠实度编造个案台账'")


@pytest.mark.asyncio
async def test_init_ch04_db_idempotent():
    mock_engine = MagicMock()
    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_engine.dispose = AsyncMock()

    executed = await init_ch04_db(engine_override=mock_engine)
    assert len(executed) == 3
    assert mock_conn.execute.call_count == 3
