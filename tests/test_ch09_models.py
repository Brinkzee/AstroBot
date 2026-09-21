import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine, select, inspect, text
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.db.session import Base
import app.models
from app.models.review_queue import ReviewQueue, ReviewStatus
from app.models.eval_run import EvalRun, TriggeredBy
from app.models.low_confidence import LowConfidenceQuestion, LowConfidenceSource, record_low_confidence
from scripts.init_ch09_db import split_sql_statements, parse_ddl_statements, init_ch09_db


def test_models_init_export():
    assert hasattr(app.models, "ReviewQueue")
    assert hasattr(app.models, "ReviewStatus")
    assert hasattr(app.models, "EvalRun")
    assert hasattr(app.models, "TriggeredBy")
    assert "ReviewQueue" in app.models.__all__
    assert "ReviewStatus" in app.models.__all__
    assert "EvalRun" in app.models.__all__
    assert "TriggeredBy" in app.models.__all__
    assert app.models.ReviewQueue is ReviewQueue
    assert app.models.ReviewStatus is ReviewStatus
    assert app.models.EvalRun is EvalRun
    assert app.models.TriggeredBy is TriggeredBy


def test_create_review_queue_item():
    # 枚举值测试
    assert ReviewStatus.PENDING.value == "待审"
    assert ReviewStatus.APPROVED.value == "通过"
    assert ReviewStatus.REJECTED.value == "驳回"

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        item = ReviewQueue(
            normalized_question="退货运费谁出？",
            ai_suggested_answer="质量问题商家包邮，非质量问题买家自理。",
        )
        assert item.normalized_question == "退货运费谁出？"
        assert item.occurrence_count == 1
        assert item.review_status == "待审"
        assert item.approved_answer is None

        session.add(item)
        session.commit()

        assert item.id is not None
        assert item.created_at is not None
        assert item.updated_at is not None

        # 更新状态与核准答案
        item.review_status = ReviewStatus.APPROVED.value
        item.approved_answer = "质量问题包退，7天无理由买家承担运费。"
        item.occurrence_count += 1
        session.commit()

        queried = session.get(ReviewQueue, item.id)
        assert queried is not None
        assert queried.review_status == "通过"
        assert queried.occurrence_count == 2
        assert queried.approved_answer == "质量问题包退，7天无理由买家承担运费。"


def test_create_eval_run_item():
    # 枚举值测试
    assert TriggeredBy.CRON.value == "定时"
    assert TriggeredBy.MANUAL.value == "手动"

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        metrics_data = {
            "recall_at_k": 0.88,
            "mrr": 0.76,
            "faithfulness": 0.94,
        }
        run = EvalRun(
            triggered_by=TriggeredBy.CRON.value,
            dataset_size=50,
            metrics=metrics_data,
        )
        assert run.triggered_by == "定时"
        assert run.dataset_size == 50
        assert run.metrics == metrics_data

        session.add(run)
        session.commit()

        assert run.id is not None
        assert run.created_at is not None

        # 查询断言
        saved = session.scalars(select(EvalRun).where(EvalRun.id == run.id)).one()
        assert saved.dataset_size == 50
        assert saved.metrics["faithfulness"] == 0.94


@pytest.mark.asyncio
async def test_low_confidence_question_with_chunks_and_review_id():
    # 1. 验证 record_low_confidence 异步辅助函数及新参数
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    sample_chunks = [
        {"chunk_id": 101, "score": 0.62, "text": "售后退换政策说明..."},
        {"chunk_id": 102, "score": 0.58, "text": "发票开具常见问题..."},
    ]

    res = await record_low_confidence(
        db=mock_session,
        raw_question="买完东西怎么开发票啊？",
        source=LowConfidenceSource.RETRIEVAL_LOW_CONF,
        conversation_id=42,
        reason="检索得分均低于阈值 0.7",
        retrieved_chunks=sample_chunks,
        matched_review_id=999,
    )

    assert isinstance(res, LowConfidenceQuestion)
    assert res.raw_question == "买完东西怎么开发票啊？"
    assert res.source == "retrieval_low_conf"
    assert res.conversation_id == 42
    assert res.reason == "检索得分均低于阈值 0.7"
    assert res.retrieved_chunks == sample_chunks
    assert res.matched_review_id == 999

    mock_session.add.assert_called_once_with(res)
    mock_session.commit.assert_awaited_once()
    mock_session.refresh.assert_awaited_once_with(res)

    # 2. 验证 SQLite 内存中的 ORM 双向关联及字段持久化
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        rq = ReviewQueue(
            normalized_question="发票如何开具？",
            ai_suggested_answer="请在订单中心点击申请开票。",
        )
        session.add(rq)
        session.commit()

        lcq = LowConfidenceQuestion(
            raw_question="买完东西怎么开发票啊？",
            source="retrieval_low_conf",
            conversation_id=42,
            reason="检索得分均低于阈值 0.7",
            retrieved_chunks=sample_chunks,
            matched_review_id=rq.id,
        )
        session.add(lcq)
        session.commit()

        # 重新查询验证
        saved_lcq = session.get(LowConfidenceQuestion, lcq.id)
        assert saved_lcq is not None
        assert saved_lcq.matched_review_id == rq.id
        assert saved_lcq.retrieved_chunks == sample_chunks

        # 正向关联: lcq.review_queue
        assert saved_lcq.review_queue is not None
        assert saved_lcq.review_queue.id == rq.id
        assert saved_lcq.review_queue.normalized_question == "发票如何开具？"

        # 反向关联: rq.raw_questions
        saved_rq = session.get(ReviewQueue, rq.id)
        assert saved_rq is not None
        assert len(saved_rq.raw_questions) == 1
        assert saved_rq.raw_questions[0].id == lcq.id
        assert saved_rq.raw_questions[0].raw_question == "买完东西怎么开发票啊？"


def test_parse_ddl_statements():
    ddl_path = Path(__file__).resolve().parent.parent / "sql" / "ch09-ddl.sql"
    assert ddl_path.exists()
    statements = parse_ddl_statements(ddl_path)
    assert len(statements) >= 3

    set_stmts = [s for s in statements if s.upper().startswith("SET ")]
    assert len(set_stmts) >= 1

    create_stmts = [s for s in statements if "CREATE TABLE" in s.upper()]
    assert len(create_stmts) == 2
    for cs in create_stmts:
        assert "IF NOT EXISTS" in cs.upper()

    alter_stmts = [s for s in statements if "ALTER TABLE" in s.upper()]
    assert len(alter_stmts) == 1
    assert "low_confidence_questions" in alter_stmts[0]


@pytest.mark.asyncio
async def test_init_ch09_db_with_mock_engine():
    mock_engine = MagicMock()
    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_engine.dispose = AsyncMock()

    executed = await init_ch09_db(engine_override=mock_engine)
    assert len(executed) >= 3
    assert mock_conn.execute.call_count >= 3


@pytest.mark.asyncio
async def test_migration_script_idempotent():
    engine = create_engine("sqlite:///:memory:")

    # 先模拟已有的 ch04 数据库状态（存在 low_confidence_questions 表，但无 ch09 新增列）
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS low_confidence_questions (
          id              INTEGER PRIMARY KEY AUTOINCREMENT,
          conversation_id INTEGER NULL,
          raw_question    TEXT NOT NULL,
          source          TEXT NOT NULL,
          reason          TEXT NULL,
          created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """))
        conn.execute(text("""
        INSERT INTO low_confidence_questions (raw_question, source, reason)
        VALUES ('老用户原话', 'retrieval_low_conf', '旧记录')
        """))

    # 首次执行 ch09 迁移
    executed1 = await init_ch09_db(engine_override=engine)
    assert len(executed1) >= 3

    insp = inspect(engine)
    tables = insp.get_table_names()
    assert "review_queue" in tables
    assert "eval_runs" in tables
    assert "low_confidence_questions" in tables

    # 检查 review_queue 列
    rq_cols = {c["name"] for c in insp.get_columns("review_queue")}
    expected_rq_cols = {
        "id", "normalized_question", "ai_suggested_answer",
        "occurrence_count", "review_status", "approved_answer",
        "created_at", "updated_at"
    }
    assert expected_rq_cols.issubset(rq_cols)

    # 检查 eval_runs 列
    eval_cols = {c["name"] for c in insp.get_columns("eval_runs")}
    expected_eval_cols = {"id", "triggered_by", "dataset_size", "metrics", "created_at"}
    assert expected_eval_cols.issubset(eval_cols)

    # 检查 low_confidence_questions 是否成功被 ALTER 增加了两列
    lcq_cols = {c["name"] for c in insp.get_columns("low_confidence_questions")}
    assert "retrieved_chunks" in lcq_cols
    assert "matched_review_id" in lcq_cols

    # 验证向新表和新列中插入及查询数据
    with engine.begin() as conn:
        conn.execute(text("""
        INSERT INTO review_queue (normalized_question, ai_suggested_answer, occurrence_count, review_status)
        VALUES ('退货运费谁出？', '质量问题商家包邮', 1, '待审')
        """))
        conn.execute(text("""
        INSERT INTO eval_runs (triggered_by, dataset_size, metrics)
        VALUES ('定时', 10, '{"faithfulness": 0.9}')
        """))
        conn.execute(text("""
        INSERT INTO low_confidence_questions (raw_question, source, retrieved_chunks, matched_review_id)
        VALUES ('新用户原话', 'retrieval_low_conf', '[{"chunk_id": 1}]', 1)
        """))

    # 二次执行 ch09 迁移，验证幂等性
    executed2 = await init_ch09_db(engine_override=engine)
    assert len(executed2) >= 3

    with engine.connect() as conn:
        rq_cnt = conn.execute(text("SELECT COUNT(*) FROM review_queue")).scalar()
        assert rq_cnt == 1
        eval_cnt = conn.execute(text("SELECT COUNT(*) FROM eval_runs")).scalar()
        assert eval_cnt == 1
        lcq_cnt = conn.execute(text("SELECT COUNT(*) FROM low_confidence_questions")).scalar()
        assert lcq_cnt == 2
