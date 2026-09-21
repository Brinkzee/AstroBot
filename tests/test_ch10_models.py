import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine, select, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.db.session import Base
import app.models
from app.models.low_confidence import LowConfidenceQuestion
from app.models.topic_classification import TopicClassification
from scripts.init_ch10_db import split_sql_statements, parse_ddl_statements, init_ch10_db


def test_models_init_export():
    assert hasattr(app.models, "TopicClassification")
    assert "TopicClassification" in app.models.__all__
    assert app.models.TopicClassification is TopicClassification


@pytest.fixture
async def memory_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_topic_classification_model_lifecycle(memory_db: AsyncSession):
    # 1. 创建低置信度问题
    lcq = LowConfidenceQuestion(
        raw_question="买大了想退",
        source="retrieval_low_conf",
        reason="相关度不足",
    )
    memory_db.add(lcq)
    await memory_db.commit()
    await memory_db.refresh(lcq)

    # 2. 关联创建主题分类
    topic = TopicClassification(
        question_id=lcq.id,
        labels=["尺码", "退换货"],
    )
    memory_db.add(topic)
    await memory_db.commit()
    await memory_db.refresh(topic)

    assert topic.id is not None
    assert topic.question_id == lcq.id
    assert topic.labels == ["尺码", "退换货"]
    assert topic.classified_at is not None

    # 3. 关联关系验证 (正向与反向)
    stmt = select(LowConfidenceQuestion).where(LowConfidenceQuestion.id == lcq.id)
    res = await memory_db.execute(stmt)
    fetched_lcq = res.scalar_one()
    assert fetched_lcq.topic_classification is not None
    assert fetched_lcq.topic_classification.labels == ["尺码", "退换货"]
    assert fetched_lcq.topic_classification.question.id == lcq.id


@pytest.mark.asyncio
async def test_topic_classification_unique_question_id(memory_db: AsyncSession):
    lcq = LowConfidenceQuestion(
        raw_question="物流到哪了？",
        source="retrieval_low_conf",
        reason="相关度不足",
    )
    memory_db.add(lcq)
    await memory_db.commit()
    await memory_db.refresh(lcq)

    topic1 = TopicClassification(
        question_id=lcq.id,
        labels=["物流"],
    )
    memory_db.add(topic1)
    await memory_db.commit()

    topic2 = TopicClassification(
        question_id=lcq.id,
        labels=["其他"],
    )
    memory_db.add(topic2)
    with pytest.raises(IntegrityError):
        await memory_db.commit()
    await memory_db.rollback()


def test_parse_ch10_ddl_statements():
    ddl_path = Path(__file__).resolve().parent.parent / "sql" / "ch10-ddl.sql"
    assert ddl_path.exists()
    statements = parse_ddl_statements(ddl_path)
    assert len(statements) >= 2

    set_stmts = [s for s in statements if s.upper().startswith("SET ")]
    assert len(set_stmts) >= 1

    create_stmts = [s for s in statements if "CREATE TABLE" in s.upper()]
    assert len(create_stmts) == 1
    assert "TOPIC_CLASSIFICATIONS" in create_stmts[0].upper()
    assert "IF NOT EXISTS" in create_stmts[0].upper()


@pytest.mark.asyncio
async def test_init_ch10_db_with_mock_engine():
    mock_engine = MagicMock()
    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_engine.dispose = AsyncMock()

    executed = await init_ch10_db(engine_override=mock_engine)
    assert len(executed) >= 2
    assert mock_conn.execute.call_count >= 2


@pytest.mark.asyncio
async def test_migration_script_idempotent():
    engine = create_engine("sqlite:///:memory:")

    # 1. 模拟底层已有的 low_confidence_questions 表
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

    # 2. 首次执行 ch10 迁移
    executed1 = await init_ch10_db(engine_override=engine)
    assert len(executed1) >= 2

    insp = inspect(engine)
    tables = insp.get_table_names()
    assert "topic_classifications" in tables

    tc_cols = {c["name"] for c in insp.get_columns("topic_classifications")}
    expected_tc_cols = {"id", "question_id", "labels", "classified_at"}
    assert expected_tc_cols.issubset(tc_cols)

    # 插入一条测试数据
    with engine.begin() as conn:
        conn.execute(text("""
        INSERT INTO low_confidence_questions (id, raw_question, source)
        VALUES (1, '测试问题', 'retrieval_low_conf')
        """))
        conn.execute(text("""
        INSERT INTO topic_classifications (question_id, labels)
        VALUES (1, '["物流", "运费"]')
        """))

    # 3. 二次执行 ch10 迁移验证幂等性
    executed2 = await init_ch10_db(engine_override=engine)
    assert len(executed2) >= 2

    with engine.connect() as conn:
        cnt = conn.execute(text("SELECT COUNT(*) FROM topic_classifications")).scalar()
        assert cnt == 1
