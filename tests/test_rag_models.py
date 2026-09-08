import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from app.db.session import Base
from app.models import KnowledgeChunk, QAExtractionStaging
from scripts.init_ch03_db import parse_ddl_statements, init_ch03_db


def test_knowledge_chunk_attributes_and_defaults():
    chunk = KnowledgeChunk(
        category="退货规则",
        questions="如何申请退货？\n退货流程是什么？",
        answer="在订单详情页点击申请售后，选择退货退款即可。",
    )
    assert chunk.category == "退货规则"
    assert "如何申请退货？" in chunk.questions
    assert "在订单详情页" in chunk.answer
    assert chunk.is_key_clause is False
    assert chunk.vectorize_status == "pending"
    assert chunk.section_path is None
    assert chunk.content_type is None
    assert chunk.prev_chunk_id is None
    assert chunk.next_chunk_id is None
    assert chunk.vector_id is None


def test_qa_extraction_staging_attributes_and_defaults():
    staging = QAExtractionStaging(
        batch_no="batch_20260908_001",
        question="运费如何计算？",
        answer="满99元免运费。",
    )
    assert staging.batch_no == "batch_20260908_001"
    assert staging.question == "运费如何计算？"
    assert staging.answer == "满99元免运费。"
    assert staging.status == "extracted"
    assert staging.source_ref is None


def test_models_table_metadata():
    assert KnowledgeChunk.__tablename__ == "knowledge_chunks"
    assert QAExtractionStaging.__tablename__ == "qa_extraction_staging"

    # KnowledgeChunk columns
    kc_cols = set(KnowledgeChunk.__table__.columns.keys())
    expected_kc_cols = {
        "id",
        "category",
        "questions",
        "answer",
        "section_path",
        "content_type",
        "is_key_clause",
        "prev_chunk_id",
        "next_chunk_id",
        "vector_id",
        "vectorize_status",
        "created_at",
        "updated_at",
    }
    assert expected_kc_cols.issubset(kc_cols)

    # QAExtractionStaging columns
    staging_cols = set(QAExtractionStaging.__table__.columns.keys())
    expected_staging_cols = {
        "id",
        "batch_no",
        "source_ref",
        "question",
        "answer",
        "status",
        "created_at",
    }
    assert expected_staging_cols.issubset(staging_cols)

    # Primary key assertions
    assert [c.name for c in KnowledgeChunk.__table__.primary_key.columns] == ["id"]
    assert [c.name for c in QAExtractionStaging.__table__.primary_key.columns] == ["id"]

    # Nullable assertions
    table_kc = KnowledgeChunk.__table__
    assert table_kc.c.category.nullable is False
    assert table_kc.c.questions.nullable is False
    assert table_kc.c.answer.nullable is False
    assert table_kc.c.is_key_clause.nullable is False
    assert table_kc.c.vectorize_status.nullable is False

    table_staging = QAExtractionStaging.__table__
    assert table_staging.c.batch_no.nullable is False
    assert table_staging.c.question.nullable is False
    assert table_staging.c.answer.nullable is False
    assert table_staging.c.status.nullable is False

    # Foreign key assertions on knowledge_chunks
    prev_fk = list(table_kc.c.prev_chunk_id.foreign_keys)[0]
    next_fk = list(table_kc.c.next_chunk_id.foreign_keys)[0]
    assert prev_fk.target_fullname == "knowledge_chunks.id"
    assert prev_fk.ondelete == "SET NULL"
    assert next_fk.target_fullname == "knowledge_chunks.id"
    assert next_fk.ondelete == "SET NULL"

    # Index assertions
    kc_indexed_cols = {c.name for idx in table_kc.indexes for c in idx.columns}
    assert "category" in kc_indexed_cols or table_kc.c.category.index is True
    assert "vectorize_status" in kc_indexed_cols or table_kc.c.vectorize_status.index is True

    staging_indexed_cols = {c.name for idx in table_staging.indexes for c in idx.columns}
    assert "batch_no" in staging_indexed_cols or table_staging.c.batch_no.index is True
    assert "status" in staging_indexed_cols or table_staging.c.status.index is True


def test_sqlite_in_memory_crud():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Create Chunk 1
        chunk1 = KnowledgeChunk(
            category="售后/退换货",
            questions="怎么退货？\n退货有什么要求？",
            answer="商品签收7天内可申请无理由退货，需保持商品完好。",
            section_path="售后政策 > 退货指南",
            content_type="policy",
            is_key_clause=True,
        )
        session.add(chunk1)
        session.commit()
        assert chunk1.id is not None
        chunk1_id = chunk1.id

        # Create Chunk 2 linked to Chunk 1
        chunk2 = KnowledgeChunk(
            category="售后/退换货",
            questions="退货运费谁承担？",
            answer="质量问题由商家承担运费，非质量问题由买家承担。",
            section_path="售后政策 > 退货运费",
            content_type="policy",
            is_key_clause=False,
            prev_chunk_id=chunk1_id,
        )
        session.add(chunk2)
        session.commit()
        assert chunk2.id is not None
        chunk2_id = chunk2.id

        # Update Chunk 1 with next_chunk_id & vector_id
        chunk1.next_chunk_id = chunk2_id
        chunk1.vector_id = "vec_test_001"
        chunk1.vectorize_status = "done"
        session.commit()

        # Query and verify chunks
        queried_c1 = session.get(KnowledgeChunk, chunk1_id)
        assert queried_c1 is not None
        assert queried_c1.next_chunk_id == chunk2_id
        assert queried_c1.vector_id == "vec_test_001"
        assert queried_c1.vectorize_status == "done"
        assert queried_c1.is_key_clause is True

        queried_c2 = session.get(KnowledgeChunk, chunk2_id)
        assert queried_c2 is not None
        assert queried_c2.prev_chunk_id == chunk1_id

        # Create QAExtractionStaging
        st_row = QAExtractionStaging(
            batch_no="batch_20260908_test",
            source_ref="conversation_id:12345",
            question="保修期多久？",
            answer="主机保修一年，主要部件保修两年。",
            status="extracted",
        )
        session.add(st_row)
        session.commit()
        assert st_row.id is not None

        queried_st = session.get(QAExtractionStaging, st_row.id)
        assert queried_st is not None
        assert queried_st.batch_no == "batch_20260908_test"
        assert queried_st.status == "extracted"

        # Update staging status
        queried_st.status = "kept"
        session.commit()
        refetched_st = session.get(QAExtractionStaging, st_row.id)
        assert refetched_st.status == "kept"


def test_parse_ddl_statements():
    ddl_path = Path(__file__).resolve().parent.parent / "sql" / "ch03-ddl.sql"
    assert ddl_path.exists()
    statements = parse_ddl_statements(ddl_path)
    assert len(statements) >= 2
    # Ensure all CREATE TABLE statements contain IF NOT EXISTS for idempotency
    create_tables = [s for s in statements if "CREATE TABLE" in s.upper()]
    assert len(create_tables) == 2
    for ct in create_tables:
        assert "IF NOT EXISTS" in ct.upper()


@pytest.mark.asyncio
async def test_init_ch03_db_idempotent():
    mock_engine = MagicMock()
    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_engine.dispose = AsyncMock()

    executed = await init_ch03_db(engine_override=mock_engine)
    assert len(executed) >= 2
    assert mock_conn.execute.call_count == len(executed)
