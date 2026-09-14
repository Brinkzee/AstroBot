import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine, select, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.session import Base
import app.models
from app.models import Conversation, Message, ConversationSummary
from scripts.init_ch07_db import split_sql_statements, parse_ddl_statements, init_ch07_db


def test_models_init_export():
    assert hasattr(app.models, "ConversationSummary")
    assert "ConversationSummary" in app.models.__all__
    assert app.models.ConversationSummary is ConversationSummary


def test_conversation_new_attributes():
    conv = Conversation(
        user_id="u_ch07_001",
        status="进行中",
        summary="用户咨询退换货，客服已告知需订单号",
        summary_upto_msg_id=12,
        layer1_from_msg_id=8,
    )
    assert conv.user_id == "u_ch07_001"
    assert conv.status == "进行中"
    assert conv.summary == "用户咨询退换货，客服已告知需订单号"
    assert conv.summary_upto_msg_id == 12
    assert conv.layer1_from_msg_id == 8

    # Verify default nullable values
    conv_default = Conversation(user_id="u_ch07_002")
    assert conv_default.summary is None
    assert conv_default.summary_upto_msg_id is None
    assert conv_default.layer1_from_msg_id is None


def test_conversation_summary_attributes_and_defaults():
    summary_item = ConversationSummary(
        conversation_id=101,
        seq=1,
        from_msg_id=1,
        upto_msg_id=10,
        content="第1段事实摘要：用户询问耳机退换规则，客服说明7天内无理由退货要求包装完好。",
    )
    assert summary_item.conversation_id == 101
    assert summary_item.seq == 1
    assert summary_item.from_msg_id == 1
    assert summary_item.upto_msg_id == 10
    assert summary_item.content == "第1段事实摘要：用户询问耳机退换规则，客服说明7天内无理由退货要求包装完好。"
    assert summary_item.id is None


def test_models_table_metadata():
    assert Conversation.__tablename__ == "conversations"
    assert ConversationSummary.__tablename__ == "conversation_summaries"

    # Verify conversation table has new columns
    conv_cols = set(Conversation.__table__.columns.keys())
    assert {"summary", "summary_upto_msg_id", "layer1_from_msg_id"}.issubset(conv_cols)

    table_conv = Conversation.__table__
    assert table_conv.c.summary.nullable is True
    assert table_conv.c.summary_upto_msg_id.nullable is True
    assert table_conv.c.layer1_from_msg_id.nullable is True

    # Verify conversation_summaries columns
    summary_cols = set(ConversationSummary.__table__.columns.keys())
    expected_cols = {
        "id",
        "conversation_id",
        "seq",
        "from_msg_id",
        "upto_msg_id",
        "content",
        "created_at",
    }
    assert expected_cols.issubset(summary_cols)

    table_summary = ConversationSummary.__table__
    # Primary key assertions
    assert [c.name for c in table_summary.primary_key.columns] == ["id"]

    # Nullable assertions
    assert table_summary.c.conversation_id.nullable is False
    assert table_summary.c.seq.nullable is False
    assert table_summary.c.from_msg_id.nullable is False
    assert table_summary.c.upto_msg_id.nullable is False
    assert table_summary.c.content.nullable is False
    assert table_summary.c.created_at.nullable is False

    # Foreign key assertions
    summary_fk_targets = {fk.target_fullname for fk in table_summary.c.conversation_id.foreign_keys}
    assert "conversations.id" in summary_fk_targets

    # Unique constraint assertions (uk_conv_seq: conversation_id, seq)
    unique_constraints = [
        c for c in table_summary.constraints
        if getattr(c, "columns", None) and (getattr(c, "unique", False) or type(c).__name__ == "UniqueConstraint")
    ]
    unique_col_sets = [{col.name for col in uc.columns} for uc in unique_constraints]
    assert {"conversation_id", "seq"} in unique_col_sets

    # Index assertions (idx_conv_upto: conversation_id, upto_msg_id)
    index_col_sets = [{c.name for c in idx.columns} for idx in table_summary.indexes]
    assert {"conversation_id", "upto_msg_id"} in index_col_sets


def test_conversation_and_summary_relationships():
    # Relationship type check
    conv_rel = Conversation.summaries.property
    assert conv_rel.mapper.class_ is ConversationSummary
    assert conv_rel.back_populates == "conversation"

    summary_rel = ConversationSummary.conversation.property
    assert summary_rel.mapper.class_ is Conversation
    assert summary_rel.back_populates == "summaries"


def test_sqlite_in_memory_crud_and_cascade_delete():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Create conversation with summary projection fields
        conv = Conversation(
            user_id="u_ch07_test",
            status="进行中",
            summary="当前前3段摘要投影",
            summary_upto_msg_id=15,
            layer1_from_msg_id=10,
        )
        session.add(conv)
        session.commit()
        assert conv.id is not None
        conv_id = conv.id

        # Add summaries
        s1 = ConversationSummary(
            conversation_id=conv_id,
            seq=1,
            from_msg_id=1,
            upto_msg_id=5,
            content="第1段摘要",
        )
        s2 = ConversationSummary(
            conversation_id=conv_id,
            seq=2,
            from_msg_id=6,
            upto_msg_id=10,
            content="第2段摘要",
        )
        session.add_all([s1, s2])
        session.commit()
        assert s1.id is not None
        assert s2.id is not None

        # Query and verify relation
        refetched_conv = session.get(Conversation, conv_id)
        assert refetched_conv is not None
        assert len(refetched_conv.summaries) == 2
        assert refetched_conv.summaries[0].seq == 1
        assert refetched_conv.summaries[1].seq == 2
        assert refetched_conv.summaries[0].conversation is refetched_conv

        # Verify unique constraint uk_conv_seq (duplicate conversation_id + seq)
        dup_summary = ConversationSummary(
            conversation_id=conv_id,
            seq=1,
            from_msg_id=11,
            upto_msg_id=15,
            content="重复的seq=1",
        )
        session.add(dup_summary)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

        # Verify cascade delete: deleting conversation removes its summaries
        session.delete(refetched_conv)
        session.commit()

        remaining_summaries = session.scalars(
            select(ConversationSummary).where(ConversationSummary.conversation_id == conv_id)
        ).all()
        assert len(remaining_summaries) == 0


def test_parse_ddl_statements():
    ddl_path = Path(__file__).resolve().parent.parent / "sql" / "ch07-ddl.sql"
    assert ddl_path.exists()
    statements = parse_ddl_statements(ddl_path)
    assert len(statements) >= 3

    # Must contain ALTER TABLE and CREATE TABLE
    alter_stmts = [s for s in statements if "ALTER TABLE" in s.upper()]
    assert len(alter_stmts) >= 2
    assert any("summary" in s for s in alter_stmts)
    assert any("layer1_from_msg_id" in s for s in alter_stmts)

    create_stmts = [s for s in statements if "CREATE TABLE" in s.upper()]
    assert len(create_stmts) == 1
    assert "IF NOT EXISTS" in create_stmts[0].upper()
    assert "conversation_summaries" in create_stmts[0]


@pytest.mark.asyncio
async def test_init_ch07_db_with_mock_engine():
    mock_engine = MagicMock()
    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_engine.dispose = AsyncMock()

    executed = await init_ch07_db(engine_override=mock_engine)
    assert len(executed) >= 3
    assert mock_conn.execute.call_count >= 3


@pytest.mark.asyncio
async def test_init_ch07_db_sqlite_migration_idempotent():
    # Test real SQLite in-memory migration and idempotency
    engine = create_engine("sqlite:///:memory:")
    # Initialize base conversation table without ch07 columns
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id VARCHAR(64) NOT NULL,
                status VARCHAR(16) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        ))

    # First migration execution
    executed1 = await init_ch07_db(engine_override=engine)
    assert len(executed1) > 0

    insp = inspect(engine)
    cols = {c["name"] for c in insp.get_columns("conversations")}
    assert "summary" in cols
    assert "summary_upto_msg_id" in cols
    assert "layer1_from_msg_id" in cols

    tables = insp.get_table_names()
    assert "conversation_summaries" in tables

    # Second migration execution (idempotency check, should not fail)
    executed2 = await init_ch07_db(engine_override=engine)
    assert len(executed2) > 0
