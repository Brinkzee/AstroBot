import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from app.db.session import Base
from app.models import Conversation, Message, FAQ, Ticket
from scripts.seed_data import seed_all_data, SEED_FAQS


def test_model_attributes():
    conv = Conversation(user_id="u1", status="进行中")
    assert conv.user_id == "u1"
    assert conv.status == "进行中"

    msg = Message(conversation_id=1, role="user", content="hello")
    assert msg.conversation_id == 1
    assert msg.role == "user"
    assert msg.content == "hello"

    faq = FAQ(question="退货政策", answer="7天无理由", category="售后")
    assert faq.question == "退货政策"
    assert faq.answer == "7天无理由"
    assert faq.category == "售后"

    ticket = Ticket(
        ticket_no="T123",
        conversation_id=1,
        description="退款",
        ticket_type="售后",
    )
    assert ticket.ticket_no == "T123"
    assert ticket.conversation_id == 1
    assert ticket.description == "退款"
    assert ticket.ticket_type == "售后"


def test_models_table_metadata():
    assert Conversation.__tablename__ == "conversations"
    assert Message.__tablename__ == "messages"
    assert FAQ.__tablename__ == "faq"
    assert Ticket.__tablename__ == "tickets"

    # Column name assertions matching sql/ch02-ddl.sql
    conv_cols = set(Conversation.__table__.columns.keys())
    assert {"id", "user_id", "status", "created_at", "updated_at"}.issubset(conv_cols)

    msg_cols = set(Message.__table__.columns.keys())
    assert {"id", "conversation_id", "role", "content", "tool_calls", "tool_call_id", "created_at"}.issubset(msg_cols)

    faq_cols = set(FAQ.__table__.columns.keys())
    assert {"id", "question", "answer", "category", "created_at", "updated_at"}.issubset(faq_cols)

    ticket_cols = set(Ticket.__table__.columns.keys())
    assert {"ticket_no", "conversation_id", "description", "ticket_type", "status", "created_at"}.issubset(ticket_cols)

    # Primary key assertions
    assert [c.name for c in Conversation.__table__.primary_key.columns] == ["id"]
    assert [c.name for c in Message.__table__.primary_key.columns] == ["id"]
    assert [c.name for c in FAQ.__table__.primary_key.columns] == ["id"]
    assert [c.name for c in Ticket.__table__.primary_key.columns] == ["ticket_no"]

    # Foreign key assertions
    msg_fk_targets = {fk.target_fullname for fk in Message.__table__.c.conversation_id.foreign_keys}
    assert "conversations.id" in msg_fk_targets

    ticket_fk_targets = {fk.target_fullname for fk in Ticket.__table__.c.conversation_id.foreign_keys}
    assert "conversations.id" in ticket_fk_targets


def test_models_sync_sqlite_crud():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Create Conversation
        conv = Conversation(user_id="u_test_1001", status="进行中")
        session.add(conv)
        session.commit()
        assert conv.id is not None
        conv_id = conv.id

        # Create Message
        msg = Message(
            conversation_id=conv_id,
            role="assistant",
            content=None,
            tool_calls=[{"name": "query_faq", "args": {"keyword": "退货"}}],
            tool_call_id=None,
        )
        session.add(msg)
        session.commit()
        assert msg.id is not None
        assert msg.tool_calls == [{"name": "query_faq", "args": {"keyword": "退货"}}]

        # Create FAQ
        faq = FAQ(
            question="什么是会员特权？",
            answer="会员享受专属折扣与优先发货服务。",
            category="会员",
        )
        session.add(faq)
        session.commit()
        assert faq.id is not None

        # Create Ticket
        ticket = Ticket(
            ticket_no="T20260907001",
            conversation_id=conv_id,
            description="用户申请换货",
            ticket_type="售后",
            status="待处理",
        )
        session.add(ticket)
        session.commit()
        assert ticket.ticket_no == "T20260907001"

        # Verify Query
        queried_conv = session.get(Conversation, conv_id)
        assert queried_conv is not None
        assert queried_conv.user_id == "u_test_1001"

        queried_ticket = session.get(Ticket, "T20260907001")
        assert queried_ticket is not None
        assert queried_ticket.description == "用户申请换货"


def test_seed_faqs_structure():
    assert len(SEED_FAQS) == 3
    questions = [f["question"] for f in SEED_FAQS]
    assert "退货政策说明" in questions
    assert "运费标准与包邮政策" in questions
    assert "发票开具说明" in questions

    categories = {f["category"] for f in SEED_FAQS}
    assert {"售后", "物流", "财务"}.issubset(categories)


@pytest.mark.asyncio
async def test_seed_all_data_idempotency():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    # Scenario 1: Empty database, all 3 FAQs should be inserted
    mock_result_empty = MagicMock()
    mock_result_empty.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result_empty

    added_count = await seed_all_data(mock_session)
    assert added_count == 3
    assert mock_session.add.call_count == 3
    mock_session.commit.assert_awaited_once()

    # Scenario 2: Already seeded, 0 FAQs should be inserted
    mock_session.reset_mock()
    mock_result_exists = MagicMock()
    mock_result_exists.scalar_one_or_none.return_value = FAQ(
        question="退货政策说明", answer="...", category="售后"
    )
    mock_session.execute.return_value = mock_result_exists

    added_count_second_run = await seed_all_data(mock_session)
    assert added_count_second_run == 0
    assert mock_session.add.call_count == 0
