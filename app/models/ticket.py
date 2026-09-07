from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import BigInteger, Integer, String, Text, Enum, DateTime, ForeignKey, func
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
        comment="工单号,如 T20260701008",
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        ForeignKey("conversations.id"),
        nullable=False,
        comment="关联会话,可倒查当时聊了什么",
    )
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="问题描述",
    )
    ticket_type: Mapped[str] = mapped_column(
        Enum("售后", "投诉", "咨询", name="ticket_type"),
        nullable=False,
        comment="工单类型",
    )
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="ticket_status"),
        default="待处理",
        nullable=False,
        comment="处理状态",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
        comment="创建时间",
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="tickets",
    )
