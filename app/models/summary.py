from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import BigInteger, Integer, Text, DateTime, ForeignKey, UniqueConstraint, Index, func
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uk_conv_seq"),
        Index("idx_conv_upto", "conversation_id", "upto_msg_id"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        ForeignKey("conversations.id"),
        nullable=False,
    )
    seq: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="第几段,从1开始",
    )
    from_msg_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        nullable=False,
        comment="这段覆盖的消息区间,闭区间",
    )
    upto_msg_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="summaries",
    )
