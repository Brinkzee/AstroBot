from datetime import datetime
from typing import Optional, Any, TYPE_CHECKING
from sqlalchemy import BigInteger, Integer, String, Text, JSON, Enum, DateTime, ForeignKey, func
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="消息主键",
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        ForeignKey("conversations.id"),
        nullable=False,
        comment="所属会话",
    )
    role: Mapped[str] = mapped_column(
        Enum("user", "assistant", "tool", name="message_role"),
        nullable=False,
        comment="角色:用户/助手/工具结果",
    )
    content: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="消息正文,assistant 纯工具调用时可为空",
    )
    tool_calls: Mapped[Optional[Any]] = mapped_column(
        JSON,
        nullable=True,
        comment="assistant 消息带的工具调用申请单",
    )
    tool_call_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="tool 消息对应的申请单 id,回灌时对号入座",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
        comment="产生时间",
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages",
    )
