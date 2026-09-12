import enum
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import BigInteger, Integer, Text, Enum, DateTime, ForeignKey, func
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation


class LowConfidenceSource(str, enum.Enum):
    """低置信度问题入池来源枚举。"""
    RETRIEVAL_LOW_CONF = "retrieval_low_conf"
    SELF_CHECK = "self_check"
    USER_FEEDBACK = "user_feedback"


class LowConfidenceQuestion(Base):
    """低置信度问题池 ORM 实体。
    
    对齐 sql/ch04_ddl.sql: low_confidence_questions 表。
    用于在检索相关度不足、生成前自检判断知识不够或用户点踩反馈未解决时收录问题原话，
    作为后续知识扩充与数据飞轮的入口。
    """
    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键",
    )
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        ForeignKey("conversations.id", name="fk_lcq_conversation"),
        nullable=True,
        comment="来源会话",
    )
    raw_question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="用户原话,带情绪口语",
    )
    source: Mapped[str] = mapped_column(
        Enum("retrieval_low_conf", "self_check", "user_feedback", name="lcq_source"),
        nullable=False,
        index=True,
        comment="入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决",
    )
    reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="判不能的原因,留作复盘",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="入池时间",
    )

    conversation: Mapped[Optional["Conversation"]] = relationship(
        "Conversation",
        lazy="select",
    )


async def record_low_confidence(
    db: AsyncSession,
    raw_question: str,
    source: str | LowConfidenceSource,
    conversation_id: Optional[int] = None,
    reason: Optional[str] = None,
) -> LowConfidenceQuestion:
    """便捷异步函数：将低置信度问题写入 low_confidence_questions 表。

    Args:
        db: 异步数据库会话
        raw_question: 用户原话
        source: 入池来源（retrieval_low_conf / self_check / user_feedback）
        conversation_id: 关联的会话 ID
        reason: 判定原因

    Returns:
        持久化后的 LowConfidenceQuestion 实体
    """
    source_val = source.value if isinstance(source, enum.Enum) else str(source)
    record = LowConfidenceQuestion(
        raw_question=raw_question,
        source=source_val,
        conversation_id=conversation_id,
        reason=reason,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record
