import enum
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from sqlalchemy import BigInteger, Integer, Text, Enum, DateTime, ForeignKey, JSON, func
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.models.conversation import Conversation
    from app.models.review_queue import ReviewQueue


class LowConfidenceSource(str, enum.Enum):
    """低置信度问题入池来源枚举。"""
    RETRIEVAL_LOW_CONF = "retrieval_low_conf"
    SELF_CHECK = "self_check"
    USER_FEEDBACK = "user_feedback"


class LowConfidenceQuestion(Base):
    """低置信度问题池 ORM 实体。
    
    对齐 sql/ch04_ddl.sql 及 sql/ch09-ddl.sql: low_confidence_questions 表。
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
    retrieved_chunks: Mapped[Optional[Any]] = mapped_column(
        JSON,
        nullable=True,
        comment="落池时的召回片段快照:Top 几条的原文与得分,审核页展示用;没走检索为 NULL",
    )
    matched_review_id: Mapped[Optional[int]] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        ForeignKey("review_queue.id", name="fk_lcq_review", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="查重后归并到的缺口,指向 review_queue.id",
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
    review_queue: Mapped[Optional["ReviewQueue"]] = relationship(
        "ReviewQueue",
        back_populates="raw_questions",
        lazy="select",
    )


async def record_low_confidence(
    db: AsyncSession,
    raw_question: str,
    source: str | LowConfidenceSource,
    conversation_id: Optional[int] = None,
    reason: Optional[str] = None,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    matched_review_id: Optional[int] = None,
) -> LowConfidenceQuestion:
    """便捷异步函数：将低置信度问题写入 low_confidence_questions 表。

    Args:
        db: 异步数据库会话
        raw_question: 用户原话
        source: 入池来源（retrieval_low_conf / self_check / user_feedback）
        conversation_id: 关联的会话 ID
        reason: 判定原因
        retrieved_chunks: 召回片段快照
        matched_review_id: 归并到的缺口 ID

    Returns:
        持久化后的 LowConfidenceQuestion 实体
    """
    source_val = source.value if isinstance(source, enum.Enum) else str(source)
    record = LowConfidenceQuestion(
        raw_question=raw_question,
        source=source_val,
        conversation_id=conversation_id,
        reason=reason,
        retrieved_chunks=retrieved_chunks,
        matched_review_id=matched_review_id,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record
