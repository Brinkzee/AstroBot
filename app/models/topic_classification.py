from datetime import datetime, timezone
from typing import Any, List, Optional, TYPE_CHECKING
from sqlalchemy import BigInteger, Integer, DateTime, ForeignKey, JSON, func, event
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.orm import Mapped, mapped_column, relationship, Session
from app.db.session import Base

if TYPE_CHECKING:
    from app.models.low_confidence import LowConfidenceQuestion


class TopicClassification(Base):
    """主题分类结果 ORM 实体。

    对齐 sql/ch10-ddl.sql: topic_classifications 表。
    微调分类器旁路批量归类结果，按批次收录低置信度问题命中权威类目的结果，
    喂给数据飞轮后台主题分布看板。
    """
    __tablename__ = "topic_classifications"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键",
    )
    question_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        ForeignKey("low_confidence_questions.id", name="fk_topic_question"),
        nullable=False,
        unique=True,
        index=True,
        comment="归类的问题,指向 low_confidence_questions.id",
    )
    labels: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        comment="多标签,17 类权威类目里命中的若干个",
    )
    classified_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        server_default=func.now(),
        nullable=False,
        comment="归类时间",
    )

    question: Mapped["LowConfidenceQuestion"] = relationship(
        "LowConfidenceQuestion",
        back_populates="topic_classification",
        lazy="selectin",
    )


@event.listens_for(Session, "after_flush")
def _expire_lcq_topic_classification(session, flush_context):
    """当新增或修改 TopicClassification 时，使同 Session 中已加载的 LowConfidenceQuestion 关联属性失效以触发重新加载。"""
    for obj in session.new:
        if isinstance(obj, TopicClassification) and obj.question_id:
            for other in session.identity_map.values():
                if type(other).__name__ == "LowConfidenceQuestion" and getattr(other, "id", None) == obj.question_id:
                    session.expire(other, ["topic_classification"])
