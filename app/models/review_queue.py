import enum
from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING
from sqlalchemy import BigInteger, Integer, String, Text, Enum, DateTime, func
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.models.low_confidence import LowConfidenceQuestion


class ReviewStatus(str, enum.Enum):
    PENDING = "待审"
    APPROVED = "通过"
    REJECTED = "驳回"


class ReviewQueue(Base):
    """待审队列 ORM 实体。

    一行 = 一个去重后的知识缺口；查重命中就累加 occurrence_count，不新建行。
    """
    __tablename__ = "review_queue"

    def __init__(self, **kwargs):
        kwargs.setdefault("occurrence_count", 1)
        kwargs.setdefault("review_status", ReviewStatus.PENDING.value)
        super().__init__(**kwargs)

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="缺口主键,也是查重命中要返回的 matched_question_id",
    )
    normalized_question: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="标准化后的 FAQ 式问题",
    )
    ai_suggested_answer: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="模型生成的示例答案,备查",
    )
    occurrence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="出现次数,查重命中累加,越高越该优先补",
    )
    review_status: Mapped[str] = mapped_column(
        Enum("待审", "通过", "驳回", name="review_status_enum"),
        nullable=False,
        default=ReviewStatus.PENDING.value,
        index=True,
        comment="人工审核状态",
    )
    approved_answer: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="审核通过时补的核准答案,走 ch03 落库流程写回知识库",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        server_default=func.now(),
        nullable=False,
        comment="首次入队时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        nullable=False,
        comment="更新时间",
    )

    raw_questions: Mapped[List["LowConfidenceQuestion"]] = relationship(
        "LowConfidenceQuestion",
        back_populates="review_queue",
        lazy="select",
    )
