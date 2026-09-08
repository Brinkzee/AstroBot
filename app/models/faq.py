from datetime import datetime
from sqlalchemy import BigInteger, Integer, String, Text, DateTime, func
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base


class FAQ(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="FAQ 主键",
    )
    question: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="问题",
    )
    answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="答案",
    )
    category: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="分类",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="更新时间",
    )
