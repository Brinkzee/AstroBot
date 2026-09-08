from datetime import datetime
from sqlalchemy import BigInteger, Integer, String, Text, DateTime, func
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base


class QAExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"

    def __init__(self, **kwargs):
        kwargs.setdefault("status", "extracted")
        super().__init__(**kwargs)

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="暂存行主键",
    )
    batch_no: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯",
    )
    source_ref: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="来源会话 / 导出文件标识,溯源用,不入最终知识库",
    )
    question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="LLM 从会话抽出的用户问法",
    )
    answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="LLM 从会话抽出的客服答案",
    )
    status: Mapped[str] = mapped_column(
        String(16),
        default="extracted",
        server_default="extracted",
        nullable=False,
        index=True,
        comment="已抽出待去重 / 去重保留 / 去重丢弃",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
        comment="抽取写入时间",
    )
