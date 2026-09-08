from datetime import datetime
from sqlalchemy import BigInteger, Integer, String, Text, Boolean, DateTime, ForeignKey, func
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.orm import Mapped, mapped_column
from app.db.session import Base


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    def __init__(self, **kwargs):
        kwargs.setdefault("is_key_clause", False)
        kwargs.setdefault("vectorize_status", "pending")
        super().__init__(**kwargs)

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="chunk 主键,与 Milvus 集合主键对齐",
    )
    category: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="分类 / 上级标题路径,进向量化文本",
    )
    questions: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="问法或本节标题,多个问法换行分隔,进向量化文本",
    )
    answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="正文答案,进向量化文本",
    )
    section_path: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="章节路径,元数据,溯源用,不进向量",
    )
    content_type: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="内容类型:faq / policy / manual 等,元数据",
    )
    is_key_clause: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="0",
        nullable=False,
        comment="是否关键条款,0 否 1 是,元数据",
    )
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"),
        nullable=True,
        comment="前一块指针,元数据",
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"),
        nullable=True,
        comment="后一块指针,元数据",
    )
    vector_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Milvus 集合 knowledge 里的主键,写入后回填",
    )
    vectorize_status: Mapped[str] = mapped_column(
        String(16),
        default="pending",
        server_default="pending",
        nullable=False,
        index=True,
        comment="待向量化 / 已向量化,双写幂等靠它",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
        comment="更新时间",
    )
