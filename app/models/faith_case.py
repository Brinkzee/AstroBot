import enum
from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import BigInteger, Integer, String, Text, Enum, DateTime, JSON, select, func
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT, INTEGER as MYSQL_INTEGER
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class FaithCaseStatus(str, enum.Enum):
    """编造个案处置状态。"""
    UNRESOLVED = "未解决"
    RESOLVED = "已解决"
    WONT_FIX = "无需解决"


class FaithCase(Base):
    """忠实度裁判编造个案台账 ORM 实体。

    对齐 sql/ch04_ddl.sql: faith_cases 表。
    一题一行 (uk_eval_id)，记录评估过程中被裁判判定为编造（幻觉）的题目及证据快照。
    具备跨轮累积计数 (seen_count) 与解决后复发自动退回「未解决」机制。
    """
    __tablename__ = "faith_cases"

    def __init__(self, **kwargs):
        kwargs.setdefault("strategy", "hybrid_rerank")
        kwargs.setdefault("status", "未解决")
        kwargs.setdefault("seen_count", 1)
        super().__init__(**kwargs)

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键",
    )
    eval_id: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        unique=True,
        index=True,
        comment="评估集题号,如 A43;一题一行",
    )
    bucket: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        comment="题目所属桶:A_policy / B_model / C_colloquial / E_multi",
    )
    query: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="用户问题原文",
    )
    strategy: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="hybrid_rerank",
        server_default="hybrid_rerank",
        comment="产出这条答案的检索策略",
    )
    answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="被判编造的那版生成答案原文",
    )
    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="裁判给的理由:编在哪一句",
    )
    citations: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSON,
        nullable=True,
        comment="这一轮喂给模型的 Top-K 证据全集快照:[{n,chunk_id,section_path,question,answer}];答案里的角标 [n] 就是这份列表的序号,答案通常只引用其中两三条;老数据没记为 NULL",
    )
    judge_model: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="判这条的裁判模型",
    )
    status: Mapped[str] = mapped_column(
        Enum("未解决", "已解决", "无需解决", name="faith_case_status"),
        nullable=False,
        default="未解决",
        server_default="未解决",
        index=True,
        comment="处置状态,人工点按钮改",
    )
    seen_count: Mapped[int] = mapped_column(
        Integer().with_variant(MYSQL_INTEGER(unsigned=True), "mysql"),
        nullable=False,
        default=1,
        server_default="1",
        comment="被判编造的累计次数(跨轮)",
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
        comment="第一次被判编造的时间",
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="最近一次被判编造的时间",
    )
    resolution: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
        comment="处置说明:标已解决要写清怎么解决的,标无需解决要写清为什么不用改;退回未解决时清空。空着的处置在台账上等于没有交代",
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        comment="最近一次被标为已解决/无需解决的时间;复发后仍保留,用来标「复发」",
    )


async def upsert_faith_case(
    db: AsyncSession,
    eval_id: str,
    bucket: str,
    query: str,
    answer: str,
    reason: str,
    strategy: str = "hybrid_rerank",
    citations: Optional[List[Dict[str, Any]]] = None,
    judge_model: Optional[str] = None,
) -> FaithCase:
    """便捷异步函数：幂等写入或更新编造个案台账。

    - 若 eval_id 不存在：新增个案，seen_count=1，status='未解决'；
    - 若 eval_id 已存在：更新答案、裁判理由、证据快照，seen_count 自增 1，last_seen_at 刷新；
    - 若已有记录此前被标为「已解决」：重新判为编造时状态自动退回「未解决」，清空 resolution，
      并保留 resolved_at 作为复发标记。

    Args:
        db: 异步数据库会话
        eval_id: 评估集题号 (如 A43)
        bucket: 题目所属桶
        query: 用户问题
        answer: 编造答案原文
        reason: 裁判判定理由
        strategy: 检索策略
        citations: Top-K 证据快照全集
        judge_model: 裁判模型名称

    Returns:
        持久化后的 FaithCase 实体
    """
    stmt = select(FaithCase).where(FaithCase.eval_id == eval_id)
    result = await db.execute(stmt)
    case = result.scalar_one_or_none()

    now = datetime.utcnow()
    if case is None:
        case = FaithCase(
            eval_id=eval_id,
            bucket=bucket,
            query=query,
            strategy=strategy,
            answer=answer,
            reason=reason,
            citations=citations,
            judge_model=judge_model,
            status="未解决",
            seen_count=1,
            first_seen_at=now,
            last_seen_at=now,
            resolution=None,
            resolved_at=None,
        )
        db.add(case)
    else:
        case.bucket = bucket
        case.query = query
        case.strategy = strategy
        case.answer = answer
        case.reason = reason
        case.citations = citations
        if judge_model is not None:
            case.judge_model = judge_model
        case.seen_count += 1
        case.last_seen_at = now

        # 复发机制：若原状态为「已解决」，再次出现编造自动退回「未解决」
        if case.status in ("已解决", FaithCaseStatus.RESOLVED):
            case.status = "未解决"
            case.resolution = None
            # case.resolved_at 保留不变以追踪复发

    await db.commit()
    await db.refresh(case)
    return case
