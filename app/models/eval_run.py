import enum
from datetime import datetime, timezone
from typing import Any, Dict
from sqlalchemy import BigInteger, Integer, Enum, DateTime, JSON, func
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class TriggeredBy(str, enum.Enum):
    CRON = "定时"
    MANUAL = "手动"


class EvalRun(Base):
    """评估轮次 ORM 实体。

    一行 = 评估流水线跑完的一轮，各指标分数收进 metrics JSON，按时间连起来就是趋势线。
    """
    __tablename__ = "eval_runs"

    def __init__(self, **kwargs):
        kwargs.setdefault("triggered_by", TriggeredBy.CRON.value)
        super().__init__(**kwargs)

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="评估轮次主键",
    )
    triggered_by: Mapped[str] = mapped_column(
        Enum("定时", "手动", name="eval_triggered_by"),
        nullable=False,
        default=TriggeredBy.CRON.value,
        comment="这轮怎么起的:定时任务,或某次改动后手动跑",
    )
    dataset_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="这轮跑的评估集条数",
    )
    metrics: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        comment='各指标分数,如 {"recall_at_k":0.82,"mrr":0.71,"faithfulness":0.90}',
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="跑完落表时间",
    )
