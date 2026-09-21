import pytest
import asyncio
from typing import List, Dict, Any
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.db.session import Base
from app.models.eval_run import EvalRun, TriggeredBy
from app.services.eval.eval_pipeline import EvalPipelineService

@pytest.fixture
async def async_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with session_maker() as session:
        yield session

@pytest.fixture
def eval_service():
    return EvalPipelineService()

@pytest.mark.asyncio
async def test_run_eval_round_persists_to_db(async_session):
    service = EvalPipelineService()
    
    # 模拟运行一次评测
    run = await service.run_eval_round(
        db=async_session,
        triggered_by="手动",
        sample_limit=5
    )
    
    assert run.id is not None
    assert run.triggered_by == "手动"
    assert run.dataset_size == 5
    assert "recall_at_3" in run.metrics
    assert "recall_at_5" in run.metrics
    assert "recall_at_10" in run.metrics
    assert "mrr" in run.metrics
    assert "faithfulness" in run.metrics

@pytest.mark.asyncio
async def test_two_eval_rounds_calculates_deltas(async_session):
    service = EvalPipelineService()
    
    # 第一次运行
    run1 = await service.run_eval_round(db=async_session, sample_limit=2)
    # 第二次运行 (稍微改一点让它能比较)
    run2 = await service.run_eval_round(db=async_session, sample_limit=3)
    
    runs = await service.get_recent_runs_with_deltas(db=async_session, limit=10)
    assert len(runs) == 2
    
    # 最近的在前面
    assert runs[0]["id"] == run2.id
    assert runs[1]["id"] == run1.id
    
    # 检查 deltas 是否正确
    # 对于最新的 run，它应该有基于上一轮（run1）计算出的 delta
    latest = runs[0]
    assert "deltas" in latest
    # delta = current_val - prior_val
    for metric in ["recall_at_3", "mrr", "faithfulness"]:
        expected_delta = latest["metrics"][metric] - runs[1]["metrics"][metric]
        assert latest["deltas"][metric] == pytest.approx(expected_delta)

def test_run_eval_pipeline_cli(tmp_path):
    # 测试 CLI 是否能正常运行
    import subprocess
    import sys
    import os
    
    script_path = Path("scripts/run_eval_pipeline.py").absolute()
    db_path = tmp_path / "test_cli.db"
    
    # Init DB
    from sqlalchemy import create_engine
    from app.db.session import Base
    sync_engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(sync_engine)
    
    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    
    result = subprocess.run(
        [sys.executable, str(script_path), "--samples", "3"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    
    assert result.returncode == 0
    # stdout 里应该有格式化的报告表格和可能的警告
    assert "Current Score" in result.stdout
    assert "Delta" in result.stdout
