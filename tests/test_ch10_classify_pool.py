import argparse
from datetime import datetime, timezone
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.db.session import Base
from app.models.low_confidence import LowConfidenceQuestion
from app.models.topic_classification import TopicClassification
from scripts.classify_pool import (
    call_classifier_service,
    fetch_unclassified_questions,
    parse_args,
    run_classify_pool,
    main,
)


@pytest.fixture
async def memory_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_classify_pool_min_batch_and_force(memory_db: AsyncSession):
    """测试 min_batch 门槛拦截与 --force 强制执行。"""
    # 插入 3 条低置信度问题
    for i in range(3):
        memory_db.add(LowConfidenceQuestion(raw_question=f"测试问题{i}", source="self_check"))
    await memory_db.commit()

    # 1. 默认 min_batch=10，不足时跳过执行并友好提示
    result = await run_classify_pool(db=memory_db, min_batch=10, force=False)
    assert result["processed"] == 0
    assert result["total_unclassified"] == 3
    assert result["skipped_due_to_min_batch"] is True

    # 验证数据库中尚未插入分类记录
    res = await memory_db.execute(select(TopicClassification))
    assert len(res.scalars().all()) == 0

    # 2. force=True，即使不足 10 条也强制执行处理
    mock_resp = [{"labels": ["商品信息"], "scores": {"商品信息": 0.95}}]
    with patch("scripts.classify_pool.call_classifier_service", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        result = await run_classify_pool(db=memory_db, min_batch=10, force=True)
        assert result["processed"] == 3
        assert result["skipped_due_to_min_batch"] is False
        assert result["topic_counts"] == {"商品信息": 3}

    # 验证落库
    res = await memory_db.execute(select(TopicClassification))
    items = res.scalars().all()
    assert len(items) == 3
    for item in items:
        assert item.labels == ["商品信息"]


@pytest.mark.asyncio
async def test_classify_pool_idempotency(memory_db: AsyncSession):
    """测试重跑幂等性：基于 question_id 唯一约束，已归类的行不重复归类、不报错。"""
    # 插入 4 条低置信度问题
    for i in range(4):
        memory_db.add(LowConfidenceQuestion(raw_question=f"幂等测试问题{i}", source="retrieval_low_conf"))
    await memory_db.commit()

    mock_resp = [
        {"labels": ["物流查询"], "scores": {"物流查询": 0.9}},
        {"labels": ["尺码建议"], "scores": {"尺码建议": 0.88}},
        {"labels": ["退换货规则"], "scores": {"退换货规则": 0.92}},
        {"labels": ["优惠活动"], "scores": {"优惠活动": 0.85}},
    ]
    with patch("scripts.classify_pool.call_classifier_service", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        result1 = await run_classify_pool(db=memory_db, min_batch=2, force=False)
        assert result1["processed"] == 4
        assert result1["total_unclassified"] == 4

    # 检查数据库总记录数
    res = await memory_db.execute(select(TopicClassification))
    assert len(res.scalars().all()) == 4

    # 再次重跑：无未归类记录，返回 processed=0，无报错，不重复插入
    with patch("scripts.classify_pool.call_classifier_service", new_callable=AsyncMock) as mock_call:
        result2 = await run_classify_pool(db=memory_db, min_batch=2, force=True)
        assert result2["processed"] == 0
        assert result2["total_unclassified"] == 0
        mock_call.assert_not_called()

    # 数据库记录依然为 4
    res2 = await memory_db.execute(select(TopicClassification))
    assert len(res2.scalars().all()) == 4


@pytest.mark.asyncio
async def test_classify_pool_batch_chunking(memory_db: AsyncSession):
    """测试分批调用 :8110 (batch_size 逻辑)。"""
    for i in range(5):
        memory_db.add(LowConfidenceQuestion(raw_question=f"分批测试问题{i}", source="user_feedback"))
    await memory_db.commit()

    called_batches = []

    async def fake_call(service_url: str, texts: list):
        called_batches.append(list(texts))
        return [{"labels": ["服务投诉"], "scores": {"服务投诉": 0.9}}] * len(texts)

    with patch("scripts.classify_pool.call_classifier_service", side_effect=fake_call):
        # min_batch=5, batch_size=2 -> 应该调用 3 次 (2, 2, 1)
        result = await run_classify_pool(db=memory_db, min_batch=5, batch_size=2, force=False)
        assert result["processed"] == 5
        assert len(called_batches) == 3
        assert len(called_batches[0]) == 2
        assert len(called_batches[1]) == 2
        assert len(called_batches[2]) == 1


@pytest.mark.asyncio
async def test_classify_pool_service_connection_error(memory_db: AsyncSession):
    """测试连接服务失败时的异常与友好提示。"""
    memory_db.add(LowConfidenceQuestion(raw_question="网络异常测试", source="self_check"))
    await memory_db.commit()

    with patch("scripts.classify_pool.call_classifier_service", side_effect=ConnectionError("无法连接到分类推理服务")):
        with pytest.raises(ConnectionError, match="无法连接到分类推理服务"):
            await run_classify_pool(db=memory_db, force=True)


@pytest.mark.asyncio
async def test_call_classifier_service_http():
    """测试 call_classifier_service HTTP POST 调用与解析。"""
    import httpx

    fake_payload = {
        "results": [
            {"text": "这件衣服偏大吗", "labels": ["尺码建议"], "scores": {"尺码建议": 0.96}},
            {"text": "发什么快递", "labels": ["物流查询"], "scores": {"物流查询": 0.91}},
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_payload
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        results = await call_classifier_service(
            service_url="http://127.0.0.1:8110",
            texts=["这件衣服偏大吗", "发什么快递"],
        )
        assert len(results) == 2
        assert results[0]["labels"] == ["尺码建议"]
        assert results[1]["labels"] == ["物流查询"]
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"] == {"texts": ["这件衣服偏大吗", "发什么快递"]}


def test_classify_pool_cli_args():
    """测试命令行参数解析。"""
    args = parse_args(["--min-batch", "20", "--force", "--batch-size", "16", "--service-url", "http://localhost:8110"])
    assert args.min_batch == 20
    assert args.force is True
    assert args.batch_size == 16
    assert args.service_url == "http://localhost:8110"

    default_args = parse_args([])
    assert default_args.min_batch == 10
    assert default_args.force is False
    assert default_args.batch_size == 32
    assert default_args.service_url == "http://127.0.0.1:8110"


@pytest.mark.asyncio
async def test_main_cli_execution():
    """测试 CLI main() 入口函数调度。"""
    with patch("scripts.classify_pool.parse_args") as mock_parse, \
         patch("scripts.classify_pool.run_classify_pool", new_callable=AsyncMock) as mock_run:
        mock_parse.return_value = argparse.Namespace(
            min_batch=10,
            force=False,
            batch_size=32,
            service_url="http://127.0.0.1:8110",
        )
        mock_run.return_value = {
            "total_unclassified": 0,
            "processed": 0,
            "skipped_due_to_min_batch": True,
            "topic_counts": {},
            "elapsed_sec": 0.01,
        }
        exit_code = await main()
        assert exit_code == 0
        mock_run.assert_called_once()
