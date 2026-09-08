"""RAG 评估套件单元测试。

测试覆盖：
1. compute_retrieval_metrics: 精确测试 Recall@3, Recall@5, Recall@10 与 MRR 计算公式的数学正确性；
2. evaluate_faithfulness: 忠实度打分与编造理由提取；
3. 编造个案持久化与复发回退：当题目判定为编造（Faithfulness < 1.0）时，自动调用 upsert_faith_case
   幂等写入 faith_cases 表并保存 citations 快照；若原状态为「已解决」，断言其状态自动回退为「未解决」；
4. evaluate_sample: 支持正常题与 D_absent 拒答题分类评估；
5. run_comparative_eval: 支持少量样本在四策略间跑出指标矩阵与渲染 Markdown 报告。

所有测试完全在内存中确定性运行，绝不发起真实外部网络 I/O。
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.models.faith_case import FaithCase, FaithCaseStatus
from app.services.rag.evaluator import (
    BucketMetrics,
    EvaluationReport,
    RAGEvaluator,
    StrategyMetrics,
    compute_retrieval_metrics,
    evaluate_faithfulness,
    render_markdown_report,
)


def test_compute_retrieval_metrics_single_section_rank_1():
    """测试单目标小节：在 Rank 1 命中。"""
    docs = [
        {"section_path": "售后政策 > 退货流程", "chunk_id": 101},
        {"section_path": "售后政策 > 运费规则", "chunk_id": 102},
        {"section_path": "售后政策 > 维修时效", "chunk_id": 103},
    ]
    metrics = compute_retrieval_metrics(docs, expect_section=["退货流程"])
    assert metrics["recall@3"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr"] == 1.0


def test_compute_retrieval_metrics_single_section_rank_4():
    """测试单目标小节：在 Rank 4 命中（不在 Top-3，但在 Top-5 和 Top-10 内）。"""
    docs = [
        {"section_path": "售后政策 > 运费规则", "chunk_id": 101},
        {"section_path": "售后政策 > 发票说明", "chunk_id": 102},
        {"section_path": "售后政策 > 会员福利", "chunk_id": 103},
        {"section_path": "售后政策 > 换货流程", "chunk_id": 104},
        {"section_path": "售后政策 > 保修说明", "chunk_id": 105},
    ]
    metrics = compute_retrieval_metrics(docs, expect_section=["换货流程"])
    assert metrics["recall@3"] == 0.0
    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr"] == 0.25  # 1 / 4 = 0.25


def test_compute_retrieval_metrics_single_section_rank_8():
    """测试单目标小节：在 Rank 8 命中（不在 Top-5，但在 Top-10 内）。"""
    docs = [{"section_path": f"无关小节 > 章节{i}", "chunk_id": i} for i in range(1, 8)]
    docs.append({"section_path": "售后政策 > 价格保护", "chunk_id": 108})
    metrics = compute_retrieval_metrics(docs, expect_section=["价格保护"])
    assert metrics["recall@3"] == 0.0
    assert metrics["recall@5"] == 0.0
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr"] == pytest.approx(1.0 / 8, rel=1e-4)


def test_compute_retrieval_metrics_single_section_no_hit():
    """测试单目标小节：未命中任何相关章节。"""
    docs = [{"section_path": f"无关小节 > 章节{i}", "chunk_id": i} for i in range(1, 11)]
    metrics = compute_retrieval_metrics(docs, expect_section=["人工客服"])
    assert metrics["recall@3"] == 0.0
    assert metrics["recall@5"] == 0.0
    assert metrics["recall@10"] == 0.0
    assert metrics["mrr"] == 0.0


def test_compute_retrieval_metrics_multi_section_groups():
    """测试多目标小节（跨条目 E_multi）：多个必命中组的 Recall 与 MRR 计算。"""
    docs = [
        {"section_path": "商品规格 > PRO-X99", "chunk_id": 1},
        {"section_path": "售后政策 > 人工客服", "chunk_id": 2},
        {"section_path": "会员权益 > 积分说明", "chunk_id": 3},
        {"section_path": "售后政策 > 运费说明", "chunk_id": 4},
        {"section_path": "售后政策 > 处理时限", "chunk_id": 5},
    ]
    # 组 1：人工客服 -> Rank 2 (1/2 = 0.5)
    # 组 2：处理时限 -> Rank 5 (1/5 = 0.2)
    metrics = compute_retrieval_metrics(
        docs,
        expect_section=["人工客服", "处理时限"],
        expect_sections_all=[["人工客服"], ["处理时限"]],
    )
    # Recall@3: 组1(2<=3)命中，组2(5<=3)未命中 -> 1/2 = 0.5
    assert metrics["recall@3"] == 0.5
    # Recall@5: 组1(2<=5)命中，组2(5<=5)命中 -> 2/2 = 1.0
    assert metrics["recall@5"] == 1.0
    # Recall@10: 2/2 = 1.0
    assert metrics["recall@10"] == 1.0
    # MRR: (0.5 + 0.2) / 2 = 0.35
    assert metrics["mrr"] == pytest.approx(0.35, rel=1e-4)


def test_evaluate_faithfulness_supported_answer():
    """测试忠实度评估：回答完全基于证据，判定忠实。"""
    citations = [
        {
            "n": 1,
            "chunk_id": 10,
            "section_path": "售后政策 > 退货流程",
            "question": "退货政策是什么样的",
            "answer": "支持7天无理由退货，商品需保持包装完好不影响二次销售。",
        }
    ]
    query = "退货规则是什么？"
    answer = "根据官方规范，支持7天无理由退货，商品需保持包装完好不影响二次销售[1]。"
    res = evaluate_faithfulness(query, answer, citations)
    assert res["faithful"] is True
    assert res["score"] == 1.0
    assert "印证" in res["reason"] or "支持" in res["reason"]


def test_evaluate_faithfulness_hallucinated_answer():
    """测试忠实度评估：回答包含未提及的私下赔付/额外代金券等违规编造。"""
    citations = [
        {
            "n": 1,
            "chunk_id": 10,
            "section_path": "售后政策 > 退货流程",
            "question": "退货政策是什么样的",
            "answer": "支持7天无理由退货，商品需保持包装完好不影响二次销售。",
        }
    ]
    query = "退货有什么补偿吗？"
    answer = "支持7天无理由退货[1]，另外客服会为您私下补偿100元代金券并承诺退款秒级立即到账。"
    res = evaluate_faithfulness(query, answer, citations)
    assert res["faithful"] is False
    assert res["score"] < 1.0
    assert any(w in res["reason"] for w in ["补偿", "代金券", "立即到账", "编造", "未提及"])


@pytest.mark.asyncio
async def test_faith_case_persistence_new_case():
    """测试编造个案持久化：Faithfulness < 1.0 时自动幂等写入 faith_cases 表。"""
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    # 模拟未命中既有记录（新个案）
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    evaluator = RAGEvaluator(is_mock=True)

    sample = {
        "id": "A43",
        "bucket": "A_policy",
        "query": "退款多久到账？",
        "expect_section": ["退款时效"],
        "expect_points": ["1-3个工作日"],
        "should_refuse": False,
    }

    # 构造编造的评估结果触发落库
    with patch(
        "app.services.rag.evaluator.evaluate_faithfulness",
        return_value={
            "faithful": False,
            "score": 0.0,
            "reason": "编造了当天立即可用且赠送100元代金券",
        },
    ):
        res = await evaluator.evaluate_sample(
            sample=sample, strategy="hybrid_rerank", db=mock_db
        )

    assert res["faithfulness"] == 0.0
    assert res["faithful"] is False
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()

    created_case = mock_db.add.call_args[0][0]
    assert isinstance(created_case, FaithCase)
    assert created_case.eval_id == "A43"
    assert created_case.bucket == "A_policy"
    assert created_case.status == "未解决"
    assert created_case.seen_count == 1
    assert "100元代金券" in created_case.reason
    assert isinstance(created_case.citations, list)


@pytest.mark.asyncio
async def test_faith_case_persistence_recurrence_rollback():
    """测试编造个案复发回退：原为「已解决」的个案再次发生编造，自动退回「未解决」。"""
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.refresh = AsyncMock()

    prior_resolved_at = datetime(2026, 9, 1, 10, 0, 0)
    existing_case = FaithCase(
        eval_id="A43",
        bucket="A_policy",
        query="退款多久到账？",
        strategy="hybrid_rerank",
        answer="旧编造答案",
        reason="旧编造理由",
        status="已解决",
        seen_count=1,
        resolution="已修改 FAQ 条款并重新测试",
        resolved_at=prior_resolved_at,
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing_case
    mock_db.execute.return_value = mock_result

    evaluator = RAGEvaluator(is_mock=True)

    sample = {
        "id": "A43",
        "bucket": "A_policy",
        "query": "退款多久到账？",
        "expect_section": ["退款时效"],
        "expect_points": ["1-3个工作日"],
        "should_refuse": False,
    }

    with patch(
        "app.services.rag.evaluator.evaluate_faithfulness",
        return_value={
            "faithful": False,
            "score": 0.0,
            "reason": "复发：仍编造了秒级到账",
        },
    ):
        await evaluator.evaluate_sample(
            sample=sample, strategy="hybrid_rerank", db=mock_db
        )

    # 断言复发机制生效：
    # 1. 状态由「已解决」自动退回为「未解决」
    # 2. resolution 说明清空
    # 3. resolved_at 保留不变作为复发标记
    # 4. seen_count 增加到 2
    assert existing_case.status == "未解决"
    assert existing_case.resolution is None
    assert existing_case.resolved_at == prior_resolved_at
    assert existing_case.seen_count == 2
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_evaluate_sample_d_absent_refusal():
    """测试 D_absent 知识库外超纲提问的拒答判定。"""
    evaluator = RAGEvaluator(is_mock=True)

    sample = {
        "id": "D1",
        "bucket": "D_absent",
        "query": "你们卖不卖活体宠物,能在这买猫吗",
        "expect_section": [],
        "expect_points": [],
        "should_refuse": True,
    }

    res = await evaluator.evaluate_sample(sample=sample, strategy="hybrid_rerank")
    assert res["is_refusal_sample"] is True
    assert res["refused"] is True


@pytest.mark.asyncio
async def test_evaluate_strategy_aggregation():
    """测试按策略批量评测及分桶指标聚合。"""
    evaluator = RAGEvaluator(is_mock=True)

    samples = [
        {
            "id": "A1",
            "bucket": "A_policy",
            "query": "退货政策是什么样的",
            "expect_section": ["退货流程"],
            "expect_points": ["7 天无理由"],
            "should_refuse": False,
        },
        {
            "id": "B1",
            "bucket": "B_model",
            "query": "PRO-X99防水等级是多少",
            "expect_section": ["PRO-X99规格"],
            "expect_points": ["5ATM"],
            "should_refuse": False,
        },
        {
            "id": "D1",
            "bucket": "D_absent",
            "query": "你们卖火箭吗",
            "expect_section": [],
            "expect_points": [],
            "should_refuse": True,
        },
    ]

    metrics = await evaluator.evaluate_strategy("hybrid_rerank", samples)
    assert isinstance(metrics, StrategyMetrics)
    assert metrics.strategy == "hybrid_rerank"
    assert "A_policy" in metrics.bucket_metrics
    assert "B_model" in metrics.bucket_metrics
    assert "D_absent" in metrics.bucket_metrics

    # D 桶应有拒答率
    d_metrics = metrics.bucket_metrics["D_absent"]
    assert d_metrics.refusal_rate == 1.0

    # overall 应正确聚合
    assert metrics.overall is not None
    assert metrics.overall.sample_count == 3


@pytest.mark.asyncio
async def test_run_comparative_eval_and_markdown_report():
    """测试四策略对比评测流水线并生成 Markdown 报告。"""
    evaluator = RAGEvaluator(is_mock=True)

    # 对 4 策略运行每桶 1 题的快速微缩评测
    report = await evaluator.run_comparative_eval(
        strategies=["vector_only", "bm25_only", "hybrid", "hybrid_rerank"],
        samples_per_bucket=1,
    )

    assert isinstance(report, EvaluationReport)
    assert len(report.strategies) == 4
    for s in ["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]:
        assert s in report.strategies

    md = render_markdown_report(report)
    assert "# AstroBot RAG Chapter 4 四策略对比评测报告" in md
    assert "Recall@3" in md
    assert "Recall@5" in md
    assert "Recall@10" in md
    assert "MRR" in md
    assert "Faithfulness" in md
    assert "vector_only" in md
    assert "hybrid_rerank" in md
