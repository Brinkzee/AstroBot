"""Tests for Chapter 10 model evaluation, tiered redlines, error attribution, and threshold replay."""

import json
from pathlib import Path
import numpy as np
import pytest

from app.services.classifier.taxonomy import (
    TAXONOMY_17,
    STRICT_TIER,
    MEDIUM_TIER,
    LOOSE_TIER,
)
from scripts.evaluate_classifier import (
    categorize_errors,
    verify_error_accounting_balance,
    check_tiered_redlines,
    compute_confusion_matrices,
    generate_error_markdown,
    evaluate_dataset,
)
from scripts.scan_threshold_replay import replay_threshold_scan


def test_categorize_errors_accounting():
    """Test 3-way error categorization (under-tag, over-tag, displaced) and mathematical balance."""
    samples = [
        {
            "text": "买大了想退",
            "true_labels": ["尺码", "退换货"],
            "pred_labels": ["退换货"],  # 漏打尺码 (under-tag: 1)
        },
        {
            "text": "什么时候发货",
            "true_labels": ["物流"],
            "pred_labels": ["物流", "运费"],  # 多打运费 (over-tag: 1)
        },
        {
            "text": "能不能修一下",
            "true_labels": ["保修维修"],
            "pred_labels": ["退换货"],  # 错位 (displaced: 1 -> 漏打保修维修 + 多打退换货)
        },
    ]

    categorized = categorize_errors(samples)
    assert categorized["under_tag_count"] == 1  # 漏打
    assert categorized["over_tag_count"] == 1   # 多打
    assert categorized["displaced_count"] == 1  # 错位
    assert len(categorized["error_samples"]) == 3

    # 数学闭环验证
    # 混淆矩阵中惩罚总笔数 = 漏打(1) + 多打(1) + 2 * 错位 = 4
    total_penalty = verify_error_accounting_balance(categorized)
    assert total_penalty == 4


def test_error_accounting_mathematical_balance_complex_cases():
    """Test error accounting on complex multi-label permutations."""
    samples = [
        # 完全正确 (0 errors)
        {
            "text": "正确样本",
            "true_labels": ["退换货", "发票"],
            "pred_labels": ["退换货", "发票"],
        },
        # 错位 2 次 + 漏打 1 次 (true=3, pred=2, 0 common)
        {
            "text": "复杂错位样本",
            "true_labels": ["退换货", "物流", "尺码"],
            "pred_labels": ["运费", "优惠活动"],
        },
        # 多打 2 次
        {
            "text": "纯多打样本",
            "true_labels": ["发票"],
            "pred_labels": ["发票", "支付", "账号"],
        },
        # 纯漏打 2 次
        {
            "text": "纯漏打样本",
            "true_labels": ["商品信息", "保修维修"],
            "pred_labels": [],
        },
    ]

    categorized = categorize_errors(samples)
    # 样本 1: 0 errors
    # 样本 2: miss=3, extra=2 -> displaced=2, under=1, over=0 (penalty = 1 + 0 + 2*2 = 5)
    # 样本 3: miss=0, extra=2 -> displaced=0, under=0, over=2 (penalty = 2)
    # 样本 4: miss=2, extra=0 -> displaced=0, under=2, over=0 (penalty = 2)
    # Total: under=3, over=2, displaced=2 -> penalty = 3 + 2 + 4 = 9

    assert categorized["under_tag_count"] == 3
    assert categorized["over_tag_count"] == 2
    assert categorized["displaced_count"] == 2
    total_penalty = verify_error_accounting_balance(categorized)
    assert total_penalty == 9


def test_tiered_redlines_status_and_prompts():
    """Test strict (>=0.90), medium (>=0.80), and loose (no line, '—') tiered redlines."""
    # 模拟各类指标
    per_class_f1 = {}
    for cat in TAXONOMY_17:
        if cat in STRICT_TIER:
            # 退换货达标 (0.92), 物流不达标 (0.85)
            per_class_f1[cat] = 0.92 if cat == "退换货" else 0.85
        elif cat in MEDIUM_TIER:
            # 运费达标 (0.82), 优惠活动不达标 (0.75)
            per_class_f1[cat] = 0.82 if cat == "运费" else 0.75
        else:
            # 宽档不论分数多高多低，一律画 '—'
            per_class_f1[cat] = 0.95

    tiered_report = check_tiered_redlines(per_class_f1)

    # 严档检查
    assert tiered_report["退换货"]["status"] == "pass"
    assert tiered_report["退换货"]["status_symbol"] == "✅"
    assert tiered_report["退换货"]["tier"] == "strict"
    assert tiered_report["退换货"]["redline"] == 0.90

    assert tiered_report["物流"]["status"] == "fail"
    assert tiered_report["物流"]["status_symbol"] == "❌"
    assert "先回头搞数据" in tiered_report["物流"]["message"]

    # 中档检查
    assert tiered_report["运费"]["status"] == "pass"
    assert tiered_report["运费"]["status_symbol"] == "✅"
    assert tiered_report["运费"]["tier"] == "medium"
    assert tiered_report["运费"]["redline"] == 0.80

    assert tiered_report["优惠活动"]["status"] == "fail"
    assert tiered_report["优惠活动"]["status_symbol"] == "❌"
    assert "先回头搞数据" in tiered_report["优惠活动"]["message"]

    # 宽档检查：严禁画 ✅，必须统一画 '—'
    for cat in LOOSE_TIER:
        assert tiered_report[cat]["status"] == "none"
        assert tiered_report[cat]["status_symbol"] == "—"
        assert tiered_report[cat]["status_symbol"] != "✅"
        assert tiered_report[cat]["tier"] == "loose"
        assert tiered_report[cat]["redline"] is None


def test_confusion_matrices_computation():
    """Test 2x2 confusion matrix (TP, FP, TN, FN) computation for each of 17 classes."""
    num_samples = 10
    num_classes = len(TAXONOMY_17)
    y_true = np.zeros((num_samples, num_classes), dtype=int)
    y_pred = np.zeros((num_samples, num_classes), dtype=int)

    # 类目 0: 2 TP, 1 FP, 2 FN, 5 TN
    y_true[0:4, 0] = 1  # 4 positives
    y_pred[0:2, 0] = 1  # 2 TP
    y_pred[4, 0] = 1    # 1 FP (true is 0)
    # fn = 4 - 2 = 2
    # tn = 10 - 2 - 1 - 2 = 5

    matrices = compute_confusion_matrices(y_true, y_pred, taxonomy=TAXONOMY_17)
    cat0_matrix = matrices[TAXONOMY_17[0]]

    assert cat0_matrix["tp"] == 2
    assert cat0_matrix["fp"] == 1
    assert cat0_matrix["fn"] == 2
    assert cat0_matrix["tn"] == 5
    assert cat0_matrix["tp"] + cat0_matrix["fp"] + cat0_matrix["fn"] + cat0_matrix["tn"] == num_samples


def test_confusion_matrix_and_error_attribution_closed_loop():
    """Test that sum(FP + FN) across all 17 classes strictly equals under + over + 2*displaced."""
    # 构建 3 个样本的真实与预测
    y_true = np.zeros((3, len(TAXONOMY_17)), dtype=int)
    y_pred = np.zeros((3, len(TAXONOMY_17)), dtype=int)

    # 样本 0: 漏打
    idx_size = TAXONOMY_17.index("尺码")
    idx_ret = TAXONOMY_17.index("退换货")
    y_true[0, idx_size] = 1
    y_true[0, idx_ret] = 1
    y_pred[0, idx_ret] = 1

    # 样本 1: 多打
    idx_logistics = TAXONOMY_17.index("物流")
    idx_freight = TAXONOMY_17.index("运费")
    y_true[1, idx_logistics] = 1
    y_pred[1, idx_logistics] = 1
    y_pred[1, idx_freight] = 1

    # 样本 2: 错位
    idx_repair = TAXONOMY_17.index("保修维修")
    y_true[2, idx_repair] = 1
    y_pred[2, idx_ret] = 1

    matrices = compute_confusion_matrices(y_true, y_pred, taxonomy=TAXONOMY_17)
    total_fp_fn = sum(m["fp"] + m["fn"] for m in matrices.values())

    samples = [
        {"text": "s0", "true_labels": ["尺码", "退换货"], "pred_labels": ["退换货"]},
        {"text": "s1", "true_labels": ["物流"], "pred_labels": ["物流", "运费"]},
        {"text": "s2", "true_labels": ["保修维修"], "pred_labels": ["退换货"]},
    ]
    categorized = categorize_errors(samples)
    total_penalty = verify_error_accounting_balance(categorized, total_fp_fn)

    assert total_penalty == total_fp_fn
    assert total_penalty == 4


def test_generate_error_markdown_content():
    """Test error samples markdown report format and content."""
    samples = [
        {
            "text": "买大了想退",
            "true_labels": ["尺码", "退换货"],
            "pred_labels": ["退换货"],
        },
        {
            "text": "能不能修一下",
            "true_labels": ["保修维修"],
            "pred_labels": ["退换货"],
        },
    ]
    categorized = categorize_errors(samples)
    md_content = generate_error_markdown(categorized, total_fp_fn=3)

    assert "# 错因分析与判错样本账本" in md_content
    assert "数学闭环" in md_content
    assert "买大了想退" in md_content
    assert "能不能修一下" in md_content
    assert "修归保修维修、退归退换货" in md_content


def test_scan_threshold_replay_matches_threshold_json():
    """Test that replaying 9 candidate lines finds the exact optimal threshold in threshold.json."""
    threshold_path = Path("data/ch10/threshold.json")
    assert threshold_path.exists(), "data/ch10/threshold.json must exist"

    with open(threshold_path, "r", encoding="utf-8") as f:
        saved_threshold_data = json.load(f)

    target_thresh = saved_threshold_data["threshold"]
    replay_result = replay_threshold_scan(
        val_scores_file="data/ch10/val_scores.json",
        threshold_file="data/ch10/threshold.json",
    )

    assert replay_result["best_threshold"] == target_thresh
    assert len(replay_result["grid_table"]) == 9
    assert replay_result["matched"] is True
