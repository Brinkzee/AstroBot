#!/usr/bin/env python3
"""Tiered redlines evaluation, 3-way error attribution, and reporting.

Chapter 10: Multi-label topic classification.
Evaluates model on the test dataset with:
- 17-class Precision, Recall, F1, Support, Macro-F1, Micro-F1
- 3-tier redlines (Strict >= 0.90, Medium >= 0.80, Loose '—')
- 17-class multilabel 2x2 confusion matrices (TP, FP, TN, FN)
- 3-way error attribution (under-tag, over-tag, displaced)
- Mathematical closed loop: under + over + 2 * displaced == sum(FP + FN)
- Outputs:
  - reports/ch10_evaluation_report.json
  - data/ch10/reports/error_samples.md
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.services.classifier.taxonomy import (
    CATEGORY_BOUNDARIES,
    LOOSE_TIER,
    MEDIUM_TIER,
    STRICT_TIER,
    TAXONOMY_17,
    get_tier_for_category,
)
from scripts.train_classifier import MultiLabelDataset, load_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_classifier")


def categorize_errors(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform 3-way error attribution on sample predictions.

    For each sample:
    - true_labels: ground truth labels (T)
    - pred_labels: predicted labels (P)
    - Missed (under-tag): T - P
    - Extra (over-tag): P - T
    - Displaced: min(|T - P|, |P - T|) -> counts as 1 displacement (2 penalty points: 1 missed + 1 extra)
    - Remaining under-tag: |T - P| - displaced (1 penalty point)
    - Remaining over-tag: |P - T| - displaced (1 penalty point)

    Total penalty for sample: under + over + 2 * displaced == |T - P| + |P - T| == FN_s + FP_s.

    Args:
        samples: List of sample dicts with keys 'text', 'true_labels', 'pred_labels'.

    Returns:
        Dict containing counts of under-tag, over-tag, displaced, total penalty, and detailed records.
    """
    total_under_tag = 0
    total_over_tag = 0
    total_displaced = 0
    error_samples: List[Dict[str, Any]] = []

    for item in samples:
        text = item.get("text", "")
        true_labels = item.get("true_labels", [])
        pred_labels = item.get("pred_labels", [])

        true_set = set(true_labels)
        pred_set = set(pred_labels)

        if true_set == pred_set:
            continue

        missed = sorted(list(true_set - pred_set))
        extra = sorted(list(pred_set - true_set))

        displaced_count = min(len(missed), len(extra))
        under_tag_count = len(missed) - displaced_count
        over_tag_count = len(extra) - displaced_count
        penalty = under_tag_count + over_tag_count + 2 * displaced_count

        total_under_tag += under_tag_count
        total_over_tag += over_tag_count
        total_displaced += displaced_count

        # Check for near-boundary pairs (e.g. 保修维修 <-> 退换货)
        boundary_notes = []
        for m in missed:
            for e in extra:
                boundary_key = (m, e)
                if boundary_key in CATEGORY_BOUNDARIES:
                    boundary_notes.append(f"{m} ↔ {e}: {CATEGORY_BOUNDARIES[boundary_key]}")

        if not boundary_notes:
            for m in missed:
                if m in CATEGORY_BOUNDARIES:
                    boundary_notes.append(f"{m}: {CATEGORY_BOUNDARIES[m]}")

        error_samples.append({
            "text": text,
            "true_labels": true_labels,
            "pred_labels": pred_labels,
            "missed_labels": missed,
            "extra_labels": extra,
            "displaced_count": displaced_count,
            "under_tag_count": under_tag_count,
            "over_tag_count": over_tag_count,
            "penalty": penalty,
            "boundary_note": "；".join(boundary_notes) if boundary_notes else "—",
        })

    total_penalty = total_under_tag + total_over_tag + 2 * total_displaced

    return {
        "total_error_samples": len(error_samples),
        "under_tag_count": total_under_tag,
        "over_tag_count": total_over_tag,
        "displaced_count": total_displaced,
        "total_penalty": total_penalty,
        "error_samples": error_samples,
    }


def verify_error_accounting_balance(
    categorized: Dict[str, Any],
    total_fp_fn: Optional[int] = None,
) -> int:
    """Verify that error accounting strictly satisfies mathematical balance.

    Balance condition:
    total_penalty = under_tag + over_tag + 2 * displaced == sum(FP + FN) in confusion matrix.

    Args:
        categorized: Output from categorize_errors.
        total_fp_fn: Optional sum of FP + FN across confusion matrices to assert against.

    Returns:
        int: Total penalty count.
    """
    total_penalty = (
        categorized["under_tag_count"]
        + categorized["over_tag_count"]
        + 2 * categorized["displaced_count"]
    )

    if total_fp_fn is not None:
        assert total_penalty == total_fp_fn, (
            f"Error accounting mathematical imbalance: "
            f"penalty ({total_penalty} = {categorized['under_tag_count']} + "
            f"{categorized['over_tag_count']} + 2*{categorized['displaced_count']}) "
            f"!= confusion matrix FP+FN ({total_fp_fn})"
        )

    return total_penalty


def check_tiered_redlines(
    per_class_f1: Dict[str, float],
    taxonomy: List[str] = TAXONOMY_17,
) -> Dict[str, Dict[str, Any]]:
    """Check F1 scores against 3-tier redlines.

    - Strict tier (5 classes): F1 >= 0.90. Passed: ✅, Failed: ❌ + '先回头搞数据'
    - Medium tier (8 classes): F1 >= 0.80. Passed: ✅, Failed: ❌ + '先回头搞数据'
    - Loose tier (4 classes): No line. Status: '—' (STRICTLY NO ✅)

    Args:
        per_class_f1: Dict mapping category name to F1 score.
        taxonomy: List of categories.

    Returns:
        Dict mapping category name to tier evaluation result.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for cat in taxonomy:
        tier = get_tier_for_category(cat)
        f1 = float(per_class_f1.get(cat, 0.0))

        if tier == "strict":
            redline = 0.90
            passed = f1 >= redline
            status = "pass" if passed else "fail"
            symbol = "✅" if passed else "❌"
            message = "达标" if passed else f"先回头搞数据 (F1: {f1:.2f} < {redline:.2f})"
        elif tier == "medium":
            redline = 0.80
            passed = f1 >= redline
            status = "pass" if passed else "fail"
            symbol = "✅" if passed else "❌"
            message = "达标" if passed else f"先回头搞数据 (F1: {f1:.2f} < {redline:.2f})"
        else:
            redline = None
            passed = None
            status = "none"
            symbol = "—"
            message = "不设线（宽档）"

        results[cat] = {
            "category": cat,
            "tier": tier,
            "redline": redline,
            "f1": round(f1, 4),
            "status": status,
            "status_symbol": symbol,
            "passed": passed,
            "message": message,
        }

    return results


def compute_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    taxonomy: List[str] = TAXONOMY_17,
) -> Dict[str, Dict[str, int]]:
    """Compute 2x2 confusion matrix (TP, FP, TN, FN) for each category.

    Args:
        y_true: Binary ground truth array of shape (N, num_classes).
        y_pred: Binary predicted array of shape (N, num_classes).
        taxonomy: List of category names.

    Returns:
        Dict mapping category name to {'tp': int, 'fp': int, 'tn': int, 'fn': int}.
    """
    matrices: Dict[str, Dict[str, int]] = {}
    num_classes = y_true.shape[1]

    for c in range(num_classes):
        cat = taxonomy[c] if c < len(taxonomy) else f"class_{c}"
        yt = y_true[:, c]
        yp = y_pred[:, c]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))

        matrices[cat] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    return matrices


def generate_error_markdown(categorized: Dict[str, Any], total_fp_fn: int) -> str:
    """Generate Markdown report for error samples and 3-way error attribution."""
    lines: List[str] = []
    lines.append("# 错因分析与判错样本账本 (Error Samples Report)")
    lines.append("")
    lines.append("> 本报告自动导出测试集预测与标准答案不一致的样本，按「漏打 / 多打 / 错位」三向记账，并实施混淆矩阵数学闭环验证。")
    lines.append("")
    lines.append("## 1. 错因三向记账与数学闭环汇总")
    lines.append("")
    lines.append("| 指标项 | 统计数值 | 惩罚权重点数 | 说明 |")
    lines.append("| :--- | :---: | :---: | :--- |")
    lines.append(f"| **错例总条数** | {categorized['total_error_samples']} | — | 测试集上预测与标准答案不符的样本数 |")
    lines.append(f"| **漏打笔数 (放跑)** | {categorized['under_tag_count']} | +1 笔 / 条 | 真实有、预测无 (真实类目 FN +1) |")
    lines.append(f"| **多打笔数 (冤枉)** | {categorized['over_tag_count']} | +1 笔 / 条 | 真实无、预测有 (误判类目 FP +1) |")
    lines.append(f"| **错位笔数 (位移)** | {categorized['displaced_count']} | +2 笔 / 条 | 真实该打 A 没打却打了 B (FN +1 & FP +1) |")
    lines.append(f"| **总惩罚笔数** | **{categorized['total_penalty']}** | — | `漏打 + 多打 + 2 * 错位` |")
    lines.append(f"| **混淆矩阵 (FP + FN) 总数** | **{total_fp_fn}** | — | 17 类多标签混淆矩阵误判之和 |")

    is_balanced = categorized["total_penalty"] == total_fp_fn
    balance_badge = "✅ 100% 严格平衡" if is_balanced else "❌ 失衡告警"
    lines.append(f"| **数学闭环验证** | **{balance_badge}** | — | `总惩罚笔数 == 混淆矩阵 (FP + FN)` |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. 判错样本逐条明细")
    lines.append("")
    lines.append("| 序号 | 样本问句 | 真实标签 | 预测标签 | 错因归因 | 惩罚 | 近邻边界与说明 |")
    lines.append("| :---: | :--- | :--- | :--- | :--- | :---: | :--- |")

    for i, err in enumerate(categorized["error_samples"], 1):
        text = err["text"].replace("|", "\\|")
        true_lbl = ", ".join(err["true_labels"]) or "无"
        pred_lbl = ", ".join(err["pred_labels"]) or "无"

        attr_parts = []
        if err["displaced_count"] > 0:
            attr_parts.append(f"错位({err['displaced_count']})")
        if err["under_tag_count"] > 0:
            attr_parts.append(f"漏打[{','.join(err['missed_labels'])}]")
        if err["over_tag_count"] > 0:
            attr_parts.append(f"多打[{','.join(err['extra_labels'])}]")
        attr_str = " + ".join(attr_parts) or "其他"

        penalty = err["penalty"]
        note = err["boundary_note"].replace("|", "\\|")

        lines.append(f"| {i} | {text} | `{true_lbl}` | `{pred_lbl}` | {attr_str} | +{penalty} | {note} |")

    lines.append("")
    return "\n".join(lines)


def evaluate_dataset(
    model: torch.nn.Module,
    tokenizer: Any,
    records: List[Dict[str, Any]],
    threshold: float,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 128,
    taxonomy: List[str] = TAXONOMY_17,
) -> Dict[str, Any]:
    """Run full evaluation on dataset records and generate structured metrics."""
    dataset = MultiLabelDataset(records, tokenizer, max_length=max_length, taxonomy=taxonomy)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits)

            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.vstack(all_targets) if all_targets else np.empty((0, len(taxonomy)))
    y_probs = np.vstack(all_probs) if all_probs else np.empty((0, len(taxonomy)))
    y_pred = (y_probs >= threshold).astype(int)

    # 1. Per-class metrics and confusion matrices
    confusion_matrices = compute_confusion_matrices(y_true, y_pred, taxonomy=taxonomy)

    per_class_f1: Dict[str, float] = {}
    per_class_precision: Dict[str, float] = {}
    per_class_recall: Dict[str, float] = {}
    per_class_support: Dict[str, int] = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cat in taxonomy:
        cm = confusion_matrices[cat]
        tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
        support = int(tp + fn)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        per_class_precision[cat] = prec
        per_class_recall[cat] = rec
        per_class_f1[cat] = f1
        per_class_support[cat] = support

        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0
    macro_precision = float(np.mean(list(per_class_precision.values()))) if per_class_precision else 0.0
    macro_recall = float(np.mean(list(per_class_recall.values()))) if per_class_recall else 0.0

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0

    # 2. Tiered redlines evaluation
    tiered_results = check_tiered_redlines(per_class_f1, taxonomy=taxonomy)

    # 3. Predict labels for error samples extraction
    samples_with_preds = []
    for i, rec in enumerate(records):
        pred_label_indices = np.where(y_pred[i] == 1)[0]
        pred_labels = [taxonomy[idx] for idx in pred_label_indices if idx < len(taxonomy)]
        samples_with_preds.append({
            "text": rec.get("text", ""),
            "true_labels": rec.get("labels", []),
            "pred_labels": pred_labels,
        })

    # 4. Error categorization & mathematical balance
    categorized = categorize_errors(samples_with_preds)
    total_fp_fn = sum(cm["fp"] + cm["fn"] for cm in confusion_matrices.values())
    verify_error_accounting_balance(categorized, total_fp_fn)

    # 5. Build per_class detailed records
    per_class_records = []
    for cat in taxonomy:
        t_info = tiered_results[cat]
        per_class_records.append({
            "category": cat,
            "tier": t_info["tier"],
            "redline": t_info["redline"],
            "precision": round(per_class_precision[cat], 4),
            "recall": round(per_class_recall[cat], 4),
            "f1": round(per_class_f1[cat], 4),
            "support": per_class_support[cat],
            "status": t_info["status"],
            "status_symbol": t_info["status_symbol"],
            "passed": t_info["passed"],
            "message": t_info["message"],
            "confusion_matrix": confusion_matrices[cat],
        })

    return {
        "summary": {
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "micro_f1": round(micro_f1, 4),
            "micro_precision": round(micro_precision, 4),
            "micro_recall": round(micro_recall, 4),
            "total_samples": len(records),
            "total_error_samples": categorized["total_error_samples"],
        },
        "threshold": round(float(threshold), 2),
        "per_class": per_class_records,
        "confusion_matrices": confusion_matrices,
        "error_accounting": {
            "total_error_samples": categorized["total_error_samples"],
            "under_tag_count": categorized["under_tag_count"],
            "over_tag_count": categorized["over_tag_count"],
            "displaced_count": categorized["displaced_count"],
            "total_penalty": categorized["total_penalty"],
            "confusion_matrix_sum": total_fp_fn,
            "balanced": categorized["total_penalty"] == total_fp_fn,
        },
        "error_samples": categorized["error_samples"],
    }


def evaluate_classifier(
    model_dir: str = "data/ch10/best_model",
    test_file: str = "mewhelp-ch10-dataset/test.jsonl",
    threshold_file: str = "data/ch10/threshold.json",
    report_file: str = "reports/ch10_evaluation_report.json",
    error_report_file: str = "data/ch10/reports/error_samples.md",
    device_name: str = "auto",
    batch_size: int = 16,
    max_length: int = 128,
) -> Dict[str, Any]:
    """Execute full evaluation pipeline and export JSON and Markdown reports."""
    # 1. Device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    # 2. Threshold
    threshold_path = Path(threshold_file)
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            thresh_data = json.load(f)
        threshold = float(thresh_data.get("threshold", 0.45))
        candidate_scores = thresh_data.get("candidate_scores", {})
    else:
        threshold = 0.45
        candidate_scores = {}
    logger.info(f"Loaded decision threshold: {threshold:.2f} from {threshold_file}")

    # 3. Model & Tokenizer
    logger.info(f"Loading model and tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    # 4. Test dataset
    logger.info(f"Loading test dataset from: {test_file}")
    records = load_jsonl(test_file)
    logger.info(f"Loaded {len(records)} test samples.")

    # 5. Evaluate
    eval_result = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        records=records,
        threshold=threshold,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        taxonomy=TAXONOMY_17,
    )

    # 6. Save reports
    eval_result["candidate_scores"] = candidate_scores
    eval_result["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved evaluation report to: {report_path.resolve()}")

    error_path = Path(error_report_file)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    categorized_data = {
        "total_error_samples": eval_result["error_accounting"]["total_error_samples"],
        "under_tag_count": eval_result["error_accounting"]["under_tag_count"],
        "over_tag_count": eval_result["error_accounting"]["over_tag_count"],
        "displaced_count": eval_result["error_accounting"]["displaced_count"],
        "total_penalty": eval_result["error_accounting"]["total_penalty"],
        "error_samples": eval_result["error_samples"],
    }
    md_content = generate_error_markdown(
        categorized_data,
        total_fp_fn=eval_result["error_accounting"]["confusion_matrix_sum"],
    )
    with open(error_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved error attribution report to: {error_path.resolve()}")

    return eval_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 10 Classifier Evaluation & Error Attribution")
    parser.add_argument("--model-dir", type=str, default="data/ch10/best_model")
    parser.add_argument("--test-file", type=str, default="mewhelp-ch10-dataset/test.jsonl")
    parser.add_argument("--threshold-file", type=str, default="data/ch10/threshold.json")
    parser.add_argument("--report-file", type=str, default="reports/ch10_evaluation_report.json")
    parser.add_argument("--error-report-file", type=str, default="data/ch10/reports/error_samples.md")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)

    args = parser.parse_args()

    result = evaluate_classifier(
        model_dir=args.model_dir,
        test_file=args.test_file,
        threshold_file=args.threshold_file,
        report_file=args.report_file,
        error_report_file=args.error_report_file,
        device_name=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    summary = result["summary"]
    ea = result["error_accounting"]

    print("\n" + "=" * 60)
    print("           第十章 分类器分档红线评测总览")
    print("=" * 60)
    print(f"Micro-F1: {summary['micro_f1']:.4f} | Macro-F1: {summary['macro_f1']:.4f}")
    print(f"测试样本总数: {summary['total_samples']} | 判错样本数: {ea['total_error_samples']}")
    print(f"错因三向: 漏打 {ea['under_tag_count']} | 多打 {ea['over_tag_count']} | 错位 {ea['displaced_count']}")
    balance_str = "[PASS] 100% 平衡" if ea["balanced"] else "[FAIL] 失衡"
    print(f"数学闭环验证: {balance_str} (总惩罚 {ea['total_penalty']} == 混淆矩阵 FP+FN {ea['confusion_matrix_sum']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
