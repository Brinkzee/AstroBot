#!/usr/bin/env python3
"""Validation threshold replay and grid scanning script.

Chapter 10: Multi-label topic classification.
Replays threshold scanning across 9 candidate thresholds (0.30 - 0.70, step 0.05)
and asserts that the optimal threshold strictly matches data/ch10/threshold.json.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.services.classifier.taxonomy import TAXONOMY_17

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scan_threshold_replay")

# Default 9 candidate thresholds: 0.30 to 0.70 with step 0.05
DEFAULT_CANDIDATES = [round(0.30 + i * 0.05, 2) for i in range(9)]


def compute_micro_macro_f1(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    """Compute Micro-F1 and Macro-F1 at a specific threshold.

    Args:
        y_true: Binary ground truth array of shape (N, num_classes).
        y_pred_probs: Predicted probability array of shape (N, num_classes).
        threshold: Decision threshold.

    Returns:
        Tuple of (micro_f1, macro_f1).
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    num_classes = y_true.shape[1]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_class_f1 = []

    for c in range(num_classes):
        yt = y_true[:, c]
        yp = y_pred[:, c]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        total_tp += tp
        total_fp += fp
        total_fn += fn

        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        per_class_f1.append(f1)

    micro_f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    return float(micro_f1), float(macro_f1)


def load_validation_scores(val_scores_file: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load validation ground truth and predicted probabilities.

    Args:
        val_scores_file: Path to validation scores JSON file.

    Returns:
        Tuple of (y_true, y_pred_probs, texts).
    """
    path = Path(val_scores_file)
    if not path.exists():
        raise FileNotFoundError(f"Validation scores file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if not samples:
        raise ValueError(f"No samples found in {path}")

    num_samples = len(samples)
    num_classes = len(TAXONOMY_17)
    y_true = np.zeros((num_samples, num_classes), dtype=int)
    y_probs = np.zeros((num_samples, num_classes), dtype=float)
    texts = []

    cat_to_idx = {cat: idx for idx, cat in enumerate(TAXONOMY_17)}

    for i, s in enumerate(samples):
        texts.append(s.get("text", ""))
        for lbl in s.get("true_labels", []):
            if lbl in cat_to_idx:
                y_true[i, cat_to_idx[lbl]] = 1

        scores_dict = s.get("scores", {})
        for cat, score in scores_dict.items():
            if cat in cat_to_idx:
                y_probs[i, cat_to_idx[cat]] = float(score)

    return y_true, y_probs, texts


def replay_threshold_scan(
    val_scores_file: Union[str, Path] = "data/ch10/val_scores.json",
    threshold_file: Union[str, Path] = "data/ch10/threshold.json",
    candidates: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Replay scanning 9 candidate thresholds and verify against threshold.json.

    Args:
        val_scores_file: Path to validation scores file.
        threshold_file: Path to target threshold.json.
        candidates: List of candidate thresholds (default 0.30 to 0.70 with step 0.05).

    Returns:
        Dict containing best_threshold, best_score, grid_table, matched, target_threshold.
    """
    if candidates is None:
        candidates = DEFAULT_CANDIDATES

    thresh_path = Path(threshold_file)
    if not thresh_path.exists():
        raise FileNotFoundError(f"Target threshold file not found: {thresh_path.resolve()}")

    with open(thresh_path, "r", encoding="utf-8") as f:
        target_data = json.load(f)

    target_threshold = float(target_data["threshold"])
    target_score = float(target_data.get("score", 0.0))

    y_true, y_probs, _ = load_validation_scores(val_scores_file)

    grid_table = []
    best_threshold = candidates[0]
    best_micro_f1 = -1.0
    best_macro_f1 = -1.0

    for t in candidates:
        micro_f1, macro_f1 = compute_micro_macro_f1(y_true, y_probs, t)
        grid_table.append({
            "threshold": round(t, 2),
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
        })

        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            best_macro_f1 = macro_f1
            best_threshold = t

    matched = abs(best_threshold - target_threshold) < 1e-4

    return {
        "best_threshold": round(best_threshold, 2),
        "best_score": round(best_micro_f1, 4),
        "best_macro_f1": round(best_macro_f1, 4),
        "target_threshold": round(target_threshold, 2),
        "target_score": round(target_score, 4),
        "matched": matched,
        "grid_table": grid_table,
    }


def print_grid_table(result: Dict[str, Any]) -> None:
    """Print formatted grid scanning table to stdout."""
    grid = result["grid_table"]
    best_t = result["best_threshold"]
    target_t = result["target_threshold"]
    matched = result["matched"]

    print("\n" + "=" * 50)
    print("      第十章 判定阈值候选线重演扫描表")
    print("=" * 50)
    print(f"| {'候选阈值':^10} | {'Micro-F1':^12} | {'Macro-F1':^12} | {'状态':^8} |")
    print("|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 10 + "|")

    for row in grid:
        t = row["threshold"]
        micro = row["micro_f1"]
        macro = row["macro_f1"]
        marker = " *最优*" if abs(t - best_t) < 1e-4 else ""
        print(f"| {t:^12.2f} | {micro:^14.4f} | {macro:^14.4f} | {marker:^10} |")

    print("=" * 50)
    print(f"重演最优阈值: {best_t:.2f} (Micro-F1: {result['best_score']:.4f})")
    print(f"标准目标阈值: {target_t:.2f} (data/ch10/threshold.json)")
    if matched:
        print("[PASS] 验证通过：重演选线与运行时阈值 100% 完全一致！")
    else:
        print("[FAIL] 验证失败：重演最优阈值与 threshold.json 不一致！")
    print("=" * 50 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 10 Threshold Replay & Grid Scanning")
    parser.add_argument(
        "--val-scores",
        type=str,
        default="data/ch10/val_scores.json",
        help="Path to validation scores JSON file",
    )
    parser.add_argument(
        "--threshold-file",
        type=str,
        default="data/ch10/threshold.json",
        help="Path to threshold.json",
    )

    args = parser.parse_args()

    result = replay_threshold_scan(
        val_scores_file=args.val_scores,
        threshold_file=args.threshold_file,
    )

    print_grid_table(result)

    assert result["matched"], (
        f"Threshold replay mismatch: scanned best {result['best_threshold']} != "
        f"target {result['target_threshold']} in {args.threshold_file}"
    )


if __name__ == "__main__":
    main()
