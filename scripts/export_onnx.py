#!/usr/bin/env python3
"""ONNX Dynamic Axes Export and Dual-Model Consistency Hard Gate.

Chapter 10: Multi-label topic classification.

Features:
- Exports fine-tuned PyTorch RoBERTa model to ONNX with dynamic axes:
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size"}
    }
- Exports tokenizer artifacts (tokenizer.json, vocab.txt, config) and threshold.json
- Dual-model consistency hard gate:
    1. Logits absolute difference: |logits_torch - logits_onnx| < 1e-4
    2. Thresholded 17-class prediction labels: 100% exact match
    3. Fails immediately with exception if either gate is violated
"""

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.services.classifier.taxonomy import TAXONOMY_17

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("export_onnx")

DEFAULT_SAMPLE_QUERIES = [
    "买大了想退货退款",
    "快递怎么还没到？物流信息一直不更新",
    "请问身高175体重140穿什么尺码合适",
    "麻烦开一张增值税专用发票",
    "收到衣服有破洞，线头也很多，质量太差了",
    "退换货的运费谁承担？有没有运费险",
    "双十一有什么满减优惠活动吗",
    "昨天刚买今天就降价了，能不能申请价保退差价",
    "微信支付失败，一直提示网络异常",
    "收货地址填错了，麻烦帮我修改一下订单地址",
    "这款黑色L码什么时候补货上架",
    "请问这件衣服是什么面料材质？可以机洗吗",
    "商品坏了怎么申请售后保修维修",
    "账号绑定手机号怎么修改",
    "会员积分怎么兑换优惠券",
    "商品非常满意，给个好评",
    "人工客服电话是多少",
]


def load_threshold(threshold_input: Union[str, Path, dict, float, int]) -> Union[float, Dict[str, float]]:
    """Load threshold value or dictionary from a JSON file, dict, or scalar."""
    if isinstance(threshold_input, (int, float)):
        return float(threshold_input)

    if isinstance(threshold_input, dict):
        if "threshold" in threshold_input and isinstance(threshold_input["threshold"], (int, float)):
            return float(threshold_input["threshold"])
        return threshold_input

    path = Path(threshold_input)
    if not path.exists():
        logger.warning(f"Threshold file not found at {path}, using default 0.5")
        return 0.5

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "threshold" in data:
        return float(data["threshold"])
    return data


def verify_onnx_torch_consistency(
    torch_logits: Union[np.ndarray, torch.Tensor],
    onnx_logits: np.ndarray,
    thresholds: Union[float, dict, list, np.ndarray],
    categories: Optional[List[str]] = None,
    tol: float = 1e-4,
    raise_on_fail: bool = False,
) -> Tuple[bool, float, bool]:
    """Verify logits tolerance and label matching between PyTorch and ONNX Runtime.

    Hard gate criteria:
    1. Absolute diff: max(|logits_torch - logits_onnx|) < tol (default 1e-4)
    2. Label match: 100% exact equality after applying classification thresholds

    Returns:
        Tuple[bool, float, bool]: (passed, max_diff, label_match)
    """
    if categories is None:
        categories = TAXONOMY_17

    if isinstance(torch_logits, torch.Tensor):
        torch_logits_np = torch_logits.detach().cpu().numpy()
    else:
        torch_logits_np = np.asarray(torch_logits, dtype=np.float32)

    onnx_logits_np = np.asarray(onnx_logits, dtype=np.float32)

    if torch_logits_np.shape != onnx_logits_np.shape:
        raise ValueError(
            f"Shape mismatch: torch_logits {torch_logits_np.shape} vs onnx_logits {onnx_logits_np.shape}"
        )

    # Criterion 1: Logits tolerance
    abs_diff = np.abs(torch_logits_np - onnx_logits_np)
    max_diff = float(np.max(abs_diff))
    within_tol = bool(max_diff < tol)

    # Compute probabilities via sigmoid
    probs_torch = 1.0 / (1.0 + np.exp(-torch_logits_np))
    probs_onnx = 1.0 / (1.0 + np.exp(-onnx_logits_np))

    # Parse threshold
    if isinstance(thresholds, (int, float)):
        thresh_arr = float(thresholds)
    elif isinstance(thresholds, dict):
        if "threshold" in thresholds and isinstance(thresholds["threshold"], (int, float)):
            thresh_arr = float(thresholds["threshold"])
        else:
            thresh_arr = np.array([thresholds.get(cat, 0.5) for cat in categories], dtype=np.float32)
    elif isinstance(thresholds, (list, np.ndarray)):
        thresh_arr = np.asarray(thresholds, dtype=np.float32)
    else:
        thresh_arr = 0.5

    torch_preds = (probs_torch >= thresh_arr).astype(int)
    onnx_preds = (probs_onnx >= thresh_arr).astype(int)

    # Criterion 2: 100% exact match
    label_match = bool(np.array_equal(torch_preds, onnx_preds))
    passed = bool(within_tol and label_match)

    if raise_on_fail and not passed:
        msg = (
            f"Dual-model consistency check failed: "
            f"max_diff={max_diff:.6e} (tolerance={tol:.6e}, passed={within_tol}), "
            f"label_match={label_match} (100% required)"
        )
        logger.error(f"[HARD GATE VIOLATION] {msg}")
        raise RuntimeError(msg)

    return passed, max_diff, label_match


def export_onnx(
    model_dir: str = "data/ch10/best_model",
    output_dir: str = "data/ch10/onnx",
    threshold_file: str = "data/ch10/threshold.json",
    tolerance: float = 1e-4,
    opset_version: int = 17,
    sample_queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Export fine-tuned PyTorch model to ONNX with dynamic axes and verify consistency."""
    model_path = Path(model_dir)
    output_path = Path(output_dir)
    thresh_path = Path(threshold_file)

    output_path.mkdir(parents=True, exist_ok=True)
    onnx_model_file = output_path / "model.onnx"

    logger.info("=" * 60)
    logger.info("Starting ONNX Dynamic Axes Export & Dual-Model Consistency Verification")
    logger.info(f"Source Model Dir: {model_path.resolve()}")
    logger.info(f"Target Output Dir: {output_path.resolve()}")
    logger.info(f"Threshold File: {thresh_path.resolve()}")
    logger.info(f"Tolerance: {tolerance}")
    logger.info("=" * 60)

    # 1. Load PyTorch model and tokenizer
    logger.info("Loading PyTorch model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    # 2. Export to ONNX with dynamic axes
    logger.info("Exporting to ONNX with dynamic axes...")
    dummy_input_ids = torch.ones((1, 32), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 32), dtype=torch.long)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size"},
    }

    import inspect
    export_kwargs = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": dynamic_axes,
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_model_file),
        **export_kwargs,
    )
    logger.info(f"ONNX model written to: {onnx_model_file} (Size: {onnx_model_file.stat().st_size / (1024*1024):.2f} MB)")

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(onnx_model_file))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph validation passed (onnx.checker.check_model OK).")

    # 3. Export tokenizer artifacts and threshold.json
    logger.info("Saving tokenizer artifacts and threshold to target directory...")
    tokenizer.save_pretrained(str(output_path))
    try:
        tokenizer.save_vocabulary(str(output_path))
    except Exception as e:
        logger.debug(f"save_vocabulary note: {e}")

    # Copy or generate vocab.txt in target directory
    source_vocab = model_path / "vocab.txt"
    target_vocab = output_path / "vocab.txt"
    if source_vocab.exists() and not target_vocab.exists():
        shutil.copy2(source_vocab, target_vocab)
    elif not target_vocab.exists() and hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
        sorted_tokens = [tok for tok, _ in sorted(tokenizer.vocab.items(), key=lambda x: x[1])]
        with target_vocab.open("w", encoding="utf-8") as f:
            for token in sorted_tokens:
                f.write(token + "\n")
        logger.info(f"Generated vocab.txt ({len(sorted_tokens)} tokens) to: {target_vocab}")

    # Copy threshold.json
    target_thresh_file = output_path / "threshold.json"
    if thresh_path.exists():
        shutil.copy2(thresh_path, target_thresh_file)
        logger.info(f"Copied threshold.json to: {target_thresh_file}")
    else:
        logger.warning(f"Source threshold file {thresh_path} not found, creating default.")
        default_thresh_payload = {
            "threshold": 0.5,
            "metric": "default",
            "score": 0.0,
        }
        with target_thresh_file.open("w", encoding="utf-8") as f:
            json.dump(default_thresh_payload, f, ensure_ascii=False, indent=2)

    threshold_data = load_threshold(target_thresh_file)

    # 4. Dual-model consistency verification (Hard Gate)
    logger.info("Executing Dual-Model Consistency Hard Gate verification...")
    if sample_queries is None:
        sample_queries = DEFAULT_SAMPLE_QUERIES

    tokenized_samples = tokenizer(
        sample_queries,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        torch_out = model(
            input_ids=tokenized_samples["input_ids"],
            attention_mask=tokenized_samples["attention_mask"],
        )
        torch_logits = torch_out.logits.cpu().numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(str(onnx_model_file), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": tokenized_samples["input_ids"].numpy(),
        "attention_mask": tokenized_samples["attention_mask"].numpy(),
    }
    ort_outputs = session.run(["logits"], ort_inputs)
    onnx_logits = ort_outputs[0]

    # Verify hard gate
    passed, max_diff, label_match = verify_onnx_torch_consistency(
        torch_logits=torch_logits,
        onnx_logits=onnx_logits,
        thresholds=threshold_data,
        categories=TAXONOMY_17,
        tol=tolerance,
        raise_on_fail=True,
    )

    logger.info("=" * 60)
    logger.info("Dual-Model Consistency Hard Gate Results:")
    logger.info(f"- Tested Samples: {len(sample_queries)}")
    logger.info(f"- Max Logits Absolute Difference: {max_diff:.6e} (Threshold: < {tolerance:.6e}) -> PASS")
    logger.info(f"- Multi-label Predictions 100% Match: {label_match} -> PASS")
    logger.info(f"- Hard Gate Overall Status: {'PASSED' if passed else 'FAILED'}")
    logger.info("=" * 60)

    return {
        "exported": True,
        "model_path": str(onnx_model_file),
        "output_dir": str(output_path),
        "max_diff": max_diff,
        "tolerance": tolerance,
        "consistency_passed": passed,
        "label_match": label_match,
        "tested_samples": len(sample_queries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 10 ONNX Dynamic Axes Export & Hard Gate")
    parser.add_argument("--model-dir", type=str, default="data/ch10/best_model", help="Path to PyTorch model")
    parser.add_argument("--output-dir", type=str, default="data/ch10/onnx", help="Path to export ONNX artifacts")
    parser.add_argument("--threshold-file", type=str, default="data/ch10/threshold.json", help="Path to threshold.json")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Logits tolerance threshold")
    parser.add_argument("--opset-version", type=int, default=17, help="ONNX opset version")

    args = parser.parse_args()

    try:
        export_onnx(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            threshold_file=args.threshold_file,
            tolerance=args.tolerance,
            opset_version=args.opset_version,
        )
    except Exception as e:
        logger.error(f"ONNX export or verification failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
