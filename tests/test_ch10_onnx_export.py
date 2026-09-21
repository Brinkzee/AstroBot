"""Tests for Chapter 10 ONNX dynamic axes export and dual-model consistency hard gate.

Tests:
1. verify_onnx_torch_consistency logic with passing case (tolerance and label match)
2. verify_onnx_torch_consistency tolerance failure case
3. verify_onnx_torch_consistency label mismatch failure case
4. verify_onnx_torch_consistency threshold input formats (float, dict, per-class)
5. verify_onnx_torch_consistency raise_on_fail behavior
6. ONNX export artifact generation (model.onnx, tokenizer files, threshold.json)
7. ONNX dynamic axes support (variable batch_size and variable seq_len)
8. End-to-end dual-model consistency check between PyTorch and ONNX Runtime
"""

import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.services.classifier.taxonomy import TAXONOMY_17
from scripts.export_onnx import (
    export_onnx,
    load_threshold,
    verify_onnx_torch_consistency,
)


def test_verify_onnx_torch_consistency_logic():
    """Positive test: within tolerance (<1e-4) and 100% label match."""
    torch_logits = np.array([
        [2.5, -1.2, 0.5, -3.0],
        [-0.5, 3.1, -2.0, 1.8],
    ])
    # Very small diff (< 1e-4)
    onnx_logits = np.array([
        [2.50001, -1.20002, 0.49999, -3.00001],
        [-0.50002, 3.10001, -1.99998, 1.80001],
    ])
    categories = ["退换货", "物流", "尺码", "发票"]
    thresholds = {"退换货": 0.5, "物流": 0.5, "尺码": 0.5, "发票": 0.5}

    passed, max_diff, label_match = verify_onnx_torch_consistency(
        torch_logits, onnx_logits, thresholds, categories, tol=1e-4
    )

    assert passed is True
    assert label_match is True
    assert max_diff < 1e-4


def test_verify_onnx_torch_consistency_tolerance_fail():
    """Negative test 1: max diff exceeds tolerance (>=1e-4) even if labels match."""
    torch_logits = np.array([
        [2.5, -1.2, 0.5, -3.0],
    ])
    # Diff is 2e-4 > 1e-4
    onnx_logits = np.array([
        [2.5002, -1.2000, 0.5000, -3.0000],
    ])
    categories = ["退换货", "物流", "尺码", "发票"]
    thresholds = 0.5

    passed, max_diff, label_match = verify_onnx_torch_consistency(
        torch_logits, onnx_logits, thresholds, categories, tol=1e-4
    )

    assert passed is False
    assert label_match is True
    assert max_diff >= 1e-4


def test_verify_onnx_torch_consistency_label_mismatch_fail():
    """Negative test 2: logits diff is small but straddles threshold, causing label mismatch."""
    # Sigmoid(0.0) = 0.5
    # Let threshold = 0.5
    # torch logit = 0.00002 -> prob > 0.5 -> label 1
    # onnx logit = -0.00002 -> prob < 0.5 -> label 0
    # diff = 4e-5 < 1e-4, but labels do NOT match!
    torch_logits = np.array([[0.00002, -2.0]])
    onnx_logits = np.array([[-0.00002, -2.0]])
    categories = ["退换货", "物流"]
    thresholds = 0.5

    passed, max_diff, label_match = verify_onnx_torch_consistency(
        torch_logits, onnx_logits, thresholds, categories, tol=1e-4
    )

    assert passed is False
    assert label_match is False
    assert max_diff < 1e-4


def test_verify_onnx_torch_consistency_threshold_formats():
    """Test support for multiple threshold formats: float, dict with 'threshold', dict per-class, array."""
    torch_logits = np.array([[1.0, -1.0]])
    onnx_logits = np.array([[1.00001, -0.99999]])
    categories = ["退换货", "物流"]

    # Format 1: float
    p1, diff1, m1 = verify_onnx_torch_consistency(torch_logits, onnx_logits, 0.45, categories)
    assert p1 and m1

    # Format 2: dict with "threshold"
    p2, diff2, m2 = verify_onnx_torch_consistency(
        torch_logits, onnx_logits, {"threshold": 0.45, "score": 0.8}, categories
    )
    assert p2 and m2

    # Format 3: dict per class
    p3, diff3, m3 = verify_onnx_torch_consistency(
        torch_logits, onnx_logits, {"退换货": 0.45, "物流": 0.5}, categories
    )
    assert p3 and m3

    # Format 4: list / numpy array
    p4, diff4, m4 = verify_onnx_torch_consistency(
        torch_logits, onnx_logits, [0.45, 0.5], categories
    )
    assert p4 and m4


def test_verify_onnx_torch_consistency_raises_when_requested():
    """Test that raise_on_fail=True raises RuntimeError when check fails."""
    torch_logits = np.array([[2.5, -1.2]])
    onnx_logits = np.array([[2.6, -1.2]])  # large diff
    categories = ["退换货", "物流"]

    with pytest.raises(RuntimeError, match="Dual-model consistency check failed"):
        verify_onnx_torch_consistency(
            torch_logits, onnx_logits, 0.5, categories, tol=1e-4, raise_on_fail=True
        )


def test_load_threshold(tmp_path):
    """Test loading threshold from file."""
    thresh_file = tmp_path / "threshold.json"
    thresh_file.write_text(json.dumps({"threshold": 0.45, "metric": "micro_f1"}), encoding="utf-8")

    val = load_threshold(thresh_file)
    assert val == 0.45


def test_onnx_export_pipeline_and_artifacts(tmp_path):
    """Test full ONNX export pipeline: artifacts generated and valid ONNX structure."""
    model_dir = Path("data/ch10/best_model")
    threshold_file = Path("data/ch10/threshold.json")
    output_dir = tmp_path / "onnx_export"

    if not (model_dir / "model.safetensors").exists():
        pytest.skip("data/ch10/best_model does not exist, skipping integration export test")

    result = export_onnx(
        model_dir=str(model_dir),
        output_dir=str(output_dir),
        threshold_file=str(threshold_file),
        tolerance=1e-4,
    )

    assert result["exported"] is True
    assert (output_dir / "model.onnx").exists()
    assert (output_dir / "model.onnx").stat().st_size > 10 * 1024 * 1024  # > 10MB
    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "tokenizer_config.json").exists()
    assert (output_dir / "vocab.txt").exists()
    assert (output_dir / "threshold.json").exists()

    # Verify ONNX model validity with onnx.checker
    onnx_model = onnx.load(str(output_dir / "model.onnx"))
    onnx.checker.check_model(onnx_model)


def test_onnx_dynamic_axes_variable_batch_and_seq(tmp_path):
    """Test exported ONNX model supports variable batch sizes and variable sequence lengths."""
    model_dir = Path("data/ch10/best_model")
    threshold_file = Path("data/ch10/threshold.json")
    output_dir = tmp_path / "onnx_dynamic"

    if not (model_dir / "model.safetensors").exists():
        pytest.skip("data/ch10/best_model does not exist, skipping dynamic axes test")

    export_onnx(
        model_dir=str(model_dir),
        output_dir=str(output_dir),
        threshold_file=str(threshold_file),
    )

    session = ort.InferenceSession(str(output_dir / "model.onnx"), providers=["CPUExecutionProvider"])

    # Test case 1: batch_size=1, seq_len=16
    input_ids_1 = np.ones((1, 16), dtype=np.int64)
    attn_mask_1 = np.ones((1, 16), dtype=np.int64)
    outputs_1 = session.run(["logits"], {"input_ids": input_ids_1, "attention_mask": attn_mask_1})
    assert outputs_1[0].shape == (1, 17)

    # Test case 2: batch_size=3, seq_len=32
    input_ids_2 = np.ones((3, 32), dtype=np.int64)
    attn_mask_2 = np.ones((3, 32), dtype=np.int64)
    outputs_2 = session.run(["logits"], {"input_ids": input_ids_2, "attention_mask": attn_mask_2})
    assert outputs_2[0].shape == (3, 17)

    # Test case 3: batch_size=2, seq_len=64
    input_ids_3 = np.ones((2, 64), dtype=np.int64)
    attn_mask_3 = np.ones((2, 64), dtype=np.int64)
    outputs_3 = session.run(["logits"], {"input_ids": input_ids_3, "attention_mask": attn_mask_3})
    assert outputs_3[0].shape == (2, 17)


def test_onnx_torch_real_model_consistency_gate(tmp_path):
    """Verify PyTorch and ONNX Runtime predictions match 100% and logits diff < 1e-4 on real texts."""
    model_dir = Path("data/ch10/best_model")
    threshold_file = Path("data/ch10/threshold.json")
    output_dir = tmp_path / "onnx_real_gate"

    if not (model_dir / "model.safetensors").exists():
        pytest.skip("data/ch10/best_model does not exist, skipping real consistency test")

    export_result = export_onnx(
        model_dir=str(model_dir),
        output_dir=str(output_dir),
        threshold_file=str(threshold_file),
    )
    assert export_result["consistency_passed"] is True
    assert export_result["max_diff"] < 1e-4
    assert export_result["label_match"] is True
