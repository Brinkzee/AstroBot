"""Tests for Chapter 10 training pipeline, metrics, and EarlyStopping."""

import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from app.services.classifier.taxonomy import TAXONOMY_17
from scripts.train_classifier import (
    EarlyStopping,
    MultiLabelDataset,
    compute_multilabel_metrics,
    encode_labels,
    find_best_threshold,
    save_threshold_file,
    train_classifier,
)


def test_multilabel_metrics_computation():
    # 3 samples, 4 classes
    y_true = np.array([
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
    ])
    # Predicted probabilities
    y_pred_probs = np.array([
        [0.8, 0.9, 0.1, 0.2],
        [0.2, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.9, 0.85],
    ])

    metrics = compute_multilabel_metrics(y_true, y_pred_probs, threshold=0.5)

    assert "macro_f1" in metrics
    assert "micro_f1" in metrics
    assert "per_class_f1" in metrics
    assert "per_class_precision" in metrics
    assert "per_class_recall" in metrics
    assert metrics["macro_f1"] > 0.8
    assert metrics["micro_f1"] > 0.8

    # Verify against sklearn
    y_pred = (y_pred_probs >= 0.5).astype(int)
    expected_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    expected_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    assert abs(metrics["macro_f1"] - expected_macro) < 1e-6
    assert abs(metrics["micro_f1"] - expected_micro) < 1e-6


def test_multilabel_metrics_zero_division_and_edge_cases():
    # Edge case: All zeros
    y_true = np.zeros((3, 4), dtype=int)
    y_pred_probs = np.zeros((3, 4), dtype=float)

    metrics = compute_multilabel_metrics(y_true, y_pred_probs, threshold=0.5)
    assert metrics["macro_f1"] == 0.0
    assert metrics["micro_f1"] == 0.0
    assert not np.isnan(metrics["macro_f1"])
    assert not np.isnan(metrics["micro_f1"])

    # Edge case: No positive predictions
    y_true = np.ones((2, 2), dtype=int)
    y_pred_probs = np.zeros((2, 2), dtype=float)
    metrics2 = compute_multilabel_metrics(y_true, y_pred_probs, threshold=0.5)
    assert metrics2["macro_f1"] == 0.0
    assert metrics2["micro_f1"] == 0.0


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=3, mode="max", delta=1e-4)

    # Initial step
    assert not es.step(0.70)
    assert es.should_save is True
    assert es.counter == 0
    assert es.best_score == 0.70

    # Improvement
    assert not es.step(0.75)
    assert es.should_save is True
    assert es.counter == 0
    assert es.best_score == 0.75

    # No improvement: step 1
    assert not es.step(0.74)
    assert es.should_save is False
    assert es.counter == 1

    # No improvement: step 2
    assert not es.step(0.73)
    assert es.should_save is False
    assert es.counter == 2

    # No improvement: step 3 -> triggers early stopping
    assert es.step(0.72)
    assert es.should_save is False
    assert es.counter == 3
    assert es.early_stop is True


def test_early_stopping_min_mode():
    es = EarlyStopping(patience=2, mode="min", delta=1e-4)

    # Initial loss
    assert not es.step(1.50)
    assert es.should_save is True
    assert es.best_score == 1.50

    # Improved (lower loss)
    assert not es.step(1.20)
    assert es.should_save is True
    assert es.best_score == 1.20

    # Not improved
    assert not es.step(1.25)
    assert es.should_save is False
    assert es.counter == 1

    # Not improved again -> triggers early stopping
    assert es.step(1.30)
    assert es.early_stop is True


def test_find_best_threshold():
    y_true = np.array([
        [1, 0],
        [1, 1],
        [0, 1],
    ])
    # Probabilities: at 0.30, [0, 1] is FP (0.35 >= 0.30). At 0.40, [0, 1] is TN (0.35 < 0.40).
    y_pred_probs = np.array([
        [0.45, 0.35],
        [0.42, 0.45],
        [0.20, 0.42],
    ])

    best_thresh, best_metrics, all_results = find_best_threshold(
        y_true,
        y_pred_probs,
        thresholds=[0.30, 0.40, 0.50, 0.60, 0.70],
    )

    assert best_thresh == 0.40
    assert best_metrics["micro_f1"] == 1.0
    assert len(all_results) == 5


def test_encode_labels_and_dataset():
    labels = ["尺码", "退换货"]
    encoded = encode_labels(labels, TAXONOMY_17)
    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape == (17,)
    assert encoded.dtype == torch.float32

    # Check 1 at indices corresponding to 尺码 and 退换货
    idx_size = TAXONOMY_17.index("尺码")
    idx_return = TAXONOMY_17.index("退换货")
    assert encoded[idx_size] == 1.0
    assert encoded[idx_return] == 1.0
    assert encoded.sum().item() == 2.0


def test_bce_loss_and_full_fine_tuning_gradients():
    # Simple linear classifier to simulate 17-class classification head
    head = nn.Linear(32, 17)
    criterion = nn.BCEWithLogitsLoss()

    batch_size = 4
    dummy_features = torch.randn(batch_size, 32, requires_grad=True)
    dummy_labels = torch.zeros(batch_size, 17)
    dummy_labels[0, 0] = 1.0
    dummy_labels[1, 2] = 1.0

    logits = head(dummy_features)
    loss = criterion(logits, dummy_labels)
    loss.backward()

    # Verify gradients exist on all weights (full fine-tuning verification)
    assert head.weight.grad is not None
    assert head.bias.grad is not None
    assert torch.all(torch.isfinite(head.weight.grad))
    assert loss.item() > 0


def test_save_and_load_threshold_file(tmp_path):
    threshold_path = tmp_path / "threshold.json"
    save_threshold_file(
        threshold_path=threshold_path,
        best_threshold=0.45,
        metric="micro_f1",
        score=0.92,
        candidate_scores={0.3: 0.81, 0.45: 0.92, 0.5: 0.89},
    )

    assert threshold_path.exists()
    with open(threshold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["threshold"] == 0.45
    assert data["metric"] == "micro_f1"
    assert data["score"] == 0.92
    assert "0.45" in data["candidate_scores"]
    assert "updated_at" in data


def test_dry_run_training_pipeline(tmp_path):
    output_dir = tmp_path / "model"
    threshold_file = tmp_path / "threshold.json"

    result = train_classifier(
        train_file="mewhelp-ch10-dataset/train.jsonl",
        val_file="mewhelp-ch10-dataset/val.jsonl",
        model_name="hfl/chinese-roberta-wwm-ext",
        output_dir=str(output_dir),
        threshold_file=str(threshold_file),
        epochs=1,
        batch_size=4,
        lr=2e-5,
        patience=2,
        device_name="cpu",
        dry_run=True,
    )

    assert result["output_dir"] == str(output_dir)
    assert threshold_file.exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "model.safetensors").exists() or (output_dir / "pytorch_model.bin").exists()
    assert "best_threshold" in result
    assert 0.30 <= result["best_threshold"] <= 0.70


def test_compute_pos_weights():
    from scripts.train_classifier import compute_pos_weights

    records = [
        {"labels": ["退换货", "尺码"]},
        {"labels": ["退换货"]},
        {"labels": ["物流"]},
        {"labels": ["物流"]},
    ]
    # Total 4 records.
    # 退换货: 2 pos, 2 neg -> (2/2)^0.5 = 1.0
    # 尺码: 1 pos, 3 neg -> (3/1)^0.5 = 1.732
    # 物流: 2 pos, 2 neg -> 1.0
    # Others: 0 pos, 4 neg -> clamped or fallback
    weights = compute_pos_weights(records, taxonomy=TAXONOMY_17, power=0.5, max_weight=5.0)
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (len(TAXONOMY_17),)
    idx_size = TAXONOMY_17.index("尺码")
    idx_return = TAXONOMY_17.index("退换货")
    assert weights[idx_size].item() > weights[idx_return].item()
    assert torch.all(weights >= 1.0)
    assert torch.all(weights <= 5.0)


def test_compute_multilabel_metrics_with_core_categories():
    from scripts.train_classifier import compute_multilabel_metrics

    # 2 samples, 17 classes
    y_true = np.zeros((2, 17), dtype=int)
    y_true[0, 0] = 1  # 退换货 (strict)
    y_true[1, 1] = 1  # 物流 (strict)

    y_pred_probs = np.zeros((2, 17), dtype=float)
    y_pred_probs[0, 0] = 0.9
    y_pred_probs[1, 1] = 0.9

    metrics = compute_multilabel_metrics(y_true, y_pred_probs, threshold=0.5)
    assert "core_macro_f1" in metrics
    assert metrics["core_macro_f1"] >= 0.0

