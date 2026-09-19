#!/usr/bin/env python3
"""RoBERTa-wwm-ext multi-label classifier training pipeline.

Features:
- Base model: hfl/chinese-roberta-wwm-ext
- 17-class multi-label classification with BCEWithLogitsLoss
- Full fine-tuning (strictly NO LoRA / QLoRA)
- AdamW optimizer with weight decay L2 regularization
- EarlyStopping with patience=3 monitoring val_macro_f1
- Optimal threshold scanning on validation set
- Artifacts: data/ch10/best_model/ and data/ch10/threshold.json
- CLI support including --dry-run for fast smoke testing
"""

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from app.services.classifier.taxonomy import TAXONOMY_17

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_classifier")


class EarlyStopping:
    """Early stopping handler monitoring a metric to halt training when not improving."""

    def __init__(
        self,
        patience: int = 3,
        mode: str = "max",
        delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.mode = mode.lower()
        if self.mode not in ("max", "min"):
            raise ValueError(f"Unsupported mode: {mode}. Expected 'max' or 'min'.")
        self.delta = delta
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.should_save: bool = False

    def step(self, current_score: float) -> bool:
        """Update tracker with current metric score.

        Returns:
            bool: True if early stopping threshold has been reached.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.should_save = True
            self.counter = 0
            return False

        if self.mode == "max":
            improved = current_score > (self.best_score + self.delta)
        else:
            improved = current_score < (self.best_score - self.delta)

        if improved:
            self.best_score = current_score
            self.should_save = True
            self.counter = 0
        else:
            self.should_save = False
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def encode_labels(labels: List[str], taxonomy: List[str] = TAXONOMY_17) -> torch.Tensor:
    """Encode a list of label names into a multi-hot binary float tensor."""
    label_to_idx = {name: idx for idx, name in enumerate(taxonomy)}
    encoded = torch.zeros(len(taxonomy), dtype=torch.float32)
    for label in labels:
        if label in label_to_idx:
            encoded[label_to_idx[label]] = 1.0
    return encoded


class MultiLabelDataset(Dataset):
    """Dataset for Chinese multi-label classification."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 128,
        taxonomy: List[str] = TAXONOMY_17,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.taxonomy = taxonomy

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.records[idx]
        text = item.get("text", "")
        labels = item.get("labels", [])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_tensor = encode_labels(labels, self.taxonomy)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_tensor,
        }


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: Union[float, np.ndarray] = 0.5,
) -> Dict[str, Any]:
    """Compute Macro-F1, Micro-F1, Per-class F1, Precision, and Recall.

    Mathematical definition:
    - Precision_c = TP_c / (TP_c + FP_c)
    - Recall_c = TP_c / (TP_c + FN_c)
    - F1_c = 2 * TP_c / (2 * TP_c + FP_c + FN_c)
    - Macro-F1 = mean(F1_c)
    - Micro-F1 = 2 * sum(TP_c) / (2 * sum(TP_c) + sum(FP_c) + sum(FN_c))

    Returns:
        Dict with macro_f1, micro_f1, per_class_f1, per_class_precision,
        per_class_recall, per_class_support, etc.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_probs = np.asarray(y_pred_probs, dtype=float)

    if isinstance(threshold, (int, float)):
        y_pred = (y_pred_probs >= threshold).astype(int)
    else:
        threshold = np.asarray(threshold, dtype=float)
        y_pred = (y_pred_probs >= threshold).astype(int)

    num_classes = y_true.shape[1] if len(y_true.shape) > 1 else 1

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_support = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for c in range(num_classes):
        yt = y_true[:, c]
        yp = y_pred[:, c]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        support = int(np.sum(yt == 1))

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        per_class_precision.append(float(prec))
        per_class_recall.append(float(rec))
        per_class_f1.append(float(f1))
        per_class_support.append(support)

    macro_precision = float(np.mean(per_class_precision)) if per_class_precision else 0.0
    macro_recall = float(np.mean(per_class_recall)) if per_class_recall else 0.0
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0

    return {
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_support": per_class_support,
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, Dict[str, Any], Dict[float, Dict[str, Any]]]:
    """Scan candidate thresholds to select the optimal threshold based on Micro-F1."""
    if thresholds is None:
        thresholds = [round(0.30 + i * 0.05, 2) for i in range(9)]  # 0.30 to 0.70

    best_threshold = thresholds[0]
    best_micro_f1 = -1.0
    best_metrics: Dict[str, Any] = {}
    all_results: Dict[float, Dict[str, Any]] = {}

    for t in thresholds:
        metrics = compute_multilabel_metrics(y_true, y_pred_probs, threshold=t)
        all_results[t] = metrics
        if metrics["micro_f1"] > best_micro_f1:
            best_micro_f1 = metrics["micro_f1"]
            best_threshold = t
            best_metrics = metrics

    return best_threshold, best_metrics, all_results


def save_threshold_file(
    threshold_path: Union[str, Path],
    best_threshold: float,
    metric: str = "micro_f1",
    score: float = 0.0,
    candidate_scores: Optional[Dict[Any, Any]] = None,
) -> None:
    """Save selected optimal threshold and candidates to JSON file."""
    path = Path(threshold_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    formatted_candidates = {}
    if candidate_scores:
        for k, v in candidate_scores.items():
            key_str = f"{float(k):.2f}"
            val_score = v.get("micro_f1", v) if isinstance(v, dict) else v
            formatted_candidates[key_str] = round(float(val_score), 4)

    from datetime import timezone
    payload = {
        "threshold": round(float(best_threshold), 2),
        "metric": metric,
        "score": round(float(score), 4),
        "candidate_scores": formatted_candidates,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Threshold file saved to: {path}")


def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    path = Path(file_path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model on a dataloader with BCEWithLogitsLoss.

    Returns:
        avg_loss, all_true_labels, all_pred_probs
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            total_loss += loss.item() * len(input_ids)
            total_samples += len(input_ids)

            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    y_true = np.vstack(all_targets) if all_targets else np.empty((0, 17))
    y_pred_probs = np.vstack(all_probs) if all_probs else np.empty((0, 17))

    return avg_loss, y_true, y_pred_probs


def train_classifier(
    train_file: str = "mewhelp-ch10-dataset/train.jsonl",
    val_file: str = "mewhelp-ch10-dataset/val.jsonl",
    model_name: str = "hfl/chinese-roberta-wwm-ext",
    output_dir: str = "data/ch10/best_model",
    threshold_file: str = "data/ch10/threshold.json",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    patience: int = 3,
    device_name: str = "auto",
    dry_run: bool = False,
    max_length: int = 128,
) -> Dict[str, Any]:
    """Run full fine-tuning with early stopping and optimal threshold selection."""
    output_path = Path(output_dir)
    threshold_path = Path(threshold_file)
    output_path.mkdir(parents=True, exist_ok=True)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading training data from: {train_file}")
    train_records = load_jsonl(train_file)
    logger.info(f"Loading validation data from: {val_file}")
    val_records = load_jsonl(val_file)

    if dry_run:
        logger.info("[DRY-RUN] Smoke test mode active: truncating data to 8 samples.")
        train_records = train_records[:8]
        val_records = val_records[:8]
        epochs = 1

    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(TAXONOMY_17),
        problem_type="multi_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    # Ensure full fine-tuning (strictly NO LoRA / QLoRA)
    for param in model.parameters():
        param.requires_grad = True

    model.to(device)

    # Datasets and Loaders
    train_dataset = MultiLabelDataset(train_records, tokenizer, max_length=max_length)
    val_dataset = MultiLabelDataset(val_records, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer & Criterion
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    early_stopping = EarlyStopping(patience=patience, mode="max")

    best_val_macro_f1 = -1.0
    best_val_metrics: Dict[str, Any] = {}
    best_threshold = 0.5
    candidate_scores: Dict[float, Dict[str, Any]] = {}

    logger.info("=" * 60)
    logger.info(f"Start training: {epochs} epochs, full fine-tuning, {len(train_records)} train samples")
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(input_ids)
            train_samples += len(input_ids)

            if dry_run and step >= 2:
                break

        avg_train_loss = train_loss / max(train_samples, 1)

        # Validation evaluation
        val_loss, y_true_val, y_pred_probs_val = evaluate_model(model, val_loader, device)

        # Scan thresholds on validation set
        thresh, val_metrics, cand_results = find_best_threshold(y_true_val, y_pred_probs_val)
        val_macro_f1 = val_metrics["macro_f1"]
        val_micro_f1 = val_metrics["micro_f1"]

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Val Micro-F1: {val_micro_f1:.4f} (at thresh={thresh:.2f})"
        )

        should_stop = early_stopping.step(val_macro_f1)

        if early_stopping.should_save or dry_run:
            logger.info(f"[+] Optimal model update (Val Macro-F1: {val_macro_f1:.4f}). Saving to {output_path}...")
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            best_val_macro_f1 = val_macro_f1
            best_val_metrics = val_metrics
            best_threshold = thresh
            candidate_scores = cand_results

            save_threshold_file(
                threshold_path=threshold_path,
                best_threshold=best_threshold,
                metric="micro_f1",
                score=val_micro_f1,
                candidate_scores=candidate_scores,
            )

        if should_stop:
            logger.info(f"[!] Early stopping triggered at epoch {epoch}. Patience of {patience} exhausted.")
            break

    logger.info("=" * 60)
    logger.info(f"Training completed. Best Val Macro-F1: {best_val_macro_f1:.4f}, Best Threshold: {best_threshold:.2f}")
    logger.info(f"Model saved at: {output_path.resolve()}")
    logger.info(f"Threshold saved at: {threshold_path.resolve()}")
    logger.info("=" * 60)

    return {
        "best_macro_f1": best_val_macro_f1,
        "best_threshold": best_threshold,
        "output_dir": str(output_path),
        "threshold_file": str(threshold_path),
        "metrics": best_val_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 10 RoBERTa Fine-tuning")
    parser.add_argument("--train-file", type=str, default="mewhelp-ch10-dataset/train.jsonl")
    parser.add_argument("--val-file", type=str, default="mewhelp-ch10-dataset/val.jsonl")
    parser.add_argument("--model-name", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--output-dir", type=str, default="data/ch10/best_model")
    parser.add_argument("--threshold-file", type=str, default="data/ch10/threshold.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Quick smoke test 1 step")
    parser.add_argument("--max-length", type=int, default=128)

    args = parser.parse_args()

    train_classifier(
        train_file=args.train_file,
        val_file=args.val_file,
        model_name=args.model_name,
        output_dir=args.output_dir,
        threshold_file=args.threshold_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device_name=args.device,
        dry_run=args.dry_run,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
