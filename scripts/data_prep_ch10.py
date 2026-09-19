#!/usr/bin/env python3
"""Chapter 10 Data preparation, cleaning, augmentation and zero-leakage check."""

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.classifier.taxonomy import (
    TAXONOMY_17,
    check_dataset_leakage,
)

# 常见错别字修正字典
TYPO_CORRECTIONS: Dict[str, str] = {
    "降介": "降价",
    "退款到长": "退款到账",
    "退款到长": "退款到账",
    "发标": "发票",
    "快弟": "快递",
    "尺玛": "尺码",
    "换货留程": "换货流程",
    "运费险低扣": "运费险抵扣",
}

# 同义词替换表（用于训练集轻量数据增强）
SYNONYMS: List[Tuple[str, str]] = [
    ("差价", "差额"),
    ("便宜", "优惠"),
    ("退款", "退钱"),
    ("发货", "寄出"),
    ("快递", "包裹"),
    ("质量", "品质"),
    ("怎么", "如何"),
]

# 句末语气词微调
PARTICLE_REPLACEMENTS: List[Tuple[str, str]] = [
    ("吗？", "呀？"),
    ("呢？", "吗？"),
    ("呀？", "呢？"),
    ("请问", "麻烦问下"),
    ("麻烦问下", "请问"),
]


def load_dataset(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """加载 JSONL 数据集."""
    records = []
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_dataset(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """将数据保存为 JSONL 文件."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def clean_and_mask_text(text: str) -> str:
    """文本清洗与脱敏.
    
    1. 手机号脱敏为 [手机号]
    2. 订单号脱敏为 [订单号]
    3. 用户姓名脱敏为 [姓名]
    4. 错别字修复
    5. 空白规范化
    """
    if not text:
        return ""

    # 1. 手机号脱敏 (匹配 11 位手机号，可选带 +86/0086)
    text = re.sub(r"(?:(?:\+|00)86)?(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号]", text)

    # 2. 订单号脱敏 (连续 14~24 位数字，不依赖英文单词边界 \b)
    text = re.sub(r"(?<!\d)\d{14,24}(?!\d)", "[订单号]", text)

    # 3. 姓名脱敏 (匹配 "我叫XXX"、"姓名：XXX" 等模式)
    text = re.sub(
        r"(?:我叫|我是|姓名[：:\s]*|用户[：:\s]*)([\u4e00-\u9fa5]{2,4})",
        lambda m: m.group(0).replace(m.group(1), "[姓名]"),
        text,
    )

    # 4. 修复错别字
    for typo, correction in TYPO_CORRECTIONS.items():
        if typo in text:
            text = text.replace(typo, correction)

    # 5. 空白字符整理
    text = re.sub(r"\s+", " ", text).strip()

    return text


def augment_train_data(
    train_data: List[Dict[str, Any]],
    max_aug_per_sample: int = 1,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """训练集数据增强（严格禁止扩充验证集或测试集）.
    
    采用同义词替换与句式/语气助词微调，保持多标签标注一致。
    """
    rng = random.Random(seed)
    augmented_dataset: List[Dict[str, Any]] = list(train_data)
    seen_texts = {item.get("text", "") for item in train_data}

    for item in train_data:
        original_text = item.get("text", "")
        labels = item.get("labels", [])
        if not original_text:
            continue

        candidates = []

        # 尝试同义词替换
        for word, syn in SYNONYMS:
            if word in original_text:
                new_text = original_text.replace(word, syn, 1)
                if new_text != original_text and new_text not in seen_texts:
                    candidates.append(new_text)
            elif syn in original_text:
                new_text = original_text.replace(syn, word, 1)
                if new_text != original_text and new_text not in seen_texts:
                    candidates.append(new_text)

        # 尝试语气助词替换
        for p_from, p_to in PARTICLE_REPLACEMENTS:
            if p_from in original_text:
                new_text = original_text.replace(p_from, p_to, 1)
                if new_text != original_text and new_text not in seen_texts:
                    candidates.append(new_text)

        if candidates:
            rng.shuffle(candidates)
            for new_text in candidates[:max_aug_per_sample]:
                seen_texts.add(new_text)
                augmented_dataset.append({
                    "text": new_text,
                    "labels": list(labels),
                })

    return augmented_dataset


def stratified_split(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """分层抽样划分数据集为 train, val, test."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    rng = random.Random(seed)

    # 按照 primary label (即第一个标签) 分组
    label_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        labels = item.get("labels", [])
        primary_label = labels[0] if labels else "其他"
        label_groups.setdefault(primary_label, []).append(item)

    train_data: List[Dict[str, Any]] = []
    val_data: List[Dict[str, Any]] = []
    test_data: List[Dict[str, Any]] = []

    for label, group in label_groups.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_data.extend(group[:n_train])
        val_data.extend(group[n_train:n_train + n_val])
        test_data.extend(group[n_train + n_val:])

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)

    return train_data, val_data, test_data


def run_leakage_check(dataset_dir: Union[str, Path]) -> int:
    """运行三份考卷零泄漏自检并输出详细报告."""
    dir_path = Path(dataset_dir)
    train_path = dir_path / "train.jsonl"
    val_path = dir_path / "val.jsonl"
    test_path = dir_path / "test.jsonl"

    print("=" * 60)
    print(f"[*] 执行三份考卷零泄漏自检: {dir_path.resolve()}")
    print("=" * 60)

    try:
        train_data = load_dataset(train_path)
        val_data = load_dataset(val_path)
        test_data = load_dataset(test_path)
    except FileNotFoundError as e:
        print(f"[!] 错误: 数据集文件缺失 - {e}")
        return 1

    report = check_dataset_leakage(train_data, val_data, test_data)

    print(f"[-] 训练集样本数: {report['train_count']}")
    print(f"[-] 验证集样本数: {report['val_count']}")
    print(f"[-] 测试集样本数: {report['test_count']}")
    print("-" * 60)
    print(f"[-] 训练集 ∩ 验证集 重叠数: {report['train_val_overlap']}")
    print(f"[-] 训练集 ∩ 测试集 重叠数: {report['train_test_overlap']}")
    print(f"[-] 验证集 ∩ 测试集 重叠数: {report['val_test_overlap']}")
    print("-" * 60)

    if report["passed"]:
        print("[+] 零泄漏硬闸自检结果: PASSED (重叠数严格为 0)")
        print("=" * 60)
        return 0
    else:
        print(f"[!] 零泄漏硬闸自检结果: FAILED (发现 {len(report['overlap_samples'])} 条重叠样本)")
        for idx, sample in enumerate(report["overlap_samples"][:10], 1):
            print(f"    {idx}. {sample}")
        if len(report["overlap_samples"]) > 10:
            print(f"    ... 还有 {len(report['overlap_samples']) - 10} 条")
        print("=" * 60)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 10 Data Prep & Leakage Check")
    parser.add_argument(
        "--check-leakage",
        action="store_true",
        default=True,
        help="执行零泄漏自检 (默认启用)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="mewhelp-ch10-dataset",
        help="数据集所在目录 (默认: mewhelp-ch10-dataset)",
    )

    args = parser.parse_args()

    if args.check_leakage:
        exit_code = run_leakage_check(args.dataset_dir)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
