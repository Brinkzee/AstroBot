import json
import os
from pathlib import Path
import pytest

from app.services.classifier.taxonomy import (
    TAXONOMY_17,
    STRICT_TIER,
    MEDIUM_TIER,
    LOOSE_TIER,
    CATEGORY_BOUNDARIES,
    TIER_MAPPING,
    get_tier_for_category,
    check_dataset_leakage,
)
from scripts.data_prep_ch10 import (
    clean_and_mask_text,
    augment_train_data,
    load_dataset,
)


def test_taxonomy_17_categories_and_tiers():
    assert len(TAXONOMY_17) == 17
    assert TAXONOMY_17[0] == "退换货"
    assert TAXONOMY_17[1] == "物流"
    assert TAXONOMY_17[2] == "尺码"
    assert TAXONOMY_17[3] == "发票"

    # 验证三档划分
    assert len(STRICT_TIER) == 5
    assert len(MEDIUM_TIER) == 8
    assert len(LOOSE_TIER) == 4
    assert len(STRICT_TIER) + len(MEDIUM_TIER) + len(LOOSE_TIER) == 17

    assert get_tier_for_category("退换货") == "strict"
    assert get_tier_for_category("价保") == "medium"
    assert get_tier_for_category("其他") == "loose"
    assert get_tier_for_category("不存在的类目") == "unknown"

    # 验证近邻边界字典
    assert "修归保修维修、退归退换货" in str(CATEGORY_BOUNDARIES)
    assert "物流管货" in str(CATEGORY_BOUNDARIES) or "运费管钱" in str(CATEGORY_BOUNDARIES)
    assert "价保是补差价" in str(CATEGORY_BOUNDARIES) or "优惠活动是券和满减" in str(CATEGORY_BOUNDARIES)


def test_leakage_check_passes_on_disjoint_sets():
    train = [{"text": "衣服偏小想换大一号", "labels": ["尺码", "退换货"]}]
    val = [{"text": "快递到哪了", "labels": ["物流"]}]
    test = [{"text": "怎么开发票", "labels": ["发票"]}]

    report = check_dataset_leakage(train, val, test)
    assert report["passed"] is True
    assert report["train_val_overlap"] == 0
    assert report["train_test_overlap"] == 0
    assert report["val_test_overlap"] == 0
    assert len(report["overlap_samples"]) == 0


def test_leakage_check_fails_on_overlap():
    train = [{"text": "衣服偏小想换大一号", "labels": ["尺码", "退换货"]}]
    val = [{"text": "衣服偏小想换大一号", "labels": ["尺码"]}]
    test = [{"text": "怎么开发票", "labels": ["发票"]}]

    report = check_dataset_leakage(train, val, test)
    assert report["passed"] is False
    assert report["train_val_overlap"] == 1
    assert "衣服偏小想换大一号" in report["overlap_samples"]


def test_leakage_check_on_real_mewhelp_dataset():
    dataset_dir = Path("mewhelp-ch10-dataset")
    assert dataset_dir.exists(), f"Dataset directory {dataset_dir} does not exist"

    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "val.jsonl"
    test_file = dataset_dir / "test.jsonl"

    assert train_file.exists()
    assert val_file.exists()
    assert test_file.exists()

    train_data = load_dataset(train_file)
    val_data = load_dataset(val_file)
    test_data = load_dataset(test_file)

    assert len(train_data) == 2594
    assert len(val_data) == 161
    assert len(test_data) == 161

    # 零泄漏硬闸自检：重叠必须严格为 0
    report = check_dataset_leakage(train_data, val_data, test_data)
    assert report["passed"] is True
    assert report["train_val_overlap"] == 0
    assert report["train_test_overlap"] == 0
    assert report["val_test_overlap"] == 0
    assert len(report["overlap_samples"]) == 0


def test_clean_and_mask_text():
    raw_text = "我叫张三，手机号13812345678，订单号1234567890123456，之前买的降介了能退款吗？"
    cleaned = clean_and_mask_text(raw_text)
    assert "13812345678" not in cleaned
    assert "[手机号]" in cleaned
    assert "1234567890123456" not in cleaned
    assert "[订单号]" in cleaned
    assert "降介" not in cleaned
    assert "降价" in cleaned


def test_augment_train_data():
    train_sample = [
        {"text": "购买后发现有差价怎么申请？", "labels": ["价保"]}
    ]
    augmented = augment_train_data(train_sample)
    # 数据增强后样本数量增加
    assert len(augmented) > len(train_sample)
    # 所有样本标签保持一致
    for item in augmented:
        assert item["labels"] == ["价保"]
