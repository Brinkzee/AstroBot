"""17-Class authoritative taxonomy, 3-tier redlines, and zero-leakage hard gate.

Chapter 10: Multi-label topic classification.
"""

from typing import Any, Dict, List, Set, Union


# 17 类权威类目常量定义：四大领头（退换货、物流、尺码、发票）及全部 17 类
TAXONOMY_17: List[str] = [
    "退换货",
    "物流",
    "尺码",
    "发票",
    "质量问题",
    "运费",
    "优惠活动",
    "价保",
    "支付",
    "订单修改",
    "库存补货",
    "商品信息",
    "保修维修",
    "账号",
    "会员积分",
    "评价",
    "其他",
]

# 三档容错红线类目
# 严档 (Strict, 5 类): F1 >= 0.90
STRICT_TIER: List[str] = [
    "退换货",
    "物流",
    "尺码",
    "发票",
    "质量问题",
]

# 中档 (Medium, 8 类): F1 >= 0.80
MEDIUM_TIER: List[str] = [
    "运费",
    "优惠活动",
    "价保",
    "支付",
    "订单修改",
    "库存补货",
    "商品信息",
    "保修维修",
]

# 宽档 (Loose, 4 类): 不设线，状态画 —
LOOSE_TIER: List[str] = [
    "账号",
    "会员积分",
    "评价",
    "其他",
]

# 档位 F1 目标阈值
TIER_THRESHOLDS: Dict[str, Union[float, None]] = {
    "strict": 0.90,
    "medium": 0.80,
    "loose": None,
}

# 类目到档位映射
TIER_MAPPING: Dict[str, str] = {
    **{cat: "strict" for cat in STRICT_TIER},
    **{cat: "medium" for cat in MEDIUM_TIER},
    **{cat: "loose" for cat in LOOSE_TIER},
}

# 近邻类目边界说明字典
CATEGORY_BOUNDARIES: Dict[Union[str, tuple], str] = {
    ("保修维修", "退换货"): "修归保修维修、退归退换货",
    ("退换货", "保修维修"): "修归保修维修、退归退换货",
    ("物流", "运费"): "物流管货、运费管钱",
    ("运费", "物流"): "运费管钱、物流管货",
    ("价保", "优惠活动"): "价保是补差价、优惠活动是券和满减",
    ("优惠活动", "价保"): "优惠活动是券和满减、价保是补差价",
    "保修维修": "修归保修维修、退归退换货",
    "退换货": "修归保修维修、退归退换货",
    "物流": "物流管货、运费管钱",
    "运费": "运费管钱、物流管货",
    "价保": "价保是补差价、优惠活动是券和满减",
    "优惠活动": "优惠活动是券和满减、价保是补差价",
}


def get_tier_for_category(category: str) -> str:
    """获取指定类目的红线档位 ('strict', 'medium', 'loose', 'unknown')."""
    return TIER_MAPPING.get(category, "unknown")


def _extract_texts(dataset: Any) -> List[str]:
    """从数据集列表提取文本列表，支持 dict 与 str 格式."""
    texts = []
    for item in dataset:
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = str(item)
        texts.append(text.strip())
    return texts


def check_dataset_leakage(
    train_data: Any,
    val_data: Any,
    test_data: Any,
) -> Dict[str, Any]:
    """三份考卷零泄漏硬闸自检.
    
    严禁任何样本文本交叉：
    - train ∩ val == 0
    - train ∩ test == 0
    - val ∩ test == 0
    
    Returns:
        包含 passed, train_val_overlap, train_test_overlap, val_test_overlap, overlap_samples 等字段的自检报告字典.
    """
    train_texts = _extract_texts(train_data)
    val_texts = _extract_texts(val_data)
    test_texts = _extract_texts(test_data)

    train_set: Set[str] = set(train_texts)
    val_set: Set[str] = set(val_texts)
    test_set: Set[str] = set(test_texts)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    all_overlaps = train_val_overlap | train_test_overlap | val_test_overlap
    passed = len(all_overlaps) == 0

    return {
        "passed": passed,
        "train_count": len(train_texts),
        "val_count": len(val_texts),
        "test_count": len(test_texts),
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "overlap_samples": sorted(list(all_overlaps)),
        "train_val_overlap_samples": sorted(list(train_val_overlap)),
        "train_test_overlap_samples": sorted(list(train_test_overlap)),
        "val_test_overlap_samples": sorted(list(val_test_overlap)),
    }


# 别名兼容
check_data_leakage = check_dataset_leakage
