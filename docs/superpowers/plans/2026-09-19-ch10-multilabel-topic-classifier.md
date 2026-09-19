# 第十章：多标签主题分类器、ONNX独立推理服务与九项实证验收实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为客服系统微调哈工大讯飞 `chinese-roberta-wwm-ext` 多标签主题分类器，搭建独立轻量 ONNX 推理服务（:8110）与旁路批处理归类流水线，并通过飞轮后台主题分布看板及九项实证验收系统（四页独立路由 + 白名单作业运行器）实现端到端生产级交付。

**Architecture:** 数据端以 17 类权威术语表为基准，对齐近邻边界与多标签规则，实施三份考卷零泄漏硬闸；训练端采用 RoBERTa 全参微调 + BCEWithLogitsLoss + 早停机制，产出分档容错红线（严 0.9 / 中 0.8 / 宽 —）报告与错因三向记账（漏打/多打/错位）；部署端采用 `torch.onnx.export` 动态轴导出并通过 Logits 容差与标签一致性双硬闸，由 FastAPI + onnxruntime 提供独立轻量服务（:8110）；应用端通过 `classify_pool.py` 旁路按批（`--min-batch 10`）将未归类低置信度问题写表 `topic_classifications`；前端采用纯 HTML/CSS 打造五页后台看板（`/topic-distribution`, `/acceptance`, `/acceptance/eval`, `/acceptance/data`, `/acceptance/errors`）及中立安全作业调度器（`/api/jobs`）。

**Tech Stack:** Python 3.12, PyTorch, HuggingFace Transformers, ONNX, ONNX Runtime, Tokenizers, Scikit-learn, FastAPI, SQLAlchemy (Async), MySQL/aiomysql, SQLite/aiosqlite (测试双模), Makefile.

**Spec:** [`docs/superpowers/specs/2026-09-19-ch10-multilabel-topic-classifier-design.md`](file:///d:/PycharmProjects/AstroBot/docs/superpowers/specs/2026-09-19-ch10-multilabel-topic-classifier-design.md)

## Global Constraints

- 17 类权威类目：退换货、物流、尺码、发票、质量问题、运费、优惠活动、价保、支付、订单修改、库存补货、商品信息、保修维修、账号、会员积分、评价、其他。
- 近邻边界：修归保修维修、退归退换货；运费管钱、物流管货；价保是补差价、优惠活动是券和满减。
- 零泄漏硬闸：训练集、验证集、测试集文本交集必须严格为 0。
- 分档红线：严档（退换货、物流、尺码、发票、质量问题）F1 $\ge 0.90$；中档（运费、优惠活动、价保、支付、订单修改、库存补货、商品信息、保修维修）F1 $\ge 0.80$；宽档（账号、会员积分、评价、其他）不设线，画 `—`。
- 错因记账：漏打（放跑 +1）、多打（冤枉 +1）、错位（放跑 +1 & 冤枉 +1），总错例数与混淆矩阵笔数平衡。
- 推理服务（:8110）：独立进程，仅依赖 FastAPI + onnxruntime + tokenizers，严禁引入 torch / transformers 依赖。
- 旁路批处理：主链路实时对话完全不调用分类器；默认 `--min-batch 10`，不足不跑，`--force` 强制；基于 `question_id` 唯一键幂等。
- 作业运行器：命令白名单硬编码，前端无 shell 注入可能；命令一律走 make，日志落 `log/acceptance/<job>.log`；`start_new_session=True` 进程树收割。
- 产物缺失：API 统一返回 `present=false` + `make_target`，前端长出重跑按钮，严禁白屏。

---

### Task 1: 依赖安装、数据库 DDL 与 ORM 模型演进

**Files:**
- Modify: `requirements.txt`
- Create: `app/models/topic_classification.py`
- Modify: `app/models/low_confidence.py`
- Modify: `app/models/__init__.py`
- Create: `scripts/init_ch10_db.py`
- Modify: `run.py`
- Test: `tests/test_ch10_models.py`

**Interfaces:**
- Consumes: `sql/ch10-ddl.sql`, `app.db.session.Base`, `LowConfidenceQuestion`
- Produces: `TopicClassification` ORM model, `init_ch10_db()` idempotent migration

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ch10_models.py
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.db.session import Base
from app.models.low_confidence import LowConfidenceQuestion
from app.models.topic_classification import TopicClassification

@pytest.fixture
async def memory_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()

@pytest.mark.asyncio
async def test_topic_classification_model_lifecycle(memory_db: AsyncSession):
    # 1. 创建低置信度问题
    lcq = LowConfidenceQuestion(
        raw_question="买大了想退",
        source="retrieval_low_conf",
        reason="相关度不足",
    )
    memory_db.add(lcq)
    await memory_db.commit()
    await memory_db.refresh(lcq)

    # 2. 关联创建主题分类
    topic = TopicClassification(
        question_id=lcq.id,
        labels=["尺码", "退换货"],
    )
    memory_db.add(topic)
    await memory_db.commit()
    await memory_db.refresh(topic)

    assert topic.id is not None
    assert topic.question_id == lcq.id
    assert topic.labels == ["尺码", "退换货"]
    assert topic.classified_at is not None

    # 3. 关联关系验证
    stmt = select(LowConfidenceQuestion).where(LowConfidenceQuestion.id == lcq.id)
    res = await memory_db.execute(stmt)
    fetched_lcq = res.scalar_one()
    assert fetched_lcq.topic_classification is not None
    assert fetched_lcq.topic_classification.labels == ["尺码", "退换货"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ch10_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.topic_classification'`

- [ ] **Step 3: Implement dependencies, ORM model and DB migration**

1. 在 `requirements.txt` 中添加依赖：
```text
torch>=2.2.0
transformers>=4.40.0
onnx>=1.16.0
onnxruntime>=1.17.0
tokenizers>=0.19.0
scikit-learn>=1.4.0
```
并在当前环境通过 pip 安装所需库。

2. 编写 `app/models/topic_classification.py`：
```python
from datetime import datetime
from typing import Any, List, Optional, TYPE_CHECKING
from sqlalchemy import BigInteger, Integer, DateTime, ForeignKey, JSON, func
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base

if TYPE_CHECKING:
    from app.models.low_confidence import LowConfidenceQuestion

class TopicClassification(Base):
    __tablename__ = "topic_classifications"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        comment="主键",
    )
    question_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite"),
        ForeignKey("low_confidence_questions.id", name="fk_topic_question"),
        nullable=False,
        unique=True,
        index=True,
        comment="归类的问题,指向 low_confidence_questions.id",
    )
    labels: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        comment="多标签,17 类权威类目里命中的若干个",
    )
    classified_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        server_default=func.now(),
        nullable=False,
        comment="归类时间",
    )

    question: Mapped["LowConfidenceQuestion"] = relationship(
        "LowConfidenceQuestion",
        back_populates="topic_classification",
        lazy="select",
    )
```

3. 在 `app/models/low_confidence.py` 中增加反向关联：
```python
    topic_classification: Mapped[Optional["TopicClassification"]] = relationship(
        "TopicClassification",
        back_populates="question",
        uselist=False,
        lazy="select",
    )
```
并在 `app/models/__init__.py` 导出 `TopicClassification`。

4. 编写 `scripts/init_ch10_db.py`：支持 MySQL 与 SQLite 双模幂等 DDL 执行，并在 `run.py` 自愈检查中注册。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ch10_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt app/models/ scripts/init_ch10_db.py run.py tests/test_ch10_models.py
git commit -m "feat(db): implement TopicClassification model and ch10 db migration"
```

---

### Task 2: 17 类权威术语表规范与三份考卷零泄漏硬闸

**Files:**
- Create: `app/services/classifier/taxonomy.py`
- Create: `scripts/data_prep_ch10.py`
- Test: `tests/test_ch10_data_prep.py`

**Interfaces:**
- Consumes: `mewhelp-ch10-dataset/*.jsonl`
- Produces: `TAXONOMY_17`, `CATEGORY_BOUNDARIES`, `TIER_MAPPING`, `check_data_leakage(train, val, test) -> LeakageReport`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ch10_data_prep.py
import pytest
from app.services/classifier/taxonomy import (
    TAXONOMY_17,
    STRICT_TIER,
    MEDIUM_TIER,
    LOOSE_TIER,
    get_tier_for_category,
    check_dataset_leakage,
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

def test_leakage_check_passes_on_disjoint_sets():
    train = [{"text": "衣服偏小想换大一号", "labels": ["尺码", "退换货"]}]
    val = [{"text": "快递到哪了", "labels": ["物流"]}]
    test = [{"text": "怎么开发票", "labels": ["发票"]}]

    report = check_dataset_leakage(train, val, test)
    assert report["passed"] is True
    assert report["train_val_overlap"] == 0
    assert report["train_test_overlap"] == 0
    assert report["val_test_overlap"] == 0

def test_leakage_check_fails_on_overlap():
    train = [{"text": "衣服偏小想换大一号", "labels": ["尺码", "退换货"]}]
    val = [{"text": "衣服偏小想换大一号", "labels": ["尺码"]}]
    test = [{"text": "怎么开发票", "labels": ["发票"]}]

    report = check_dataset_leakage(train, val, test)
    assert report["passed"] is False
    assert report["train_val_overlap"] == 1
    assert "衣服偏小想换大一号" in report["overlap_samples"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ch10_data_prep.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement taxonomy and data preparation script**

1. 编写 `app/services/classifier/taxonomy.py`：
   - 定义 17 类常量数组 `TAXONOMY_17`；
   - 定义边界字典 `CATEGORY_BOUNDARIES`；
   - 定义三档分类 `STRICT_TIER` (0.9), `MEDIUM_TIER` (0.8), `LOOSE_TIER` (—)；
   - 实现 `check_dataset_leakage(train_data, val_data, test_data)`。
2. 编写 `scripts/data_prep_ch10.py`：
   - 读取 `mewhelp-ch10-dataset/` 验证零泄漏硬闸；
   - 提供从 `low_confidence_questions` 抽取、清洗、预标与数据增强的复用工具函数。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ch10_data_prep.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/classifier/taxonomy.py scripts/data_prep_ch10.py tests/test_ch10_data_prep.py
git commit -m "feat(classifier): implement 17-class taxonomy and data leakage hard gate"
```

---

### Task 3: RoBERTa 全参微调、早停与最优权重保存

**Files:**
- Create: `scripts/train_classifier.py`
- Test: `tests/test_ch10_training_pipeline.py`

**Interfaces:**
- Consumes: `mewhelp-ch10-dataset/{train,val}.jsonl`, `hfl/chinese-roberta-wwm-ext`
- Produces: `data/ch10/best_model/` (PyTorch weights, config), `data/ch10/threshold.json`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ch10_training_pipeline.py
import os
import pytest
from scripts.train_classifier import compute_multilabel_metrics, EarlyStopping

def test_multilabel_metrics_computation():
    import numpy as np
    # 模拟 3 个样本，4 个类别
    y_true = np.array([
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
    ])
    # 预测概率
    y_pred_probs = np.array([
        [0.8, 0.9, 0.1, 0.2],
        [0.2, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.9, 0.85],
    ])
    metrics = compute_multilabel_metrics(y_true, y_pred_probs, threshold=0.5)
    assert "macro_f1" in metrics
    assert "micro_f1" in metrics
    assert metrics["macro_f1"] > 0.8

def test_early_stopping_trigger():
    es = EarlyStopping(patience=3, mode="max")
    assert not es.step(0.70)
    assert not es.step(0.75) # 提升
    assert not es.step(0.74) # 没提升 1
    assert not es.step(0.73) # 没提升 2
    assert es.step(0.72)     # 没提升 3 -> 触发早停
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ch10_training_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement training script with EarlyStopping & BCEWithLogitsLoss**

编写 `scripts/train_classifier.py`：
- 支持参数 `--epochs 10 --lr 2e-5 --batch-size 16 --patience 3 --model-name hfl/chinese-roberta-wwm-ext`；
- 使用 `AutoModelForSequenceClassification.from_pretrained(..., num_labels=17, problem_type="multi_label_classification")`；
- `AdamW(weight_decay=0.01)` 正则化；
- 在验证集上寻找最佳 Micro-F1 对应阈值，保存到 `data/ch10/threshold.json`；
- 最优模型保存至 `data/ch10/best_model/`。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ch10_training_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/train_classifier.py tests/test_ch10_training_pipeline.py
git commit -m "feat(training): implement RoBERTa full fine-tuning with early stopping"
```

---

### Task 4: 分档红线评测、错因三向归因与判定阈值复算

**Files:**
- Create: `scripts/evaluate_classifier.py`
- Create: `scripts/scan_threshold_replay.py`
- Test: `tests/test_ch10_evaluation.py`

**Interfaces:**
- Consumes: `data/ch10/best_model/`, `mewhelp-ch10-dataset/test.jsonl`, `data/ch10/threshold.json`
- Produces: `reports/ch10_evaluation_report.json`, `data/ch10/reports/error_samples.md`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ch10_evaluation.py
import pytest
from scripts.evaluate_classifier import categorize_errors, verify_error_accounting_balance

def test_categorize_errors_accounting():
    samples = [
        {
            "text": "买大了想退",
            "true_labels": ["尺码", "退换货"],
            "pred_labels": ["退换货"], # 漏打尺码
        },
        {
            "text": "什么时候发货",
            "true_labels": ["物流"],
            "pred_labels": ["物流", "运费"], # 多打运费
        },
        {
            "text": "能不能修一下",
            "true_labels": ["保修维修"],
            "pred_labels": ["退换货"], # 错位：保修维修漏打，退换货多打
        },
    ]
    categorized = categorize_errors(samples)
    assert categorized["under_tag_count"] == 1  # 漏打
    assert categorized["over_tag_count"] == 1   # 多打
    assert categorized["displaced_count"] == 1  # 错位

    # 数学闭环验证
    # 混淆矩阵中惩罚总笔数 = 漏打(1) + 多打(1) + 错位(2) = 4
    total_penalty = verify_error_accounting_balance(categorized)
    assert total_penalty == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ch10_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement evaluation and threshold replay scripts**

1. 编写 `scripts/evaluate_classifier.py`：
   - 在 `test.jsonl` 上计算 17 类的 Precision / Recall / F1 / Support；
   - 套用三档红线（严 $\ge 0.9$，中 $\ge 0.8$，宽 `—`）；
   - 输出 17 类混淆矩阵；
   - 提取预测错误样本，进行漏打/多打/错位分类，输出 `data/ch10/reports/error_samples.md`；
   - 完整指标落盘 `reports/ch10_evaluation_report.json`。
2. 编写 `scripts/scan_threshold_replay.py`：
   - 载入验证集得分，在 $0.30 \sim 0.70$（步长 0.05）扫描 9 个候选线；
   - 输出扫描表，断言最优线与 `threshold.json` 一致。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ch10_evaluation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_classifier.py scripts/scan_threshold_replay.py tests/test_ch10_evaluation.py
git commit -m "feat(eval): implement tiered redlines, error accounting, and threshold replay"
```

---

### Task 5: ONNX 动态轴导出与双模型一致性硬闸

**Files:**
- Create: `scripts/export_onnx.py`
- Test: `tests/test_ch10_onnx_export.py`

**Interfaces:**
- Consumes: `data/ch10/best_model/`
- Produces: `data/ch10/onnx/model.onnx`, `data/ch10/onnx/tokenizer.json`, `data/ch10/onnx/vocab.txt`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ch10_onnx_export.py
import pytest
import numpy as np
from scripts.export_onnx import verify_onnx_torch_consistency

def test_verify_onnx_torch_consistency_logic():
    # 模拟一致
    torch_logits = np.array([[2.5, -1.2, 0.5]])
    onnx_logits = np.array([[2.50001, -1.20002, 0.49999]])
    thresholds = {"类目1": 0.5, "类目2": 0.5, "类目3": 0.5}
    categories = ["类目1", "类目2", "类目3"]

    passed, max_diff, label_match = verify_onnx_torch_consistency(
        torch_logits, onnx_logits, thresholds, categories, tol=1e-4
    )
    assert passed is True
    assert label_match is True
    assert max_diff < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ch10_onnx_export.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ONNX export and verification**

编写 `scripts/export_onnx.py`：
- 使用 `torch.onnx.export` 导出 `data/ch10/onnx/model.onnx`；
- 配置动态轴：`dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}, "attention_mask": {0: "batch_size", 1: "seq_len"}, "logits": {0: "batch_size"}}`；
- 导出 Tokenizer 词表与配置到 `data/ch10/onnx/`；
- 执行双模型一致性验证，不一致直接抛出异常退出。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ch10_onnx_export.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/export_onnx.py tests/test_ch10_onnx_export.py
git commit -m "feat(onnx): implement dynamic axes export and dual-model consistency hard gate"
```

---

### Task 6: 独立轻量推理服务 (:8110) 与 PID 治理

**Files:**
- Create: `app/services/classifier_service/__init__.py`
- Create: `app/services/classifier_service/server.py`
- Create: `scripts/classifier_service.py`
- Test: `tests/test_ch10_classifier_service.py`

**Interfaces:**
- Consumes: `data/ch10/onnx/model.onnx`, `data/ch10/threshold.json`
- Produces: HTTP `:8110` (`GET /healthz`, `POST /classify`), PID 文件 `data/ch10/classifier.pid`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ch10_classifier_service.py
import pytest
from httpx import AsyncClient, ASGITransport
from app.services/classifier_service/server import create_classifier_app

@pytest.mark.asyncio
async def test_classifier_service_endpoints():
    app = create_classifier_app(mock_mode=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. 探活
        res = await client.get("/healthz")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

        # 2. 分类
        res = await client.post("/classify", json={"texts": ["买大了想退"]})
        assert res.status_code == 200
        data = res.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert "labels" in data["results"][0]
        assert "scores" in data["results"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ch10_classifier_service.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement lightweight ONNX classifier service and CLI manager**

1. 编写 `app/services/classifier_service/server.py`：
   - 仅使用 `FastAPI`, `onnxruntime`, `tokenizers`，零 torch/transformers 依赖；
   - 载入 ONNX 模型与 `threshold.json`；
   - `GET /healthz` 与 `POST /classify`。
2. 编写 `scripts/classifier_service.py`：
   - 接收 `start`, `stop`, `status` 指令；
   - `start` 时写入 PID 到 `data/ch10/classifier.pid`；
   - `stop` 时安全终止进程并清理 PID。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ch10_classifier_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/classifier_service/ scripts/classifier_service.py tests/test_ch10_classifier_service.py
git commit -m "feat(service): implement lightweight ONNX inference service and PID manager"
```

---

### Task 7: 旁路批量归类流水线 (`scripts/classify_pool.py`)

**Files:**
- Create: `scripts/classify_pool.py`
- Test: `tests/test_ch10_classify_pool.py`

**Interfaces:**
- Consumes: `low_confidence_questions`, `http://127.0.0.1:8110/classify`
- Produces: 批量插入 `topic_classifications`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ch10_classify_pool.py
import pytest
from unittest.mock import AsyncMock, patch
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.db.session import Base
from app.models.low_confidence import LowConfidenceQuestion
from app.models.topic_classification import TopicClassification
from scripts.classify_pool import run_classify_pool

@pytest.fixture
async def memory_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()

@pytest.mark.asyncio
async def test_classify_pool_min_batch_and_force(memory_db: AsyncSession):
    # 插入 3 条低置信度问题
    for i in range(3):
        memory_db.add(LowConfidenceQuestion(raw_question=f"测试问题{i}", source="self_check"))
    await memory_db.commit()

    # 1. 默认 min_batch=10，不足时不处理
    result = await run_classify_pool(db=memory_db, min_batch=10, force=False)
    assert result["processed"] == 0
    assert result["skipped_due_to_min_batch"] is True

    # 2. force=True，执行处理
    mock_resp = [{"labels": ["商品信息"], "scores": {"商品信息": 0.95}}]
    with patch("scripts.classify_pool.call_classifier_service", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        result = await run_classify_pool(db=memory_db, min_batch=10, force=True)
        assert result["processed"] == 3

    # 验证落库
    res = await memory_db.execute(select(TopicClassification))
    items = res.scalars().all()
    assert len(items) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ch10_classify_pool.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement bypass batch classification pipeline**

编写 `scripts/classify_pool.py`：
- 查询 `low_confidence_questions` 中尚未归类的记录；
- 校验 `--min-batch` 约束（不足提示退出，除非 `--force`）；
- 调用 `:8110/classify`；
- 批量插入 `topic_classifications`；
- 打印归类统计摘要。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ch10_classify_pool.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/classify_pool.py tests/test_ch10_classify_pool.py
git commit -m "feat(flywheel): implement bypass batch topic classification pipeline"
```

---

### Task 8: 白名单作业运行器与中立 `/api/jobs` 接口

**Files:**
- Create: `app/core/jobs.py`
- Create: `app/api/jobs.py`
- Modify: `Makefile`
- Modify: `main.py`
- Test: `tests/test_ch10_jobs_runner.py`

**Interfaces:**
- Consumes: Makefile targets
- Produces: `JobRegistry`, `POST /api/jobs/start`, `GET /api/jobs/{job_id}`, `GET /api/jobs/{job_id}/logs`, `POST /api/jobs/{job_id}/stop`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ch10_jobs_runner.py
import pytest
from app.core.jobs import JobRegistry, start_job, get_job_status
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_job_runner_whitelist_and_reentrancy():
    # 1. 拒绝非白名单作业
    with pytest.raises(ValueError, match="not in whitelist"):
        await start_job("malicious_command")

    # 2. 正常注册作业能够查询
    assert "classify-pool" in JobRegistry.WHITELIST

@pytest.mark.asyncio
async def test_jobs_api_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 尝试启动一个不存在的作业 -> 400
        res = await client.post("/api/jobs/start", json={"job_name": "unknown"})
        assert res.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ch10_jobs_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement JobRegistry, runner, and neutral API**

1. 编写 `app/core/jobs.py`：
   - 注册白名单：`classifier-up`, `classifier-down`, `train-ch10`, `export-onnx`, `eval-ch10`, `replay-threshold`, `classify-pool`, `data-prep-ch10`；
   - 标记 heavy 作业；
   - 日志输出至 `log/acceptance/<job>.log`；
   - `start_new_session=True`，支持安全 killpg / 进程树收割。
2. 编写 `app/api/jobs.py`：
   - 挂载路由 `/api/jobs`；
   - 启动、状态轮询、日志读取、终止。
3. 更新 `Makefile` 补齐所有 target。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ch10_jobs_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/core/jobs.py app/api/jobs.py Makefile main.py tests/test_ch10_jobs_runner.py
git commit -m "feat(jobs): implement neutral whitelist job runner and jobs API"
```

---

### Task 9: 验收 API 体系与只读端点 (`app/api/acceptance.py`)

**Files:**
- Create: `app/api/acceptance.py`
- Modify: `main.py`
- Test: `tests/test_ch10_acceptance_api.py`

**Interfaces:**
- Consumes: `reports/ch10_evaluation_report.json`, `data/ch10/reports/error_samples.md`, `mewhelp-ch10-dataset/`
- Produces: `/api/acceptance/{overview,eval,data,errors,service,classify}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ch10_acceptance_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_acceptance_api_graceful_missing():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 当产物不存在时，不报 500，而是返回 present=false + make_target
        res = await client.get("/api/acceptance/overview")
        assert res.status_code == 200
        data = res.json()
        assert "gates" in data
        assert "passed_count" in data

        res_eval = await client.get("/api/acceptance/eval")
        assert res_eval.status_code == 200
        assert "present" in res_eval.json()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ch10_acceptance_api.py -v`
Expected: FAIL with `404 Not Found`

- [ ] **Step 3: Implement acceptance API endpoints**

编写 `app/api/acceptance.py`：
- `GET /api/acceptance/overview`：九项闸门三态判定（`pass`/`fail`/`missing`）；
- `GET /api/acceptance/eval`：读取评测报告，返回 micro vs macro、17 类红线、混淆矩阵；
- `GET /api/acceptance/data`：读取考卷盘点、零泄漏硬闸自检结果、产物状态；
- `GET /api/acceptance/errors`：读取错例分析与三向记账统计；
- `GET /api/acceptance/service`：探活 `:8110`；
- `POST /api/acceptance/classify`：代理 `:8110` 单句试分类。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ch10_acceptance_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/acceptance.py main.py tests/test_ch10_acceptance_api.py
git commit -m "feat(api): implement acceptance read-only API and proxy endpoints"
```

---

### Task 10: 前端五页看板与后台集成 (Vibe Coding)

**Files:**
- Create: `app/static/acceptance.css`
- Create: `app/static/acceptance.js`
- Create: `app/static/admin.js`
- Create: `app/static/topic_distribution.html`
- Create: `app/static/acceptance.html`
- Create: `app/static/acceptance_eval.html`
- Create: `app/static/acceptance_data.html`
- Create: `app/static/acceptance_errors.html`
- Modify: `app/static/index.html`
- Modify: `main.py`

**Interfaces:**
- Consumes: `/api/acceptance/*`, `/api/jobs/*`, `/api/topics/distribution`
- Produces: 5 张页面路由及后台导航

- [ ] **Step 1: Implement topic distribution API and page**
  - 在 `app/api/acceptance.py` 或 `app/api/topic_routes.py` 实现 `GET /api/topics/distribution`；
  - 编写 `app/static/topic_distribution.html`：纯 HTML/CSS 绘制 17 类目横向柱状图，Top 3 类目高亮标红，一键触发 `classify-pool` 作业并带日志弹窗与自动刷新。

- [ ] **Step 2: Implement shared CSS and JS**
  - `app/static/acceptance.css`：卡片网格、三态闸门徽标（绿 pass / 红 fail / 灰 missing）、纯 CSS 条形图与混淆矩阵热力色卡；
  - `app/static/acceptance.js`：作业触发、日志轮询窗口、单句试分类交互；
  - `app/static/admin.js`：后台统一主导航条（知识库、观测成本、飞轮待审、主题分布、实证验收）与子页 Tab 栏。

- [ ] **Step 3: Implement four acceptance pages**
  - `/acceptance` (`acceptance.html`)：九项实证闸门卡、总闸 N/9 统计、重跑按钮；
  - `/acceptance/eval` (`acceptance_eval.html`)：Micro vs Macro 对照表、17 类红线卡、9 候选线扫描明细、17 类混淆矩阵、单句试分类演示；
  - `/acceptance/data` (`acceptance_data.html`)：语料血缘图、零泄漏硬闸卡、考卷分布、产物状态；
  - `/acceptance/errors` (`acceptance_errors.html`)：错因三向记账统计卡、错例列表逐条对照。

- [ ] **Step 4: Mount routes and verify static pages in browser/HTTP client**
  - 在 `main.py` 中挂载对应 HTML 路由；
  - 验证页面均能 200 访问，无 JS 语法错误。

- [ ] **Step 5: Commit**

```bash
git add app/static/ main.py
git commit -m "feat(ui): implement topic distribution and four acceptance admin pages"
```

---

### Task 11: 端到端全链路验收测试与全量回归套件

**Files:**
- Create: `tests/test_ch10_acceptance.py`
- Modify: `dev-notes/ch10.md`

**Interfaces:**
- Consumes: All Task 1-10 deliverables
- Produces: Complete passing test suite, final verification evidence

- [ ] **Step 1: Write deep end-to-end acceptance tests**

编写 `tests/test_ch10_acceptance.py`，覆盖：
1. `test_gate_1_data_leakage_zero_overlap`（零泄漏硬闸）
2. `test_gate_2_tiered_f1_redlines`（17 类分档 F1 红线与宽档画 `—`）
3. `test_gate_3_error_accounting_balance`（漏打/多打/错位三向记账数学闭环）
4. `test_gate_4_threshold_replay_consistency`（9 候选线扫描复算一致性）
5. `test_gate_5_onnx_torch_logits_and_labels_match`（ONNX 容差与标签 100% 一致）
6. `test_gate_6_classifier_service_live_and_multi_intent`（:8110 探活与「买大了想退」命中尺码+退换货）
7. `test_gate_7_bypass_batch_classification_idempotency`（旁路批处理落库与幂等）
8. `test_gate_8_acceptance_api_tri_state_and_fallback`（API 闸门三态与优雅降级）
9. `test_gate_9_job_runner_security_and_process_cleanup`（白名单安全与防重入）

- [ ] **Step 2: Run all chapter 10 tests and full regression**

Run: `pytest tests/test_ch10_*.py -v`
Expected: All tests 100% PASS

- [ ] **Step 3: Update dev-notes/ch10.md with all milestones and verification evidence**

- [ ] **Step 4: Commit**

```bash
git add tests/test_ch10_acceptance.py dev-notes/ch10.md
git commit -m "test(acceptance): complete chapter 10 end-to-end acceptance test suite"
```
