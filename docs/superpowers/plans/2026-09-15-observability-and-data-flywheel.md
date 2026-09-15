# Chapter 09: Observability, Cost Control, and Data Flywheel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为客服系统接入 Langfuse 全链路 Trace 观测与意图维度 Token 成本统计，升级基于 Ch04 评估集校准的多信号置信度闸门并实现三路落池带召回片段快照，构建 FAQ 标准化与查重去重飞轮流水线并闭环双写回流知识库，建立多指标自动化评估时序趋势，打造后台「观测与成本」(/observability) 和「待审队列」(/review-queue) 管理面板。

**Architecture:** 采用模块解耦分层架构。LangGraph 编译期单点注入 Langfuse 回调并支持宽容降级；意图节点运行时动态向 Trace 根绑定分类元数据；置信度闸门结合精排分、有效证据数、分差梯度综合判定；飞轮批处理通过 FAQ 标准化提示词与两阶段查重归并至 `review_queue`，审核通过走 Ch03 `KnowledgeDualWriter` 写入 MySQL 和 Milvus；评估流水线复用 Ch04 评估集持久化至 `eval_runs` 表；后台通过只读 API `GET /api/observability/overview` 呈现三块权威报表，重跑通过白名单作业调度器 `JobRunner` 安全调度。

**Tech Stack:** Python 3.12, FastAPI, LangGraph, Langfuse Python SDK (langchain callback), SQLAlchemy (async), MySQL / SQLite, Milvus-Lite, BGE-M3 Embedding, BGE-Reranker-v2-m3, Pytest.

**Spec:** [docs/superpowers/specs/2026-09-15-observability-and-data-flywheel-design.md](file:///d:/PycharmProjects/AstroBot/docs/superpowers/specs/2026-09-15-observability-and-data-flywheel-design.md)

## Global Constraints
- 全程遵循 Superpowers 规范，优先 TDD，纯 Prompt/数据类任务拿评估集或标注样例跑验证，前端部分走 Vibe Coding；
- 过程留痕：在 `dev-notes/ch09.md` 中按阶段追记（关键原话、产出物路径、拒绝纠偏、翻车返工）；
- 涉及第三方库一律以最新接口为准，API 版本对齐；
- 杜绝“第二个真相”，指标由脚本落入 JSON 权威产物，API 仅读取不重算；三块报表互不连坐、宽容降级；
- 作业重跑一律走 `JobRunner` 白名单作业名，前端绝不传递 shell 命令片段。

---

### Task 1: 数据库演进与 DDL 迁移脚本

**Files:**
- Create: `app/models/review_queue.py`
- Create: `app/models/eval_run.py`
- Modify: `app/models/low_confidence.py`
- Modify: `app/models/__init__.py`
- Create: `scripts/init_ch09_db.py`
- Test: `tests/test_ch09_models.py`

**Interfaces:**
- Consumes: `app/db/session.py:Base`, `sql/ch09-ddl.sql`
- Produces: `ReviewQueue`, `ReviewStatus`, `EvalRun`, `TriggeredBy`, `LowConfidenceQuestion.retrieved_chunks`, `LowConfidenceQuestion.matched_review_id`, `init_ch09_db()`

- [ ] **Step 1: Write failing tests for Ch09 models and migrations**
```python
# tests/test_ch09_models.py
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.review_queue import ReviewQueue, ReviewStatus
from app.models.eval_run import EvalRun, TriggeredBy
from app.models.low_confidence import LowConfidenceQuestion, record_low_confidence

@pytest.mark.asyncio
async def test_create_review_queue_item(db_session: AsyncSession):
    item = ReviewQueue(
        normalized_question="商品退换货规则是什么？",
        ai_suggested_answer="支持7天无理由退货。",
        occurrence_count=1,
        review_status=ReviewStatus.PENDING,
    )
    db_session.add(item)
    await db_session.commit()
    await db_session.refresh(item)
    assert item.id is not None
    assert item.review_status == ReviewStatus.PENDING
    assert item.occurrence_count == 1

@pytest.mark.asyncio
async def test_create_eval_run_item(db_session: AsyncSession):
    run = EvalRun(
        triggered_by=TriggeredBy.MANUAL,
        dataset_size=50,
        metrics={"recall_at_3": 0.85, "mrr": 0.75, "faithfulness": 0.90},
    )
    db_session.add(run)
    await db_session.commit()
    await db_session.refresh(run)
    assert run.id is not None
    assert run.metrics["recall_at_3"] == 0.85

@pytest.mark.asyncio
async def test_low_confidence_question_with_chunks_and_review_id(db_session: AsyncSession):
    queue_item = ReviewQueue(
        normalized_question="运费险如何理赔？",
        ai_suggested_answer="系统自动审核后赔付。",
        review_status=ReviewStatus.PENDING,
    )
    db_session.add(queue_item)
    await db_session.commit()
    await db_session.refresh(queue_item)

    chunks = [{"text": "运费险条款...", "score": 0.32, "section": "售后"}]
    lcq = await record_low_confidence(
        db=db_session,
        raw_question="运费险怎么赔啊？",
        source="retrieval_low_conf",
        reason="evidence_confidence_below_threshold",
        retrieved_chunks=chunks,
        matched_review_id=queue_item.id,
    )
    assert lcq.id is not None
    assert lcq.retrieved_chunks == chunks
    assert lcq.matched_review_id == queue_item.id
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ch09_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.review_queue'`

- [ ] **Step 3: Implement models and migration script**
1. Implement `app/models/review_queue.py` defining `ReviewStatus` enum and `ReviewQueue` model.
2. Implement `app/models/eval_run.py` defining `TriggeredBy` enum and `EvalRun` model.
3. Update `app/models/low_confidence.py` adding `retrieved_chunks` (JSON nullable) and `matched_review_id` (ForeignKey to review_queue.id, nullable) fields, update `record_low_confidence`.
4. Export in `app/models/__init__.py`.
5. Implement `scripts/init_ch09_db.py` to inspect and execute SQLite/MySQL DDL idempotently.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ch09_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/models/review_queue.py app/models/eval_run.py app/models/low_confidence.py app/models/__init__.py scripts/init_ch09_db.py tests/test_ch09_models.py
git commit -m "feat(db): implement review_queue and eval_runs models with ch09 migrations"
```

---

### Task 2: Langfuse 观测管理与编译期回调注入

**Files:**
- Create: `app/services/observability/__init__.py`
- Create: `app/services/observability/langfuse_service.py`
- Modify: `app/config.py`
- Modify: `app/services/workflow/engine.py`
- Test: `tests/test_ch09_langfuse.py`

**Interfaces:**
- Consumes: `app/config.py:settings`
- Produces: `LangfuseManager.get_callback_handler()`, `LangfuseManager.update_current_trace()`, compiled graph with callback

- [ ] **Step 1: Write failing tests for Langfuse callback integration and fallback**
```python
# tests/test_ch09_langfuse.py
import pytest
from app.services.observability.langfuse_service import LangfuseManager
from app.services.workflow.engine import build_workflow_graph

def test_langfuse_disabled_graceful_fallback(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", None)
    monkeypatch.setattr("app.config.settings.langfuse_secret_key", None)
    handler = LangfuseManager.get_callback_handler()
    assert handler is None

def test_workflow_compilation_with_langfuse_hook():
    graph = build_workflow_graph()
    assert graph is not None
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ch09_langfuse.py -v`
Expected: FAIL with missing module/method.

- [ ] **Step 3: Implement LangfuseManager and compile hook**
1. Add `langfuse_public_key`, `langfuse_secret_key`, `langfuse_host`, `langfuse_enabled` to `app/config.py`.
2. Implement `app/services/observability/langfuse_service.py` with `get_callback_handler()`, `update_current_trace()`, and connection health check.
3. In `build_workflow_graph` of `app/services/workflow/engine.py`, attach `.with_config({"callbacks": [handler]})` when handler is available.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ch09_langfuse.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/config.py app/services/observability/ app/services/workflow/engine.py tests/test_ch09_langfuse.py
git commit -m "feat(observability): integrate Langfuse callback into LangGraph compilation with graceful fallback"
```

---

### Task 3: 意图元数据注入与 Cost Control 统计脚本

**Files:**
- Modify: `app/services/workflow/nodes/intent_nodes.py`
- Create: `scripts/check_intent_costs.py`
- Create: `app/services/observability/cost_analytics.py`
- Test: `tests/test_ch09_cost_analytics.py`

**Interfaces:**
- Consumes: LangGraph intent output, Langfuse client / local traces
- Produces: `reports/cost_by_intent.json`, `CostAnalyticsService.get_intent_costs()`

- [ ] **Step 1: Write failing tests for Cost Analytics & Intent binding**
```python
# tests/test_ch09_cost_analytics.py
import os
import json
import pytest
from app.services.observability.cost_analytics import CostAnalyticsService

def test_cost_analytics_aggregation():
    sample_records = [
        {"intent": "faq", "prompt_tokens": 100, "completion_tokens": 50, "model": "gpt-4o"},
        {"intent": "faq", "prompt_tokens": 200, "completion_tokens": 80, "model": "gpt-4o"},
        {"intent": "refund", "prompt_tokens": 500, "completion_tokens": 300, "model": "gpt-4o"},
    ]
    report = CostAnalyticsService.aggregate_records(sample_records)
    assert len(report["items"]) == 2
    # Refund is more expensive -> should be top 1
    assert report["items"][0]["intent"] == "refund"
    assert report["items"][0]["total_tokens"] == 800
    assert report["items"][1]["intent"] == "faq"
    assert report["items"][1]["total_tokens"] == 430
    assert report["total_tokens"] == 1230
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ch09_cost_analytics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Intent Metadata binding and Cost Analytics script**
1. In `intent_node` (`app/services/workflow/nodes/intent_nodes.py`), call `LangfuseManager.update_current_trace(metadata={"intent": intent}, tags=[intent])`.
2. Implement `app/services/observability/cost_analytics.py` providing calculation of prompt/completion tokens, pricing, and intent rankings.
3. Implement `scripts/check_intent_costs.py` writing out `reports/cost_by_intent.json`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ch09_cost_analytics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/services/workflow/nodes/intent_nodes.py app/services/observability/cost_analytics.py scripts/check_intent_costs.py tests/test_ch09_cost_analytics.py
git commit -m "feat(cost): implement intent metadata injection and cost control analytics"
```

---

### Task 4: 多信号证据置信度闸门与网格校准脚本

**Files:**
- Modify: `app/services/workflow/nodes/gate.py`
- Modify: `app/config.py`
- Create: `scripts/calibrate_confidence_gate.py`
- Test: `tests/test_ch09_confidence_gate.py`

**Interfaces:**
- Consumes: `state["retrieved_docs"]`, `tests/data/eval_ch04.jsonl`
- Produces: `confidence_gate(state) -> "pass" | "fallback"`, `compute_evidence_confidence(docs) -> float`, `reports/confidence_calibration.json`

- [ ] **Step 1: Write failing tests for multi-signal evidence confidence**
```python
# tests/test_ch09_confidence_gate.py
import pytest
from app.services.workflow.nodes.gate import compute_evidence_confidence, confidence_gate

def test_compute_evidence_confidence_empty():
    assert compute_evidence_confidence([]) == 0.0

def test_compute_evidence_confidence_high_score_and_margin():
    docs = [
        {"score": 0.85, "text": "高质量文档1"},
        {"score": 0.30, "text": "普通文档2"},
        {"score": 0.10, "text": "低分文档3"},
    ]
    score = compute_evidence_confidence(docs)
    assert score >= 0.50

def test_confidence_gate_fallback_when_below_threshold():
    low_docs = [{"score": 0.20, "text": "弱匹配文档"}]
    state = {"retrieved_docs": low_docs}
    assert confidence_gate(state) == "fallback"

def test_confidence_gate_pass_when_above_threshold():
    high_docs = [{"score": 0.88, "text": "核心退款条款"}]
    state = {"retrieved_docs": high_docs}
    assert confidence_gate(state) == "pass"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ch09_confidence_gate.py -v`
Expected: FAIL

- [ ] **Step 3: Implement multi-signal evidence confidence and calibration script**
1. Add `evidence_confidence_threshold: float = Field(default=0.40)` to `app/config.py`.
2. In `app/services/workflow/nodes/gate.py`, implement `compute_evidence_confidence(docs)` using top1 score, valid count, and score margin; update `confidence_gate(state)` using `settings.evidence_confidence_threshold`.
3. Implement `scripts/calibrate_confidence_gate.py` running grid search on `tests/data/eval_ch04.jsonl`, generating `reports/confidence_calibration.json`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ch09_confidence_gate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/services/workflow/nodes/gate.py app/config.py scripts/calibrate_confidence_gate.py tests/test_ch09_confidence_gate.py
git commit -m "feat(gate): implement multi-signal evidence confidence gate and calibration script"
```

---

### Task 5: 低置信度三路落池与证据快照回捞

**Files:**
- Modify: `app/services/workflow/nodes/knowledge_node.py`
- Modify: `app/services/rag/generator.py`
- Modify: `app/api/routes.py`
- Test: `tests/test_ch09_pool_ingestion.py`

**Interfaces:**
- Consumes: LowConfidenceQuestion model, DB session
- Produces: `POST /api/chat/feedback`, `knowledge_fallback_node` snapshot persistence, `self_check` snapshot persistence

- [ ] **Step 1: Write failing tests for 3-entry low confidence ingestion**
```python
# tests/test_ch09_pool_ingestion.py
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select
from main import app
from app.models.low_confidence import LowConfidenceQuestion

@pytest.mark.asyncio
async def test_feedback_thumbs_down_endpoint(db_session):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/api/chat/feedback", json={
            "conversation_id": 1,
            "feedback_type": "down",
            "reason": "回答不满意",
            "query": "你们包邮吗？"
        })
        assert resp.status_code == 200
        res = resp.json()
        assert res["status"] == "success"

    q = (await db_session.execute(
        select(LowConfidenceQuestion).where(LowConfidenceQuestion.source == "user_feedback")
    )).scalars().all()
    assert len(q) > 0
    assert q[-1].raw_question == "你们包邮吗？"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ch09_pool_ingestion.py -v`
Expected: FAIL with 404 or missing endpoint.

- [ ] **Step 3: Implement snapshot capture in fallback node, generator, and feedback API**
1. In `app/services/workflow/nodes/knowledge_node.py`, pass `retrieved_chunks` (Top docs text & score) into `LowConfidenceQuestion`.
2. In `app/services/rag/generator.py`, pass `retrieved_chunks` when `useful=False`.
3. In `app/api/routes.py`, implement `POST /api/chat/feedback` recovering previous query and recall chunks, saving to `low_confidence_questions` with `source='user_feedback'`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ch09_pool_ingestion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/services/workflow/nodes/knowledge_node.py app/services/rag/generator.py app/api/routes.py tests/test_ch09_pool_ingestion.py
git commit -m "feat(flywheel): implement snapshot capture across 3 low-confidence entry points and feedback API"
```

---

### Task 6: 飞轮批处理流水线（标准化改写 + 两阶段查重归并）

**Files:**
- Create: `app/services/flywheel/__init__.py`
- Create: `app/services/flywheel/pipeline.py`
- Create: `scripts/run_flywheel_pipeline.py`
- Test: `tests/test_ch09_flywheel_pipeline.py`

**Interfaces:**
- Consumes: pending `LowConfidenceQuestion` rows
- Produces: `FlywheelPipeline.process_pending_questions(db) -> dict`, `scripts/run_flywheel_pipeline.py`

- [ ] **Step 1: Write failing tests for Flywheel normalization and deduplication**
```python
# tests/test_ch09_flywheel_pipeline.py
import pytest
from sqlalchemy import select
from app.models.low_confidence import LowConfidenceQuestion
from app.models.review_queue import ReviewQueue
from app.services.flywheel.pipeline import FlywheelPipeline

@pytest.mark.asyncio
async def test_flywheel_pipeline_creates_and_merges_review_item(db_session):
    # 1. 插入第一条原话
    q1 = LowConfidenceQuestion(raw_question="这衣服掉色很厉害怎么办？", source="retrieval_low_conf")
    db_session.add(q1)
    await db_session.commit()

    # 运行飞轮
    res1 = await FlywheelPipeline.process_pending_questions(db_session)
    assert res1["processed_count"] == 1
    assert res1["new_created"] == 1

    # 验证 review_queue 新增 1 条
    items = (await db_session.execute(select(ReviewQueue))).scalars().all()
    assert len(items) == 1
    assert items[0].occurrence_count == 1
    review_id = items[0].id

    # 2. 插入第二条同义原话
    q2 = LowConfidenceQuestion(raw_question="衣服洗了严重褪色能退吗？", source="user_feedback")
    db_session.add(q2)
    await db_session.commit()

    res2 = await FlywheelPipeline.process_pending_questions(db_session)
    assert res2["processed_count"] == 1
    assert res2["merged_count"] == 1

    # 验证累加计数且未新建行
    await db_session.refresh(items[0])
    assert items[0].occurrence_count == 2
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ch09_flywheel_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement FlywheelPipeline and run_flywheel_pipeline script**
1. Implement FAQ normalization prompt to rewrite colloquial question into standardized question + sample answer.
2. Implement semantic deduplication against existing `review_queue` items.
3. Update `occurrence_count` on match or create new `ReviewQueue` row, linking `matched_review_id`.
4. Implement `scripts/run_flywheel_pipeline.py`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ch09_flywheel_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/services/flywheel/ scripts/run_flywheel_pipeline.py tests/test_ch09_flywheel_pipeline.py
git commit -m "feat(flywheel): implement FAQ normalization and deduplication pipeline"
```

---

### Task 7: 待审队列管理与 Ch03 知识库双写回流

**Files:**
- Create: `app/api/review_queue_routes.py`
- Modify: `app/api/routes.py`
- Test: `tests/test_ch09_review_queue_api.py`

**Interfaces:**
- Consumes: `KnowledgeDualWriter`, `ReviewQueue`
- Produces: `GET /api/review-queue`, `GET /api/review-queue/{id}`, `POST /api/review-queue/{id}/approve`, `POST /api/review-queue/{id}/reject`

- [ ] **Step 1: Write failing tests for Review Queue approval & Ch03 knowledge writeback**
```python
# tests/test_ch09_review_queue_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select
from main import app
from app.models.review_queue import ReviewQueue, ReviewStatus
from app.models.knowledge import KnowledgeChunk

@pytest.mark.asyncio
async def test_review_queue_approval_dual_writes_to_kb(db_session):
    item = ReviewQueue(
        normalized_question="双十一保价退差价流程是什么？",
        ai_suggested_answer="活动结束后3天内联系客服退差价。",
        occurrence_count=3,
        review_status=ReviewStatus.PENDING,
    )
    db_session.add(item)
    await db_session.commit()
    await db_session.refresh(item)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(f"/api/review-queue/{item.id}/approve", json={
            "approved_answer": "在订单详情页点击申请保价，系统自动核验差价并原路返还。"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "approved"

    # 验证 review_queue 状态变为通过
    await db_session.refresh(item)
    assert item.review_status == ReviewStatus.APPROVED
    assert "申请保价" in item.approved_answer

    # 验证 knowledge_chunks 已成功写入
    chunks = (await db_session.execute(
        select(KnowledgeChunk).where(KnowledgeChunk.questions.contains("双十一保价"))
    )).scalars().all()
    assert len(chunks) > 0
    assert chunks[0].content_type == "faq"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ch09_review_queue_api.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Review Queue router and KnowledgeDualWriter integration**
1. In `app/api/review_queue_routes.py`, implement GET endpoints (list, detail with raw queries & snapshots) and POST endpoints (reject, approve).
2. On approve: invoke `KnowledgeDualWriter.write_chunks` with `category="常见问题"`, `questions=normalized_question`, `answer=approved_answer`, `content_type="faq"`.
3. Include router in `main.py`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ch09_review_queue_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/api/review_queue_routes.py app/api/routes.py tests/test_ch09_review_queue_api.py
git commit -m "feat(flywheel): implement review queue approval and Ch03 knowledge writeback flow"
```

---

### Task 8: 自动化评测流水线与趋势持久化

**Files:**
- Create: `app/services/eval/__init__.py`
- Create: `app/services/eval/eval_pipeline.py`
- Create: `scripts/run_eval_pipeline.py`
- Test: `tests/test_ch09_eval_pipeline.py`

**Interfaces:**
- Consumes: `tests/data/eval_ch04.jsonl`, `app/services/rag/evaluator.py`
- Produces: `eval_runs` table records, `run_eval_pipeline()`

- [ ] **Step 1: Write failing tests for evaluation batch execution and trend recording**
```python
# tests/test_ch09_eval_pipeline.py
import pytest
from sqlalchemy import select
from app.models.eval_run import EvalRun, TriggeredBy
from app.services.eval.eval_pipeline import EvalPipelineService

@pytest.mark.asyncio
async def test_run_eval_pipeline_writes_to_eval_runs(db_session):
    run = await EvalPipelineService.run_eval_round(
        db=db_session,
        triggered_by=TriggeredBy.MANUAL,
        sample_limit=5, # 快速回归小样本
    )
    assert run.id is not None
    assert run.dataset_size == 5
    assert "recall_at_3" in run.metrics
    assert "mrr" in run.metrics
    assert "faithfulness" in run.metrics

    # 查表确认持久化
    all_runs = (await db_session.execute(select(EvalRun))).scalars().all()
    assert len(all_runs) >= 1
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ch09_eval_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement EvalPipelineService and run_eval_pipeline script**
1. Implement `app/services/eval/eval_pipeline.py` reusing metric computation from `app/services/rag/evaluator.py`.
2. Persist round result to `eval_runs` table.
3. Implement `scripts/run_eval_pipeline.py` supporting `--trigger` and `--samples` args, displaying delta against prior run.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ch09_eval_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/services/eval/ scripts/run_eval_pipeline.py tests/test_ch09_eval_pipeline.py
git commit -m "feat(eval): implement automated evaluation pipeline and eval_runs trend persistence"
```

---

### Task 9: 观测与成本 Overview 只读 API 与作业调度器注册

**Files:**
- Create: `app/api/observability.py`
- Modify: `app/services/job_runner.py`
- Modify: `main.py`
- Test: `tests/test_ch09_observability_api.py`

**Interfaces:**
- Consumes: `reports/cost_by_intent.json`, `eval_runs` table, `reports/confidence_calibration.json`, `settings.evidence_confidence_threshold`
- Produces: `GET /api/observability/overview`, registered jobs in `JobRunner`

- [ ] **Step 1: Write failing tests for /api/observability/overview**
```python
# tests/test_ch09_observability_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_observability_overview_returns_three_blocks():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/observability/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "cost_block" in data
        assert "eval_trend_block" in data
        assert "calibration_block" in data
        assert data["cost_block"]["job_name"] == "cost-analysis"
        assert data["eval_trend_block"]["job_name"] == "eval-pipeline"
        assert data["calibration_block"]["job_name"] == "calibrate-confidence"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ch09_observability_api.py -v`
Expected: FAIL

- [ ] **Step 3: Implement /api/observability/overview and register whitelist jobs**
1. In `app/services/job_runner.py`, add `cost-analysis`, `eval-pipeline`, `calibrate-confidence`, `flywheel-pipeline` into `WHITELIST_JOBS`.
2. Implement `app/api/observability.py` with `GET /api/observability/overview`: reads the 3 blocks independently without re-calculating, handles missing files with `present=false` + hint, detects threshold mismatch between recommended and active.
3. Register router in `main.py`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ch09_observability_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/api/observability.py app/services/job_runner.py main.py tests/test_ch09_observability_api.py
git commit -m "feat(api): implement read-only observability overview and register whitelist jobs"
```

---

### Task 10: 前端交互与看板实现 (Vibe Coding)

**Files:**
- Create: `app/static/observability.html`
- Create: `app/static/review_queue.html`
- Modify: `app/static/index.html`
- Modify: `app/static/kb.html`
- Modify: `app/static/rag_eval.html`
- Modify: `main.py`

**Interfaces:**
- Consumes: `/api/observability/overview`, `/api/jobs`, `/api/review-queue`, `/api/chat/feedback`
- Produces: HTML routes `/observability` and `/review-queue`, feedback 👎 button hook, unified nav bars and cards.

- [ ] **Step 1: Mount HTML routes in main.py**
Add `/observability` and `/review-queue` endpoints in `main.py`.

- [ ] **Step 2: Implement app/static/observability.html**
Three cards (Cost, Eval Trends, Calibration), job rerun buttons linked to `/api/jobs`, mismatch warning banner, delta trend rendering.

- [ ] **Step 3: Implement app/static/review_queue.html**
Status tabs (全部, 待审, 通过, 驳回), table with occurrence count, expandable row with raw queries and recall chunk snapshots, approve modal with knowledge base writeback, reject button, run flywheel batch button.

- [ ] **Step 4: Hook thumbs-down feedback in app/static/index.html**
In `handleFeedback(btn, 'down')`, post to `/api/chat/feedback` and render success toast.

- [ ] **Step 5: Update navigation links across all admin pages**
Add "观测与成本" and "飞轮待审" links to header nav bars in `kb.html`, `rag_eval.html`, and `observability.html`.

- [ ] **Step 6: Commit**
```bash
git add app/static/observability.html app/static/review_queue.html app/static/index.html app/static/kb.html app/static/rag_eval.html main.py
git commit -m "feat(ui): implement observability dashboard, review queue admin UI, and feedback hook"
```

---

### Task 11: 端到端全链路验收测试与全量回归套件

**Files:**
- Create: `tests/test_ch09_acceptance.py`
- Modify: `dev-notes/ch09.md`

**Interfaces:**
- Runs full workflow: Fallback -> Pool -> Flywheel -> Review -> KB Writeback -> Answered correctly; Feedback 👎; 2 Eval rounds trend comparison.

- [ ] **Step 1: Write E2E Acceptance Test covering all 6 acceptance criteria**
```python
# tests/test_ch09_acceptance.py
import pytest
from httpx import AsyncClient, ASGITransport
from main import app
from app.services.workflow.engine import WorkflowEngine
from app.services.flywheel.pipeline import FlywheelPipeline

@pytest.mark.asyncio
async def test_ch09_full_flywheel_lifecycle(db_session):
    # 1. 问一个知识库没有的问题，获得兜底回复并落入问题池
    engine = WorkflowEngine()
    out = await engine.run(conversation_id=901, query="超光速曲率引擎质保期几年？", db=db_session)
    assert out.get("confidence_passed") is False
    assert "暂未收录" in out.get("response_text", "")

    # 2. 运行飞轮批处理，问题进入待审队列
    res = await FlywheelPipeline.process_pending_questions(db_session)
    assert res["processed_count"] >= 1

    # 3. 审核通过并双写入知识库
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        q_list = (await ac.get("/api/review-queue")).json()
        target = [item for item in q_list["items"] if "曲率引擎" in item["normalized_question"]][0]
        approve_resp = await ac.post(f"/api/review-queue/{target['id']}/approve", json={
            "approved_answer": "超光速曲率引擎享有星际联邦3个恒星年质保。"
        })
        assert approve_resp.status_code == 200

    # 4. 同一个问题再问，成功检索并答对（飞轮转完一整圈）
    out2 = await engine.run(conversation_id=902, query="超光速曲率引擎质保期几年？", db=db_session)
    assert "3个恒星年" in out2.get("response_text", "")

@pytest.mark.asyncio
async def test_eval_pipeline_two_rounds_trend(db_session):
    from app.services.eval.eval_pipeline import EvalPipelineService
    from app.models.eval_run import TriggeredBy
    r1 = await EvalPipelineService.run_eval_round(db=db_session, triggered_by=TriggeredBy.MANUAL, sample_limit=3)
    r2 = await EvalPipelineService.run_eval_round(db=db_session, triggered_by=TriggeredBy.MANUAL, sample_limit=3)
    assert r1.id != r2.id
```

- [ ] **Step 2: Run all tests in the repository for full regression**
Run: `pytest tests/ -v`
Expected: 100% PASS

- [ ] **Step 3: Commit and update dev-notes/ch09.md**
```bash
git add tests/test_ch09_acceptance.py dev-notes/ch09.md
git commit -m "test(acceptance): add chapter 9 end-to-end flywheel and evaluation acceptance tests"
```
