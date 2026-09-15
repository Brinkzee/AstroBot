# 第九章系统设计规范：可观测性监控、成本透视与数据飞轮闭环 (Design Spec)

本文档定义第九章的技术架构、数据模型、API 契约、飞轮闭环与观测后台设计规范。

---

## 1. 目标与边界

### 1.1 核心目标
1. **Langfuse 全链路 Trace 观测**：LangGraph 编译期单点注入回调，请求全链路（Prompt、工具调用、知识检索、Token 消耗、耗时）可观测。
2. **意图维度的 Token 成本控制 (Cost Control)**：将意图识别结果动态写入 Trace 根元数据，终端与后台可视化统计各意图的 Token 开销及最烧钱分类。
3. **校准置信度闸门与证据快照落池**：升级置信度判定为多信号算子（Top1 相关性分、有效证据数量、Top1-Top2 分差），离线网格搜索校准阈值；三路入口（低置信度拦截、自评不足、用户点踩 👎）原子化写入 `low_confidence_questions` 表，并保存召回片段快照 `retrieved_chunks`。
4. **数据飞轮批处理与知识库回流**：标准化改写口语原话、两阶段语义查重归并至 `review_queue`；人工审核通过后调用既有 `KnowledgeDualWriter` 双写落库（MySQL + Milvus），使系统答对原先答不上的问题。
5. **自动化评估流水线与趋势落表**：复用 Ch04 评估集与检索（Recall@K, MRR）及生成（Faithfulness）指标，将每轮评估持久化至 `eval_runs`，展示历史下滑预警。
6. **观测与成本后台看板 (`/observability`) 与待审管理页 (`/review-queue`)**：将三张终端报表搬进后台，支持一键安全作业重跑；提供统一导航与阈值未回填预警。

### 1.2 本章不做的边界
- 暂不做低置信度问题按主题归类的微调分类器（留待后续微调章节）。

---

## 2. 数据库与实体设计

对齐 `sql/ch09-ddl.sql`，支持 SQLite 与 MySQL 双数据库方言兼容。

### 2.1 待审队列表 `review_queue`
```sql
CREATE TABLE review_queue (
  id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '缺口主键,也是查重命中要返回的 matched_review_id',
  normalized_question VARCHAR(512)    NOT NULL                COMMENT '标准化后的 FAQ 式问题',
  ai_suggested_answer TEXT            NULL                    COMMENT '模型生成的示例答案,备查',
  occurrence_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '出现次数,查重命中累加,越高越该优先补',
  review_status       ENUM('待审','通过','驳回') NOT NULL DEFAULT '待审' COMMENT '人工审核状态',
  approved_answer     TEXT            NULL                    COMMENT '审核通过时补的核准答案,走 ch03 落库流程写回知识库',
  created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '首次入队时间',
  updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_review_status (review_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='飞轮待审队列';
```
- ORM 映射模型：`app/models/review_queue.py` (`ReviewQueue`, `ReviewStatus`)

### 2.2 评估轮次表 `eval_runs`
```sql
CREATE TABLE eval_runs (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '评估轮次主键',
  triggered_by ENUM('定时','手动') NOT NULL DEFAULT '定时' COMMENT '这轮怎么起的:定时任务,或某次改动后手动跑',
  dataset_size INT UNSIGNED    NOT NULL                COMMENT '这轮跑的评估集条数',
  metrics      JSON            NOT NULL                COMMENT '各指标分数,如 {"recall_at_k":0.82,"mrr":0.71,"faithfulness":0.90}',
  created_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '跑完落表时间',
  PRIMARY KEY (id),
  KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='自动化评估流水线轮次结果';
```
- ORM 映射模型：`app/models/eval_run.py` (`EvalRun`, `TriggeredBy`)

### 2.3 低置信度表扩展 `low_confidence_questions`
在现有模型 `app/models/low_confidence.py` 中新增两列：
- `retrieved_chunks`: `JSON`（可为 NULL），存落池时的召回片段快照（Top 几条的原文和得分），审核页给人看；未走检索（闲聊、业务数据等）为 NULL。
- `matched_review_id`: `BIGINT UNSIGNED`（可为 NULL），外键指向 `review_queue.id` (`ON DELETE SET NULL`)，记录查重后归并到的缺口。
- 迁移脚本 `scripts/init_ch09_db.py` 负责无损且幂等地增列与建表。

---

## 3. Langfuse 观测与成本控制架构

### 3.1 Langfuse 客户端管理 (`app/services/observability/langfuse_service.py`)
- **配置项** (`app/config.py`):
  - `langfuse_public_key`: 公钥
  - `langfuse_secret_key`: 私钥
  - `langfuse_host`: 服务端地址（默认 `http://localhost:3000`）
  - `langfuse_enabled`: 自动根据是否配置 Key 决定启用
- **宽容降级原则**：未配置 Key 或网络不可达时，`get_langfuse_callback()` 返回 `None`，主流程完全正常执行，不阻断对话与单元测试。

### 3.2 LangGraph 编译期单点注入
在 `app/services/workflow/engine.py` 的 `build_workflow_graph()` 中：
```python
memory = checkpointer or MemorySaver()
compiled = builder.compile(checkpointer=memory)
langfuse_handler = get_langfuse_callback()
if langfuse_handler is not None:
    return compiled.with_config({"callbacks": [langfuse_handler]})
return compiled
```

### 3.3 意图元数据与 Trace 根绑定
在 `app/services/workflow/nodes/intent_nodes.py` 中，当意图识别完成后：
```python
LangfuseService.update_current_trace(
    metadata={"intent": intent_name},
    tags=[intent_name]
)
```

### 3.4 成本统计脚本与权威产物
- 运行脚本：`scripts/check_intent_costs.py`
  - 尝试从 Langfuse API 拉取 Trace，若无外网或本地环境则从审计/工作流运行数据中计算聚合；
  - 权威产物：`reports/cost_by_intent.json`，格式包括各意图的 `total_calls`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `estimated_cost_usd`, `cost_percentage`，按 `total_tokens` 降序排列。

---

## 4. 证据置信度闸门与多入口快照落池

### 4.1 多信号证据置信度公式 (`app/services/workflow/nodes/gate.py`)
结合精排 Top1 得分、有效证据数、Top1 与 Top2 分差梯度：
$$\text{evidence\_confidence} = w_1 \cdot \text{top1\_score} + w_2 \cdot \min(\frac{\text{valid\_count}}{K}, 1.0) + w_3 \cdot \max(\text{score\_margin}, 0.0)$$
默认权重基准：$w_1 = 0.60, w_2 = 0.25, w_3 = 0.15$。

### 4.2 离线网格校准脚本 (`scripts/calibrate_confidence_gate.py`)
- 评估样本：`tests/data/eval_ch04.jsonl`（50 条样本，区分有效知识与越界 `D_absent`）；
- 算法：网格搜索遍历截断阈值，计算 Youden's J 统计量（Sensitivity + Specificity - 1）与 F1-Score，输出最优推荐阈值；
- 权威产物：`reports/confidence_calibration.json`，包含 `recommended_threshold`、`youden_j`、`best_f1`、`sample_count` 及校准时间。
- 联动配置：在 `app/config.py` 中加入 `evidence_confidence_threshold: float = 0.40`。

### 4.3 三路落池与快照回捞
1. **检索证据置信度低**：
   - 闸门位置不动，仍卡在 `knowledge_retrieval` 之后、`main_agent` 之前；
   - 拦下流转至 `knowledge_fallback` 节点，返回统一兜底话术；
   - 原子化持久化至 `low_confidence_questions`，`source="retrieval_low_conf"`，`retrieved_chunks` 存 Top 检索候选原文与得分。
2. **生成自评知识不够**：
   - `RAGControlledGenerator.evaluate_citations_usefulness` 判 `useful=False` 时；
   - 写入 `low_confidence_questions`，`source="self_check"`，`retrieved_chunks` 存入当前引用的 citations。
3. **用户反馈没解决 (👎)**：
   - 接口：`POST /api/chat/feedback`
   - 参数：`{"conversation_id": 123, "feedback_type": "down"}`
   - 回捞逻辑：根据会话回溯上一轮问答；若该轮触发了检索，将当轮召回片段写入 `retrieved_chunks`；若当轮为闲聊或纯业务操作未走检索，则 `retrieved_chunks = NULL`。

---

## 5. 数据飞轮流水线与知识库回流

### 5.1 批处理流水线 (`app/services/flywheel/pipeline.py`)
- 扫描 `low_confidence_questions` 中 `matched_review_id IS NULL` 的记录；
- **标准化改写**：调用 LLM FAQ 提示词将口语问题提炼为标准问法，并生成一份 `ai_suggested_answer`；
- **两阶段查重**：
  - 候选初筛：提取队列中处于 `'待审'` 状态的记录；
  - 语义精判：LLM 判定新问题与候选问题是否为同一客服咨询诉求；
- **缺口维护**：
  - 命中已存缺口：`occurrence_count = occurrence_count + 1`，将命中 ID 回填至 `matched_review_id`；
  - 未命中：在 `review_queue` 中新建记录，初态 `'待审'`，`occurrence_count = 1`，将新 ID 回填至 `matched_review_id`。

### 5.2 后台审核与 Ch03 原生双写落库
- 接口：
  - `GET /api/review-queue`：分页与状态过滤（待审/通过/驳回），按 `occurrence_count DESC` 排序；
  - `GET /api/review-queue/{id}`：详情展示，联动返回归并的所有原话与各自的 `retrieved_chunks` 快照；
  - `POST /api/review-queue/{id}/reject`：更新状态为 `'驳回'`；
  - `POST /api/review-queue/{id}/approve`：
    - 传入人工确认或润色后的 `approved_answer`；
    - 状态置为 `'通过'`；
    - 构造 `DocChunk(category="常见问题", questions=normalized_question, answer=approved_answer, content_type="faq")`；
    - 调用 Ch03 `KnowledgeDualWriter.write_chunks` 写入 MySQL `knowledge_chunks` 并在 Milvus 中同步完成向量化与 Upsert；
    - 再次提问即可命中新知识并正确回答，达成飞轮闭环。

---

## 6. 自动化评测流水线与趋势落表

### 6.1 评测执行 (`scripts/run_eval_pipeline.py` & `app/services/eval/eval_pipeline.py`)
- 输入：`tests/data/eval_ch04.jsonl`
- 计算指标：
  - 检索段：`Recall@3`, `Recall@5`, `Recall@10`, `MRR`
  - 生成段：`Faithfulness`
- 写入表：`eval_runs`
  - `triggered_by`: `'手动'` / `'定时'`
  - `dataset_size`: 评测样例总数
  - `metrics`: 指标 JSON 对象
  - `created_at`: 当前时间

### 6.2 趋势对比与下滑预警
- 每次运行比对上一轮记录，输出各指标 Delta 差值，标明指标上升或下滑趋势。

---

## 7. 观测与成本后台 (`/observability`) 架构

### 7.1 只读 API 规范 (`app/api/observability.py`)
`GET /api/observability/overview`：三块互不连坐，独立 try-catch，绝不重算。

返回结构示例：
```json
{
  "cost_block": {
    "present": true,
    "source_file": "reports/cost_by_intent.json",
    "updated_at": "2026-09-15T21:00:00",
    "job_name": "cost-analysis",
    "missing_hint": "先聊几句攒 trace 再重跑",
    "data": [...]
  },
  "eval_trend_block": {
    "present": true,
    "runs_count": 5,
    "job_name": "eval-pipeline",
    "missing_hint": "按一次就是趋势的第一个点",
    "data": [...]
  },
  "calibration_block": {
    "present": true,
    "job_name": "calibrate-confidence",
    "missing_hint": "去扫一遍",
    "recommended_threshold": 0.42,
    "active_threshold": 0.40,
    "threshold_mismatch": true,
    "data": {...}
  }
}
```

### 7.2 安全白名单作业调度器注册 (`app/services/job_runner.py`)
在 `WHITELIST_JOBS` 中统一登记本章脚本：
- `"cost-analysis"`: `[sys.executable, "scripts/check_intent_costs.py"]`
- `"eval-pipeline"`: `[sys.executable, "scripts/run_eval_pipeline.py", "--trigger", "manual"]`
- `"calibrate-confidence"`: `[sys.executable, "scripts/calibrate_confidence_gate.py"]`
- `"flywheel-pipeline"`: `[sys.executable, "scripts/run_flywheel_pipeline.py"]`

前端只能传入 `job_name`，由 `POST /api/jobs` 执行与轮询日志，绝不传递任何外部 shell 字符串。

### 7.3 前端页面与导航
- 新增 `app/static/observability.html`（路由 `/observability`）；
- 新增 `app/static/review_queue.html`（路由 `/review-queue`）；
- 在各后台页导航条统一加入「观测与成本」和「飞轮待审」入口；
- 若 `threshold_mismatch === true`，在页面顶部及卡片上醒目标注“推荐阈值与在用阈值不一致，请回填配置”。

---

## 8. 验收与测试验证策略
1. **单元测试与 TDD**：
   - 数据模型与双向外键验证 (`tests/test_ch09_models.py`)
   - 置信度闸门多信号计算与拦截断言 (`tests/test_ch09_gate.py`)
   - 飞轮标准化与查重去重流水线 (`tests/test_ch09_flywheel.py`)
   - 评测流水线与趋势落库 (`tests/test_ch09_eval_pipeline.py`)
   - 观测与成本 Overview 只读 API (`tests/test_ch09_observability_api.py`)
2. **端到端飞轮闭环测试 (`tests/test_ch09_acceptance.py`)**：
   - 知识库缺失问题 -> 置信度拦截 -> 待审入队 -> 审核通过落库 -> 再次提问答对（飞轮转满一整圈）；
   - 聊天页 👎 反馈 -> 快照回捞入池；
   - 连续运行 2 轮评估流水线，校验 `eval_runs` 趋势对比。
