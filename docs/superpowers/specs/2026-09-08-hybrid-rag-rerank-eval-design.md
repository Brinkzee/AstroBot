# 架构设计规范 (Design Spec) - 第四章：RAG 进阶 · 混合检索、重排与质量评估体系

## 1. 目标与背景

本设计致力于全面升级客服系统的检索召回与生成质量体系。通过引入 **Milvus 2.5 原生 BM25 全文检索** 与 **BGE-M3 语义向量** 的两路召回及 **RRF 融合**，结合 **BGE-Reranker-v2-m3** 交叉编码重排与 **首尾放置（Lost in the Middle）** 提示词编排，突破传统单一向量检索在型号匹配与长尾专有名词上的局限；同时建立基于证据引用溯源、前置知识充分度自评与显式拒答的生成防护机制，配合 `low_confidence_questions` 与 `faith_cases` 两张业务表，依托 `tests/data/eval_ch04.jsonl` 真实测试集构建覆盖四种检索策略的离线评测体系与前端溯源/反馈交互。

---

## 2. 系统总体架构

系统采用模块化分层流水线架构，从请求输入至前端呈现涵盖以下子系统：

```
[ 用户提问 / 评估集 Query ]
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Query 理解模块 (QueryProcessor)                          │
│    • 口语模糊问法归一化 (standard_query)                     │
│    • 检索侧同义词扩展 (expanded_keywords -> bm25_query)     │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
       [standard_query]                  [bm25_query]
                │                             │
                ▼                             ▼
┌───────────────────────────────┐ ┌───────────────────────────┐
│ BGE-M3 Dense 向量生成         │ │ Milvus 2.5 原生 BM25 检索 │
│ (1024 维 Dense Vector)        │ │ (jieba 分词器 Sparse 向量)│
└───────────────┬───────────────┘ └───────────┬───────────────┘
                │                             │
                ▼ (Top-50 召回)               ▼ (Top-50 召回)
┌─────────────────────────────────────────────────────────────┐
│ 2. Milvus hybrid_search 粗排 (RRFRanker k=60)                │
│    • 标量元数据前置过滤 (品类 category 等)                   │
│    • RRF 倒数排名融合合并出 Top-50 候选集                   │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. BGE-Reranker-v2-m3 精排重排器 (BGERerankerClient)        │
│    • Query-Document Cross-Encoder 语义相关度打分            │
│    • 精排截取 Top-10 知识块                                 │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 上下文编排与引用对齐 (Lost in the Middle)                 │
│    • 首尾放置排布: [D1, D3, D5, D7, D9, D10, D8, D6, D4, D2]│
│    • 生成 1~10 引用快照 citations 元数据                    │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. 生成质量控制与自评守卫 (RAGGenerator)                    │
│    • 阶段 1 (自评): 模型判断知识是否充足 (useful)           │
│      - useful=False: 拒答并落库 low_confidence_questions   │
│    • 阶段 2 (受控生成): 注入负面约束，流式输出带 [n] 角标答案│
└───────────────────────────────┬─────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌───────────────────────────────┐ ┌───────────────────────────┐
│ 6. 前端渲染与交互 (Web Chat)  │ │ 7. 离线评估与台账(Evaluator)│
│    • SSE 下发 citations 事件  │ │ • 4 策略对比(Recall/MRR) │
│    • [n] 角标点击弹窗显示原文 │ │ • Faithfulness 裁判评测   │
│    • 👍/👎 满意度反馈点亮锁定 │ │ • 编造个案落库 faith_cases│
└───────────────────────────────┘ └───────────────────────────┘
```

---

## 3. 详细设计与数据契约

### 3.1 数据库与存储层设计

#### 3.1.1 MySQL 表结构（对齐 `sql/ch04_ddl.sql`）
1. **`low_confidence_questions`（低置信度问题池）**
   - `id`: BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY
   - `conversation_id`: BIGINT UNSIGNED NULL, 外键关联 `conversations(id)`
   - `raw_question`: TEXT NOT NULL, 用户原始提问
   - `source`: ENUM('retrieval_low_conf', 'self_check', 'user_feedback') NOT NULL
   - `reason`: TEXT NULL, 判不能回答的原因
   - `created_at`: DATETIME DEFAULT CURRENT_TIMESTAMP
   - 索引: `idx_source`, `idx_created_at`
2. **`faith_cases`（编造个案台账）**
   - `id`: BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY
   - `eval_id`: VARCHAR(16) NOT NULL UNIQUE (唯一索引 `uk_eval_id`)
   - `bucket`: VARCHAR(24) NOT NULL ('A_policy', 'B_model', 'C_colloquial', 'E_multi')
   - `query`: VARCHAR(512) NOT NULL
   - `strategy`: VARCHAR(24) NOT NULL DEFAULT 'hybrid_rerank'
   - `answer`: TEXT NOT NULL, 被判编造的答案原文
   - `reason`: TEXT NOT NULL, 裁判判定的具体理由
   - `citations`: JSON NULL, 喂给模型的 Top-K 证据全集快照 `[{n, chunk_id, section_path, question, answer}]`
   - `judge_model`: VARCHAR(64) NULL
   - `status`: ENUM('未解决', '已解决', '无需解决') NOT NULL DEFAULT '未解决'
   - `seen_count`: INT UNSIGNED NOT NULL DEFAULT 1
   - `first_seen_at`: DATETIME DEFAULT CURRENT_TIMESTAMP
   - `last_seen_at`: DATETIME DEFAULT CURRENT_TIMESTAMP
   - `resolution`: VARCHAR(300) NULL, 处置说明
   - `resolved_at`: DATETIME NULL
   - **台账幂等逻辑**：
     - 若 `eval_id` 已存在，更新 `answer`, `reason`, `citations`, `last_seen_at`，`seen_count += 1`；
     - 若原状态为 `'已解决'`，更新时状态自动退回 `'未解决'`，标记复发。

#### 3.1.2 Milvus 2.5 混合检索集合 Schema 升级
- **集合名称**：`knowledge`
- **字段定义**：
  - `id`: `DataType.INT64`, 主键, `auto_id=False`（严格与 MySQL `knowledge_chunks.id` 1:1 数值对齐）
  - `chunk_text`: `DataType.VARCHAR(65535)`, 开启 `enable_analyzer=True`, `analyzer_params={"type": "jieba"}`
  - `sparse_vector`: `DataType.SPARSE_FLOAT_VECTOR`
  - `vector`: `DataType.FLOAT_VECTOR(dim=1024)`
  - `category`: `DataType.VARCHAR(128)`
  - `section_path`: `DataType.VARCHAR(512)`
  - `questions`: `DataType.VARCHAR(2048)`
  - `answer`: `DataType.VARCHAR(8192)`
  - `content_type`: `DataType.VARCHAR(32)`
  - `is_key_clause`: `DataType.BOOL`
- **BM25 函数**：
  ```python
  bm25_fn = Function(
      name="chunk_bm25",
      function_type=FunctionType.BM25,
      input_field_names=["chunk_text"],
      output_field_names=["sparse_vector"],
  )
  ```
- **索引配置**：
  - `vector`: `metric_type="COSINE"`, `index_type="FLAT"`
  - `sparse_vector`: `metric_type="BM25"`, `index_type="AUTOINDEX"`
- **数据灌库脚本 (`scripts/reindex_ch04_knowledge.py`)**：从 MySQL 读取所有 `knowledge_chunks`，组装 `chunk_text = f"【类目】{category}\n【标准问法】{questions}\n【解答】{answer}"`，写入 Milvus 触发分词与双向量索引构建。

---

### 3.2 检索与重排流水线设计

#### 3.2.1 Query 理解（`app/services/rag/query_processor.py`）
- **职责**：将自然语言口语归一化为标准电商术语，并在检索侧生成关键词与同义词扩展。
- **输出 DTO**：
  ```python
  @dataclass
  class QueryUnderstandingResult:
      original_query: str
      standard_query: str
      expanded_keywords: List[str]
      bm25_query: str  # 拼接 standard_query 与 expanded_keywords
  ```
- **降级支持**：离线测试或 Mock 模式下，直接回退为原始问题，保障 100% 可用。

#### 3.2.2 Milvus 原生 BM25 + Dense RRF 混合检索
- **检索接口**：
  ```python
  def hybrid_search(
      self,
      dense_vector: List[float],
      bm25_text: str,
      category_filter: Optional[str] = None,
      top_k_per_route: int = 50,
      limit: int = 50,
      collection_name: str = "knowledge"
  ) -> List[Dict]
  ```
- **双路构造**：
  - Dense 路：`AnnSearchRequest(data=[dense_vector], anns_field="vector", param={"metric_type": "COSINE"}, limit=50, expr=filter_expr)`
  - BM25 路：`AnnSearchRequest(data=[bm25_text], anns_field="sparse_vector", param={"metric_type": "BM25"}, limit=50, expr=filter_expr)`
- **融合规则**：`RRFRanker(k=60)`，计算倒数排名得分并取 Top-50。
- **解耦方法**：保留 `search_dense(...)` 与 `search_bm25(...)` 单路方法，专供四策略对比评测。

#### 3.2.3 BGE-Reranker-v2-m3 重排客户端（`app/services/rag/reranker.py`）
- **客户端**：`BGERerankerClient`
- **模型**：`BAAI/bge-reranker-v2-m3`
- **生产调用**：通过 `huggingface_hub.InferenceClient` 官方 Router 端点发送批量请求 `inputs: [{"text": query, "text_pair": doc}, ...]`，解析 Cross-Encoder logit 得分。
- **重试与降级**：内置 3 次重试及指数退避；离线或未配置网络时自动切换基于词重叠与 Jaccard/Dense 组合的确定性 Mock 逻辑，确保单测可靠。
- **截断**：由 50 条精排出 Top-10。

#### 3.2.4 首尾放置（Lost in the Middle）与证据序号打标
- 对重排后的 10 条按降序 $D_1, D_2, \dots, D_{10}$ 重排为：
  `[D_1, D_3, D_5, D_7, D_9, D_{10}, D_8, D_6, D_4, D_2]`
- 赋予角标序号 $n \in [1, 10]$，组装 `CitationItem` 列表：
  ```python
  {
      "n": 1,
      "chunk_id": 12,
      "section_path": "售后政策 > 退货流程",
      "question": "退货政策是什么样的",
      "answer": "支持7天无理由退货..."
  }
  ```

---

### 3.3 生成质量控制与流式交互设计

#### 3.3.1 两阶段生成编排（`app/services/rag/generator.py`）
- **阶段 1：知识充分度前置自评（Self-Check）**
  - 输入：用户原提问与排布好的 Top-10 知识证据；
  - 判定：
    - 若候选集为空或匹配度极低：直接判不足，`source='retrieval_low_conf'`；
    - 否则调用结构化提示词判别：`{"useful": bool, "reason": str}`；
  - 动作（`useful == False`）：
    1. 写入 `low_confidence_questions` 表；
    2. 流式输出礼貌拒答：“非常抱歉，当前知识库中暂未收录相关信息，已为您登记至后台人工处理...”；
    3. 截断后续生成，禁止硬编。
- **阶段 2：受控生成与引用角标注入（`useful == True`）**
  - **负面知识约束**：
    - 严禁承诺退款到账精确时间（只能表明平台审核后原路退回，到账以发卡行为准）；
    - 严禁承诺私下额外赔付或非规则赠品；
    - 严禁根据常识随意无据外推。
  - **角标规范**：每一句陈述必须后接 `[1]` 或 `[1][2]`，且序号必须存在于喂入的证据序号列表中。
  - **流式事件**：
    1. 生成前发射 `{"event_type": "citations", "citations": [...]}`；
    2. 逐 token 发射 `{"event_type": "text", "content": chunk}`。

---

### 3.4 评估体系与个案台账设计

#### 3.4.1 评估集规格（`tests/data/eval_ch04.jsonl`）
- 总量 300 题，5 大分桶：
  1. `A_policy` (60题): 基础政策条款，验证语义召回；
  2. `B_model` (60题): 具体型号与专有名词（如 PRO-X99），检验 BM25 优势；
  3. `C_colloquial` (60题): 口语模糊提问，检验 Query 改写；
  4. `E_multi` (60题): 跨条目复杂问题，检验混合与重排；
  5. `D_absent` (60题): 知识库外与低置信度问题（`should_refuse=True`），检验自评拒答入池。

#### 3.4.2 评测指标与四策略切流
- **四种策略**：
  1. `vector_only` (Dense Top-10)
  2. `bm25_only` (Milvus BM25 Top-10)
  3. `hybrid` (Dense 50 + BM25 50 RRF Top-10)
  4. `hybrid_rerank` (Dense 50 + BM25 50 RRF Top-50 -> BGE-Reranker-v2-m3 Top-10)
- **检索段指标**：
  - $\text{Recall@K} = \frac{|\text{Retrieved@K} \cap \text{ExpectSection}|}{|\text{ExpectSection}|}$ ($K \in \{3, 5, 10\}$)
  - $\text{MRR} = \frac{1}{\text{rank}_{\text{first\_hit}}}$
- **生成段指标**：
  - $\text{Faithfulness}$（忠实度）：LLM-as-a-Judge 评估关键陈述是否有证据支持，提取编造陈述与原因；
  - 编造判定入库：自动触发 `faith_cases` 跨轮幂等更新（已解决自动回退未解决）。
- **执行脚本**：`scripts/run_ch04_evaluation.py`，生成报告至 `reports/ch04_evaluation_report.md`。

---

### 3.5 前端交互设计（Vibe Coding）

#### 3.5.1 引用编号点击与抽屉溯源
- 解析文本中的 `\[(\d+)\]` 正则，渲染为样式化微标签 `<span class="cite-pill" onclick="...">[1]</span>`；
- 点击后右侧滑出抽屉或浮层，展示证据序号、所属章节路径、标准问题与回答原文，支持便捷关闭。

#### 3.5.2 满意度反馈（👍/👎）
- 位于每条 Bot 消息左下角；
- 点击任一按钮即高亮所选状态、淡入呈现「已反馈」标签并同时设置 `disabled` 一次性锁定；
- 纯前端采集状态，预留数据飞轮上报 Hook。

---

## 4. 验收标准与验证方案

1. **四策略对比报告能跑出数字**：执行 `python scripts/run_ch04_evaluation.py`，产出包含 4 策略在 5 个分桶下的 Recall@K、MRR 与 Faithfulness 完整对比矩阵；
2. **问带具体型号的问题 BM25 那路能命中**：测试 `B_model` 型号题目（如 PRO-X99），验证 BM25 召回命中真实 chunk；
3. **答案引用编号能定位回原文**：聊天界面输出 `[1]`，点击后能精准展开对应 chunk 的章节路径与原文；
4. **问知识库没有的内容得到明确拒答且入池**：测试 `D_absent` 或超纲提问，得到标准拒答，且 `low_confidence_questions` 表成功新增记录；
5. **全量单测通过**：包括既有 121 项测试及新增的各模块单元测试，100% PASS。
