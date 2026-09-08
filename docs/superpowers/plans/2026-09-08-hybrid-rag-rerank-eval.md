# 实施计划 (Implementation Plan) - 第四章：RAG 进阶 · 混合检索、重排与质量评估体系

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 Milvus 2.5 原生 BM25 全文检索、BGE-M3 密集向量及 RRF 融合实现双路混合召回，接入 BGE-Reranker-v2-m3 精排与首尾放置上下文编排，配合前置知识充分度自检、拒答入池与负面知识控制，依托 `tests/data/eval_ch04.jsonl` 构建四策略对比评估体系与 `faith_cases` 编造台账，并在前端提供可点击引用溯源抽屉与点赞/点踩点亮锁定交互。

**Architecture:** 采用高内聚、易单测的模块化分层流水线架构。Query 理解内聚在检索管道内部，Milvus 原生 BM25 与 Dense 并行召回 Top-50 并经 RRF 融合，交叉编码重排输出 Top-10 并按首尾放置（Lost in the Middle）编排证据元数据；生成层两阶段受控判别（自检不足入 `low_confidence_questions` 并拒答，充足则流式生成带角标答案）；离线评测支持四策略对比与裁判忠实度评估，编造个案幂等落库 `faith_cases`；前端以 Vibe Coding 模式快速交付角标与反馈交互。

**Tech Stack:** FastAPI, SQLAlchemy 2.0 (Async), MySQL 8.0, PyMilvus 3.0.1 / Milvus-Lite (BM25 Function + jieba analyzer, RRFRanker), Hugging Face Hub (BAAI/bge-reranker-v2-m3 & BAAI/bge-m3), LangChain Core, Pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-hybrid-rag-rerank-eval-design.md`

## Global Constraints

- **Python 运行环境**: Python 3.12, 依赖已安装的 `pymilvus==3.0.1`, `milvus_lite==3.2.1`, `huggingface_hub==1.21.0`。
- **Milvus BM25 分词器**: 必须采用底层已验证支持的 `analyzer_params={"type": "jieba"}`，禁止使用非法的 `"chinese"` 类型。
- **重排模型**: 严格使用 `BAAI/bge-reranker-v2-m3`，基于 Hugging Face Router 批量请求 `inputs: [{"text": q, "text_pair": doc}, ...]`，并提供确定性本地 Mock 降级支持离线单测。
- **数据库与表结构**: 严格对齐 `sql/ch04_ddl.sql`，表名为 `low_confidence_questions` 与 `faith_cases`，外键关联 `conversations(id)`。
- **评估数据集**: 严格使用已存在的 `tests/data/eval_ch04.jsonl`（包含 300 题，5 大分桶）。
- **零破坏原则**: 现有 `query_faq` 工具与 `ChatService` 必须保持对外接口兼容，既有 121 个测试必须全部保持 PASS。

---

### Task 1: ORM 数据模型与 DDL 初始化基础设施 (`low_confidence_questions` & `faith_cases`)

**Files:**
- Create: `app/models/low_confidence.py`
- Create: `app/models/faith_case.py`
- Modify: `app/models/__init__.py`
- Create: `scripts/init_ch04_db.py`
- Test: `tests/test_ch04_models.py`

**Interfaces:**
- Consumes: `app.models.conversation.Conversation`, `app.db.session.Base`
- Produces: `LowConfidenceQuestion`, `FaithCase`, `record_low_confidence()`, `upsert_faith_case()`

- [ ] **Step 1: 编写数据模型失败测试**
编写 `tests/test_ch04_models.py`，验证：
1. `LowConfidenceQuestion` 模型各字段映射与 `source` 枚举验证（`retrieval_low_conf`, `self_check`, `user_feedback`）；
2. `FaithCase` 模型各字段映射、唯一键 `uk_eval_id` 约束、`citations` JSON 字段序列化；
3. `upsert_faith_case()` 幂等更新逻辑：新题新增（`seen_count=1`, `status='未解决'`），已存在题更新最新快照且 `seen_count += 1`，若原状态为 `'已解决'` 则自动回退为 `'未解决'` 并标记复发。

- [ ] **Step 2: 运行测试验证失败**
运行 `pytest tests/test_ch04_models.py -v`，预期报 `ImportError: cannot import name 'LowConfidenceQuestion'`。

- [ ] **Step 3: 实现模型与数据库初始化脚本**
1. 实现 `app/models/low_confidence.py`：定义 `LowConfidenceQuestion` ORM 类及便捷写入函数 `record_low_confidence(db, ...)`；
2. 实现 `app/models/faith_case.py`：定义 `FaithCase` ORM 类及幂等更新函数 `upsert_faith_case(db, ...)`；
3. 修改 `app/models/__init__.py`：导出新增模型；
4. 实现 `scripts/init_ch04_db.py`：读取并执行 `sql/ch04_ddl.sql`，提供幂等建表保障。

- [ ] **Step 4: 运行测试验证通过**
运行 `pytest tests/test_ch04_models.py -v`，确保全部测试 PASS。

- [ ] **Step 5: 提交代码**
```bash
git add app/models/low_confidence.py app/models/faith_case.py app/models/__init__.py scripts/init_ch04_db.py tests/test_ch04_models.py
git commit -m "feat(models): add low_confidence_questions and faith_cases models with ddl runner"
```

---

### Task 2: Milvus 2.5 混合集合 Schema 升级与全量数据迁移刷库

**Files:**
- Modify: `app/services/rag/milvus_client.py:38-150`
- Create: `scripts/reindex_ch04_knowledge.py`
- Test: `tests/test_rag_milvus_bm25.py`

**Interfaces:**
- Consumes: `app.models.knowledge.KnowledgeChunk`, `pymilvus.Function`, `pymilvus.FunctionType.BM25`, `pymilvus.RRFRanker`
- Produces: `MilvusKnowledgeStore.hybrid_search()`, `MilvusKnowledgeStore.search_dense()`, `MilvusKnowledgeStore.search_bm25()`

- [ ] **Step 1: 编写 Milvus BM25 与 Hybrid 检索失败测试**
编写 `tests/test_rag_milvus_bm25.py`，使用独立临时 sqlite/milvus-lite db：
1. 验证集合创建后包含 `chunk_text`（jieba 分词）、`sparse_vector`（BM25 函数输出）、`vector`（1024维）；
2. 插入测试数据后，`search_bm25("PRO-X99")` 能够精准命中包含型号的文本；
3. 验证 `hybrid_search(dense_vector, bm25_text, category_filter, top_k_per_route=50, limit=50)` 能正常通过 `RRFRanker(k=60)` 输出排序结果，且支持 `category` 标量过滤；
4. 验证 `search_dense(...)` 单路向量检索功能保持正常。

- [ ] **Step 2: 运行测试验证失败**
运行 `pytest tests/test_rag_milvus_bm25.py -v`，预期报缺少 `hybrid_search` 或 schema 字段不匹配错误。

- [ ] **Step 3: 升级 MilvusKnowledgeStore 并编写重建索引脚本**
1. 修改 `app/services/rag/milvus_client.py`：
   - 升级 `init_collection`：添加 `chunk_text` (VARCHAR, `enable_analyzer=True`, `analyzer_params={"type": "jieba"}`), `sparse_vector` (SPARSE_FLOAT_VECTOR), `bm25_function` (FunctionType.BM25)；为 `sparse_vector` 添加 `BM25` AUTOINDEX；
   - 实现 `hybrid_search(...)`：构造 dense 与 bm25 的 `AnnSearchRequest`，调用 `self.client.hybrid_search(..., ranker=RRFRanker(k=60))`；
   - 显式暴露 `search_dense(...)` 与 `search_bm25(...)`；
   - 保证原有 `upsert` 方法写入时若传入 `chunk_text` 则自动由 Milvus 内置分词器生成稀疏向量。
2. 编写 `scripts/reindex_ch04_knowledge.py`：从 MySQL `knowledge_chunks` 读取全量知识，重新格式化 `chunk_text`，重新生成 Dense 向量，批量写入 Milvus `knowledge` 集合。

- [ ] **Step 4: 运行测试验证通过**
运行 `pytest tests/test_rag_milvus_bm25.py -v`，验证全部测试 PASS。

- [ ] **Step 5: 提交代码**
```bash
git add app/services/rag/milvus_client.py scripts/reindex_ch04_knowledge.py tests/test_rag_milvus_bm25.py
git commit -m "feat(milvus): upgrade knowledge collection to native bm25 and hybrid search"
```

---

### Task 3: BGE-Reranker-v2-m3 重排客户端与 Lost in the Middle 首尾重排

**Files:**
- Create: `app/services/rag/reranker.py`
- Create: `app/services/rag/reorder.py`
- Test: `tests/test_rag_reranker.py`

**Interfaces:**
- Consumes: `huggingface_hub.InferenceClient`, candidate documents list
- Produces: `BGERerankerClient.rerank()`, `lost_in_the_middle_reorder()`, `build_citation_items()`

- [ ] **Step 1: 编写重排与首尾放置失败测试**
编写 `tests/test_rag_reranker.py`：
1. 测试 `BGERerankerClient` 在 Mock 模式下的排序行为（相关文本得分显著高于不相关文本）；
2. 测试网络异常重试与指数退避机制；
3. 测试 `lost_in_the_middle_reorder(docs)`：输入 10 条按相关度降序排列的文档 $[D_1, D_2, \dots, D_{10}]$，输出顺序严格符合 $[D_1, D_3, D_5, D_7, D_9, D_{10}, D_8, D_6, D_4, D_2]$（最高分在最前，次高分在最后）；
4. 测试 `build_citation_items(docs)` 输出标准的 `[{n, chunk_id, section_path, question, answer}]` 引用快照。

- [ ] **Step 2: 运行测试验证失败**
运行 `pytest tests/test_rag_reranker.py -v`，预期报 `ImportError: cannot import name 'BGERerankerClient'`。

- [ ] **Step 3: 实现 BGERerankerClient 与 reorder 算法**
1. 实现 `app/services/rag/reranker.py`：
   - 封装 `BGERerankerClient`，对接 Hugging Face Router 批量推理 `inputs: [{"text": q, "text_pair": doc}, ...]`；
   - 内置指数退避重试（3 次）与确定性 Mock 降级（基于词覆盖与 Jaccard 相似度打分）；
   - 提供 `async def rerank(query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]`。
2. 实现 `app/services/rag/reorder.py`：
   - 实现 `lost_in_the_middle_reorder(docs: List[Dict]) -> List[Dict]`；
   - 实现 `build_citation_items(docs: List[Dict]) -> List[Dict]`，为各文档分配 `n=1..K` 的角标并提取 `section_path`、问答原文。

- [ ] **Step 4: 运行测试验证通过**
运行 `pytest tests/test_rag_reranker.py -v`，验证全部测试 PASS。

- [ ] **Step 5: 提交代码**
```bash
git add app/services/rag/reranker.py app/services/rag/reorder.py tests/test_rag_reranker.py
git commit -m "feat(rerank): add bge-reranker-v2-m3 client and lost-in-the-middle reordering"
```

---

### Task 4: Query 理解模块 (口语改写归一 + 检索侧同义词扩展)

**Files:**
- Create: `app/prompts/query_understanding.py`
- Create: `app/services/rag/query_processor.py`
- Test: `tests/test_rag_query_processor.py`

**Interfaces:**
- Consumes: `app.llm.get_chat_model`
- Produces: `QueryProcessor.process()`, `QueryUnderstandingResult`

- [x] **Step 1: 编写 Query 理解失败测试**
编写 `tests/test_rag_query_processor.py`：
1. 测试口语化问法（如“我买了那个pro x99的手表，要是用着不顺心能退吗”）能被改写归一出含“星光PRO-X99”、“退货政策”的标准问法 `standard_query`；
2. 测试同义词与型号扩展 `expanded_keywords` 包含关键实体；
3. 测试拼接生成的 `bm25_query`（包含归一问题与扩展关键词）；
4. 测试大模型调用异常或 Mock 模式下的平滑降级（回退为原始 query）。

- [x] **Step 2: 运行测试验证失败**
运行 `pytest tests/test_rag_query_processor.py -v`，预期报 `ImportError: cannot import name 'QueryProcessor'`。

- [x] **Step 3: 实现 QueryProcessor 与 Prompt**
1. 实现 `app/prompts/query_understanding.py`：设计严格结构化 JSON 提示词，指导 LLM 输出 `standard_query` 与 `expanded_keywords`；
2. 实现 `app/services/rag/query_processor.py`：
   - 定义 `QueryUnderstandingResult` 数据类；
   - 实现 `QueryProcessor.aprocess(query: str) -> QueryUnderstandingResult`，异步调用轻量 LLM 并解析结构化结果；
   - 增加错误捕获与 Mock 兜底。

- [x] **Step 4: 运行测试验证通过**
运行 `pytest tests/test_rag_query_processor.py -v`，验证全部测试 PASS。

- [x] **Step 5: 提交代码**
```bash
git add app/prompts/query_understanding.py app/services/rag/query_processor.py tests/test_rag_query_processor.py
git commit -m "feat(rag): add query understanding processor for normalization and synonym expansion"
```

---

### Task 5: 进阶检索管道串联升级与 `query_faq` 工具适配

**Files:**
- Create: `app/services/rag/advanced_retriever.py`
- Modify: `app/tools/business_tools.py:100-140`
- Test: `tests/test_rag_advanced_retriever.py`

**Interfaces:**
- Consumes: `QueryProcessor`, `MilvusKnowledgeStore`, `BGEEmbeddingClient`, `BGERerankerClient`, `reorder`
- Produces: `AdvancedKnowledgeRetriever.retrieve_with_strategy()`, `AdvancedKnowledgeRetriever.retrieve_top_k()`, `query_faq` (升级实现但契约保持)

- [ ] **Step 1: 编写进阶检索管道失败测试**
编写 `tests/test_rag_advanced_retriever.py`：
1. 验证 `retrieve_with_strategy(query, strategy="hybrid_rerank")` 完整链路（Query改写 $\to$ Dense 50 + BM25 50 $\to$ RRF $\to$ BGE 重排 Top-10 $\to$ 首尾重排 $\to$ 带有 citations 元数据）；
2. 验证支持切换 4 种策略：`vector_only`, `bm25_only`, `hybrid`, `hybrid_rerank`，且出参结构完全一致；
3. 验证 `category_filter` 品类过滤生效；
4. 验证 `query_faq.ainvoke({"keyword": "..."})` 工具调用契约 100% 保持兼容，输出包含格式化问答文本与引用说明。

- [ ] **Step 2: 运行测试验证失败**
运行 `pytest tests/test_rag_advanced_retriever.py -v`，预期报缺少 `AdvancedKnowledgeRetriever`。

- [ ] **Step 3: 实现 AdvancedKnowledgeRetriever 与 query_faq 升级**
1. 实现 `app/services/rag/advanced_retriever.py`：
   - 编排完整 RAG 检索管线；
   - 提供 `retrieve_with_strategy(query, strategy, category_filter, top_k)` 统一接口；
   - 产出排布好的候选列表与 `citations` 快照；
2. 修改 `app/tools/business_tools.py` 中的 `query_faq`：将底层替换为 `AdvancedKnowledgeRetriever`，入参和出参文本保持不变。

- [ ] **Step 4: 运行测试验证通过**
运行 `pytest tests/test_rag_advanced_retriever.py -v` 及既有 `tests/test_business_tools.py`，验证全部通过。

- [ ] **Step 5: 提交代码**
```bash
git add app/services/rag/advanced_retriever.py app/tools/business_tools.py tests/test_rag_advanced_retriever.py
git commit -m "feat(retriever): implement advanced hybrid rag retriever with seamless query_faq tool upgrade"
```

---

### Task 6: 两阶段生成质量控制与自评守卫 (Self-Check, 拒答入池, 负面知识约束, 引用角标生成)

**Files:**
- Create: `app/prompts/rag_qa.py`
- Create: `app/services/rag/generator.py`
- Modify: `app/services/chat_service.py:140-310`
- Test: `tests/test_rag_generator.py`

**Interfaces:**
- Consumes: `AdvancedKnowledgeRetriever`, `LowConfidenceQuestion`, `ChatService`
- Produces: `RAGControlledGenerator.self_check()`, `RAGControlledGenerator.stream_generate()`, SSE `citations` 事件

- [x] **Step 1: 编写生成质量控制失败测试**
编写 `tests/test_rag_generator.py`：
1. 测试知识不足或超纲提问时，`self_check` 返回 `useful=False`，异步向 `low_confidence_questions` 表插入记录（`source='self_check'` 或 `'retrieval_low_conf'`，包含 `reason`），并返回显式拒答文本；
2. 测试知识充分时，`self_check` 返回 `useful=True`；
3. 测试受控生成 Prompt 包含严格的负面知识红线（禁止承诺退款到账精确时间、禁止承诺私下赔付）及引用角标格式规范 `[n]`；
4. 测试 `ChatService.stream_chat` 在触发知识库检索时，首帧下发 `{"event_type": "citations", "citations": [...]}`。

- [x] **Step 2: 运行测试验证失败**
运行 `pytest tests/test_rag_generator.py -v`，预期报缺少 `RAGControlledGenerator`。

- [x] **Step 3: 实现受控生成器与 ChatService 编排**
1. 实现 `app/prompts/rag_qa.py`：定义自检判别 Prompt 与带负面红线约束的引用问答 Prompt；
2. 实现 `app/services/rag/generator.py`：实现 `RAGControlledGenerator`，封装两阶段自检与流式生成逻辑；
3. 修改 `app/services/chat_service.py`：在执行知识工具后，若触发进阶 RAG 逻辑，发射 `citations` 事件，并应用自评拒答或受控流式生成。

- [x] **Step 4: 运行测试验证通过**
运行 `pytest tests/test_rag_generator.py -v` 及 `tests/test_chat_service.py`，确保全部测试 PASS。

- [x] **Step 5: 提交代码**
```bash
git add app/prompts/rag_qa.py app/services/rag/generator.py app/services/chat_service.py app/services/rag/__init__.py tests/test_rag_generator.py
git commit -m "feat(rag): add two-phase self-eval guardrails, refusal pool logging, and citation generation"
```

---

### Task 7: 评估体系构建与四策略对比评测 (`eval_ch04.jsonl`, Recall@K, MRR, Faithfulness, `faith_cases` 编造台账)

**Files:**
- Create: `app/services/rag/evaluator.py`
- Create: `scripts/run_ch04_evaluation.py`
- Test: `tests/test_rag_evaluator.py`

**Interfaces:**
- Consumes: `tests/data/eval_ch04.jsonl`, `AdvancedKnowledgeRetriever`, `FaithCase`
- Produces: `RAGEvaluator.evaluate_strategy()`, `RAGEvaluator.run_comparative_eval()`, `reports/ch04_evaluation_report.md`

- [x] **Step 1: 编写评估指标与台账落库失败测试**
编写 `tests/test_rag_evaluator.py`：
1. 验证 `Recall@K` 与 `MRR` 计算函数的数学准确性；
2. 验证 Faithfulness 裁判打分与编造判定逻辑；
3. 验证当某道题被判定为编造时，能调用 `upsert_faith_case` 幂等写入 `faith_cases` 表，保存完整 `citations` 快照；若该题原为「已解决」，能自动回退为「未解决」（复发追踪）。

- [x] **Step 2: 运行测试验证失败**
运行 `pytest tests/test_rag_evaluator.py -v`，预期报缺少 `RAGEvaluator`。

- [x] **Step 3: 实现 RAGEvaluator 与对比评测脚本**
1. 实现 `app/services/rag/evaluator.py`：
   - 加载 `tests/data/eval_ch04.jsonl`，按分桶统计；
   - 实现检索指标计算（基于 `expect_section` 命中判断）与 LLM 忠实度裁判；
   - 编造个案自动持久化至 `faith_cases` 表；
2. 实现 `scripts/run_ch04_evaluation.py`：
   - 支持命令行参数 `--strategies vector_only bm25_only hybrid hybrid_rerank`；
   - 支持 `--sample` 采样或全量跑；
   - 控制台渲染对比表格，并写入 `reports/ch04_evaluation_report.md`。

- [x] **Step 4: 运行测试验证通过**
运行 `pytest tests/test_rag_evaluator.py -v`，验证全部通过。

- [x] **Step 5: 提交代码**
```bash
git add app/services/rag/evaluator.py scripts/run_ch04_evaluation.py tests/test_rag_evaluator.py
git commit -m "feat(eval): add comparative evaluation suite and faith_cases tracking ledger"
```

---

### Task 8: 前端配套交互升级 (Vibe Coding: 可点引用编号浮层/抽屉 + 👍/👎 点亮锁定满意度反馈)

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes: SSE `event_type == 'citations'`, `event_type == 'text'`
- Produces: UI 引用编号可点角标、右侧抽屉式原文溯源卡片、消息左下角点赞/点踩点亮锁定交互

- [ ] **Step 1: 在 index.html 中实现 citations SSE 帧接收与角标转换**
1. 在 JS SSE 接收循环中捕获 `data.event_type === 'citations'`，将快照数据保存在对应消息 DOM 节点上；
2. 在流式文本渲染与完成时，使用正则 `replace(/\[(\d+)\]/g, ...)` 将 `[1]` 转换为可点击徽章：
   `<button class="cite-pill" onclick="openCitationDrawer(event, this, 1)">[1]</button>`。

- [ ] **Step 2: 实现引用抽屉 / 模态卡片 (Citation Drawer)**
1. 添加抽屉与半透明蒙层 HTML 与 CSS；
2. 点击角标时，根据角标序号从消息绑定的 citations 中查找对应条目，动态渲染：
   - 证据序号：`【证据 1】`；
   - 章节路径面包屑：`📚 章节：售后政策 > 退货流程`；
   - 知识库问答原文：标准问法、所属品类与解答内容；
3. 支持点击右上角关闭按钮或点击蒙层平滑收起。

- [ ] **Step 3: 实现每段回答左下角满意度反馈 (👍/👎 点亮锁定)**
1. 在每段 Bot 最终气泡的左下角渲染反馈容器：
   `<div class="feedback-bar"><button class="btn-fb-thumb up" onclick="handleFeedback(this, 'up')">👍</button><button class="btn-fb-thumb down" onclick="handleFeedback(this, 'down')">👎</button><span class="fb-tip"></span></div>`；
2. 实现 `handleFeedback(btn, type)`：
   - 点赞时高亮为品牌蓝，点踩时高亮为柔和红；
   - 旁边淡入显示「已反馈」文字；
   - 将当前回答下两个按钮同时设为 `disabled`，彻底锁定不可更改；
   - 纯前端采集状态，预留控制台日志与本地缓存。

- [ ] **Step 4: 本地网页端交互视觉走查**
打开浏览器访问 `http://127.0.0.1:8000`，实际发起提问，验证角标点击弹窗与点赞点踩锁定效果。

- [ ] **Step 5: 提交代码**
```bash
git add app/static/index.html
git commit -m "feat(ui): add clickable citation drawer and thumbs-up/down feedback locking"
```

---

### Task 9: 端到端综合业务验收测试 (四大验收标准系统化验证)

**Files:**
- Create: `tests/test_ch04_acceptance.py`

**Interfaces:**
- Consumes: 全量系统模块
- Produces: 4 大核心验收标准的完整自动化回归测试

- [ ] **Step 1: 编写全量端到端验收测试用例**
编写 `tests/test_ch04_acceptance.py`，严格对应用户四大验收标准：
1. **验收标准 1（四策略对比报告能跑出数字）**：运行评估套件子集，验证能成功计算出各策略的 Recall@K、MRR 与 Faithfulness 指标矩阵；
2. **验收标准 2（问带具体型号的问题 BM25 那路能命中）**：针对“星光PRO-X99智能手表充电规格”，验证 BM25 单路及 Hybrid 均能准确命中对应 chunk；
3. **验收标准 3（答案引用编号能定位回原文）**：验证回答文本包含 `[n]` 角标，且与下发的 `citations` 快照中 `n` 对应的 chunk 原文和 `section_path` 严格吻合；
4. **验收标准 4（问知识库没有的内容得到明确拒答且入池）**：针对超纲提问，大模型给出标准拒答语，且在 `low_confidence_questions` 表中查询到新插入的记录（`raw_question` 与 `reason` 明确）。

- [ ] **Step 2: 运行端到端验收测试**
运行 `pytest tests/test_ch04_acceptance.py -v`，验证 4 项验收标准全部 PASS。

- [ ] **Step 3: 运行全量测试套件回归验证**
运行 `pytest -v`，验证包含 Chapter 1~3 的 121 个既有测试以及 Chapter 4 的所有新增测试，达到 100% PASS。

- [ ] **Step 4: 提交代码**
```bash
git add tests/test_ch04_acceptance.py
git commit -m "test(acceptance): add end-to-end acceptance tests for ch04 requirements"
```
