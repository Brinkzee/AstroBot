# Task 5 Execution Report: 历史客服对话知识挖掘管道

## 1. 执行概述
- **任务目标**: 构建从客服历史对话流水分批抽取问答对、落入 `qa_extraction_staging` 中转暂存表、执行规则精确去重与 BGE-M3 语义聚类去重、最终将保留项无缝转入知识库双写落库的完整离线挖掘管道。
- **执行状态**: DONE
- **Git Commit**: `c36c6aedaf54033d2442178525c99e949e409dab`
- **提交信息**: `feat(rag): add dialogue qa mining pipeline with staging and semantic deduplication`
- **测试结果**:
  - `pytest tests/test_rag_miner.py -v`: 7/7 passed (100%)
  - 全量回归 `pytest`: 95/95 passed (100%)

---

## 2. TDD 流程执行证据

### 步骤 1: 编写失败测试 (RED)
- 新建测试文件 `tests/test_rag_miner.py`，覆盖：
  1. `test_qa_extraction_pydantic_schema_and_parser`: 验证 Prompt Pydantic Schema (`ExtractedQAPair`, `ExtractedQAList`)，以及容错解析器对标准 JSON 数组、带 Markdown 代码块、`{"items": [...]}` 包装格式与异常文本的处理能力；
  2. `test_fetch_dialogue_batches_grouping`: 验证按 `conversation_id` 聚合对话流水、过滤 tool 中间消息、多会话分批打包以及生成唯一批次号 `BATCH_{timestamp}_{idx}`；
  3. `test_extract_and_stage`: 验证调用大模型结构化抽取问答并即刻批量落入 `qa_extraction_staging` 表（初始状态 `extracted`）；
  4. `test_deduplicate_staging_exact_match`: 验证规则层精确去重，相同问题保留答案字数更长更详尽的问答对，重复项置为 `discarded`；
  5. `test_deduplicate_staging_semantic_similarity`: 验证语义层 BGE-M3 向量相似度聚类去重（余弦相似度 $\ge 0.92$ 判定为同义重复，保留详尽答案，标记 `kept` / `discarded`）；
  6. `test_ingest_kept_chunks`: 验证将 `status = 'kept'` 的条目映射为 `DocChunk`（`category="历史问答挖掘"`, `content_type="mined_qa"`），并通过 `KnowledgeDualWriter` 双写落入 MySQL 与 Milvus；
  7. `test_run_mining_pipeline_end_to_end`: 验证 CLI 脚本主流程 `run_mining_pipeline` 端到端集成运行。

### 步骤 2: 验证测试失败 (RED Verification)
- 运行 `pytest tests/test_rag_miner.py -v`
- 结果: `ModuleNotFoundError: No module named 'app.prompts.qa_extraction'`
- 确认因为核心代码尚未实现而预期失败。

### 步骤 3: 编写最小实现代码 (GREEN)
- `app/prompts/qa_extraction.py`:
  - 定义 `ExtractedQAPair` 与 `ExtractedQAList` Pydantic 模型；
  - 编写 `QA_EXTRACTION_SYSTEM_PROMPT`，强化剔除寒暄与客套、脱敏隐私（单号、姓名、地址等）、提炼通用问答以及强制输出标准 JSON 格式；
  - 实现 `qa_extraction_prompt` (LangChain ChatPromptTemplate)；
  - 实现健壮的 `parse_qa_extraction_output`，优雅解析各类合法及代码块包裹的 JSON。
- `app/services/rag/miner.py`:
  - 实现 `DialogueKnowledgeMiner` 核心类；
  - `fetch_dialogue_batches`: 从 `Message` 中提取有效对话并按会话分组，分批赋予唯一批次号 `BATCH_{timestamp}_{idx}`；
  - `_call_llm_for_qa`: 封装统一 LLM 提示词执行接口，兼顾异步与测试 Mock；
  - `extract_and_stage`: 解析大模型抽取结果并批量写入 `qa_extraction_staging`；
  - `extract_batch_dialogues`: 批量驱动会话抽取与暂存写入；
  - `deduplicate_staging`: 两阶段去重（规则层文本清洗与精确去重 + 语义层 BGE-M3 向量余弦相似度 $\ge 0.92$ 聚类去重），优先保留详细答案，分别标记 `status` 为 `kept` 与 `discarded`；
  - `ingest_kept_chunks`: 过滤 `status = 'kept'` 记录，转化为 `content_type="mined_qa"` 的 `DocChunk`，调用 `KnowledgeDualWriter` 双写落库。
- `app/services/rag/__init__.py`:
  - 导出 `DialogueKnowledgeMiner`，纳入 `__all__`。
- `scripts/mine_dialogues.py`:
  - 实现完整的 CLI 调度工具，提供 `run_mining_pipeline` 核心函数与命令行入口，支持 `--step all/extract/dedup/ingest` 分阶段或全量调度。

### 步骤 4: 验证测试全部通过 (GREEN Verification)
- 运行 `pytest tests/test_rag_miner.py -v`:
  - `test_qa_extraction_pydantic_schema_and_parser PASSED`
  - `test_fetch_dialogue_batches_grouping PASSED`
  - `test_extract_and_stage PASSED`
  - `test_deduplicate_staging_exact_match PASSED`
  - `test_deduplicate_staging_semantic_similarity PASSED`
  - `test_ingest_kept_chunks PASSED`
  - `test_run_mining_pipeline_end_to_end PASSED`
  - 7 passed in 7.94s (100%).
- 运行全量测试套件 `pytest`:
  - 95 passed in 19.79s (从原先 88 passed 增至 95 passed，0 失败，100% 通过)。

### 步骤 5: Git 提交
- 命令:
  ```bash
  git add app/prompts/qa_extraction.py app/services/rag/ scripts/mine_dialogues.py tests/test_rag_miner.py
  git commit -m "feat(rag): add dialogue qa mining pipeline with staging and semantic deduplication"
  ```
- Commit Hash: `c36c6aedaf54033d2442178525c99e949e409dab`

---

## 3. 交付清单
- `app/prompts/qa_extraction.py` (新增)
- `app/services/rag/miner.py` (新增)
- `app/services/rag/__init__.py` (修改 - 导出 `DialogueKnowledgeMiner`)
- `scripts/mine_dialogues.py` (新增)
- `tests/test_rag_miner.py` (新增)

---

## 4. Return Contract
- **Status**: DONE
- **Commits**: `c36c6aedaf54033d2442178525c99e949e409dab`
- **Test summary**: 95/95 passing (tests/test_rag_miner.py: 7/7 passing)
- **Concerns**: None
