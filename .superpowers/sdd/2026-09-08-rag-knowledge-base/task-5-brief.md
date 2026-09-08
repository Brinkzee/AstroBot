# Task 5 Brief: 历史客服对话知识挖掘管道

## 1. 任务目标
构建从客服历史对话流水分批抽取问答对、落入 `qa_extraction_staging` 中转暂存表、执行规则与 BGE-M3 语义聚类去重、最终将保留项无缝转入知识库双写落库的完整离线挖掘管道。

## 2. 涉及文件
- Create: `app/prompts/qa_extraction.py`
- Create: `app/services/rag/miner.py`
- Create: `scripts/mine_dialogues.py`
- Modify: `app/services/rag/__init__.py` (导出 `DialogueKnowledgeMiner`)
- Test: `tests/test_rag_miner.py`

## 3. 全局约束与业务规范
- 分批防串味：按会话 `conversation_id` 聚合流水，每批打上唯一 `batch_no`，分批喂给 LLM，杜绝上下文串扰；
- 暂存表隔离：抽取结果第一时间写入 `qa_extraction_staging`（初始状态 `extracted`），将长时 LLM 抽取与后续去重彻底解耦；
- 整体去重算法：
  - 规则层：文本清洗与精确去重；
  - 语义层：BGE-M3 向量化计算问题两两余弦相似度，相似度 $\ge 0.92$ 视为同义重复，保留答案更详尽的问答对，状态标记为 `kept`，重复项标记为 `discarded`；
- 入库转化：将 `status = 'kept'` 的条目映射为 `DocChunk`（`content_type="mined_qa"`，`questions` 填真实问法），调用 `KnowledgeDualWriter` 双写落库；
- 遵循 TDD：先写测试 -> 验证失败 -> 写最小实现 -> 验证全部通过。

## 4. 关键接口与功能要求

### 4.1 抽取 Prompt 与数据 Schema (`app/prompts/qa_extraction.py`)
- Pydantic 模型：`ExtractedQAPair(question: str, answer: str)`、`ExtractedQAList(items: List[ExtractedQAPair])`；
- Prompt 明确指令：剔除日常寒暄（“你好”、“在吗”等），脱敏去除具体单号/人名/地址等隐私，提炼出标准通用问答对，强制输出规范 JSON。

### 4.2 `DialogueKnowledgeMiner` (`app/services/rag/miner.py`)
- `__init__(llm: Optional[Any] = None, embedding_client: Optional[BGEEmbeddingClient] = None)`
- `async fetch_dialogue_batches(session: AsyncSession, batch_size: int = 20) -> List[Tuple[str, List[Dict]]]`:
  - 从 `conversations` 与 `messages` 聚合每个会话的有效角色消息（user, assistant）；
  - 每 `batch_size` 个会话打包为一个批次，生成唯一 `batch_no = f"BATCH_{timestamp}_{idx}"`。
- `async extract_and_stage(session: AsyncSession, dialogue_text: str, source_ref: str, batch_no: str) -> List[QAExtractionStaging]`:
  - 触发 LLM 抽取（支持注入 Mock 或真实 LLM），Pydantic 校验解析后批量写入 `qa_extraction_staging`（状态 `extracted`）。
- `async deduplicate_staging(session: AsyncSession, batch_no: Optional[str] = None, similarity_threshold: float = 0.92) -> Tuple[int, int]`:
  - 查询暂存表中 `status = 'extracted'` 记录；
  - 执行文本规则清洗 + BGE-M3 余弦相似度比较（$\ge 0.92$）；
  - 更新数据库状态：优质保留项置为 `'kept'`，重复项置为 `'discarded'`；
  - 返回 `(kept_count, discarded_count)`。
- `async ingest_kept_chunks(session: AsyncSession, dual_writer: KnowledgeDualWriter, batch_no: Optional[str] = None) -> int`:
  - 查询暂存表中 `status = 'kept'` 记录；
  - 转为 `DocChunk` 实体列表，调用 `dual_writer.write_chunks(session, chunks)` 完成双写落库；
  - 返回入库成功的知识块数量。

### 4.3 离线抽取脚本 (`scripts/mine_dialogues.py`)
- 完整的 CLI 调度工具，串联读取、分批抽取、暂存落库、聚类去重与最终知识入库。

## 5. 验收标准
- `pytest tests/test_rag_miner.py -v` 100% 通过；
- 测试必须包含：Prompt JSON Schema 校验、分批会话聚合、暂存表抽取落库、相似度聚类去重流转（kept / discarded 状态划分）、以及 kept 转入知识库双写全流程；
- 全量回归保持 100% 通过。
