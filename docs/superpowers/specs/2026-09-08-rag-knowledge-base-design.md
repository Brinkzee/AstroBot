# 系统设计规范 (Design Spec) - 第三章：RAG 基础与知识库向量语义检索升级

## 1. 背景与目标
在第二章中，客服系统通过 Function Calling 工具链实现了查数据与业务操作闭环，其中 `query_faq` 工具采用 SQL LIKE 关键词查表实现。关键词匹配面对口语化、同义替换表达（例如用户问“邮费是多少”而知识库记录的是“运费规范”）时极易发生召回落空。

本章通过 Superpowers 流程为 AstroBot 智能客服系统构建正式的 RAG 知识库子系统，将 `query_faq` 的内部实现从关键词查表升级为 BGE-M3 Dense 向量语义检索，同时保持对外工具出入参契约 100% 不变。

---

## 2. 核心技术选型与运行环境

- **向量嵌入模型 (Embedding)**：`BAAI/bge-m3`（1024 维 Dense 语义向量），通过 `huggingface_hub.InferenceClient` 驱动，使用 `.env` 中的 `HUGGINGFACE_TOKEN`。
- **向量数据库 (Vector DB)**：`Milvus-Lite` (`milvus_lite` + `pymilvus` 3.x 本地嵌入式存储，持久化到 `./data/milvus/astro_bot.db`)，集合名 `knowledge`，度量方式 `COSINE`。
- **原文权威源 (Authoritative Source)**：MySQL 8.0 (表 `knowledge_chunks` 与暂存表 `qa_extraction_staging`)，通过 SQLAlchemy 2.0 异步引擎访问。
- **对话挖掘驱动 (LLM)**：兼容 OpenAI 协议的模型接口（基于 `.env` 中的大语言模型配置），输出结构化 JSON 问答对。
- **本章明确不包含**：关键词召回、稀疏向量/混合检索、重排 (Reranker)；本章专注跑通 Dense 向量单路检索体系。

---

## 3. 数据层与表结构规范 (对齐 `sql/ch03-ddl.sql`)

### 3.1 MySQL 数据表规范

1. **`knowledge_chunks` 表 (知识库原文权威源)**
   - `id`: BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY (与 Milvus 集合主键 1:1 对齐)
   - `category`: VARCHAR(255) NOT NULL (分类 / 上级标题路径，参与向量化)
   - `questions`: TEXT NOT NULL (真实问法或本节标题，多个问法换行分隔，参与向量化)
   - `answer`: TEXT NOT NULL (正文答案，参与向量化)
   - `section_path`: VARCHAR(512) NULL (章节溯源路径，元数据，不进向量)
   - `content_type`: VARCHAR(32) NULL (内容类型: faq / policy / manual / mined_qa 等，元数据)
   - `is_key_clause`: TINYINT(1) NOT NULL DEFAULT 0 (是否关键/免责条款，元数据)
   - `prev_chunk_id`: BIGINT UNSIGNED NULL (前一块指针，元数据，外键自引用)
   - `next_chunk_id`: BIGINT UNSIGNED NULL (后一块指针，元数据，外键自引用)
   - `vector_id`: VARCHAR(64) NULL (Milvus 集合写入后回填的向量主键)
   - `vectorize_status`: ENUM('pending', 'done') NOT NULL DEFAULT 'pending' (双写状态与断点续跑标识)
   - `created_at` / `updated_at`: DATETIME

2. **`qa_extraction_staging` 表 (历史对话抽取问答中转暂存表)**
   - `id`: BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY
   - `batch_no`: VARCHAR(64) NOT NULL (抽取批次号，如 `BATCH_20260908_001`)
   - `source_ref`: VARCHAR(255) NULL (来源会话标识，如 `conv_1001`)
   - `question`: TEXT NOT NULL (LLM 从会话中提炼的用户提问)
   - `answer`: TEXT NOT NULL (LLM 从会话中提炼的客服标准回答)
   - `status`: ENUM('extracted', 'kept', 'discarded') NOT NULL DEFAULT 'extracted'
   - `created_at`: DATETIME

### 3.2 Milvus `knowledge` 集合规范

- **Collection Name**: `knowledge`
- **Fields**:
  - `id`: INT64, Primary Key, `auto_id=False` (数值对齐 `knowledge_chunks.id`)
  - `vector`: FLOAT_VECTOR, `dim=1024`
  - `category`: VARCHAR(255)
  - `content_type`: VARCHAR(32)
  - `is_key_clause`: BOOL
  - `chunk_text`: VARCHAR(4096) (存储拼接后的向量文本，便于召回时直接读取)
- **Index**:
  - Index Type: `FLAT`
  - Metric Type: `COSINE`

---

## 4. 离线建库·结构感知文档切分引擎 (`app/services/rag/splitter.py`)

### 4.1 标题层级感知与结构构建
- 维护 Markdown 标题栈（`#` 至 `######`）；
- 若当前小节为政策/规则/手册（无天然问题）：
  - `category` = 祖先标题路径（如 `售后指南 > 退货退款政策`）；
  - `questions` = 当前章节标题（如 `退货运费补贴规则`）；
  - `content_type` = `'policy'` 或 `'manual'`；
- 若解析到 Markdown 形式的问答（`问：... 答：...` 或 `Q: ... A: ...`）：
  - `questions` = 真实问法；
  - `content_type` = `'faq'`；
- 关键条款识别：标题或正文命中【重要提示】、【特别说明】、【不可退换】等关键词时，标记 `is_key_clause = 1`。

### 4.2 超长内容递归切分与句号对齐重叠 (不留半截话)
- 设定单块预算 `chunk_size = 500` 字符，重叠窗口 `overlap_size = 80` 字符；
- 递归分块优先级：`\n\n`（自然段） $\to$ `\n`（行） $\to$ 句子标点；
- **句号对齐切分算法**：
  - 在重叠滑动窗口区域 `[chunk_end - overlap_size, chunk_end]` 内，向前后扫描最近的断句标点（`。`、`！`、`？`、`；`、`\n`）；
  - 切分点强制对齐并锚定在该断句标点后，保证前一个 Chunk 的结尾与后一个 Chunk 的重叠起始段皆为完备语句，彻底杜绝断章腰斩。

### 4.3 大表格按行切分与表头强制广播复制
- 精准识别 Markdown 表格块（首行 Header + 次行 Divider `|---|---|` + 数据行 Row）；
- 当表格行数过多超过 chunk 大小时，按行拆分为多个子 chunk；
- **强制规则**：每个拆分出来的表格子 chunk 最上方均完整复制原表格的表头与分隔行，紧跟对应的数据行片段，确保任意切块单独向量化时语义不丢失字段含义。

### 4.4 块间前后指针绑定
- 切分产出的 Chunk 序列在落库前建立内存双向链表，在 MySQL 写入生成自增主键后，两阶段回填更新 `prev_chunk_id` 与 `next_chunk_id`。

---

## 5. 离线建库·历史对话挖掘管道 (`app/services/rag/miner.py`)

### 5.1 分批读取会话
- 从 MySQL `conversations` 与 `messages` 中按 `conversation_id` 聚合有效对话流水；
- 设定分批窗口（每批 10~20 个完整会话），生成唯一批次号 `batch_no = f"BATCH_{timestamp}_{idx}"`，避免上下文溢出与多会话信息混淆。

### 5.2 LLM 结构化抽取与暂存
- 编写专用 Prompt (`app/prompts/qa_extraction.py`)，指令大模型剥离口语闲聊与敏感单号，提炼通用 Q&A；
- 强制模型输出 JSON 数组 `[{"question": "...", "answer": "..."}]` 并由 Pydantic 校验；
- 抽取结果即刻批量写入 `qa_extraction_staging` 表，初始状态为 `status = 'extracted'`。

### 5.3 整体去重与转入知识库
- 抽取阶段结束后，触发整体去重任务：
  1. 文本规范化与精确去重；
  2. BGE-M3 语义相似度聚类去重（余弦相似度 $\ge 0.92$ 的同义问题归并）；
- 质量较高条目更新暂存表为 `status = 'kept'`，重复或无价值条目置为 `status = 'discarded'`；
- 将 `status = 'kept'` 的记录映射为待入库 Chunk（`category = '历史问答挖掘'`, `content_type = 'mined_qa'`），提交双写管道。

---

## 6. 双写落库与断点续跑补齐 (`app/services/rag/dual_writer.py`)

### 6.1 向量化文本拼接
- 待向量化输入严格拼装为一段密集语义文本：
  ```python
  text_to_embed = f"分类：{category}\n问题：{questions}\n内容：{answer}"
  ```
- 其余元数据仅保存在数据库和 Milvus Payload，不输入向量模型。

### 6.2 严格两阶段双写落库
1. **阶段 1 (MySQL 权威源)**：
   - 插入 `knowledge_chunks` 表，置 `vectorize_status = 'pending'`, `vector_id = NULL`；
   - 提交后获得全局唯一自增 `id`，更新绑定相邻块的 `prev_chunk_id` / `next_chunk_id`。
2. **阶段 2 (BGE-M3 向量化)**：
   - 调用 `BGEEmbeddingClient.embed_documents(...)` 批量生成 1024 维向量；
   - 内置重试机制抵御瞬时网络抖动。
3. **阶段 3 (Milvus 集合 Upsert)**：
   - 以 MySQL 的 `id` 直接作为 Milvus 的 `id`，执行 `upsert`；
   - 保证主键幂等，即使重复写入也不会出现重复向量。
4. **阶段 4 (状态回填转 `done`)**：
   - 执行 `UPDATE knowledge_chunks SET vector_id = str(id), vectorize_status = 'done' WHERE id = :id`。

### 6.3 断点续跑与异常捡起补偿 (`repair_pending_chunks`)
- 提供自愈函数 `repair_pending_chunks(session)`：
  - 扫描 `SELECT * FROM knowledge_chunks WHERE vectorize_status = 'pending'`；
  - 将所有遗留块重新送入 BGE-M3 向量化并 upsert 到 Milvus，回填 `vector_id` 并将状态更新为 `'done'`；
- 无论建库脚本在任何时间点异常崩溃中断，下次重跑均能 100% 自动捡起补齐，达到数据最终一致性。

---

## 7. 在线语义检索与 `query_faq` 无缝升级 (`app/services/rag/retriever.py`)

### 7.1 工具契约零变更
- 工具函数签名、Docstring、参数注解保持 100% 兼容：
  ```python
  @tool
  async def query_faq(keyword: str) -> str:
      """查询常见问题解答 (FAQ) 知识库。当用户咨询平台规则、退换货政策、发票开具、运费标准等常见业务规范时调用此工具。

      Args:
          keyword: 检索关键词，例如 '退货'、'运费'、'发票' 等
      """
  ```
- 返回字符串输出格式保持一致：
  ```text
  1. 问：{questions}
     答：{answer}

  2. 问：...
     答：...
  ```
  未命中时返回：`未找到与【{keyword}】相关的常见问题解答。`

### 7.2 内部检索流改造
- 废弃原 SQL `WHERE question LIKE %kw% OR answer LIKE %kw%`；
- 执行流程：
  1. `query_vector = await embedding_client.aembed_query(keyword)`；
  2. `milvus_client.search("knowledge", data=[query_vector], limit=3, output_fields=...)`；
  3. 过滤相似度低于阈值（如 `0.35`）的噪声结果；
  4. 组装 Top-3 结果为标准序号问答文本返回给 Agent 回灌。

---

## 8. 验证计划与验收标准覆盖

### 8.1 自动化测试矩阵
- `tests/test_rag_splitter.py`：单测验证 Markdown 标题层级感知、句号对齐重叠不留半截话、大表格按行拆分且每块强制复制表头、关键条款标记。
- `tests/test_rag_miner.py`：单测验证分批对话流水聚合、LLM 问答抽取 Schema 校验、暂存表落库、相似度与规则去重。
- `tests/test_rag_dual_writer.py`：单测验证 MySQL 与 Milvus 两阶段双写流程、状态翻转、主键幂等覆盖。
- `tests/test_rag_resume.py`：针对验收标准 2，故意模拟中断制造 `pending` 状态块，调用恢复补齐程序，验证漏向量化的块被正确补全且状态转 `done`。
- `tests/test_rag_retriever.py`：单测验证 BGE-M3 Dense 检索与 `query_faq` 工具适配。

### 8.2 业务验收标准覆盖
1. **验收标准 1 (语义泛化召回)**：
   - 运行端到端测试，输入“邮费是多少”等换说法提问；
   - 验证 `query_faq` 不再发生 SQL LIKE 查表落空，成功从向量库召回运费说明，模型最终答复包含正确运费规则。
2. **验收标准 2 (断点补齐)**：
   - 运行建库中断模拟脚本，打断流程后重跑，验证漏向量化块全部补齐，双写最终一致。
