# Task 4 Brief: 双写落库与断点续跑管理器

## 1. 任务目标
实现 MySQL `knowledge_chunks` 原文权威源与 Milvus-Lite `knowledge` 向量库的四阶段严格双写落库、双向链表前后指针绑定，以及断点重跑补偿修复机制 `repair_pending_chunks`，确保意外中断重跑后能够自动捡起所有遗留的 `pending` 块并最终一致。同时提供离线构建入口 `scripts/build_knowledge_base.py`。

## 2. 涉及文件
- Create: `app/services/rag/dual_writer.py`
- Create: `scripts/build_knowledge_base.py`
- Modify: `app/services/rag/__init__.py` (导出 `KnowledgeDualWriter`)
- Test: `tests/test_rag_dual_writer.py`
- Test: `tests/test_rag_resume.py`

## 3. 全局约束与业务规范
- 向量化拼装严格规范：`f"分类：{category}\n问题：{questions}\n内容：{answer}"`，其余元数据（`section_path`, `content_type`, `is_key_clause`, `prev_chunk_id`, `next_chunk_id`）只落库与存 Payload，绝不进向量化文本；
- 状态机流转：初始固定为 `'pending'`，Milvus 成功落库后回填 `vector_id = str(chunk.id)` 并更新为 `'done'`；
- 幂等性：以 MySQL 自增 `id` 作为 Milvus 的 `id`，Milvus 执行 `upsert` 具备天然主键幂等性；
- 遵循 TDD：先编写双写与断点自愈失败测试 -> 验证失败 -> 编写实现 -> 验证测试通过。

## 4. 关键接口与功能要求

### 4.1 文本拼接标准
```python
def compose_embedding_text(category: str, questions: str, answer: str) -> str:
    return f"分类：{category}\n问题：{questions}\n内容：{answer}"
```

### 4.2 `KnowledgeDualWriter` (`app/services/rag/dual_writer.py`)
- `__init__(store: Optional[MilvusKnowledgeStore] = None, embedding_client: Optional[BGEEmbeddingClient] = None, milvus_uri: Optional[str] = None)`
- `async write_chunks(session: AsyncSession, chunks: List[DocChunk]) -> List[KnowledgeChunk]`:
  1. 批量保存到 `knowledge_chunks` 表，状态全部设为 `'pending'`；
  2. `await session.flush()` 获取生成的自增 `id`，并按顺序更新 `prev_chunk_id` 与 `next_chunk_id`；
  3. `await session.commit()` 确立 MySQL 原文权威源落库成功；
  4. 遍历这批 chunk，使用 `compose_embedding_text` 组装文本，调用 `embedding_client.aembed_documents` 获取 1024 维向量；
  5. 组装 Milvus records（`id=chunk.id`, `vector=vec`, `category`, `questions`, `answer`, `chunk_text`, `content_type`, `is_key_clause`），调用 `store.upsert(records)`；
  6. 更新 MySQL: `vector_id = str(chunk.id)`, `vectorize_status = 'done'`，提交事务；
  7. 返回落库成功的实体列表。
- `async repair_pending_chunks(session: AsyncSession) -> int`:
  - 查询 `SELECT * FROM knowledge_chunks WHERE vectorize_status = 'pending'`；
  - 若无待处理块，返回 0；
  - 若有，调用嵌入模型生成向量，upsert 到 Milvus，回填 `vector_id` 并更新状态为 `'done'`；
  - 提交事务并返回修复的块数量。

### 4.3 `scripts/build_knowledge_base.py`
- 扫描指定目录（默认 `data/kb/`）下的所有 `.md` 文件；
- 使用 `MarkdownStructureSplitter` 解析切分为 `DocChunk`；
- 调用 `KnowledgeDualWriter.write_chunks` 完成双写；
- 任务执行前后自动调用 `repair_pending_chunks` 确保遗留块清零；
- 命令行支持 `--kb-dir`, `--clean`, `--repair-only` 等参数。

## 5. 验收标准
- `pytest tests/test_rag_dual_writer.py tests/test_rag_resume.py -v` 100% 通过；
- 测试验证重点：
  1. 完整双写流程后，MySQL 状态全部为 `'done'`，且 Milvus 中记录数一致；
  2. 中断自愈验证：人为制造 `pending` 块，执行 `repair_pending_chunks` 后，漏向量化的块被自动捡起补齐，状态转为 `'done'`（对齐验收标准 2）；
- 全量回归保持 100% 通过。
