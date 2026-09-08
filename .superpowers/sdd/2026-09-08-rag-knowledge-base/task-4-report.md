# Task 4 Execution Report: 双写落库与断点续跑管理器

## 1. 任务概述
- **任务目标**: 实现 MySQL `knowledge_chunks` 原文权威源与 Milvus-Lite `knowledge` 向量库的四阶段严格双写落库、双向链表前后指针绑定，以及断点重跑补偿修复机制 `repair_pending_chunks`，确保意外中断重跑后能够自动捡起所有遗留的 `pending` 块并达到最终一致。同时提供离线构建入口 `scripts/build_knowledge_base.py`。
- **执行分支**: `feat/ch03-rag-knowledge-base`
- **提交哈希**: `b6940b1fc49c9f6e5b4fdfa08e0e945632bb5ecd`

## 2. 变更文件清单
1. `app/services/rag/dual_writer.py`: 新增 `compose_embedding_text` 与 `KnowledgeDualWriter` 类；
2. `app/services/rag/__init__.py`: 导出 `KnowledgeDualWriter` 与 `compose_embedding_text`；
3. `scripts/build_knowledge_base.py`: 新增离线知识库构建与断点续跑命令行工具；
4. `tests/test_rag_dual_writer.py`: 编写覆盖文本拼接标准、空切片防御、四阶段双写、双向链表连结、Milvus 载荷存储、主键幂等覆写及建库脚本的 6 个测试用例；
5. `tests/test_rag_resume.py`: 编写覆盖空补偿、中断遗留 pending 块补偿捡起、混合状态隔离修复、写流程偶发失败自愈续跑的 4 个测试用例。

## 3. TDD 执行过程记录

### 3.1 RED 阶段 (失败用例确立)
- 编写测试文件 `tests/test_rag_dual_writer.py` 与 `tests/test_rag_resume.py`，导入目标服务与函数；
- 执行 `pytest tests/test_rag_dual_writer.py tests/test_rag_resume.py -v`；
- 验证失败确立：因模块未创建，抛出 `ModuleNotFoundError: No module named 'app.services.rag.dual_writer'`，测试收集阶段退出（Exit Code 1）。

### 3.2 GREEN 阶段 (最小代码实现)
- **`app/services/rag/dual_writer.py`**:
  - `compose_embedding_text`: 严格按 `f"分类：{category}\n问题：{questions}\n内容：{answer}"` 规范拼接密集语义文本；
  - `KnowledgeDualWriter.__init__`: 注入或默认创建 `MilvusKnowledgeStore` 与 `BGEEmbeddingClient`；
  - `write_chunks`:
    1. 阶段 1：批量插入 `KnowledgeChunk` 实体，状态全部置为 `'pending'`；
    2. 阶段 2：`await session.flush()` 生成自增 ID，按批次顺序双向绑定 `prev_chunk_id` 与 `next_chunk_id`；
    3. 阶段 3：`await session.commit()` 确立 MySQL 原文权威源落库成功；
    4. 阶段 4：调用 `aembed_documents` 异步生成 1024 维向量；
    5. 阶段 5：组装包含 `id`, `vector`, `category`, `questions`, `answer`, `chunk_text`, `content_type`, `is_key_clause`, `section_path`, `prev_chunk_id`, `next_chunk_id` 的记录，调用 `store.upsert(records)`；
    6. 阶段 6：更新 MySQL 记录 `vector_id = str(chunk.id)`, `vectorize_status = 'done'` 并提交事务；
  - `repair_pending_chunks`: 查询所有 `vectorize_status == 'pending'` 的块，重新向量化并 upsert 到 Milvus，回填 `vector_id` 并将状态更新为 `'done'`，提交事务后返回修复条数；
  - `close`: 释放向量存储句柄与资源；
- **`app/services/rag/__init__.py`**:
  - 导出 `KnowledgeDualWriter` 与 `compose_embedding_text`；
- **`scripts/build_knowledge_base.py`**:
  - 扫描指定知识库目录（默认 `data/kb/`）下的 `.md` 文件；
  - 使用 `MarkdownStructureSplitter` 解析切分为 `DocChunk`；
  - 调用 `KnowledgeDualWriter.write_chunks` 双写；
  - 在任务执行前和执行后分别调用 `repair_pending_chunks`，确保存量与新增块 100% 向量化完毕；
  - 命令行参数支持 `--kb-dir`, `--clean`, `--repair-only`, `--milvus-uri`, `--chunk-size`, `--overlap-size`。

### 3.3 REFACTOR & 验证阶段
- 专项测试验证：`pytest tests/test_rag_dual_writer.py tests/test_rag_resume.py -v` -> 10/10 PASSED；
- 全量回归验证：`pytest` -> 88/88 PASSED (原 78 个测试 + 本任务新增 10 个测试全部通过)；
- Git 提交：按规范完成提交并核对工作区状态。

## 4. 测试与验证结果

### 专项测试结果
```text
tests/test_rag_dual_writer.py::test_compose_embedding_text_standard PASSED
tests/test_rag_dual_writer.py::test_dual_writer_empty_chunks PASSED
tests/test_rag_dual_writer.py::test_dual_writer_full_flow PASSED
tests/test_rag_dual_writer.py::test_dual_writer_idempotent_upsert PASSED
tests/test_rag_dual_writer.py::test_build_knowledge_base_pipeline PASSED
tests/test_rag_dual_writer.py::test_build_knowledge_base_clean_and_repair_only PASSED
tests/test_rag_resume.py::test_repair_pending_chunks_empty PASSED
tests/test_rag_resume.py::test_repair_pending_chunks_interrupted_recovery PASSED
tests/test_rag_resume.py::test_repair_pending_chunks_mixed_status PASSED
tests/test_rag_resume.py::test_write_chunks_failure_and_subsequent_repair PASSED

====================== 10 passed, 42 warnings in 12.02s =======================
```

### 全量回归结果
```text
tests\test_after_sale.py ....                                            [  4%]
tests\test_api.py .....                                                  [ 10%]
tests\test_api_chat_stream.py ....                                       [ 14%]
tests\test_business_tools.py ........                                    [ 23%]
tests\test_chat_service.py .......                                       [ 31%]
tests\test_config.py ...                                                 [ 35%]
tests\test_db_session.py ...                                             [ 38%]
tests\test_models.py .....                                               [ 44%]
tests\test_prompts.py ...                                                [ 47%]
tests\test_rag_dual_writer.py ......                                     [ 54%]
tests\test_rag_embedding_milvus.py ...........                           [ 67%]
tests\test_rag_models.py ......                                          [ 73%]
tests\test_rag_resume.py ....                                            [ 78%]
tests\test_rag_splitter.py .......                                       [ 86%]
tests\test_session_manager.py ...                                        [ 89%]
tests\test_tool_executor.py .........                                    [100%]

====================== 88 passed, 49 warnings in 15.90s =======================
```

## 5. 返回契约 (Return Contract)
- **Status**: DONE
- **Commits**: `b6940b1fc49c9f6e5b4fdfa08e0e945632bb5ecd`
- **Test summary**: 88/88 passing
- **Concerns**: None
