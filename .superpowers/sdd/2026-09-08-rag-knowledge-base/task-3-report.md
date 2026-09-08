# Task 3 Execution Report: BGE-M3 向量嵌入客户端与 Milvus-Lite 向量库管理器

## 1. 任务概述
- **任务目标**: 实现 BGE-M3 1024 维 Dense 语义向量嵌入客户端（`BGEEmbeddingClient`）与基于 Milvus-Lite 的向量库管理器（`MilvusKnowledgeStore`），支持向量批量生成、MySQL chunk ID 幂等覆盖 Upsert、余弦相似度 Top-K 检索、标量过滤及本地资源安全释放。
- **执行分支**: `feat/ch03-rag-knowledge-base`
- **提交哈希**: `a73dc70d27b56fcc60f0fd740a5e7188ad190eb3`

## 2. 变更文件清单
1. `app/config.py`: 新增 `milvus_uri`（默认 `./data/milvus/astro_bot.db`）与 `huggingface_token` 字段，提供 `MILVUS_URI` 与 `HUGGINGFACE_TOKEN` 大写兼容属性；
2. `app/services/rag/embedding.py`: 新增 `BGEEmbeddingClient` 类；
3. `app/services/rag/milvus_client.py`: 新增 `MilvusKnowledgeStore` 类；
4. `app/services/rag/__init__.py`: 导出 `BGEEmbeddingClient`、`MilvusKnowledgeStore`、`MarkdownStructureSplitter`、`DocChunk`；
5. `tests/test_rag_embedding_milvus.py`: 编写覆盖嵌入客户端与 Milvus-Lite 全生命周期的 11 个自动化测试用例。

## 3. TDD 执行过程记录

### 3.1 RED 阶段 (失败用例确立)
- 编写测试文件 `tests/test_rag_embedding_milvus.py`，导入目标服务类；
- 执行 `pytest tests/test_rag_embedding_milvus.py -v`；
- 验证失败确立：因模块未创建，抛出 `ModuleNotFoundError: No module named 'app.services.rag.embedding'`，测试收集失败（Exit Code 1）。

### 3.2 GREEN 阶段 (最小代码实现)
- **`app/config.py`**:
  - 添加 `milvus_uri: str = Field(default="./data/milvus/astro_bot.db")` 与兼容 property `MILVUS_URI`；
  - 添加 `huggingface_token: Optional[str] = Field(default=None)` 与兼容 property `HUGGINGFACE_TOKEN`；
- **`app/services/rag/embedding.py` (`BGEEmbeddingClient`)**:
  - `__init__`: 优先级解析 Token，支持真实 `InferenceClient` 与内置确定性 Mock 模式（`is_mock`）；
  - `embed_documents`: 支持批处理（`batch_size`），网络抖动时内置指数退避重试（最大 3 次），重试失败后优雅降级为 Mock 兜底向量；
  - `embed_query`: 单句封装，返回 1024 维 float 列表；
  - `aembed_query` / `aembed_documents`: 使用 `asyncio.to_thread` 提供无阻塞异步包装；
- **`app/services/rag/milvus_client.py` (`MilvusKnowledgeStore`)**:
  - `__init__`: 自动创建本地父目录，初始化 `pymilvus.MilvusClient`；
  - `init_collection`: 集合参数配置为 `dimension=1024`、`primary_field_name="id"`、`id_type="int"`、`auto_id=False`、`vector_field_name="vector"`、`metric_type="COSINE"`、`enable_dynamic_field=True`；
  - `upsert`: 批量插入并按主键覆盖，返回成功写入条数；
  - `search`: 基于余弦相似度检索，展开并返回 payload 字段，排除大数组 `vector`，支持 `min_score` 相似度阈值过滤与表达式过滤；
  - `count`: 统计当前集合记录数；
  - `close`: 释放 MilvusClient，并针对 Windows 平台调用 `server_manager_instance.release_server`，彻底解除 SQLite/DB 句柄锁；支持 `__enter__` / `__exit__` 上下文管理。

### 3.3 REFACTOR & 验证阶段
- 单元测试验证：`pytest tests/test_rag_embedding_milvus.py -v` -> 11/11 PASSED；
- 全量回归验证：`pytest` -> 78/78 PASSED (覆盖原有 67 个测试 + 新增 11 个测试)。

## 4. 测试与验证结果
```text
tests\test_rag_embedding_milvus.py::test_config_milvus_uri_and_token PASSED
tests\test_rag_embedding_milvus.py::test_rag_package_exports PASSED
tests\test_rag_embedding_milvus.py::test_bge_embedding_client_mock_query PASSED
tests\test_rag_embedding_milvus.py::test_bge_embedding_client_mock_documents_batch PASSED
tests\test_rag_embedding_milvus.py::test_bge_embedding_client_async_methods PASSED
tests\test_rag_embedding_milvus.py::test_bge_embedding_client_real_with_mocked_hf PASSED
tests\test_rag_embedding_milvus.py::test_bge_embedding_client_retry_and_fallback_on_error PASSED
tests\test_rag_embedding_milvus.py::test_milvus_knowledge_store_lifecycle_and_init PASSED
tests\test_rag_embedding_milvus.py::test_milvus_knowledge_store_upsert_and_overwrite PASSED
tests\test_rag_embedding_milvus.py::test_milvus_knowledge_store_search_cosine_and_filter PASSED
tests\test_rag_embedding_milvus.py::test_milvus_knowledge_store_context_manager PASSED

======================= 78 passed, 7 warnings in 5.19s ========================
```

## 5. 返回契约 (Return Contract)
- **Status**: DONE
- **Commits**: `a73dc70d27b56fcc60f0fd740a5e7188ad190eb3`
- **Test summary**: 78/78 passing
- **Concerns**: None
