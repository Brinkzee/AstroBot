# Task 1 Execution Report: ORM 数据模型与 DDL 初始化基础设施

## 1. 执行概述
- **任务目标**: 实现 `knowledge_chunks` 与 `qa_extraction_staging` 的 SQLAlchemy ORM 数据模型，保证对齐 `sql/ch03-ddl.sql`，同时在 SQLite (内存测试) 与 MySQL (生产 WSL2) 均能无损运行，并提供数据库表初始化脚本 `scripts/init_ch03_db.py`。
- **执行状态**: DONE
- **Git Commit**: `a112f6c6c20d9b1932b29c6af8de5c7b29d2b01b`
- **提交信息**: `feat(models): add knowledge_chunks and qa_extraction_staging models`
- **测试结果**: 
  - `pytest tests/test_rag_models.py -v`: 6/6 passed (100%)
  - 全量回归 `pytest`: 60/60 passed (100%)

---

## 2. TDD 流程执行证据

### 步骤 1: 编写失败测试 (RED)
- 新建测试文件 `tests/test_rag_models.py`，覆盖：
  1. `test_knowledge_chunk_attributes_and_defaults`: 测试字段映射与默认值；
  2. `test_qa_extraction_staging_attributes_and_defaults`: 测试暂存表字段映射与默认值；
  3. `test_models_table_metadata`: 测试表名、主外键、非空约束、索引等元数据；
  4. `test_sqlite_in_memory_crud`: 在 SQLite 内存数据库中端到端验证两张表的增删改查以及自引用外键链；
  5. `test_parse_ddl_statements`: 验证 `scripts/init_ch03_db.py` 解析 DDL 并注入 `CREATE TABLE IF NOT EXISTS` 幂等安全特性；
  6. `test_init_ch03_db_idempotent`: 验证初始化执行逻辑幂等性。

### 步骤 2: 验证测试失败 (RED Verification)
- 运行 `pytest tests/test_rag_models.py -v`
- 结果: `ImportError: cannot import name 'KnowledgeChunk' from 'app.models'`
- 确认因为生产代码尚未实现而预期失败。

### 步骤 3: 编写最小实现代码 (GREEN)
- `app/models/knowledge.py`:
  - 实现 `KnowledgeChunk` 模型，继承 `Base`；
  - 主键采用跨方言自增兼容类型：`BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite")`；
  - 严格映射字段：`category`, `questions`, `answer`, `section_path`, `content_type`, `is_key_clause`, `prev_chunk_id`, `next_chunk_id`, `vector_id`, `vectorize_status`, `created_at`, `updated_at`；
  - 外键设置 `ForeignKey("knowledge_chunks.id", ondelete="SET NULL")`。
- `app/models/staging.py`:
  - 实现 `QAExtractionStaging` 模型，继承 `Base`；
  - 严格映射字段：`id`, `batch_no`, `source_ref`, `question`, `answer`, `status`, `created_at`。
- `app/models/__init__.py`:
  - 统一导出 `KnowledgeChunk` 与 `QAExtractionStaging`，纳入 `__all__`。
- `scripts/init_ch03_db.py`:
  - 实现 `parse_ddl_statements`: 去除 SQL 注释并注入 `IF NOT EXISTS` 保证执行幂等；
  - 实现 `init_ch03_db`: 支持通过 async engine 安全执行 `sql/ch03-ddl.sql`，且支持测试注入 `engine_override`。

### 步骤 4: 验证测试全部通过 (GREEN Verification)
- 运行 `pytest tests/test_rag_models.py -v`:
  - `test_knowledge_chunk_attributes_and_defaults PASSED`
  - `test_qa_extraction_staging_attributes_and_defaults PASSED`
  - `test_models_table_metadata PASSED`
  - `test_sqlite_in_memory_crud PASSED`
  - `test_parse_ddl_statements PASSED`
  - `test_init_ch03_db_idempotent PASSED`
  - 6 passed.
- 运行全量测试 `pytest`:
  - 60 passed (从原先 54 passed 增至 60 passed，0 失败)。

### 步骤 5: Git 提交
- 命令:
  ```bash
  git add app/models/ tests/test_rag_models.py scripts/init_ch03_db.py
  git commit -m "feat(models): add knowledge_chunks and qa_extraction_staging models"
  ```
- Commit Hash: `a112f6c6c20d9b1932b29c6af8de5c7b29d2b01b`

---

## 3. 交付清单
- `app/models/knowledge.py` (新增)
- `app/models/staging.py` (新增)
- `app/models/__init__.py` (修改)
- `scripts/init_ch03_db.py` (新增)
- `tests/test_rag_models.py` (新增)

---

## 4. Return Contract
- **Status**: DONE
- **Commits**: `a112f6c6c20d9b1932b29c6af8de5c7b29d2b01b`
- **Test summary**: 60/60 passing (tests/test_rag_models.py: 6/6 passing)
- **Concerns**: None
