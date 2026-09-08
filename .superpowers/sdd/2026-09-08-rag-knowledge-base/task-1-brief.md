# Task 1 Brief: ORM 数据模型与 DDL 初始化基础设施

## 1. 任务目标
实现 `knowledge_chunks` 与 `qa_extraction_staging` 的 SQLAlchemy ORM 数据模型，保证对齐 `sql/ch03-ddl.sql`，同时在 SQLite (内存测试) 与 MySQL (生产 WSL2) 均能无损运行，并提供数据库表初始化脚本 `scripts/init_ch03_db.py`。

## 2. 涉及文件
- Create: `app/models/knowledge.py`
- Create: `app/models/staging.py`
- Modify: `app/models/__init__.py`
- Create: `scripts/init_ch03_db.py`
- Test: `tests/test_rag_models.py`

## 3. 全局约束
- 字段与类型严格遵循 `sql/ch03-ddl.sql`；
- 主键兼容性：使用 `BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite")`；
- 严禁盲写业务代码，严格遵循 TDD：先写测试 -> 验证失败 -> 写最小实现 -> 验证通过 -> 提交。

## 4. 具体接口与实现细节
### `KnowledgeChunk` (`app/models/knowledge.py`)
- `id`: BigIntID 自增主键；
- `category`: `String(255)`, nullable=False, index=True
- `questions`: `Text`, nullable=False
- `answer`: `Text`, nullable=False
- `section_path`: `String(512)`, nullable=True
- `content_type`: `String(32)`, nullable=True
- `is_key_clause`: `Boolean`, default=False, nullable=False
- `prev_chunk_id`: BigIntID, ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True
- `next_chunk_id`: BigIntID, ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True
- `vector_id`: `String(64)`, nullable=True
- `vectorize_status`: `String(16)`, default="pending", nullable=False, index=True
- `created_at`: `DateTime`, default=datetime.utcnow, nullable=False
- `updated_at`: `DateTime`, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False

### `QAExtractionStaging` (`app/models/staging.py`)
- `id`: BigIntID 自增主键；
- `batch_no`: `String(64)`, nullable=False, index=True
- `source_ref`: `String(255)`, nullable=True
- `question`: `Text`, nullable=False
- `answer`: `Text`, nullable=False
- `status`: `String(16)`, default="extracted", nullable=False, index=True
- `created_at`: `DateTime`, default=datetime.utcnow, nullable=False

### `scripts/init_ch03_db.py`
- 读取并执行 `sql/ch03-ddl.sql`，安全支持幂等创建 `knowledge_chunks` 与 `qa_extraction_staging` 表。

## 5. 验收标准
- `pytest tests/test_rag_models.py -v` 100% 通过；
- 全量回归 `pytest` 保持 100% 通过。
