# Task 1 Review Package

## Commit Range
778b76de4fb4a554a5fdb9660efda2409f2c4e80..a112f6c6c20d9b1932b29c6af8de5c7b29d2b01b

## Commit Log
a112f6c feat(models): add knowledge_chunks and qa_extraction_staging models

## Stat Summary
 app/models/__init__.py   |   6 +-
 app/models/knowledge.py  |  48 ++++++++++++++
 app/models/staging.py    |  33 ++++++++++
 scripts/init_ch03_db.py  |  77 ++++++++++++++++++++++
 tests/test_rag_models.py | 213 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 5 files changed, 375 insertions(+), 2 deletions(-)

## Full Diff
See git show a112f6c
- `app/models/knowledge.py`: KnowledgeChunk definition with BigIntID variant, fields: id, category, questions, answer, section_path, content_type, is_key_clause, prev_chunk_id, next_chunk_id, vector_id, vectorize_status, created_at, updated_at.
- `app/models/staging.py`: QAExtractionStaging definition with id, batch_no, source_ref, question, answer, status, created_at.
- `app/models/__init__.py`: Exported KnowledgeChunk, QAExtractionStaging.
- `scripts/init_ch03_db.py`: Parses sql/ch03-ddl.sql and executes with idempotent IF NOT EXISTS.
- `tests/test_rag_models.py`: 6 comprehensive tests covering defaults, metadata, SQLite in-memory CRUD, DDL parser, and idempotent execution.
