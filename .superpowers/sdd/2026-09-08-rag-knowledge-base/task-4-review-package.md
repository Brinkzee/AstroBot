# Task 4 Review Package

## Commit Range
63af96607864884bec565a94f093ddb938902229..b6940b1fc49c9f6e5b4fdfa08e0e945632bb5ecd

## Commit Log
b6940b1 feat(rag): implement dual-write persistence with automatic pending chunk repair

## Stat Summary
 app/services/rag/__init__.py    |   4 +
 app/services/rag/dual_writer.py | 191 ++++++++++++++++++++++++
 scripts/build_knowledge_base.py | 205 ++++++++++++++++++++++++++
 tests/test_rag_dual_writer.py   | 317 ++++++++++++++++++++++++++++++++++++++++
 tests/test_rag_resume.py        | 242 ++++++++++++++++++++++++++++++
 5 files changed, 959 insertions(+)

## Summary of Implementation
- `app/services/rag/dual_writer.py`:
  - `compose_embedding_text(category, questions, answer)`: Formats strictly as `f"分类：{category}\n问题：{questions}\n内容：{answer}"`, keeping other metadata out of the embedding vector.
  - `KnowledgeDualWriter`:
    - `write_chunks(session, chunks)`:
      1. Adds `KnowledgeChunk` entities with `vectorize_status = 'pending'`.
      2. Flushes to acquire MySQL auto-increment IDs; sets `prev_chunk_id` and `next_chunk_id` pointers between sequential chunks; commits to MySQL authoritative source.
      3. Generates 1024-dim BGE-M3 embeddings.
      4. Upserts into Milvus collection `knowledge` with `id=chunk.id`, `vector=emb`, and metadata payload (`category`, `questions`, `answer`, `chunk_text`, `content_type`, `is_key_clause`).
      5. Updates MySQL: `vector_id = str(chunk.id)`, `vectorize_status = 'done'`, and commits.
    - `repair_pending_chunks(session)`:
      - Selects all rows with `vectorize_status == 'pending'`.
      - Re-embeds with BGE-M3 and upserts into Milvus (idempotent overwrite).
      - Backfills `vector_id = str(chunk.id)` and sets `vectorize_status = 'done'`.
      - Returns count of repaired chunks.
- `scripts/build_knowledge_base.py`:
  - CLI script to ingest Markdown files from `data/kb/` into the knowledge base.
  - Automatically runs `repair_pending_chunks` at start and end.
  - Supports `--clean`, `--repair-only`, `--kb-dir`.
- `tests/test_rag_dual_writer.py` and `tests/test_rag_resume.py`:
  - 10 tests verifying full dual-write, pointer links, payload persistence, simulated interruption, pending chunk compensation, mixed states, and CLI execution.
