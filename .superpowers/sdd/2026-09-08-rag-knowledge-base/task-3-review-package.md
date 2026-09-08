# Task 3 Review Package

## Commit Range
2d592ede27e9933c600802a988ea3071b0d70135..a73dc70d27b56fcc60f0fd740a5e7188ad190eb3

## Commit Log
a73dc70 feat(rag): add bge-m3 embedding client and milvus-lite store manager

## Stat Summary
 app/config.py                      |  18 +++
 app/services/rag/__init__.py       |  13 +-
 app/services/rag/embedding.py      | 165 ++++++++++++++++++++++
 app/services/rag/milvus_client.py  | 168 ++++++++++++++++++++++
 tests/test_rag_embedding_milvus.py | 280 +++++++++++++++++++++++++++++++++++++
 5 files changed, 643 insertions(+), 1 deletion(-)

## Summary of Implementation
- `app/config.py`: Added `milvus_uri` and `huggingface_token` fields and property getters.
- `app/services/rag/embedding.py`:
  - `BGEEmbeddingClient`:
    - 1024-dim dense vectors.
    - Uses `huggingface_hub.InferenceClient` with model `BAAI/bge-m3`.
    - Includes tenacity retry on request errors, exponential backoff, and offline fallback mode for unit testing.
    - Async methods: `aembed_query` and `aembed_documents` using `asyncio.to_thread`.
- `app/services/rag/milvus_client.py`:
  - `MilvusKnowledgeStore`:
    - Auto-creates directories for uri.
    - Creates `knowledge` collection with `dimension=1024`, `metric_type="COSINE"`, `id` as INT64 primary key (`auto_id=False`).
    - Idempotent `upsert` and vector similarity `search` with `min_score` filtering.
    - Implements Windows file lock cleanup in `close()` and supports context manager (`__enter__` / `__exit__`).
- `tests/test_rag_embedding_milvus.py`: 11 comprehensive tests for embedding dimension, offline fallback, async embedding, Milvus collection initialization, upsert, COSINE search, filtering, and count.
