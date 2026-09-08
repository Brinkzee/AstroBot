# Task 6 Review Package

## Commit Range
34703c3310200aaa12dc141f46b38d8f14e901e6..765dc85

## Commit Log
765dc85 feat(tools): upgrade query_faq implementation to dense vector semantic retrieval

## Stat Summary
 app/services/rag/__init__.py  |   2 +
 app/services/rag/retriever.py | 127 ++++++++++++++++++++
 app/tools/business_tools.py   |  25 +++-
 tests/test_business_tools.py  |  23 ++++
 tests/test_rag_retriever.py   | 273 ++++++++++++++++++++++++++++++++++++++++++
 5 files changed, 449 insertions(+), 1 deletion(-)

## Summary of Implementation
- `app/services/rag/retriever.py`:
  - `KnowledgeRetriever`:
    - Handles query vectorization via `BGEEmbeddingClient.aembed_query`.
    - Searches Milvus `knowledge` collection via `MilvusKnowledgeStore.search`.
    - Filters by `min_score` threshold (default `0.35`).
    - Formats output matching `1. 问：{question}\n   答：{answer}`.
    - Gracefully handles empty results and empty queries.
- `app/tools/business_tools.py`:
  - `query_faq`:
    - Signature `async def query_faq(keyword: str) -> str`, `@tool` decorator, docstring, and return contract remain 100% untouched.
    - Internally prioritizes `KnowledgeRetriever.retrieve_faq_text(kw, top_k=3)`.
    - If vector search yields no results (or before initial vector database population), falls back to SQL FAQ LIKE query for smooth backwards compatibility.
- `tests/test_rag_retriever.py`: 10 tests covering vector retrieval, formatting, threshold filtering, empty handling, tool contract immutability, and fallback mechanisms.
- `tests/test_business_tools.py`: Added vector upgrade verification test while preserving all existing business tool tests.
