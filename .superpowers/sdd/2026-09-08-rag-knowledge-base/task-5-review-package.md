# Task 5 Review Package

## Commit Range
3eeb0588d0362ddf911e0827f46ccd5b21a2290a..c36c6aedaf54033d2442178525c99e949e409dab

## Commit Log
c36c6ae feat(rag): add dialogue qa mining pipeline with staging and semantic deduplication

## Stat Summary
 app/prompts/qa_extraction.py | 126 +++++++++++++
 app/services/rag/__init__.py |   3 +
 app/services/rag/miner.py    | 353 +++++++++++++++++++++++++++++++++++
 scripts/mine_dialogues.py    | 203 ++++++++++++++++++++
 tests/test_rag_miner.py      | 432 +++++++++++++++++++++++++++++++++++++++++++
 5 files changed, 1117 insertions(+)

## Summary of Implementation
- `app/prompts/qa_extraction.py`:
  - Defined Pydantic models: `ExtractedQAPair` (with validation for clean question/answer) and `ExtractedQAList`.
  - Defined system and user prompt templates: explicit rules for removing casual greetings, privacy redaction, and strict JSON format extraction.
  - Provided `parse_qa_extraction_response` helper with regex JSON markdown cleanup.
- `app/services/rag/miner.py`:
  - `DialogueKnowledgeMiner`:
    - `fetch_dialogue_batches`: Groups messages by `conversation_id`, generates unique `batch_no`, and batches conversations.
    - `extract_and_stage`: Invokes LLM, validates output, and writes to `qa_extraction_staging` with status `extracted`.
    - `deduplicate_staging`: Two-tier deduplication:
      1. Exact string normalization deduplication.
      2. BGE-M3 semantic cosine similarity clustering (threshold 0.92).
      Updates staging statuses: `kept` vs `discarded`.
    - `ingest_kept_chunks`: Maps `kept` items to `DocChunk`s with `content_type="mined_qa"` and invokes `KnowledgeDualWriter.write_chunks`.
- `scripts/mine_dialogues.py`:
  - CLI offline mining runner supporting `--batch-size`, `--dry-run`, `--similarity-threshold`, `--skip-extraction`, `--skip-ingestion`.
- `tests/test_rag_miner.py`: 7 tests covering prompt schema parsing, markdown fence cleanup, batch dialogue grouping, staging insertion, exact + BGE-M3 deduplication, kept-to-KB dual-write, and CLI flow.
