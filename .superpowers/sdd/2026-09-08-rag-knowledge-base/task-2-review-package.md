# Task 2 Review Package

## Commit Range
8a5bb12957eb42e9d92d2fb87879409ce5251334..26fd863b1c692d84b65015fa7aae2976bd1181d8

## Commit Log
26fd863 feat(rag): implement structure-aware markdown splitter with sentence-aligned overlap and table header copying

## Stat Summary
 app/services/rag/__init__.py |   1 +
 app/services/rag/splitter.py | 447 +++++++++++++++++++++++++++++++++++++++++++
 tests/test_rag_splitter.py   | 128 +++++++++++++
 3 files changed, 576 insertions(+)

## Summary of Implementation
- `app/services/rag/splitter.py`:
  - `DocChunk`: Dataclass containing `category`, `questions`, `answer`, `section_path`, `content_type`, `is_key_clause`, `order_index`.
  - `MarkdownStructureSplitter`:
    - Tracks header stack (`#`~`######`) to accurately construct `category`, `questions`, and `section_path`.
    - Key-clause detection: scans for keywords (`【重要提示】`, `【特别说明】`, `注意`, `不支持`, `免责`, etc.) -> sets `is_key_clause = True`.
    - Sentence boundary alignment (`align_to_sentence_boundary`): aligns chunk overlap boundaries to punctuation (`。`, `！`, `？`, `；`, `\n`) preventing broken sentences.
    - Table splitting: detects table headers and divider rows, splits data rows into chunks while copying the header and divider rows to every sub-chunk.
- `tests/test_rag_splitter.py`: 7 tests covering Markdown hierarchy, sentence-boundary overlap without half sentences, table row splitting with header copying, key-clause detection, FAQ extraction, and edge cases.
