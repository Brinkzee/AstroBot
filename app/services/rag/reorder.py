"""RAG 检索候选文档首尾重排与证据引用快照生成模块。

实现规范与算法：
1. lost_in_the_middle_reorder:
   依据《Lost in the Middle》长上下文注意力衰减规律，对 Top-K 知识块进行交替折叠：
   将最高分置于最前 (index 0)，次高分置于最后 (index -1)，低分沉降至中间，
   形如 [D1, D3, D5, D7, D9, D10, D8, D6, D4, D2]。
2. build_citation_items:
   为重排后的 Top-K 候选打上自增序号 n (1..K)，提取并规范化 chunk_id、section_path、question、answer，
   输出标准引用快照列表。
"""

from typing import Any, Dict, List, Optional


def lost_in_the_middle_reorder(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对已按相关度降序排列的文档列表执行 Lost-in-the-Middle 首尾交替重排。

    算法逻辑：
    - 偶数索引项 (0, 2, 4...) 依序放入前半部分 (front)；
    - 奇数索引项 (1, 3, 5...) 放入后半部分并逆序追加 (back reversed)；
    - 最终结果为 front + list(reversed(back))。
    例如 10 条文档 [D1..D10]：
      front = [D1, D3, D5, D7, D9]
      back = [D2, D4, D6, D8, D10] -> reversed = [D10, D8, D6, D4, D2]
      output = [D1, D3, D5, D7, D9, D10, D8, D6, D4, D2]
    最高分位于索引 0，次高分位于索引 -1，最低分沉降至序列中心。

    Args:
        docs: 按相关度降序排列的文档字典列表

    Returns:
        重排后的文档字典列表
    """
    if not docs:
        return []
    if len(docs) <= 2:
        return list(docs)

    front: List[Dict[str, Any]] = []
    back: List[Dict[str, Any]] = []

    for i, doc in enumerate(docs):
        if i % 2 == 0:
            front.append(doc)
        else:
            back.append(doc)

    return front + list(reversed(back))


def build_citation_items(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """组装 Top-K 证据全集快照，分配标准序号 n in [1..len(docs)]。

    输出字段规范：
    - n: int (1-based 自增序号，与模型生成回答中的引用角标 [n] 严格对齐)
    - chunk_id: int / Any (优先取 chunk_id，缺失则取 id)
    - section_path: str (文档所属章节或类目路径)
    - question: str (标准提问文本，支持容错多行或列表形式的 questions)
    - answer: str (知识解答原文)

    Args:
        docs: 重排完成的候选知识列表

    Returns:
        标准 CitationItem 字典列表
    """
    if not docs:
        return []

    citations: List[Dict[str, Any]] = []

    for idx, doc in enumerate(docs, start=1):
        # 兼容 chunk_id 与 id
        chunk_id = doc.get("chunk_id") if "chunk_id" in doc else doc.get("id")

        # 兼容 question 与 questions (str 或 list)
        raw_q = doc.get("question")
        if raw_q is None and "questions" in doc:
            raw_qs = doc.get("questions")
            if isinstance(raw_qs, list):
                raw_q = raw_qs[0] if raw_qs else ""
            elif isinstance(raw_qs, str):
                lines = [line.strip() for line in raw_qs.splitlines() if line.strip()]
                raw_q = lines[0] if lines else raw_qs.strip()
            else:
                raw_q = str(raw_qs)
        elif raw_q is None:
            raw_q = ""

        citation: Dict[str, Any] = {
            "n": idx,
            "chunk_id": chunk_id,
            "section_path": str(doc.get("section_path") or ""),
            "question": str(raw_q),
            "answer": str(doc.get("answer") or ""),
        }
        citations.append(citation)

    return citations
