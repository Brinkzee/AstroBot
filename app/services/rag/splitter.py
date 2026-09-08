"""
Structure-aware Markdown Splitter for RAG Knowledge Base.

Supports:
- Title hierarchy tracking via heading stack (# to ######)
- Sentence boundary aligned overlap chunking (no cut sentences)
- Table row chunking with header & divider broadcasting
- Q&A FAQ extraction (Q:... A:..., 问：... 答：...)
- Key clause detection (【重要提示】, 【特别说明】, 【注意】, 【不可退换】, etc.)
"""
from dataclasses import dataclass
import re
from typing import Optional, List, Tuple


@dataclass
class DocChunk:
    category: str
    questions: str
    answer: str
    section_path: Optional[str] = None
    content_type: Optional[str] = "policy"
    is_key_clause: bool = False
    order_index: int = 0


PUNCT_CHARS = set("。！？；!?;\n")
CLOSING_CHARS = set('”"\'’）)]】 ')

KEY_CLAUSE_KEYWORDS = (
    "【重要提示】",
    "【特别说明】",
    "【注意】",
    "【不可退换】",
    "重要提示",
    "特别说明",
    "不可退换",
    "不支持",
)

HEADER_REGEX = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*$")
Q_REGEX = re.compile(r"^\s*(?:问|Q|q)[：:]\s*(.+)$")
A_REGEX = re.compile(r"^\s*(?:答|A|a)[：:]\s*(.+)$")


def find_sentence_boundaries(text: str) -> List[int]:
    """
    Find all sentence boundary positions in text.
    A boundary is the character index immediately following punctuation
    and any attached closing quotes or brackets.
    """
    boundaries = []
    n = len(text)
    i = 0
    while i < n:
        char = text[i]
        is_punct = False
        if char in PUNCT_CHARS:
            is_punct = True
        elif char == ".":
            # Exclude decimals like 3.14
            is_prev_digit = (i > 0 and text[i - 1].isdigit())
            is_next_digit = (i + 1 < n and text[i + 1].isdigit())
            if not (is_prev_digit and is_next_digit):
                is_punct = True

        if is_punct:
            end_pos = i + 1
            while end_pos < n and text[end_pos] in CLOSING_CHARS:
                end_pos += 1
            boundaries.append(end_pos)
            i = end_pos
        else:
            i += 1

    return sorted(list(set(boundaries)))


def split_text_with_sentence_boundary(
    text: str, chunk_size: int, overlap_size: int
) -> List[str]:
    """
    Split text into chunks aligned to sentence boundaries.
    Guarantees no sentence is cut in half at chunk boundaries.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    boundaries = find_sentence_boundaries(text)
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        if n - start <= chunk_size:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        target_end = start + chunk_size

        # Find boundary <= target_end (preferably the largest one > start)
        candidates_before = [b for b in boundaries if start < b <= target_end]
        if candidates_before:
            end_pos = candidates_before[-1]
        else:
            # If no boundary before target_end (single sentence > chunk_size),
            # search forward for the first boundary after target_end
            candidates_after = [b for b in boundaries if b > target_end]
            if candidates_after:
                end_pos = candidates_after[0]
            else:
                end_pos = min(target_end, n)

        chunk = text[start:end_pos].strip()
        if chunk:
            chunks.append(chunk)

        if end_pos >= n:
            break

        # Calculate next_start for overlap
        if overlap_size <= 0:
            next_start = end_pos
        else:
            ideal_start = end_pos - overlap_size
            overlap_candidates = [b for b in boundaries if start < b < end_pos]
            if overlap_candidates:
                next_start = min(overlap_candidates, key=lambda b: abs(b - ideal_start))
            else:
                next_start = end_pos

        # Skip leading whitespace at next_start
        while next_start < n and text[next_start] in (" ", "\t", "\n", "\r"):
            next_start += 1

        # Prevent infinite loops / zero progress
        if next_start <= start:
            next_start = end_pos if end_pos > start else start + 1

        start = next_start

    return chunks


def is_table_divider(line: str) -> bool:
    line_clean = line.strip()
    if not line_clean or "|" not in line_clean:
        return False
    cells = [c.strip() for c in line_clean.strip("|").split("|")]
    if not cells:
        return False
    return all(bool(c and re.match(r"^:?-+:?$", c)) for c in cells)


def is_table_row(line: str) -> bool:
    line_clean = line.strip()
    return bool(line_clean and ("|" in line_clean))


def split_table_lines(
    header_line: str, divider_line: str, rows: List[str], chunk_size: int
) -> List[str]:
    header_block = f"{header_line}\n{divider_line}"
    if not rows:
        return [header_block]

    total_table = f"{header_block}\n" + "\n".join(rows)
    if len(total_table) <= chunk_size:
        return [total_table]

    chunks = []
    current_rows = []
    current_len = len(header_block)

    for row in rows:
        row_len = len(row) + 1
        if current_rows and (current_len + row_len > chunk_size):
            chunks.append(f"{header_block}\n" + "\n".join(current_rows))
            current_rows = [row]
            current_len = len(header_block) + row_len
        else:
            current_rows.append(row)
            current_len += row_len

    if current_rows:
        chunks.append(f"{header_block}\n" + "\n".join(current_rows))

    return chunks


def parse_faqs_from_lines(lines: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Extract (intro_lines, list_of_(question, answer)) from lines."""
    intro = []
    faqs = []
    current_q = None
    current_a_lines = []

    for line in lines:
        q_match = Q_REGEX.match(line)
        if q_match:
            if current_q is not None:
                faqs.append((current_q, "\n".join(current_a_lines).strip()))
                current_a_lines = []
            current_q = q_match.group(1).strip()
        elif current_q is not None:
            a_match = A_REGEX.match(line)
            if a_match:
                current_a_lines.append(a_match.group(1).strip())
            else:
                current_a_lines.append(line)
        else:
            intro.append(line)

    if current_q is not None:
        faqs.append((current_q, "\n".join(current_a_lines).strip()))

    return intro, faqs


class MarkdownStructureSplitter:
    def __init__(self, chunk_size: int = 500, overlap_size: int = 80):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    @staticmethod
    def align_to_sentence_boundary(text: str, target_pos: int, window: int = 40) -> int:
        boundaries = find_sentence_boundaries(text)
        if not boundaries:
            return min(target_pos, len(text))
        in_window = [b for b in boundaries if abs(b - target_pos) <= window]
        if in_window:
            return min(in_window, key=lambda b: abs(b - target_pos))
        return min(boundaries, key=lambda b: abs(b - target_pos))

    @staticmethod
    def split_table(table_lines: List[str], chunk_size: int) -> List[str]:
        if len(table_lines) < 2:
            return ["\n".join(table_lines)]
        return split_table_lines(
            table_lines[0], table_lines[1], table_lines[2:], chunk_size
        )

    def _is_key_clause(self, text: str, questions: str = "") -> bool:
        check_str = f"{questions} {text}"
        return any(kw in check_str for kw in KEY_CLAUSE_KEYWORDS)

    def split_text(self, text: str) -> List[DocChunk]:
        if not text or not text.strip():
            return []

        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized_text.split("\n")

        sections: List[Tuple[List[Tuple[int, str]], List[str]]] = []
        current_stack: List[Tuple[int, str]] = []
        current_lines: List[str] = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            if not in_code_block:
                h_match = HEADER_REGEX.match(line)
                if h_match:
                    if any(l.strip() for l in current_lines):
                        sections.append((list(current_stack), current_lines))
                        current_lines = []

                    level = len(h_match.group(1))
                    title = h_match.group(2).strip()

                    while current_stack and current_stack[-1][0] >= level:
                        current_stack.pop()
                    current_stack.append((level, title))
                    continue

            current_lines.append(line)

        if any(l.strip() for l in current_lines):
            sections.append((list(current_stack), current_lines))

        # Convert sections into DocChunks
        all_chunks: List[DocChunk] = []

        for stack, sec_lines in sections:
            # Determine path & default category/questions
            if not stack:
                section_path = None
                category = "未分类"
                default_questions = "正文"
            elif len(stack) == 1:
                section_path = stack[0][1]
                category = stack[0][1]
                default_questions = stack[0][1]
            else:
                section_path = " > ".join(t for _, t in stack)
                category = " > ".join(t for _, t in stack[:-1])
                default_questions = stack[-1][1]

            # Check if section content contains FAQ
            intro_lines, faqs = parse_faqs_from_lines(sec_lines)

            if faqs:
                # Handle intro lines before FAQ if any
                intro_text = "\n".join(intro_lines).strip()
                if intro_text:
                    for part in split_text_with_sentence_boundary(
                        intro_text, self.chunk_size, self.overlap_size
                    ):
                        all_chunks.append(
                            DocChunk(
                                category=category,
                                questions=default_questions,
                                answer=part,
                                section_path=section_path,
                                content_type="policy",
                                is_key_clause=self._is_key_clause(part, default_questions),
                            )
                        )

                for q_text, a_text in faqs:
                    a_parts = split_text_with_sentence_boundary(
                        a_text, self.chunk_size, self.overlap_size
                    )
                    if not a_parts:
                        a_parts = [a_text]
                    for part in a_parts:
                        all_chunks.append(
                            DocChunk(
                                category=category,
                                questions=q_text,
                                answer=part,
                                section_path=section_path,
                                content_type="faq",
                                is_key_clause=self._is_key_clause(part, q_text),
                            )
                        )
                continue

            # Check if heading itself is an FAQ question
            h_q_match = Q_REGEX.match(default_questions)
            if h_q_match:
                faq_q = h_q_match.group(1).strip()
                sec_text = "\n".join(sec_lines).strip()
                a_match = A_REGEX.match(sec_text)
                if a_match:
                    sec_text = a_match.group(1).strip()

                for part in split_text_with_sentence_boundary(
                    sec_text, self.chunk_size, self.overlap_size
                ):
                    all_chunks.append(
                        DocChunk(
                            category=category,
                            questions=faq_q,
                            answer=part,
                            section_path=section_path,
                            content_type="faq",
                            is_key_clause=self._is_key_clause(part, faq_q),
                        )
                    )
                continue

            # Standard policy/manual section with possible tables and paragraphs
            blocks = []
            idx = 0
            curr_text_lines = []

            while idx < len(sec_lines):
                line = sec_lines[idx]
                if (
                    idx + 1 < len(sec_lines)
                    and "|" in line
                    and is_table_divider(sec_lines[idx + 1])
                ):
                    if curr_text_lines:
                        blocks.append(("text", "\n".join(curr_text_lines).strip()))
                        curr_text_lines = []

                    header = line.strip()
                    divider = sec_lines[idx + 1].strip()
                    rows = []
                    idx += 2
                    while idx < len(sec_lines) and is_table_row(sec_lines[idx]):
                        rows.append(sec_lines[idx].strip())
                        idx += 1
                    blocks.append(("table", (header, divider, rows)))
                else:
                    curr_text_lines.append(line)
                    idx += 1

            if curr_text_lines:
                blocks.append(("text", "\n".join(curr_text_lines).strip()))

            for b_type, b_data in blocks:
                if b_type == "text":
                    text_str = b_data.strip()
                    if not text_str:
                        continue
                    parts = split_text_with_sentence_boundary(
                        text_str, self.chunk_size, self.overlap_size
                    )
                    for part in parts:
                        all_chunks.append(
                            DocChunk(
                                category=category,
                                questions=default_questions,
                                answer=part,
                                section_path=section_path,
                                content_type="policy",
                                is_key_clause=self._is_key_clause(part, default_questions),
                            )
                        )
                elif b_type == "table":
                    header, divider, rows = b_data
                    table_parts = split_table_lines(
                        header, divider, rows, self.chunk_size
                    )
                    for t_part in table_parts:
                        all_chunks.append(
                            DocChunk(
                                category=category,
                                questions=default_questions,
                                answer=t_part,
                                section_path=section_path,
                                content_type="policy",
                                is_key_clause=self._is_key_clause(t_part, default_questions),
                            )
                        )

        # Assign order_index
        for idx, chunk in enumerate(all_chunks):
            chunk.order_index = idx

        return all_chunks

    def split_markdown(self, text: str) -> List[DocChunk]:
        return self.split_text(text)
