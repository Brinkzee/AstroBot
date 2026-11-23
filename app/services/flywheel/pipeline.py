import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import asc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.low_confidence import LowConfidenceQuestion
from app.models.review_queue import ReviewQueue, ReviewStatus

logger = logging.getLogger(__name__)


class FlywheelPipeline:
    @staticmethod
    def _build_normalize_prompt(raw_question: str) -> str:
        return f"""
请将以下用户的口语化提问改写为一个标准FAQ（常见问题解答）格式。
你需要输出两部分内容：
1. 标准化后的FAQ问题（应简洁明了，去除情绪词）
2. 简短的AI建议答案草稿

原话："{raw_question}"
        """

    @staticmethod
    def _parse_normalize_response(response_text: str) -> Tuple[str, str]:
        text = (response_text or "").strip()
        parts = text.split("\n\n")
        if len(parts) >= 2:
            return parts[0].strip(), parts[1].strip()
        return text, "Suggested answer not properly formatted"

    @staticmethod
    def _build_duplicate_prompt(
        normalized_question: str, candidates: List[ReviewQueue]
    ) -> str:
        candidates_text = "\n".join(
            [f"ID: {c.id} | Q: {c.normalized_question or ''}" for c in candidates]
        )
        return f"""
新问题："{normalized_question}"

候选问题列表：
{candidates_text}

请判断新问题是否与候选问题列表中某一个问题表达了相同的客户意图。
如果相同，请仅输出匹配候选问题的ID。
如果不相同，请仅输出"NONE"。
        """

    @staticmethod
    def _parse_duplicate_response(
        response_text: str, candidates: List[ReviewQueue]
    ) -> Optional[int]:
        text = (response_text or "").strip().upper()
        if not text or "NONE" in text:
            return None

        # 严格匹配独立的整数 ID（可带可选前缀 "ID:"），禁止从否定解释句中误提取数字或 "YES"
        match = re.fullmatch(r"(?:ID\s*[:：#]?\s*)?(\d+)", text)
        if match:
            matched_id = int(match.group(1))
            if any(c.id == matched_id for c in candidates):
                return matched_id

        return None

    @staticmethod
    def _extract_llm_content(resp: Any) -> str:
        if resp is None:
            return ""
        content = getattr(resp, "content", resp)
        return str(content) if content is not None else ""

    @classmethod
    async def _ainvoke_llm(cls, llm: Any, prompt: str) -> str:
        if hasattr(llm, "ainvoke"):
            resp = await llm.ainvoke(prompt)
        elif hasattr(llm, "invoke"):
            resp = await asyncio.to_thread(llm.invoke, prompt)
        else:
            raise ValueError(f"Unsupported LLM client: {type(llm)}")
        return cls._extract_llm_content(resp)

    def normalize_question(self, raw_question: str, llm=None) -> Tuple[str, str]:
        raw_str = str(raw_question or "").strip()
        if not llm:
            return raw_str, "Fallback answer"

        prompt = self._build_normalize_prompt(raw_str)
        response = self._extract_llm_content(llm.invoke(prompt))
        return self._parse_normalize_response(response)

    async def anormalize_question(self, raw_question: str, llm=None) -> Tuple[str, str]:
        raw_str = str(raw_question or "").strip()
        if not llm:
            return raw_str, "Fallback answer"

        prompt = self._build_normalize_prompt(raw_str)
        response = await self._ainvoke_llm(llm, prompt)
        return self._parse_normalize_response(response)

    def find_duplicate_question(
        self, normalized_question: str, candidates: List[ReviewQueue], llm=None
    ) -> Optional[int]:
        if not candidates:
            return None

        norm_clean = str(normalized_question or "").strip().lower()
        # Stage 1: Exact / close text match
        for cand in candidates:
            cand_q = str(cand.normalized_question or "").strip().lower()
            if cand_q and cand_q == norm_clean:
                return cand.id

        # Stage 2: LLM semantic equivalence
        if not llm:
            return None

        prompt = self._build_duplicate_prompt(str(normalized_question or ""), candidates)
        response = self._extract_llm_content(llm.invoke(prompt))
        return self._parse_duplicate_response(response, candidates)

    async def afind_duplicate_question(
        self, normalized_question: str, candidates: List[ReviewQueue], llm=None
    ) -> Optional[int]:
        if not candidates:
            return None

        norm_clean = str(normalized_question or "").strip().lower()
        for cand in candidates:
            cand_q = str(cand.normalized_question or "").strip().lower()
            if cand_q and cand_q == norm_clean:
                return cand.id

        if not llm:
            return None

        prompt = self._build_duplicate_prompt(str(normalized_question or ""), candidates)
        response = await self._ainvoke_llm(llm, prompt)
        return self._parse_duplicate_response(response, candidates)

    async def process_pending_questions(
        self, db: AsyncSession, batch_size: int = 50, llm=None
    ) -> Dict[str, Any]:
        # Fetch pending low confidence questions
        stmt = (
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id.is_(None))
            .order_by(asc(LowConfidenceQuestion.id))
            .limit(batch_size)
        )

        res = await db.execute(stmt)
        pending_questions = res.scalars().all()

        if not pending_questions:
            return {"processed_count": 0, "new_created": 0, "merged_count": 0}

        # Fetch active review queue items
        rq_stmt = select(ReviewQueue).where(
            ReviewQueue.review_status == ReviewStatus.PENDING.value
        )
        rq_res = await db.execute(rq_stmt)
        candidates = list(rq_res.scalars().all())
        logger.debug("FlywheelPipeline loaded %d pending review queue candidates", len(candidates))

        new_created = 0
        merged_count = 0
        processed_count = 0

        for q in pending_questions:
            try:
                normalized_q, suggested_a = await self.anormalize_question(q.raw_question, llm)
                matched_id = await self.afind_duplicate_question(normalized_q, candidates, llm)
                logger.debug(
                    "FlywheelPipeline processed lcq_id=%s norm=%r matched_id=%s",
                    q.id,
                    normalized_q,
                    matched_id,
                )

                if matched_id is not None:
                    q.matched_review_id = matched_id

                    # update occurrence count
                    for cand in candidates:
                        if cand.id == matched_id:
                            cand.occurrence_count = (cand.occurrence_count or 0) + 1
                            break
                    merged_count += 1
                else:
                    review_item = ReviewQueue(
                        normalized_question=normalized_q,
                        ai_suggested_answer=suggested_a,
                        occurrence_count=1,
                        review_status=ReviewStatus.PENDING.value,
                    )
                    db.add(review_item)
                    await db.flush()  # flush to get ID

                    q.matched_review_id = review_item.id
                    candidates.append(review_item)
                    new_created += 1

                processed_count += 1
            except Exception as e:
                logger.warning("FlywheelPipeline failed on question id=%s: %s", q.id, e)
                continue

        await db.commit()

        return {
            "processed_count": processed_count,
            "new_created": new_created,
            "merged_count": merged_count,
        }
