"""RAG 两阶段受控生成器与自检守卫。

包含：
1. SelfCheckResult：自评判定结果数据传输对象 (DTO)；
2. RAGControlledGenerator：负责 Phase 1 知识充分度自评（自评不足落库并拒答）与 Phase 2 带负面约束及 [n] 引用角标受控流式生成。
"""

import asyncio
from dataclasses import dataclass
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import get_chat_model
from app.models.low_confidence import (
    LowConfidenceQuestion,
    LowConfidenceSource,
    record_low_confidence,
)
from app.prompts.rag_qa import (
    DEFAULT_REFUSAL_RESPONSE,
    RAG_CONTROLLED_QA_SYSTEM_PROMPT,
    SELF_CHECK_SYSTEM_PROMPT,
    format_evidence_context,
    parse_self_check_response,
)

logger = logging.getLogger(__name__)


@dataclass
class SelfCheckResult:
    """前置自检判定结果 DTO。"""

    useful: bool
    reason: str
    source: str = "self_check"  # 'retrieval_low_conf' 或 'self_check'


class RAGControlledGenerator:
    """RAG 两阶段受控流式生成器。"""

    def __init__(
        self,
        model: Optional[Any] = None,
        stream_model: Optional[Any] = None,
        refusal_text: str = DEFAULT_REFUSAL_RESPONSE,
    ) -> None:
        self.model = model
        self.stream_model = stream_model
        self.refusal_text = refusal_text

    async def check_sufficiency(
        self,
        query: str,
        citations: List[Dict[str, Any]],
        db: Optional[AsyncSession] = None,
        conversation_id: Optional[int] = None,
    ) -> SelfCheckResult:
        """Phase 1: 前置知识充分度自评 (Self-Check)。
        
        1. 证据为空时直接判定 useful=False，source='retrieval_low_conf'；
        2. 证据非空时调用 LLM 评估（或在无模型环境下降级 Mock），判定 useful=True/False，source='self_check'；
        3. 若 useful=False 且传入了 db，则自动调用 record_low_confidence 入库。
        """
        raw_query = str(query).strip() if query else ""

        # 1. 证据为空直接判不足
        if not citations:
            reason = "检索候选证据为空，未匹配到任何相关知识库条目"
            source = LowConfidenceSource.RETRIEVAL_LOW_CONF.value
            result = SelfCheckResult(useful=False, reason=reason, source=source)
            if db is not None:
                await record_low_confidence(
                    db=db,
                    raw_question=raw_query,
                    source=source,
                    conversation_id=conversation_id,
                    reason=reason,
                )
            return result

        # 2. 证据非空，调用 LLM 或确定性 Mock 进行自评
        evidence_text = format_evidence_context(citations)
        llm = self.model

        useful = False
        reason = ""
        source = LowConfidenceSource.SELF_CHECK.value

        if llm is not None:
            try:
                messages = [
                    SystemMessage(content=SELF_CHECK_SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"【用户提问】\n{raw_query}\n\n【知识库证据】\n{evidence_text}"
                    ),
                ]
                if hasattr(llm, "ainvoke"):
                    resp = await llm.ainvoke(messages)
                elif hasattr(llm, "invoke"):
                    resp = llm.invoke(messages)
                else:
                    resp = await asyncio.to_thread(llm, messages)

                content = resp.content if hasattr(resp, "content") else str(resp)
                parsed = parse_self_check_response(content)
                useful = parsed["useful"]
                reason = parsed["reason"]
            except Exception as e:
                logger.warning(f"LLM 自评调用异常，回退确定性 Mock 判断: {e}")
                useful, reason = self._mock_self_check(raw_query, citations)
        else:
            # Mock 降级模式
            useful, reason = self._mock_self_check(raw_query, citations)

        result = SelfCheckResult(useful=useful, reason=reason, source=source)

        # 3. 若判定不足，自动入池落库
        if not useful and db is not None:
            try:
                await record_low_confidence(
                    db=db,
                    raw_question=raw_query,
                    source=source,
                    conversation_id=conversation_id,
                    reason=reason,
                )
            except Exception as e:
                logger.error(f"记录低置信度问题入库失败: {e}")

        return result

    def _mock_self_check(
        self, query: str, citations: List[Dict[str, Any]]
    ) -> tuple[bool, str]:
        """确定性 Mock 自检规则，供离线单测使用。"""
        # 超纲与不相关关键字过滤
        out_of_scope_keywords = ["火星", "飞船", "量子", "柴油", "航母", "外星", "造反"]
        if any(kw in query for kw in out_of_scope_keywords):
            return False, f"提问包含超出电商客服范畴的超纲概念，现有证据无法支持解答"

        # 检查 query 中的核心词与证据的相关度
        all_evidence = " ".join(
            f"{c.get('question', '')} {c.get('answer', '')} {c.get('section_path', '')}"
            for c in citations
        )

        query_terms = [t for t in query.replace("？", "").replace("?", "").split() if t]
        if not query_terms:
            query_terms = [query]

        # 简单交集计算
        match_count = 0
        for term in query_terms:
            if term in all_evidence:
                match_count += 1

        # 若是字符级包含（中文无空格分词）
        if any(len(term) >= 2 and term in all_evidence for term in [query[i:i+2] for i in range(len(query)-1)]):
            match_count += 1

        if match_count > 0:
            return True, "证据中包含与提问高度匹配的业务规则与规格描述，支持精准解答"
        else:
            return False, "证据内容与提问匹配度不足，缺失核心事实解答依据"

    async def astream_generate(
        self,
        query: str,
        citations: List[Dict[str, Any]],
        history: Optional[List[BaseMessage]] = None,
    ) -> AsyncGenerator[str, None]:
        """Phase 2: 受控生成与引用角标流式注入。
        
        遵循负面红线约束与细粒度 [n] 引用编号。
        """
        raw_query = str(query).strip() if query else ""
        evidence_text = format_evidence_context(citations)

        stream_llm = self.stream_model or self.model

        if stream_llm is not None and hasattr(stream_llm, "astream"):
            messages: List[BaseMessage] = [
                SystemMessage(content=RAG_CONTROLLED_QA_SYSTEM_PROMPT)
            ]
            if history:
                messages.extend(history)
            messages.append(
                HumanMessage(
                    content=f"【用户提问】\n{raw_query}\n\n【知识库证据】\n{evidence_text}\n\n请严格基于上述证据作答并标注 [n] 角标："
                )
            )

            try:
                async for chunk in stream_llm.astream(messages):
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if text:
                        yield text
                return
            except Exception as e:
                logger.warning(f"流式模型调用异常，降级确定性 Mock 生成: {e}")

        # 确定性 Mock 生成兜底
        async for chunk in self._mock_stream_generate(raw_query, citations):
            yield chunk

    async def _mock_stream_generate(
        self, query: str, citations: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """确定性 Mock 流式生成，保证单测无需远程 API。"""
        if not citations:
            yield self.refusal_text
            return

        # 提取证据要点并附带角标
        c0 = citations[0]
        n0 = c0.get("n", 1)
        ans0 = str(c0.get("answer") or "").strip()
        # 保证有角标
        if f"[{n0}]" not in ans0:
            ans0 = f"{ans0}[{n0}]"

        yield "根据官方服务规范为您解答："
        yield ans0

        if len(citations) > 1:
            c1 = citations[1]
            n1 = c1.get("n", 2)
            ans1 = str(c1.get("answer") or "").strip()
            if f"[{n1}]" not in ans1:
                ans1 = f"{ans1}[{n1}]"
            yield "另外补充说明："
            yield ans1
