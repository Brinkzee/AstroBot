"""历史客服对话知识挖掘管道服务。

负责从 MySQL 历史客服对话流水分批提取有效上下文，触发大模型结构化抽取 Q&A 问答对，
落库到 qa_extraction_staging 中转暂存表，执行规则精确去重与 BGE-M3 语义聚类去重，
并将最终保留项无缝转化为知识块，通过 KnowledgeDualWriter 双写落库至 MySQL 与 Milvus。
"""
import asyncio
from datetime import datetime
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import Message
from app.models.staging import QAExtractionStaging
from app.prompts.qa_extraction import (
    qa_extraction_prompt,
    parse_qa_extraction_output,
)
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.splitter import DocChunk

logger = logging.getLogger(__name__)


def _clean_question_text(text: str) -> str:
    """去除文本中的空白字符与标点符号，统一小写，用于规则层精确去重"""
    return re.sub(r"[\s\W_]+", "", text.lower().strip())


class DialogueKnowledgeMiner:
    """历史客服对话知识挖掘器。"""

    def __init__(
        self,
        llm: Optional[Any] = None,
        embedding_client: Optional[BGEEmbeddingClient] = None,
    ):
        if llm is None:
            from app.llm import get_chat_model
            self.llm = get_chat_model(temperature=0.0)
        else:
            self.llm = llm

        self.embedding_client = embedding_client or BGEEmbeddingClient()

    async def fetch_dialogue_batches(
        self,
        session: AsyncSession,
        batch_size: int = 20,
        conversation_ids: Optional[List[int]] = None,
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """从 conversations 与 messages 中按 conversation_id 聚合流水并分批。
        
        仅提取 user 与 assistant 角色的有效消息，排除 tool 等中间消息。
        每 batch_size 个完整会话打包为一个批次，打上唯一 batch_no = f"BATCH_{timestamp}_{idx}"。
        
        返回：
            List[Tuple[batch_no, List[conversation_dict]]]
        """
        stmt = (
            select(Message)
            .where(
                Message.role.in_(["user", "assistant"]),
                Message.content.is_not(None),
                Message.content != "",
            )
            .order_by(Message.conversation_id, Message.created_at, Message.id)
        )
        if conversation_ids:
            stmt = stmt.where(Message.conversation_id.in_(conversation_ids))

        result = await session.execute(stmt)
        messages = list(result.scalars().all())

        if not messages:
            return []

        # 按 conversation_id 分组保持先后顺序
        conv_groups: Dict[int, List[Message]] = {}
        for msg in messages:
            if msg.conversation_id not in conv_groups:
                conv_groups[msg.conversation_id] = []
            conv_groups[msg.conversation_id].append(msg)

        conversations: List[Dict[str, Any]] = []
        for conv_id, msg_list in conv_groups.items():
            lines = []
            for m in msg_list:
                role_label = "用户" if m.role == "user" else "客服"
                lines.append(f"{role_label}: {m.content.strip()}")
            dialogue_text = "\n".join(lines).strip()
            if not dialogue_text:
                continue

            conversations.append({
                "conversation_id": conv_id,
                "dialogue_text": dialogue_text,
                "source_ref": f"conversation_{conv_id}",
                "message_count": len(msg_list),
            })

        if not conversations:
            return []

        # 分批打包并赋予批次号
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        batches: List[Tuple[str, List[Dict[str, Any]]]] = []

        for idx in range(0, len(conversations), batch_size):
            batch_slice = conversations[idx : idx + batch_size]
            batch_idx = idx // batch_size + 1
            batch_no = f"BATCH_{now_str}_{batch_idx:03d}"
            batches.append((batch_no, batch_slice))

        return batches

    async def _call_llm_for_qa(self, dialogue_text: str) -> str:
        """调用大模型抽取通用问答对，返回原始输出字符串"""
        messages = qa_extraction_prompt.format_messages(dialogue_text=dialogue_text)
        if hasattr(self.llm, "ainvoke"):
            resp = await self.llm.ainvoke(messages)
            return resp.content if hasattr(resp, "content") else str(resp)
        elif hasattr(self.llm, "invoke"):
            resp = self.llm.invoke(messages)
            return resp.content if hasattr(resp, "content") else str(resp)
        elif callable(self.llm):
            res = self.llm(messages)
            if asyncio.iscoroutine(res):
                res = await res
            return res.content if hasattr(res, "content") else str(res)
        else:
            raise ValueError(f"不支持的 LLM 客户端类型: {type(self.llm)}")

    async def extract_and_stage(
        self,
        session: AsyncSession,
        dialogue_text: str,
        source_ref: str,
        batch_no: str,
    ) -> List[QAExtractionStaging]:
        """触发大模型抽取，Pydantic 校验解析后批量写入 qa_extraction_staging（状态 extracted）。"""
        if not dialogue_text or not dialogue_text.strip():
            return []

        raw_output = await self._call_llm_for_qa(dialogue_text)
        qa_pairs = parse_qa_extraction_output(raw_output)

        if not qa_pairs:
            logger.info(f"会话 {source_ref} 未抽取到有效问答对")
            return []

        staged_items: List[QAExtractionStaging] = []
        for pair in qa_pairs:
            item = QAExtractionStaging(
                batch_no=batch_no,
                source_ref=source_ref,
                question=pair.question.strip(),
                answer=pair.answer.strip(),
                status="extracted",
            )
            staged_items.append(item)

        session.add_all(staged_items)
        await session.commit()

        for item in staged_items:
            await session.refresh(item)

        logger.info(f"会话 {source_ref} 成功抽取并落库 {len(staged_items)} 条待去重问答（批次: {batch_no}）")
        return staged_items

    async def extract_batch_dialogues(
        self,
        session: AsyncSession,
        conversation_ids: Optional[List[int]] = None,
        batch_no: Optional[str] = None,
        batch_size: int = 20,
    ) -> int:
        """从指定或全部会话中提取问答并写入暂存表，返回写入暂存表的记录数"""
        batches = await self.fetch_dialogue_batches(
            session=session,
            batch_size=batch_size,
            conversation_ids=conversation_ids,
        )
        total_staged = 0
        for b_no, convs in batches:
            active_batch_no = batch_no or b_no
            for conv in convs:
                staged = await self.extract_and_stage(
                    session=session,
                    dialogue_text=conv["dialogue_text"],
                    source_ref=conv["source_ref"],
                    batch_no=active_batch_no,
                )
                total_staged += len(staged)
        return total_staged

    async def deduplicate_staging(
        self,
        session: AsyncSession,
        batch_no: Optional[str] = None,
        similarity_threshold: float = 0.92,
    ) -> Tuple[int, int]:
        """对暂存表中 status = 'extracted' 的问答执行规则与 BGE-M3 语义聚类去重。
        
        1. 规则层：文本清洗与精确去重，相同问题保留答案字数更长更详尽的问答对；
        2. 语义层：BGE-M3 余弦相似度 >= similarity_threshold 判定为同义重复，保留更优质项；
        3. 更新暂存表记录状态：保留项置为 'kept'，重复项置为 'discarded'。
        
        返回：
            Tuple[kept_count, discarded_count]
        """
        stmt = (
            select(QAExtractionStaging)
            .where(QAExtractionStaging.status == "extracted")
            .order_by(QAExtractionStaging.id)
        )
        if batch_no:
            stmt = stmt.where(QAExtractionStaging.batch_no == batch_no)

        result = await session.execute(stmt)
        extracted_items = list(result.scalars().all())

        if not extracted_items:
            logger.info("暂存表中无待去重的 extracted 记录")
            return 0, 0

        logger.info(f"开始去重暂存记录，待去重记录总数: {len(extracted_items)}")

        # 阶段 1: 规则层精确去重
        exact_groups: Dict[str, List[QAExtractionStaging]] = {}
        for item in extracted_items:
            clean_q = _clean_question_text(item.question)
            if clean_q not in exact_groups:
                exact_groups[clean_q] = []
            exact_groups[clean_q].append(item)

        candidates: List[QAExtractionStaging] = []
        for clean_q, group in exact_groups.items():
            if len(group) == 1:
                candidates.append(group[0])
            else:
                # 优先保留答案更长、更详尽的问答对
                sorted_group = sorted(
                    group,
                    key=lambda it: (len(it.answer.strip()), -it.id),
                    reverse=True,
                )
                candidates.append(sorted_group[0])
                # 重复项直接置为 discarded
                for dup in sorted_group[1:]:
                    dup.status = "discarded"

        # 阶段 2: 语义层 BGE-M3 余弦相似度聚类去重
        if not candidates:
            pass
        elif len(candidates) == 1:
            candidates[0].status = "kept"
        else:
            # 同样按答案详尽程度降序排序，确保两两冲突时优先保留更详细的回答
            candidates.sort(
                key=lambda it: (len(it.answer.strip()), -it.id),
                reverse=True,
            )
            q_texts = [it.question.strip() for it in candidates]
            vectors = await self.embedding_client.aembed_documents(q_texts)

            V = np.array(vectors, dtype=np.float32)
            # L2 归一化防抖动
            norms = np.linalg.norm(V, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            V = V / norms

            kept_candidates: List[QAExtractionStaging] = []
            kept_vectors: List[np.ndarray] = []

            for i, cand in enumerate(candidates):
                vec_i = V[i]
                is_duplicate = False
                for kv in kept_vectors:
                    sim = float(np.dot(vec_i, kv))
                    if sim >= similarity_threshold:
                        is_duplicate = True
                        break

                if is_duplicate:
                    cand.status = "discarded"
                else:
                    cand.status = "kept"
                    kept_candidates.append(cand)
                    kept_vectors.append(vec_i)

        await session.commit()

        kept_count = sum(1 for it in extracted_items if it.status == "kept")
        discarded_count = sum(1 for it in extracted_items if it.status == "discarded")

        logger.info(
            f"去重完成: 保留 (kept)={kept_count}, 丢弃 (discarded)={discarded_count}"
        )
        return kept_count, discarded_count

    async def ingest_kept_chunks(
        self,
        session: AsyncSession,
        dual_writer: KnowledgeDualWriter,
        batch_no: Optional[str] = None,
    ) -> int:
        """将暂存表中 status = 'kept' 的条目转化为 DocChunk 并通过 KnowledgeDualWriter 双写落库。
        
        返回：
            入库成功的知识块数量
        """
        stmt = (
            select(QAExtractionStaging)
            .where(QAExtractionStaging.status == "kept")
            .order_by(QAExtractionStaging.id)
        )
        if batch_no:
            stmt = stmt.where(QAExtractionStaging.batch_no == batch_no)

        result = await session.execute(stmt)
        kept_items = list(result.scalars().all())

        if not kept_items:
            logger.info("未发现 status = 'kept' 的知识块，无需入库")
            return 0

        chunks: List[DocChunk] = []
        for item in kept_items:
            sec_path = (
                f"客服对话挖掘 > {item.source_ref}"
                if item.source_ref
                else "客服对话挖掘"
            )
            chunk = DocChunk(
                category="历史问答挖掘",
                questions=item.question.strip(),
                answer=item.answer.strip(),
                section_path=sec_path,
                content_type="mined_qa",
                is_key_clause=False,
            )
            chunks.append(chunk)

        saved = await dual_writer.write_chunks(session, chunks)
        logger.info(f"成功将 {len(saved)} 条挖掘问答双写落库至知识库与向量库")
        return len(saved)
