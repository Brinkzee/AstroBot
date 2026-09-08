"""双写落库与断点续跑管理器。

实现 MySQL knowledge_chunks 原文权威源与 Milvus-Lite knowledge 向量库的四阶段双写落库、
双向链表前后指针绑定，以及断点重跑补偿修复机制 repair_pending_chunks。
"""
import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import KnowledgeChunk
from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.milvus_client import MilvusKnowledgeStore
from app.services.rag.splitter import DocChunk

logger = logging.getLogger(__name__)


def compose_embedding_text(category: str, questions: str, answer: str) -> str:
    """按统一规范拼装 Dense 向量化文本。
    
    规范格式：
    分类：{category}
    问题：{questions}
    内容：{answer}
    
    其余元数据（section_path, content_type, is_key_clause 等）只落库和存入 Payload，不进向量化文本。
    """
    return f"分类：{category}\n问题：{questions}\n内容：{answer}"


class KnowledgeDualWriter:
    """RAG 知识库双写管理器。
    
    管理 MySQL 与 Milvus 两阶段事务双写，并具备断点中断后的自愈补齐能力。
    """

    def __init__(
        self,
        store: Optional[MilvusKnowledgeStore] = None,
        embedding_client: Optional[BGEEmbeddingClient] = None,
        milvus_uri: Optional[str] = None,
    ):
        self.store = store or MilvusKnowledgeStore(uri=milvus_uri)
        self.embedding_client = embedding_client or BGEEmbeddingClient()

    async def write_chunks(
        self, session: AsyncSession, chunks: List[DocChunk]
    ) -> List[KnowledgeChunk]:
        """执行四阶段严格双写落库。
        
        1. 批量保存至 knowledge_chunks 表，初始状态为 'pending'；
        2. await session.flush() 获取自增 ID，并依序双向绑定 prev_chunk_id 与 next_chunk_id；
        3. await session.commit() 确保 MySQL 原文权威源持久化成功；
        4. 调用 BGE-M3 生成 1024 维密集语义向量；
        5. 组装 Milvus Payload records 并调用 store.upsert(records)（主键幂等）；
        6. 更新 MySQL 记录：vector_id = str(chunk.id), vectorize_status = 'done' 并提交事务；
        7. 返回已落库并向量化完毕的 KnowledgeChunk 实体列表。
        """
        if not chunks:
            return []

        # 阶段 1: 批量保存到 MySQL，初始状态固定为 'pending'
        db_chunks = [
            KnowledgeChunk(
                category=c.category,
                questions=c.questions,
                answer=c.answer,
                section_path=c.section_path,
                content_type=c.content_type,
                is_key_clause=c.is_key_clause,
                vectorize_status="pending",
                prev_chunk_id=None,
                next_chunk_id=None,
            )
            for c in chunks
        ]
        session.add_all(db_chunks)

        # 阶段 2: 获取自增 ID 并更新前后双向链表指针
        await session.flush()
        n = len(db_chunks)
        for i in range(n):
            db_chunks[i].prev_chunk_id = db_chunks[i - 1].id if i > 0 else None
            db_chunks[i].next_chunk_id = db_chunks[i + 1].id if i < n - 1 else None

        # 阶段 3: 确立 MySQL 原文权威源落库成功
        await session.commit()
        logger.info(f"MySQL 权威源成功保存 {n} 条待向量化知识块（状态: pending）")

        # 阶段 4: 拼装文本并向量化
        texts = [
            compose_embedding_text(c.category, c.questions, c.answer)
            for c in db_chunks
        ]
        vectors = await self.embedding_client.aembed_documents(texts)

        # 阶段 5: 组装 Milvus records 并执行 Upsert
        records = []
        for chunk, vec, text in zip(db_chunks, vectors, texts):
            records.append({
                "id": chunk.id,
                "vector": vec,
                "category": chunk.category,
                "questions": chunk.questions,
                "answer": chunk.answer,
                "chunk_text": text,
                "content_type": chunk.content_type,
                "is_key_clause": chunk.is_key_clause,
                "section_path": chunk.section_path,
                "prev_chunk_id": chunk.prev_chunk_id,
                "next_chunk_id": chunk.next_chunk_id,
            })
        self.store.upsert(records)
        logger.info(f"Milvus 向量库成功写入/更新 {len(records)} 条向量记录")

        # 阶段 6: 回填 vector_id 并更新状态为 'done'
        for chunk in db_chunks:
            chunk.vector_id = str(chunk.id)
            chunk.vectorize_status = "done"
        await session.commit()
        logger.info(f"MySQL 权威源成功回填 {n} 条知识块状态为 done")

        return db_chunks

    async def repair_pending_chunks(self, session: AsyncSession) -> int:
        """扫描并补偿修复所有遗留的 pending 块。
        
        用于建库或双写中断后的断点续跑：
        1. 查询所有 vectorize_status == 'pending' 的记录；
        2. 若无则返回 0；
        3. 对所有 pending 块重新向量化并 upsert 到 Milvus；
        4. 回填 vector_id 并将状态转为 'done'，提交事务；
        5. 返回修复的块数量。
        """
        stmt = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        result = await session.execute(stmt)
        pending_chunks = list(result.scalars().all())

        if not pending_chunks:
            logger.info("未发现遗留 pending 知识块，无需补偿修复")
            return 0

        logger.info(f"发现 {len(pending_chunks)} 条遗留 pending 知识块，开始断点续跑补偿...")

        texts = [
            compose_embedding_text(c.category, c.questions, c.answer)
            for c in pending_chunks
        ]
        vectors = await self.embedding_client.aembed_documents(texts)

        records = []
        for chunk, vec, text in zip(pending_chunks, vectors, texts):
            records.append({
                "id": chunk.id,
                "vector": vec,
                "category": chunk.category,
                "questions": chunk.questions,
                "answer": chunk.answer,
                "chunk_text": text,
                "content_type": chunk.content_type,
                "is_key_clause": chunk.is_key_clause,
                "section_path": chunk.section_path,
                "prev_chunk_id": chunk.prev_chunk_id,
                "next_chunk_id": chunk.next_chunk_id,
            })
        self.store.upsert(records)

        for chunk in pending_chunks:
            chunk.vector_id = str(chunk.id)
            chunk.vectorize_status = "done"
        await session.commit()

        logger.info(f"成功补全并修复 {len(pending_chunks)} 条 pending 知识块，状态已转为 done")
        return len(pending_chunks)

    def close(self):
        """关闭底层向量存储连接并释放资源"""
        if self.store is not None:
            self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
