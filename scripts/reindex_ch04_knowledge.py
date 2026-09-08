#!/usr/bin/env python3
"""全量知识库数据迁移与 Milvus 2.5 混合检索集合刷库脚本。

从 MySQL knowledge_chunks 读取权威源数据，按规范组装 chunk_text 结构化问答文本，
使用 BGEEmbeddingClient 生成 1024 维 Dense 语义向量，并在 Milvus 中重建 knowledge 集合，
通过 原生 jieba BM25 + Dense 索引批量灌库。

用法示例：
    python scripts/reindex_ch04_knowledge.py
    python scripts/reindex_ch04_knowledge.py --milvus-uri ./data/milvus/astro_bot.db --batch-size 32
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# 将项目根目录置入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import AsyncSessionLocal
from app.models.knowledge import KnowledgeChunk
from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.milvus_client import MilvusKnowledgeStore
from scripts.wsl_helper import ensure_mysql_ready

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reindex_ch04_knowledge")


def compose_chunk_text(category: str, questions: str, answer: str) -> str:
    """按规范组装包含类目、标准问法与解答的全文索引文本。
    
    格式：
    【类目】{category}
    【标准问法】{questions}
    【解答】{answer}
    """
    cat = category or ""
    q = questions or ""
    a = answer or ""
    return f"【类目】{cat}\n【标准问法】{q}\n【解答】{a}"


async def reindex_knowledge_chunks(
    session: Optional[AsyncSession] = None,
    store: Optional[MilvusKnowledgeStore] = None,
    embedding_client: Optional[BGEEmbeddingClient] = None,
    milvus_uri: Optional[str] = None,
    collection_name: str = "knowledge",
    batch_size: int = 32,
) -> Dict[str, int]:
    """从 MySQL 查询全量 knowledge_chunks 并重建 Milvus 混合检索集合。
    
    Returns:
        统计字典: {"total_chunks": int, "indexed_chunks": int}
    """
    stats = {"total_chunks": 0, "indexed_chunks": 0}

    own_store = False
    if store is None:
        store = MilvusKnowledgeStore(uri=milvus_uri)
        own_store = True

    if embedding_client is None:
        embedding_client = BGEEmbeddingClient()

    async def _execute_reindex(sess: AsyncSession):
        logger.info("从 MySQL knowledge_chunks 查询全量知识块...")
        stmt = select(KnowledgeChunk).order_by(KnowledgeChunk.id)
        res = await sess.execute(stmt)
        chunks: List[KnowledgeChunk] = list(res.scalars().all())

        stats["total_chunks"] = len(chunks)
        logger.info(f"成功查询到 {len(chunks)} 条知识块记录")

        # 重新创建集合（删除已有，建立 BM25 + Dense 双向量集合）
        logger.info(f"重新初始化 Milvus 集合 {collection_name} (drop_existing=True)...")
        store.init_collection(collection_name=collection_name, drop_existing=True)

        if not chunks:
            logger.warning("知识库记录为空，无需迁移数据。")
            return

        # 分批向量化并写入
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = [
                compose_chunk_text(c.category, c.questions, c.answer)
                for c in batch_chunks
            ]

            # 批量获取 Dense 向量
            logger.info(f"正在为批次 [{i + 1} ~ {min(i + batch_size, total)}/{total}] 生成 1024 维 Dense 语义向量...")
            batch_vectors = await embedding_client.aembed_documents(batch_texts)

            # 组装 Milvus Payload records
            records = []
            for chunk, text, vec in zip(batch_chunks, batch_texts, batch_vectors):
                records.append({
                    "id": chunk.id,
                    "vector": vec,
                    "chunk_text": text,
                    "category": chunk.category,
                    "section_path": chunk.section_path,
                    "questions": chunk.questions,
                    "answer": chunk.answer,
                    "content_type": chunk.content_type,
                    "is_key_clause": chunk.is_key_clause,
                    "prev_chunk_id": chunk.prev_chunk_id,
                    "next_chunk_id": chunk.next_chunk_id,
                })

            upserted = store.upsert(records, collection_name=collection_name)
            stats["indexed_chunks"] += upserted
            logger.info(f"批次 [{i + 1} ~ {min(i + batch_size, total)}] 写入 Milvus 成功，本批写入 {upserted} 条")

        logger.info(f"全量知识库迁移完成！总处理: {stats['total_chunks']}, 写入 Milvus: {stats['indexed_chunks']}")

    try:
        if session is not None:
            await _execute_reindex(session)
        else:
            ensure_mysql_ready(verbose=False)
            async with AsyncSessionLocal() as sess:
                await _execute_reindex(sess)
    finally:
        if own_store:
            store.close()

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="全量知识库数据迁移与 Milvus 2.5 混合检索集合刷库工具")
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default=None,
        help="Milvus 连接 URI，默认使用系统配置",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="knowledge",
        help="目标向量集合名称，默认 knowledge",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批处理大小，默认 32",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    logger.info("====== 开始执行 Milvus 2.5 混合检索集合刷库 ======")
    stats = await reindex_knowledge_chunks(
        milvus_uri=args.milvus_uri,
        collection_name=args.collection,
        batch_size=args.batch_size,
    )
    logger.info(f"====== 刷库任务完成: 扫描知识块={stats['total_chunks']}, 索引入库={stats['indexed_chunks']} ======")


if __name__ == "__main__":
    asyncio.run(main())
