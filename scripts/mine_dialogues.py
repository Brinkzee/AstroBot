#!/usr/bin/env python3
"""离线历史客服对话知识挖掘调度脚本。

支持端到端全流程执行，以及分阶段（extract 抽取暂存 / dedup 规则与语义去重 / ingest 双写落库）独立调度。

用法示例：
    # 执行完整挖掘管道（抽取 -> 去重 -> 入库）
    python scripts/mine_dialogues.py --batch-size 20 --similarity-threshold 0.92

    # 仅执行分批抽取并落入暂存表
    python scripts/mine_dialogues.py --step extract --batch-size 20

    # 仅执行暂存表聚类去重
    python scripts/mine_dialogues.py --step dedup --similarity-threshold 0.92

    # 仅将保留项（status=kept）转入知识库双写落库
    python scripts/mine_dialogues.py --step ingest
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 将项目根目录置入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import AsyncSessionLocal
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.miner import DialogueKnowledgeMiner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mine_dialogues")


async def run_mining_pipeline(
    session: Optional[AsyncSession] = None,
    miner: Optional[DialogueKnowledgeMiner] = None,
    dual_writer: Optional[KnowledgeDualWriter] = None,
    batch_size: int = 20,
    similarity_threshold: float = 0.92,
    batch_no: Optional[str] = None,
    step: str = "all",
    milvus_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """执行历史对话挖掘管道。
    
    返回统计字典：
    {
        "batches_fetched": int,
        "conversations_scanned": int,
        "staged_count": int,
        "kept_count": int,
        "discarded_count": int,
        "ingested_count": int,
    }
    """
    stats = {
        "batches_fetched": 0,
        "conversations_scanned": 0,
        "staged_count": 0,
        "kept_count": 0,
        "discarded_count": 0,
        "ingested_count": 0,
    }

    own_writer = False
    if dual_writer is None:
        dual_writer = KnowledgeDualWriter(milvus_uri=milvus_uri)
        own_writer = True

    if miner is None:
        miner = DialogueKnowledgeMiner()

    try:
        async def _execute_pipeline(sess: AsyncSession):
            # 步骤 1: 抽取与暂存 (extract)
            if step in ("all", "extract"):
                logger.info(f"开始读取历史客服对话（batch_size={batch_size}）...")
                batches = await miner.fetch_dialogue_batches(sess, batch_size=batch_size)
                stats["batches_fetched"] = len(batches)
                logger.info(f"共打包 {len(batches)} 个批次待抽取")

                for b_no, convs in batches:
                    active_batch_no = batch_no or b_no
                    stats["conversations_scanned"] += len(convs)
                    logger.info(f"正在处理批次 {active_batch_no}，包含 {len(convs)} 个会话...")

                    for conv in convs:
                        staged_items = await miner.extract_and_stage(
                            session=sess,
                            dialogue_text=conv["dialogue_text"],
                            source_ref=conv["source_ref"],
                            batch_no=active_batch_no,
                        )
                        stats["staged_count"] += len(staged_items)

                logger.info(f"抽取完成：暂存写入 {stats['staged_count']} 条问答记录")

            # 步骤 2: 规则与语义聚类去重 (dedup)
            if step in ("all", "dedup"):
                logger.info(
                    f"开始执行暂存表去重（batch_no={batch_no}, threshold={similarity_threshold}）..."
                )
                kept, discarded = await miner.deduplicate_staging(
                    session=sess,
                    batch_no=batch_no,
                    similarity_threshold=similarity_threshold,
                )
                stats["kept_count"] = kept
                stats["discarded_count"] = discarded
                logger.info(f"去重完成：保留 {kept} 条，丢弃重复 {discarded} 条")

            # 步骤 3: 转化知识块并双写落库 (ingest)
            if step in ("all", "ingest"):
                logger.info(f"开始将保留问答（status=kept）转入知识库与向量库...")
                ingested = await miner.ingest_kept_chunks(
                    session=sess,
                    dual_writer=dual_writer,
                    batch_no=batch_no,
                )
                stats["ingested_count"] = ingested
                logger.info(f"双写落库完成：共写入 {ingested} 条 mined_qa 知识块")

        if session is not None:
            await _execute_pipeline(session)
        else:
            async with AsyncSessionLocal() as sess:
                await _execute_pipeline(sess)

    finally:
        if own_writer:
            dual_writer.close()

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="历史客服对话知识挖掘与去重工具")
    parser.add_argument(
        "--step",
        choices=["all", "extract", "dedup", "ingest"],
        default="all",
        help="执行管道阶段: all=全流程, extract=仅抽取, dedup=仅去重, ingest=仅入库",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="会话分批大小，默认 20",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.92,
        help="BGE-M3 余弦相似度同义去重阈值，默认 0.92",
    )
    parser.add_argument(
        "--batch-no",
        type=str,
        default=None,
        help="指定特定批次号执行去重或入库",
    )
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default=None,
        help="自定义 Milvus-Lite 本地路径或服务连接串",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    logger.info(f"====== 开始执行客服对话挖掘任务（阶段: {args.step}） ======")
    stats = await run_mining_pipeline(
        batch_size=args.batch_size,
        similarity_threshold=args.similarity_threshold,
        batch_no=args.batch_no,
        step=args.step,
        milvus_uri=args.milvus_uri,
    )
    logger.info(
        f"====== 任务执行完毕: 分批数={stats['batches_fetched']}, "
        f"扫描会话={stats['conversations_scanned']}, "
        f"抽取暂存={stats['staged_count']}, "
        f"去重保留={stats['kept_count']}, "
        f"去重丢弃={stats['discarded_count']}, "
        f"双写落库={stats['ingested_count']} ======"
    )


if __name__ == "__main__":
    asyncio.run(main())
