#!/usr/bin/env python3
"""离线知识库构建与断点续跑工具。

扫描指定目录下的 Markdown 规范文档，通过 MarkdownStructureSplitter 进行结构感知切分，
并由 KnowledgeDualWriter 完成 MySQL 权威源与 Milvus-Lite 向量库的双写落库与自愈补偿。

用法示例：
    python scripts/build_knowledge_base.py --kb-dir data/kb
    python scripts/build_knowledge_base.py --clean
    python scripts/build_knowledge_base.py --repair-only
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

# 将项目根目录置入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import AsyncSessionLocal
from app.models.knowledge import KnowledgeChunk
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.milvus_client import MilvusKnowledgeStore
from app.services.rag.splitter import MarkdownStructureSplitter
from scripts.wsl_helper import ensure_mysql_ready

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_knowledge_base")


async def build_knowledge_base(
    kb_dir: str = "data/kb",
    clean: bool = False,
    repair_only: bool = False,
    milvus_uri: Optional[str] = None,
    chunk_size: int = 500,
    overlap_size: int = 80,
    session: Optional[AsyncSession] = None,
    writer: Optional[KnowledgeDualWriter] = None,
) -> Dict[str, int]:
    """执行知识库构建主流程与断点续跑补齐。
    
    返回统计信息：
    {
        "pre_repaired": int,
        "ingested_files": int,
        "total_chunks": int,
        "post_repaired": int,
    }
    """
    stats = {
        "pre_repaired": 0,
        "ingested_files": 0,
        "total_chunks": 0,
        "post_repaired": 0,
    }

    ensure_mysql_ready(verbose=False)

    own_writer = False
    if writer is None:
        writer = KnowledgeDualWriter(milvus_uri=milvus_uri)
        own_writer = True

    try:
        # 管理 session 生命周期
        async def _run_with_session(sess: AsyncSession):
            # 1. clean 模式：清空向量库和数据库现有数据
            if clean:
                logger.warning("启用 --clean 模式，清空向量集合与 MySQL 原文知识块...")
                writer.store.init_collection(drop_existing=True)
                await sess.execute(delete(KnowledgeChunk))
                await sess.commit()
                logger.info("已完成数据清空。")

            # 2. 任务前置自愈检查
            logger.info("执行任务前 pending 遗留块检查...")
            stats["pre_repaired"] = await writer.repair_pending_chunks(sess)
            if stats["pre_repaired"] > 0:
                logger.info(f"前置检查修复了 {stats['pre_repaired']} 条 pending 块")

            if repair_only:
                logger.info("--repair-only 模式，任务已完成。")
                return

            # 3. 扫描目录下的所有 .md 文件
            kb_path = Path(kb_dir)
            if not kb_path.is_absolute():
                kb_path = PROJECT_ROOT / kb_path

            if not kb_path.exists() or not kb_path.is_dir():
                logger.warning(f"指定知识库目录不存在: {kb_path}")
                return

            md_files = sorted(list(kb_path.glob("*.md")))
            logger.info(f"在 {kb_path} 发现 {len(md_files)} 个 Markdown 知识文档")

            splitter = MarkdownStructureSplitter(
                chunk_size=chunk_size,
                overlap_size=overlap_size,
            )

            for md_file in md_files:
                logger.info(f"正在处理文档: {md_file.name}")
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                doc_chunks = splitter.split_text(content)
                if not doc_chunks:
                    logger.info(f"文档 {md_file.name} 切分后无有效 chunk，跳过")
                    continue

                logger.info(f"文档 {md_file.name} 切分为 {len(doc_chunks)} 个 chunk，开始双写落库...")
                saved = await writer.write_chunks(sess, doc_chunks)
                stats["ingested_files"] += 1
                stats["total_chunks"] += len(saved)

            # 4. 任务后置自愈检查，确保无遗漏
            logger.info("执行任务后 pending 遗留块检查...")
            stats["post_repaired"] = await writer.repair_pending_chunks(sess)
            if stats["post_repaired"] > 0:
                logger.info(f"后置检查修复了 {stats['post_repaired']} 条 pending 块")

        if session is not None:
            await _run_with_session(session)
        else:
            async with AsyncSessionLocal() as sess:
                await _run_with_session(sess)

    finally:
        if own_writer:
            writer.close()

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="离线知识库构建与断点续跑管理工具")
    parser.add_argument(
        "--kb-dir",
        type=str,
        default="data/kb",
        help="待入库 Markdown 知识文档目录，默认 data/kb",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="构建前清空向量集合与原文知识块",
    )
    parser.add_argument(
        "--repair-only",
        action="store_true",
        help="仅执行 pending 遗留块断点续跑与补偿修复",
    )
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default=None,
        help="自定义 Milvus-Lite 本地路径或服务连接串",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="正文切分大小，默认 500",
    )
    parser.add_argument(
        "--overlap-size",
        type=int,
        default=80,
        help="重叠窗口大小，默认 80",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    logger.info("====== 开始执行知识库构建任务 ======")
    stats = await build_knowledge_base(
        kb_dir=args.kb_dir,
        clean=args.clean,
        repair_only=args.repair_only,
        milvus_uri=args.milvus_uri,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
    )
    logger.info(
        f"====== 任务执行完毕: 前置修复={stats['pre_repaired']}, "
        f"解析入库文件={stats['ingested_files']}, "
        f"写入 Chunk 总数={stats['total_chunks']}, "
        f"后置修复={stats['post_repaired']} ======"
    )


if __name__ == "__main__":
    asyncio.run(main())
