#!/usr/bin/env python3
"""第十章旁路批量归类流水线 (scripts/classify_pool.py)。

负责从 low_confidence_questions 表捞取未归类数据，
通过批次闸门（--min-batch 10，支持 --force 强制执行）控制，
分批调用独立推理服务 (:8110) 的 POST /classify 接口，
并将预测的多标签结果幂等落库写入 topic_classifications 表。

主链路零干扰：实时对话主链路绝不调用分类器，仅由本批处理流水线旁路消费。
"""

import argparse
import asyncio
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 将项目根目录置入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.models.low_confidence import LowConfidenceQuestion
from app.models.topic_classification import TopicClassification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("classify_pool")

DEFAULT_MIN_BATCH = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_SERVICE_URL = "http://127.0.0.1:8110"


async def fetch_unclassified_questions(db: AsyncSession) -> Sequence[Tuple[int, str]]:
    """查询 low_confidence_questions 中尚未归类的问题记录。

    执行查询：
    SELECT q.id, q.raw_question
    FROM low_confidence_questions q
    LEFT JOIN topic_classifications t ON q.id = t.question_id
    WHERE t.id IS NULL
    ORDER BY q.id ASC
    """
    stmt = (
        select(LowConfidenceQuestion.id, LowConfidenceQuestion.raw_question)
        .outerjoin(TopicClassification, LowConfidenceQuestion.id == TopicClassification.question_id)
        .where(TopicClassification.id.is_(None))
        .order_by(LowConfidenceQuestion.id.asc())
    )
    result = await db.execute(stmt)
    return result.all()


async def call_classifier_service(
    service_url: str,
    texts: List[str],
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """分批调用独立推理服务 (:8110) 的 POST /classify 接口获取分类结果。

    :param service_url: 分类服务基础 URL，例如 http://127.0.0.1:8110
    :param texts: 待分类文本列表
    :param timeout: 请求超时时间 (秒)
    :return: 包含 labels 与 scores 的结果字典列表
    """
    endpoint = f"{service_url.rstrip('/')}/classify"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(endpoint, json={"texts": texts})
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            elif isinstance(data, list):
                return data
            return []
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        msg = (
            f"无法连接到分类推理服务 ({service_url})。"
            f"请先启动推理服务 (例如 make classifier-up 或 python scripts/classifier_service.py start)。"
        )
        logger.error(msg)
        raise ConnectionError(msg) from e
    except httpx.HTTPStatusError as e:
        msg = f"分类推理服务返回异常状态码: {e.response.status_code}, 详情: {e.response.text}"
        logger.error(msg)
        raise RuntimeError(msg) from e
    except Exception as e:
        msg = f"调用分类推理服务失败: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e


def print_summary(
    total_unclassified: int,
    processed: int,
    topic_counts: Dict[str, int],
    elapsed_sec: float,
    skipped_due_to_min_batch: bool,
    min_batch: int,
) -> None:
    """打印控制台格式化统计摘要。"""
    print("\n" + "=" * 60)
    print("      Chapter 10 旁路批量归类流水线报告")
    print("=" * 60)
    print(f"未归类问题总数 : {total_unclassified}")
    if skipped_due_to_min_batch:
        print(f"执行状态       : [跳过] 未达到最小批次门槛 (min_batch={min_batch})")
        print("提示           : 使用 --force 参数可强制立即归类。")
    else:
        print(f"实际处理条数   : {processed}")
        print(f"耗时           : {elapsed_sec:.2f}s")
        print("各主题命中统计 :")
        if not topic_counts:
            print("  (无命中标签或未处理任何记录)")
        else:
            sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
            for topic, count in sorted_topics:
                print(f"  - {topic}: {count}")
    print("=" * 60 + "\n")


async def run_classify_pool(
    db: Optional[AsyncSession] = None,
    min_batch: int = DEFAULT_MIN_BATCH,
    force: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    service_url: str = DEFAULT_SERVICE_URL,
) -> Dict[str, Any]:
    """执行旁路批量归类流水线。

    :param db: 可选的 AsyncSession，若未提供则使用 AsyncSessionLocal()
    :param min_batch: 最小批次门槛，未达到且未开启 force 时跳过执行
    :param force: 强制执行，忽略 min_batch 门槛
    :param batch_size: 单次调用推理服务的批次大小
    :param service_url: 分类推理服务地址
    :return: 统计字典
    """
    if db is None:
        async with AsyncSessionLocal() as session:
            return await _run_classify_pool_core(
                db=session,
                min_batch=min_batch,
                force=force,
                batch_size=batch_size,
                service_url=service_url,
            )
    else:
        return await _run_classify_pool_core(
            db=db,
            min_batch=min_batch,
            force=force,
            batch_size=batch_size,
            service_url=service_url,
        )


async def _run_classify_pool_core(
    db: AsyncSession,
    min_batch: int,
    force: bool,
    batch_size: int,
    service_url: str,
) -> Dict[str, Any]:
    start_time = time.time()

    # 1. 捞取未归类数据
    unclassified = await fetch_unclassified_questions(db)
    total_unclassified = len(unclassified)
    logger.info(f"捞取到未归类低置信度问题: {total_unclassified} 条")

    # 2. 批次闸门控制
    if total_unclassified == 0:
        elapsed = round(time.time() - start_time, 4)
        print_summary(
            total_unclassified=0,
            processed=0,
            topic_counts={},
            elapsed_sec=elapsed,
            skipped_due_to_min_batch=False,
            min_batch=min_batch,
        )
        return {
            "total_unclassified": 0,
            "processed": 0,
            "skipped_due_to_min_batch": False,
            "topic_counts": {},
            "elapsed_sec": elapsed,
        }

    if total_unclassified < min_batch and not force:
        elapsed = round(time.time() - start_time, 4)
        logger.info(
            f"未归类问题数量 ({total_unclassified}) 未达到最小批次门槛 ({min_batch})，跳过执行。"
            f"使用 --force 可强制归类。"
        )
        print_summary(
            total_unclassified=total_unclassified,
            processed=0,
            topic_counts={},
            elapsed_sec=elapsed,
            skipped_due_to_min_batch=True,
            min_batch=min_batch,
        )
        return {
            "total_unclassified": total_unclassified,
            "processed": 0,
            "skipped_due_to_min_batch": True,
            "topic_counts": {},
            "elapsed_sec": elapsed,
        }

    # 3. 分批调用推理服务并幂等落库
    processed = 0
    topic_counts: Dict[str, int] = {}

    for i in range(0, total_unclassified, batch_size):
        batch = unclassified[i : i + batch_size]
        batch_texts = [row[1] for row in batch]
        batch_qids = [row[0] for row in batch]

        logger.info(f"正在处理批次 [{i + 1} ~ {i + len(batch)} / {total_unclassified}]...")
        call_results = await call_classifier_service(service_url=service_url, texts=batch_texts)

        # 兼容单条结果广播 (如 mock 仅返回单条或测试桩)
        if len(call_results) == 1 and len(batch_texts) > 1:
            call_results = [call_results[0]] * len(batch_texts)

        # 查询本批次中已存在的 question_id (幂等双重保险)
        existing_stmt = select(TopicClassification.question_id).where(
            TopicClassification.question_id.in_(batch_qids)
        )
        existing_res = await db.execute(existing_stmt)
        existing_qids = set(existing_res.scalars().all())

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for (qid, _raw_q), item in zip(batch, call_results):
            if qid in existing_qids:
                logger.debug(f"问题 ID {qid} 已存在分类记录，跳过插入。")
                continue

            labels = item.get("labels", [])
            tc = TopicClassification(
                question_id=qid,
                labels=labels,
                classified_at=now,
            )
            db.add(tc)
            processed += 1

            for label in labels:
                topic_counts[label] = topic_counts.get(label, 0) + 1

        await db.commit()

    elapsed = round(time.time() - start_time, 4)
    print_summary(
        total_unclassified=total_unclassified,
        processed=processed,
        topic_counts=topic_counts,
        elapsed_sec=elapsed,
        skipped_due_to_min_batch=False,
        min_batch=min_batch,
    )

    return {
        "total_unclassified": total_unclassified,
        "processed": processed,
        "skipped_due_to_min_batch": False,
        "topic_counts": topic_counts,
        "elapsed_sec": elapsed,
    }


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Chapter 10 旁路批量归类流水线: 从 low_confidence_questions 捞取未归类数据批量推理并落库",
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=DEFAULT_MIN_BATCH,
        help=f"最小批次门槛（默认: {DEFAULT_MIN_BATCH}），未达到且未开启 --force 时跳过归类",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制执行归类，忽略 min-batch 门槛",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"单次调用推理服务的批次大小（默认: {DEFAULT_BATCH_SIZE}）",
    )
    parser.add_argument(
        "--service-url",
        type=str,
        default=DEFAULT_SERVICE_URL,
        help=f"分类推理服务 URL（默认: {DEFAULT_SERVICE_URL}）",
    )
    return parser.parse_args(args)


async def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口函数。"""
    args = parse_args(argv)
    try:
        result = await run_classify_pool(
            min_batch=args.min_batch,
            force=args.force,
            batch_size=args.batch_size,
            service_url=args.service_url,
        )
        return 0
    except ConnectionError as e:
        logger.error(f"[ERROR] {e}")
        return 1
    except Exception as e:
        logger.exception(f"[FATAL] 批量归类流水线执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
