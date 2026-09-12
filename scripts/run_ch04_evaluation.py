#!/usr/bin/env python3
"""AstroBot Chapter 4: RAG 评估体系与四策略对比评测脚本。

支持四种检索策略评测：
- vector_only: 单路语义向量检索 (BGE-M3 Dense)
- bm25_only: 单路 Milvus 2.5 原生全文检索 (jieba BM25)
- hybrid: 双路召回 + RRF 融合
- hybrid_rerank: 双路召回 + RRF 融合 + BGE-Reranker-v2-m3 重排 + 首尾放置

用法示例：
    # 快速离线 Mock 验证（每桶采样 2 题）
    python scripts/run_ch04_evaluation.py --sample 2 --mock

    # 完整评测
    python scripts/run_ch04_evaluation.py --strategies vector_only bm25_only hybrid hybrid_rerank
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys

# 必须在初始化 logging 及任何标准流输出前强制配置 Windows UTF-8 编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 确保项目根目录在 sys.path 中
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.rag.evaluator import RAGEvaluator, render_markdown_report


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ch04_eval")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AstroBot RAG 四策略对比评估与编造台账生成工具"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["vector_only", "bm25_only", "hybrid", "hybrid_rerank"],
        help="评测的检索策略列表，默认为 vector_only bm25_only hybrid hybrid_rerank",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="每个分桶采样的题目数量（例如 --sample 5；默认 None 为全量评测）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="启用离线确定性 Mock 模式（无需外网模型和活跃 Milvus 服务，飞速跑通单测与脚本）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/ch04_evaluation_report.md",
        help="Markdown 评测报告输出路径（默认 reports/ch04_evaluation_report.md）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tests/data/eval_ch04.jsonl",
        help="评测集路径（默认 tests/data/eval_ch04.jsonl）",
    )
    return parser.parse_args()


async def run_evaluation(args):
    dataset_path = ROOT_DIR / args.dataset
    if not dataset_path.exists():
        logger.error(f"评测集文件不存在: {dataset_path}")
        return 1

    logger.info("=" * 60)
    logger.info("启动 AstroBot Chapter 4 RAG 体系对比评估")
    logger.info(f"数据集路径: {dataset_path}")
    logger.info(f"评估策略: {args.strategies}")
    logger.info(f"采样规格: {'全量 300 题' if args.sample is None else f'每桶 {args.sample} 题'}")
    logger.info(f"运行模式: {'离线确定性 Mock' if args.mock else '生产模型/服务'}")
    logger.info("=" * 60)

    evaluator = RAGEvaluator(
        eval_dataset_path=str(dataset_path),
        is_mock=args.mock,
    )

    report = await evaluator.run_comparative_eval(
        strategies=args.strategies,
        samples_per_bucket=args.sample,
    )

    # 渲染报告
    md_content = render_markdown_report(report)

    # 打印控制台表格摘要
    print("\n" + "=" * 78)
    print("评测对比矩阵摘要 (Recall@5 & MRR & Faithfulness)")
    print("=" * 78)
    header = f"{'策略 (Strategy)':<18} | {'Recall@5 (总体)':<16} | {'MRR (总体)':<14} | {'Faithfulness':<14} | {'D_absent 拒答率':<14}"
    print(header)
    print("-" * 78)

    for strat_name, sm in report.strategies.items():
        overall_r5 = f"{sm.overall.recall_at_5:.3f}" if sm.overall else "N/A"
        overall_mrr = f"{sm.overall.mrr:.3f}" if sm.overall else "N/A"
        overall_faith = f"{sm.overall.faithfulness:.3f}" if sm.overall else "N/A"
        d_metrics = sm.bucket_metrics.get("D_absent")
        refusal_rate = (
            f"{d_metrics.refusal_rate:.3f}"
            if (d_metrics and d_metrics.refusal_rate is not None)
            else "N/A"
        )
        print(
            f"{strat_name:<18} | {overall_r5:<16} | {overall_mrr:<14} | {overall_faith:<14} | {refusal_rate:<14}"
        )

    print("=" * 78 + "\n")

    # 写入报告文件
    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_content, encoding="utf-8")
    logger.info(f"评测报告已成功写入: {output_path}")

    return 0



def main():
    args = parse_args()
    exit_code = asyncio.run(run_evaluation(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
