#!/usr/bin/env python3
"""Chapter 03 端到端验收自动化脚本。

自动化执行三大核心验收：
1. 验收标准 1:「邮费是多少」换说法语义泛化召回运费说明并让客服答对；
2. 验收标准 2: 故意中断建库任务制造漏向量化数据，重跑后漏网块全部被捡起补齐（双写最终一致）；
3. 数据库与向量库 1:1 数量完全一致性对齐验证；
4. 客服大模型端到端利用 query_faq 工具回答运费规范实测。

运行方法：
    python scripts/verify_ch03_acceptance.py
"""
import asyncio
import sys
from pathlib import Path
from sqlalchemy import or_, select, func

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import AsyncSessionLocal, engine
from app.models.knowledge import KnowledgeChunk
from app.services.chat_service import ChatService
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.services.rag.milvus_client import MilvusKnowledgeStore
from app.tools.business_tools import query_faq, get_retriever
from scripts.build_knowledge_base import build_knowledge_base
from scripts.wsl_helper import ensure_mysql_ready


async def run_acceptance():
    print("=" * 75)
    print("🚀 [Chapter 03] RAG 知识库与向量语义检索端到端双重验收开始")
    print("=" * 75)

    # 1. 确保 MySQL 容器和保活进程就绪
    print("\n【步骤 1】环境与存储就绪状态检查...")
    ensure_mysql_ready(verbose=True)

    # 2. 检查并确保知识库灌库
    print("\n【步骤 2】检查知识库灌库状态...")
    async with AsyncSessionLocal() as session:
        count_res = await session.execute(
            select(func.count(KnowledgeChunk.id)).where(KnowledgeChunk.vectorize_status == "done")
        )
        existing_count = count_res.scalar() or 0

    if existing_count == 0:
        print("检测到知识库为空，正在执行全量知识库构建与双写落库...")
        stats = await build_knowledge_base(kb_dir="data/kb", clean=True)
        print(f"全量建库完成: 写入 {stats['total_chunks']} 个知识块。")
    else:
        print(f"知识库已包含 {existing_count} 条已完成向量化记录。")

    # 3. 验收标准 1: 语义泛化检索与字面匹配对比
    print("\n" + "=" * 75)
    print("【验收标准 1】「邮费是多少」换说法语义泛化召回运费说明")
    print("-" * 75)
    query_text = "邮费是多少"
    print(f"用户自然提问: \"{query_text}\" (文档原文标题为: \"商城配送与运费服务规范 / 运费标准与偏远地区配送说明\")")

    # 3.1 验证传统 SQL LIKE 发生语义缺失 (MISS)
    async with AsyncSessionLocal() as session:
        stmt = select(KnowledgeChunk).where(
            or_(
                KnowledgeChunk.questions.like(f"%{query_text}%"),
                KnowledgeChunk.answer.like(f"%{query_text}%"),
            )
        )
        res = await session.execute(stmt)
        like_chunks = res.scalars().all()

    print(f"  ▶ 传统 SQL LIKE 匹配结果: 命中了 {len(like_chunks)} 条")
    if len(like_chunks) == 0:
        print("    ↳ [符合预期] SQL LIKE 因无'邮费'字样发生语义 MISS，传统关键词无法匹配！")
    else:
        print("    ↳ [警告] SQL LIKE 意外命中了记录")

    # 3.2 验证 BGE-M3 Dense 语义向量检索命中
    faq_answer = await query_faq.ainvoke({"keyword": query_text})
    print(f"  ▶ query_faq 向量语义检索返回:\n{faq_answer}")
    assert "未找到" not in faq_answer, "Dense 向量检索未命中知识块！"
    assert ("运费" in faq_answer or "包邮" in faq_answer), "语义检索结果应包含运费或包邮条款"
    assert "88元" in faq_answer, "语义检索结果应包含满88元包邮关键规则"
    print("\n  ✅ 【验收标准 1 验证通过】: BGE-M3 成功跨越词汇鸿沟，精准命中运费服务规范！")

    # 4. 验收标准 2: 断点中断与自愈补齐（双写最终一致）
    print("\n" + "=" * 75)
    print("【验收标准 2】故意中断建库任务制造漏向量化数据，重跑后漏网块全部被捡起补齐")
    print("-" * 75)
    writer = KnowledgeDualWriter()
    test_chunk_id = None
    try:
        # 4.1 人为制造中断遗留的 pending 孤立块
        async with AsyncSessionLocal() as session:
            orphan_chunk = KnowledgeChunk(
                category="商城配送与运费服务规范",
                questions="断点续跑容灾演练问题",
                answer="容灾演练正文：模拟知识写入MySQL后因网络闪断未进入Milvus，保持pending状态。",
                section_path="商城配送与运费服务规范 > 容灾演练",
                content_type="policy",
                is_key_clause=False,
                vectorize_status="pending",
                vector_id=None,
            )
            session.add(orphan_chunk)
            await session.commit()
            await session.refresh(orphan_chunk)
            test_chunk_id = orphan_chunk.id

        print(f"  ▶ [中断注入] 模拟系统故障，已在 MySQL 中成功注入 pending 孤立块 (ID: {test_chunk_id})")

        # 4.2 执行断点续跑自愈逻辑
        print("  ▶ [自愈补偿] 启动 repair_pending_chunks 补偿扫描与重跑...")
        async with AsyncSessionLocal() as session:
            repaired_count = await writer.repair_pending_chunks(session)
        print(f"    ↳ 自愈程序检出并补偿修复了 {repaired_count} 条 pending 知识块")
        assert repaired_count >= 1, "自愈程序应至少修复 1 条 pending 块"

        # 4.3 检验状态翻转与向量 ID 对齐
        async with AsyncSessionLocal() as session:
            verified_chunk = await session.get(KnowledgeChunk, test_chunk_id)
            assert verified_chunk is not None
            assert verified_chunk.vectorize_status == "done"
            assert verified_chunk.vector_id == str(test_chunk_id)
            print(f"    ↳ 知识块 ID={test_chunk_id} 状态已成功翻转: vectorize_status='{verified_chunk.vectorize_status}', vector_id='{verified_chunk.vector_id}'")

        # 4.4 再次自愈确认无残留
        async with AsyncSessionLocal() as session:
            second_repair = await writer.repair_pending_chunks(session)
        assert second_repair == 0, "再次检查应无任何残留 pending 块"
        print("  ✅ 【验收标准 2 验证通过】: 漏向量化数据 100% 自动检出、补齐向量并最终一致！")
    finally:
        writer.close()

    # 5. 存储对齐一致性校验
    print("\n" + "=" * 75)
    print("【权威源与向量存储对齐检查】")
    print("-" * 75)
    async with AsyncSessionLocal() as session:
        mysql_count_res = await session.execute(
            select(func.count(KnowledgeChunk.id)).where(KnowledgeChunk.vectorize_status == "done")
        )
        mysql_done_count = mysql_count_res.scalar() or 0

    store = MilvusKnowledgeStore()
    try:
        milvus_count = store.count("knowledge")
    finally:
        store.close()

    print(f"  ▶ MySQL knowledge_chunks 表中 done 记录数: {mysql_done_count}")
    print(f"  ▶ Milvus knowledge 集合中向量记录总数:       {milvus_count}")
    assert mysql_done_count == milvus_count, f"MySQL 权威源与 Milvus 向量库记录不一致: {mysql_done_count} vs {milvus_count}"
    print("  ✅ 【数据对齐验证通过】: MySQL 原文权威源与 Milvus 向量库达到严格 1:1 双写一致！")

    # 6. 大模型客服端到端完整问答实测
    print("\n" + "=" * 75)
    print("【端到端实测】客服大模型调用 query_faq 回答运费政策")
    print("-" * 75)
    user_prompt = "请问商城邮费是多少？偏远地区如新疆西藏包邮吗？"
    print(f"客户输入: \"{user_prompt}\"")
    service = ChatService()
    full_response = ""
    tool_called = None

    async with AsyncSessionLocal() as session:
        async for event in service.stream_chat(session, conversation_id=None, message=user_prompt):
            etype = event.get("event_type")
            if etype == "tool_start":
                tool_called = event.get("tool_name")
                print(f"  🚀 [SSE 状态帧] 工具调用开始: {tool_called} | 参数: {event.get('args')}")
            elif etype == "tool_end":
                print(f"  ✅ [SSE 状态帧] 工具调用成功: {event.get('tool_name')}")
            elif etype == "text":
                content = event.get("content", "")
                full_response += content
                sys.stdout.write(content)
                sys.stdout.flush()

    print("\n" + "-" * 75)
    assert tool_called == "query_faq", f"预期调用 query_faq，实际调用: {tool_called}"
    assert any(kw in full_response for kw in ["88", "八十八"]), "最终回复应包含88元包邮规则"
    assert any(kw in full_response for kw in ["15", "十五", "偏远", "加收"]), "最终回复应包含偏远地区加收15元说明"
    print("  ✅ 【大模型问答验证通过】: 大模型基于向量检索结果提供了准确专业的答复！")

    # 释放数据库连接池
    await engine.dispose()

    print("\n" + "=" * 75)
    print("🎉 Chapter 03 RAG 知识库与密集向量检索四大核心验收全部成功通过！")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(run_acceptance())
