"""Chapter 03 端到端双重业务验收测试用例。

验收标准 1: 「邮费是多少」换说法语义泛化召回运费说明并让客服答对（BGE-M3 Dense 语义召回 vs SQL LIKE MISS）
验收标准 2: 故意中断建库任务制造漏向量化数据，重跑后漏网块全部被捡起补齐（双写最终一致性）
验收标准 3: 大模型端到端通过 query_faq 检索知识库并准确回答运费政策
"""

import pytest
from sqlalchemy import or_, select

from app.db.session import AsyncSessionLocal, engine
from app.models.knowledge import KnowledgeChunk
from app.services.chat_service import ChatService
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.tools.business_tools import query_faq
from scripts.wsl_helper import ensure_mysql_ready


@pytest.fixture(scope="module", autouse=True)
def setup_mysql_ready():
    """保证 WSL2 MySQL 容器已拉起且保活进程处于激活状态"""
    ensure_mysql_ready(verbose=False)


@pytest.fixture(autouse=True)
async def reset_engine_connections():
    """避免 pytest-asyncio 在多个测试函数间切换 event loop 时复用已关闭 loop 的连接与单例"""
    import gc
    from app.llm import clear_llm_cache
    from app.tools import business_tools

    clear_llm_cache()
    if hasattr(business_tools, "_retriever") and business_tools._retriever is not None:
        if hasattr(business_tools._retriever, "close"):
            business_tools._retriever.close()
        business_tools._retriever = None
    await engine.dispose()
    gc.collect()

    yield

    clear_llm_cache()
    if hasattr(business_tools, "_retriever") and business_tools._retriever is not None:
        if hasattr(business_tools._retriever, "close"):
            business_tools._retriever.close()
        business_tools._retriever = None
    await engine.dispose()
    gc.collect()




@pytest.mark.asyncio
async def test_acceptance_criteria_1_semantic_rephrase():
    """验收标准 1:「邮费是多少」这类换说法的问题,能召回运费说明并答对。

    对比验证：
    1. SQL LIKE 按"邮费是多少"查询 MySQL 原文库发生语义 MISS（原文为“运费标准与偏远地区配送说明”）；
    2. query_faq 通过 BGE-M3 Dense 向量检索成功召回运费规则，包含满88元包邮规则与偏远地区资费。
    """
    # 1. 验证传统关键词/LIKE 匹配发生语义 MISS
    async with AsyncSessionLocal() as session:
        stmt = select(KnowledgeChunk).where(
            or_(
                KnowledgeChunk.questions.like("%邮费是多少%"),
                KnowledgeChunk.answer.like("%邮费是多少%"),
            )
        )
        res = await session.execute(stmt)
        sql_chunks = res.scalars().all()
        assert len(sql_chunks) == 0, "SQL LIKE 针对同义词'邮费是多少'应发生语义 MISS"

    # 2. 验证 Dense 向量语义检索准确命中
    result = await query_faq.ainvoke({"keyword": "邮费是多少"})
    assert "未找到" not in result, f"Dense 向量检索未命中知识块: {result}"
    assert ("运费" in result or "包邮" in result), "检索结果应包含运费或包邮说明"
    assert "88元" in result, f"检索结果应命中满88元包邮条款: {result}"


@pytest.mark.asyncio
async def test_acceptance_criteria_2_resumed_pending_chunks():
    """验收标准 2: 故意中断建库任务再重跑, 漏向量化的块能被捡起补齐。

    验证流程：
    1. 人为插入一条处于 pending 状态、尚未向量化的孤立块（模拟建库中断）；
    2. 再次调用 repair_pending_chunks，验证漏网块 100% 被捡起补偿；
    3. 检查数据库中状态翻转为 'done' 且 vector_id 成功回填对齐。
    """
    writer = KnowledgeDualWriter()
    target_id = None
    try:
        # 1. 制造中断遗留的 pending 块
        async with AsyncSessionLocal() as session:
            chunk = KnowledgeChunk(
                category="商城配送与运费服务规范",
                questions="断点续跑测试问题",
                answer="断点续跑测试内容：模拟系统在写入MySQL后意外中断，未能完成Milvus写入。",
                section_path="商城配送与运费服务规范 > 异常容灾",
                content_type="policy",
                is_key_clause=False,
                vectorize_status="pending",
                vector_id=None,
            )
            session.add(chunk)
            await session.commit()
            await session.refresh(chunk)
            target_id = chunk.id

        assert target_id is not None

        # 2. 模拟系统自愈/重跑构建补偿逻辑
        async with AsyncSessionLocal() as session:
            repaired_count = await writer.repair_pending_chunks(session)
            assert repaired_count >= 1, "应至少修复 1 条遗留的 pending 块"

        # 3. 验证数据库状态翻转与向量回填
        async with AsyncSessionLocal() as session:
            repaired_chunk = await session.get(KnowledgeChunk, target_id)
            assert repaired_chunk is not None
            assert repaired_chunk.vectorize_status == "done", "知识块状态应翻转为 done"
            assert repaired_chunk.vector_id == str(target_id), "vector_id 应回填对齐 chunk.id"

        # 4. 再次执行自愈，应无遗留块
        async with AsyncSessionLocal() as session:
            second_repair = await writer.repair_pending_chunks(session)
            assert second_repair == 0, "再次自愈应返回 0"

    finally:
        writer.close()


@pytest.mark.asyncio
async def test_e2e_full_retrieval_and_answer():
    """端到端验证：客服会话大模型通过 query_faq 检索知识库并回答运费。"""
    service = ChatService()
    async with AsyncSessionLocal() as session:
        events = [
            event
            async for event in service.stream_chat(
                db=session,
                conversation_id=None,
                message="请问商城邮费是多少？新疆西藏等偏远地区包邮吗？",
            )
        ]

    # 1. 验证触发了 query_faq 工具调用
    tool_starts = [e for e in events if e.get("event_type") == "tool_start"]
    assert any(ts.get("tool_name") == "query_faq" for ts in tool_starts), "会话应触发 query_faq 工具调用"

    # 2. 验证工具调用成功结束
    tool_ends = [e for e in events if e.get("event_type") == "tool_end"]
    assert any(te.get("tool_name") == "query_faq" and te.get("success") is True for te in tool_ends)

    # 3. 验证最终模型回答结合了召回的运费政策（满88包邮、偏远地区加收15元）
    text_events = [e for e in events if e.get("event_type") == "text"]
    full_answer = "".join(e.get("content", "") for e in text_events)
    assert any(kw in full_answer for kw in ["88", "八十八"]), f"最终客服回答应提及88元包邮门槛: {full_answer}"
    assert any(kw in full_answer for kw in ["15", "十五", "偏远", "加收"]), f"最终客服回答应提及偏远地区运费规则: {full_answer}"
