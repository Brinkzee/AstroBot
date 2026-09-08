import os
import shutil
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.session import Base
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.staging import QAExtractionStaging
from app.models.knowledge import KnowledgeChunk

from app.prompts.qa_extraction import (
    ExtractedQAPair,
    ExtractedQAList,
    QA_EXTRACTION_SYSTEM_PROMPT,
    qa_extraction_prompt,
    parse_qa_extraction_output,
)
from app.services.rag.miner import DialogueKnowledgeMiner
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.services.rag.embedding import BGEEmbeddingClient
from scripts.mine_dialogues import run_mining_pipeline


class AsyncSessionAdapter:
    """Async wrapper over SQLite synchronous session for testing."""

    def __init__(self, sync_session: Session):
        self._sync = sync_session

    def add(self, obj):
        self._sync.add(obj)

    def add_all(self, objs):
        self._sync.add_all(objs)

    async def flush(self):
        self._sync.flush()

    async def commit(self):
        self._sync.commit()

    async def rollback(self):
        self._sync.rollback()

    async def refresh(self, obj):
        self._sync.refresh(obj)

    async def get(self, entity_cls, ident):
        return self._sync.get(entity_cls, ident)

    async def execute(self, stmt):
        return self._sync.execute(stmt)

    async def scalars(self, stmt):
        return self._sync.scalars(stmt)

    async def close(self):
        self._sync.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.rollback()
        await self.close()


@pytest.fixture
def sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield AsyncSessionAdapter(session)


@pytest.fixture
def temp_milvus_db(tmp_path):
    db_file = tmp_path / "milvus_miner" / "miner_test.db"
    yield str(db_file)
    from milvus_lite.server_manager import server_manager_instance
    try:
        server_manager_instance.release_server(str(db_file))
    except Exception:
        pass
    parent = db_file.parent
    if parent.exists():
        try:
            shutil.rmtree(parent, ignore_errors=True)
        except Exception:
            pass


def test_qa_extraction_pydantic_schema_and_parser():
    """验证 Prompt Pydantic Schema 定义与输出解析能力"""
    # 1. 模型实例与字段校验
    pair = ExtractedQAPair(question="运费多少钱？", answer="满88包邮。")
    assert pair.question == "运费多少钱？"
    assert pair.answer == "满88包邮。"

    qa_list = ExtractedQAList(items=[pair])
    assert len(qa_list.items) == 1
    assert qa_list.items[0].question == "运费多少钱？"

    # 2. Prompt 包含核心约束指令：去客套、脱敏隐私、通用性、JSON 输出
    assert "寒暄" in QA_EXTRACTION_SYSTEM_PROMPT or "客套" in QA_EXTRACTION_SYSTEM_PROMPT
    assert "脱敏" in QA_EXTRACTION_SYSTEM_PROMPT or "隐私" in QA_EXTRACTION_SYSTEM_PROMPT
    assert "JSON" in QA_EXTRACTION_SYSTEM_PROMPT

    # 3. 各种格式 JSON 解析能力测试
    # 3.1 标准数组 JSON
    raw_array = '[{"question": "支持7天退货吗？", "answer": "支持7天无理由退货。"}]'
    parsed = parse_qa_extraction_output(raw_array)
    assert len(parsed) == 1
    assert parsed[0].question == "支持7天退货吗？"

    # 3.2 带 Markdown 代码块的 JSON
    markdown_json = '```json\n[{"question": "多久发货？", "answer": "48小时内发货。"}]\n```'
    parsed_md = parse_qa_extraction_output(markdown_json)
    assert len(parsed_md) == 1
    assert parsed_md[0].question == "多久发货？"

    # 3.3 对象包装形式 {"items": [...]}
    wrapped_json = '{"items": [{"question": "什么快递？", "answer": "默认顺丰速运。"}]}'
    parsed_wrapped = parse_qa_extraction_output(wrapped_json)
    assert len(parsed_wrapped) == 1
    assert parsed_wrapped[0].question == "什么快递？"

    # 3.4 异常/空白输出容错
    assert parse_qa_extraction_output("") == []
    assert parse_qa_extraction_output("这是非 JSON 的文本回答") == []


@pytest.mark.asyncio
async def test_fetch_dialogue_batches_grouping(sqlite_session):
    """验证按 conversation_id 分组聚合流水、过滤 tool 消息、分批及生成唯一 batch_no"""
    # 插入 3 个会话及不同角色的消息
    c1 = Conversation(user_id="user_1", status="已结束")
    c2 = Conversation(user_id="user_2", status="已结束")
    c3 = Conversation(user_id="user_3", status="已结束")
    sqlite_session.add_all([c1, c2, c3])
    await sqlite_session.flush()

    m1 = Message(conversation_id=c1.id, role="user", content="你好，运费怎么算？")
    m2 = Message(conversation_id=c1.id, role="assistant", content="全场满88元包邮哦。")
    m3 = Message(conversation_id=c1.id, role="tool", content='{"status": "ok"}')  # tool 消息应被过滤

    m4 = Message(conversation_id=c2.id, role="user", content="衣服尺码偏大还是偏小？")
    m5 = Message(conversation_id=c2.id, role="assistant", content="此款为标准尺码，按平时尺码购买即可。")

    m6 = Message(conversation_id=c3.id, role="user", content="支持分期付款吗？")
    m7 = Message(conversation_id=c3.id, role="assistant", content="目前暂不支持分期付款。")

    sqlite_session.add_all([m1, m2, m3, m4, m5, m6, m7])
    await sqlite_session.commit()

    miner = DialogueKnowledgeMiner()
    # 每批 2 个会话，3 个会话应分成 2 个批次
    batches = await miner.fetch_dialogue_batches(sqlite_session, batch_size=2)
    assert len(batches) == 2

    batch_no_1, convs_1 = batches[0]
    batch_no_2, convs_2 = batches[1]

    assert batch_no_1.startswith("BATCH_")
    assert batch_no_2.startswith("BATCH_")
    assert batch_no_1 != batch_no_2

    assert len(convs_1) == 2
    assert len(convs_2) == 1

    # 验证对话流水格式聚合与 tool 消息过滤
    first_conv = convs_1[0]
    assert first_conv["conversation_id"] == c1.id
    assert "用户: 你好，运费怎么算？" in first_conv["dialogue_text"]
    assert "客服: 全场满88元包邮哦。" in first_conv["dialogue_text"]
    assert "status" not in first_conv["dialogue_text"]  # tool 消息被排除


@pytest.mark.asyncio
async def test_extract_and_stage(sqlite_session):
    """验证 extract_and_stage 调用 LLM 抽取并写入 qa_extraction_staging 表（状态 extracted）"""
    miner = DialogueKnowledgeMiner()

    mock_llm_output = """[
        {"question": "退货运费谁出？", "answer": "非质量问题买家出，质量问题商家出。"},
        {"question": "几点前下单当天发？", "answer": "每天下午16:00前下单当天发货。"}
    ]"""

    with patch.object(miner, "_call_llm_for_qa", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_llm_output

        staged = await miner.extract_and_stage(
            sqlite_session,
            dialogue_text="用户: 退货运费谁出？\\n客服: 非质量问题买家出...",
            source_ref="conversation_101",
            batch_no="BATCH_20260908_001",
        )

        assert len(staged) == 2
        assert staged[0].batch_no == "BATCH_20260908_001"
        assert staged[0].source_ref == "conversation_101"
        assert staged[0].status == "extracted"
        assert staged[0].question == "退货运费谁出？"
        assert staged[1].question == "几点前下单当天发？"

        # 数据库中持久化验证
        stmt = select(QAExtractionStaging).where(QAExtractionStaging.batch_no == "BATCH_20260908_001")
        res = await sqlite_session.execute(stmt)
        rows = list(res.scalars().all())
        assert len(rows) == 2
        assert all(r.status == "extracted" for r in rows)


@pytest.mark.asyncio
async def test_deduplicate_staging_exact_match(sqlite_session):
    """验证规则层精确去重：相同问题保留答案更详尽的问答对，重复项置为 discarded"""
    miner = DialogueKnowledgeMiner()
    batch_no = "BATCH_DEDUP_EXACT"

    # 插入两个文本相同的问题，但答案详尽程度不同
    r1 = QAExtractionStaging(
        batch_no=batch_no,
        source_ref="conv_1",
        question="运费如何计算？",
        answer="全场实付满88元包邮，未满88元收取8元基础运费，偏远地区加收15元。",  # 详尽版
        status="extracted",
    )
    r2 = QAExtractionStaging(
        batch_no=batch_no,
        source_ref="conv_2",
        question="运费如何计算？",
        answer="满88包邮。",  # 简短版
        status="extracted",
    )
    # 插入另一个完全不同问题
    r3 = QAExtractionStaging(
        batch_no=batch_no,
        source_ref="conv_3",
        question="支持什么支付方式？",
        answer="支持微信、支付宝和银联支付。",
        status="extracted",
    )
    sqlite_session.add_all([r1, r2, r3])
    await sqlite_session.commit()

    kept_count, discarded_count = await miner.deduplicate_staging(
        sqlite_session, batch_no=batch_no
    )
    assert kept_count == 2
    assert discarded_count == 1

    await sqlite_session.refresh(r1)
    await sqlite_session.refresh(r2)
    await sqlite_session.refresh(r3)

    assert r1.status == "kept"  # 详细答案保留
    assert r2.status == "discarded"  # 简短答案丢弃
    assert r3.status == "kept"  # 独立问题保留


@pytest.mark.asyncio
async def test_deduplicate_staging_semantic_similarity(sqlite_session):
    """验证语义层 BGE-M3 去重：相似度 >= 0.92 的同义问题归并，保留详尽答案"""
    miner = DialogueKnowledgeMiner()
    batch_no = "BATCH_DEDUP_SEMANTIC"

    # Q1 与 Q2 同义，Q3 独立
    r1 = QAExtractionStaging(
        batch_no=batch_no,
        source_ref="conv_10",
        question="退货运费谁来承担？",
        answer="商品质量问题由本店承担往返运费；个人原因退换货由买家自行承担寄回运费。",  # 详尽
        status="extracted",
    )
    r2 = QAExtractionStaging(
        batch_no=batch_no,
        source_ref="conv_11",
        question="寄回商品的邮费谁付？",
        answer="质量问题商家付，非质量问题自己付。",  # 简短
        status="extracted",
    )
    r3 = QAExtractionStaging(
        batch_no=batch_no,
        source_ref="conv_12",
        question="下单后几天内能收到商品？",
        answer="通常发货后 2-4 天送达，偏远地区预计 5-7 天。",
        status="extracted",
    )
    sqlite_session.add_all([r1, r2, r3])
    await sqlite_session.commit()

    # 构造向量使得 Q1 与 Q2 相似度 >= 0.95，与 Q3 相似度 < 0.2
    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0

    v2 = np.zeros(1024, dtype=np.float32)
    v2[0] = 0.96
    v2[1] = float(np.sqrt(1 - 0.96**2))  # 归一化，与 v1 dot = 0.96 >= 0.92

    v3 = np.zeros(1024, dtype=np.float32)
    v3[2] = 1.0  # 与 v1, v2 正交

    async def mock_aembed(texts, batch_size=16):
        res = []
        for t in texts:
            if "承担" in t:
                res.append(v1.tolist())
            elif "邮费" in t:
                res.append(v2.tolist())
            else:
                res.append(v3.tolist())
        return res

    with patch.object(miner.embedding_client, "aembed_documents", side_effect=mock_aembed):
        kept_count, discarded_count = await miner.deduplicate_staging(
            sqlite_session, batch_no=batch_no, similarity_threshold=0.92
        )
        assert kept_count == 2
        assert discarded_count == 1

        await sqlite_session.refresh(r1)
        await sqlite_session.refresh(r2)
        await sqlite_session.refresh(r3)

        assert r1.status == "kept"  # 答案更详尽保留
        assert r2.status == "discarded"  # 同义重复丢弃
        assert r3.status == "kept"  # 独立问题保留


@pytest.mark.asyncio
async def test_ingest_kept_chunks(sqlite_session, temp_milvus_db):
    """验证 ingest_kept_chunks 将 status='kept' 的条目转为 DocChunk 并通过双写落库"""
    miner = DialogueKnowledgeMiner()
    writer = KnowledgeDualWriter(milvus_uri=temp_milvus_db)
    batch_no = "BATCH_INGEST_001"

    try:
        r1 = QAExtractionStaging(
            batch_no=batch_no,
            source_ref="conv_201",
            question="保修期是多长时间？",
            answer="本店售出商品自签收之日起享有全国联保一年服务。",
            status="kept",
        )
        r2 = QAExtractionStaging(
            batch_no=batch_no,
            source_ref="conv_202",
            question="商品破损怎么处理？",
            answer="签收24小时内联系客服并提供开箱照片，免费补发或全额退款。",
            status="kept",
        )
        r3 = QAExtractionStaging(
            batch_no=batch_no,
            source_ref="conv_203",
            question="在吗？",
            answer="在的亲。",
            status="discarded",  # 已丢弃，不应入库
        )
        sqlite_session.add_all([r1, r2, r3])
        await sqlite_session.commit()

        ingested_count = await miner.ingest_kept_chunks(
            sqlite_session, dual_writer=writer, batch_no=batch_no
        )
        assert ingested_count == 2

        # 验证 MySQL 权威源写入
        stmt = select(KnowledgeChunk).where(KnowledgeChunk.content_type == "mined_qa")
        res = await sqlite_session.execute(stmt)
        db_chunks = list(res.scalars().all())
        assert len(db_chunks) == 2
        for c in db_chunks:
            assert c.category == "历史问答挖掘"
            assert c.vectorize_status == "done"
            assert c.vector_id is not None
            assert "客服对话挖掘 > conv_" in (c.section_path or "")

        # 验证 Milvus 向量库写入
        assert writer.store.count() == 2
        hits = writer.store.search(
            query_vector=writer.embedding_client.embed_query("保修多长时间？"),
            top_k=2,
        )
        assert len(hits) >= 1
        assert "保修期是多长时间？" in hits[0]["questions"]
    finally:
        writer.store.close()


@pytest.mark.asyncio
async def test_run_mining_pipeline_end_to_end(sqlite_session, temp_milvus_db):
    """验证 CLI mine_dialogues.py 脚本完整管道端到端运行"""
    # 构造历史对话流水
    c1 = Conversation(user_id="user_e2e_1", status="已结束")
    sqlite_session.add(c1)
    await sqlite_session.flush()

    m1 = Message(conversation_id=c1.id, role="user", content="退货地址在哪里？")
    m2 = Message(conversation_id=c1.id, role="assistant", content="提交售后申请后系统会自动发送退货仓库地址短信。")
    sqlite_session.add_all([m1, m2])
    await sqlite_session.commit()

    writer = KnowledgeDualWriter(milvus_uri=temp_milvus_db)
    miner = DialogueKnowledgeMiner()

    mock_llm_output = '[{"question": "退货地址在哪里？", "answer": "提交售后申请后系统会自动发送退货仓库地址短信。"}]'
    with patch.object(miner, "_call_llm_for_qa", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_llm_output

        try:
            stats = await run_mining_pipeline(
                session=sqlite_session,
                miner=miner,
                dual_writer=writer,
                batch_size=10,
                step="all",
            )
            assert stats["batches_fetched"] >= 1
            assert stats["conversations_scanned"] >= 1
            assert stats["staged_count"] == 1
            assert stats["kept_count"] == 1
            assert stats["discarded_count"] == 0
            assert stats["ingested_count"] == 1

            assert writer.store.count() == 1
        finally:
            writer.store.close()
