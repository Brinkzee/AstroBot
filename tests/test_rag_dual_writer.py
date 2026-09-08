import os
import shutil
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from unittest.mock import AsyncMock, MagicMock, patch

from app.db.session import Base
from app.models.knowledge import KnowledgeChunk
from app.services.rag.splitter import DocChunk
from app.services.rag.dual_writer import KnowledgeDualWriter, compose_embedding_text
from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.milvus_client import MilvusKnowledgeStore


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
    db_file = tmp_path / "milvus_dual_writer" / "dual_writer_test.db"
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


def test_compose_embedding_text_standard():
    """验证文本拼接严格遵循规范：分类：...\\n问题：...\\n内容：..."""
    text = compose_embedding_text(
        category="售后服务 > 退货政策",
        questions="如何申请退货？\\n退货有什么要求？",
        answer="支持7天无理由退货，商品需完好。",
    )
    expected = (
        "分类：售后服务 > 退货政策\n"
        "问题：如何申请退货？\\n退货有什么要求？\n"
        "内容：支持7天无理由退货，商品需完好。"
    )
    assert text == expected


@pytest.mark.asyncio
async def test_dual_writer_empty_chunks(sqlite_session, temp_milvus_db):
    """验证空列表输入安全返回空列表"""
    writer = KnowledgeDualWriter(milvus_uri=temp_milvus_db)
    try:
        result = await writer.write_chunks(sqlite_session, [])
        assert result == []
    finally:
        writer.store.close()


@pytest.mark.asyncio
async def test_dual_writer_full_flow(sqlite_session, temp_milvus_db):
    """验证完整的两阶段四步双写流程：
    1. MySQL 原文权威源落库（pending 状态）
    2. 自增 id 生成与双向链表 prev/next 指针正确绑定
    3. BGE-M3 向量化并 upsert 到 Milvus
    4. MySQL 状态回填更新为 done，且 vector_id = str(id)
    """
    writer = KnowledgeDualWriter(milvus_uri=temp_milvus_db)
    try:
        chunks = [
            DocChunk(
                category="退货规则",
                questions="退货时效是几天？",
                answer="支持7天无理由退换货。",
                section_path="售后政策 > 退货时效",
                content_type="policy",
                is_key_clause=True,
            ),
            DocChunk(
                category="退货规则",
                questions="退货运费谁承担？",
                answer="质量问题商家承担，非质量问题买家承担。",
                section_path="售后政策 > 运费承担",
                content_type="policy",
                is_key_clause=False,
            ),
            DocChunk(
                category="退货规则",
                questions="什么情况下不可退货？",
                answer="定制类商品和贴身衣物拆封后不可退货。",
                section_path="售后政策 > 特殊商品",
                content_type="policy",
                is_key_clause=True,
            ),
        ]

        saved_entities = await writer.write_chunks(sqlite_session, chunks)
        assert len(saved_entities) == 3

        # 验证 MySQL 实体属性
        c0, c1, c2 = saved_entities[0], saved_entities[1], saved_entities[2]

        assert c0.id is not None
        assert c1.id is not None
        assert c2.id is not None

        # 验证双向链表前后指针正确连结
        assert c0.prev_chunk_id is None
        assert c0.next_chunk_id == c1.id

        assert c1.prev_chunk_id == c0.id
        assert c1.next_chunk_id == c2.id

        assert c2.prev_chunk_id == c1.id
        assert c2.next_chunk_id is None

        # 验证状态机全部推进到 'done' 且 vector_id 回填
        for c in saved_entities:
            assert c.vectorize_status == "done"
            assert c.vector_id == str(c.id)

        # 验证 Milvus 存储记录数量与主键幂等
        assert writer.store.count() == 3

        # 检索测试 Milvus 中的 Payload
        query_vec = writer.embedding_client.embed_query("退货运费谁付？")
        hits = writer.store.search(query_vector=query_vec, top_k=3)
        assert len(hits) == 3
        hit_ids = {h["id"] for h in hits}
        assert hit_ids == {c0.id, c1.id, c2.id}

        # 验证 Payload 包含 chunk_text 与元数据
        target_hit = next(h for h in hits if h["id"] == c1.id)
        assert target_hit["category"] == "退货规则"
        assert "运费" in target_hit["questions"]
        assert "质量问题商家承担" in target_hit["answer"]
        assert "分类：退货规则" in target_hit["chunk_text"]
    finally:
        writer.store.close()


@pytest.mark.asyncio
async def test_dual_writer_idempotent_upsert(sqlite_session, temp_milvus_db):
    """验证相同 chunk 再次调用写入口时，Milvus 主键 upsert 保持幂等"""
    writer = KnowledgeDualWriter(milvus_uri=temp_milvus_db)
    try:
        chunks = [
            DocChunk(
                category="包邮说明",
                questions="多少钱包邮？",
                answer="满88元包邮。",
                section_path="物流 > 包邮",
                content_type="policy",
                is_key_clause=False,
            )
        ]
        res1 = await writer.write_chunks(sqlite_session, chunks)
        assert len(res1) == 1
        assert writer.store.count() == 1

        # 再次对已写入 chunk 重新 upsert Milvus
        rec = {
            "id": res1[0].id,
            "vector": [0.05] * 1024,
            "category": "包邮说明",
            "questions": "多少钱包邮？",
            "answer": "全场满88元包邮（更新版）。",
            "chunk_text": "分类：包邮说明\n问题：多少钱包邮？\n内容：全场满88元包邮（更新版）。",
            "content_type": "policy",
            "is_key_clause": True,
        }
        writer.store.upsert([rec])
        # 总数应维持 1 条（主键去重覆写）
        assert writer.store.count() == 1
    finally:
        writer.store.close()


@pytest.mark.asyncio
async def test_build_knowledge_base_pipeline(sqlite_session, temp_milvus_db, tmp_path):
    """测试 build_knowledge_base 完整脚本流程：扫描文件、切分、双写、前后自愈"""
    from scripts.build_knowledge_base import build_knowledge_base

    kb_dir = tmp_path / "test_kb_files"
    kb_dir.mkdir(parents=True, exist_ok=True)
    sample_md = kb_dir / "sample_policy.md"
    sample_md.write_text(
        "# 退款政策\n"
        "## 退款到账时间\n"
        "退款通常在 1-3 个工作日内原路退回支付账户。\n"
        "## 退款注意事项\n"
        "【特别说明】优惠券退款后不予退还。\n",
        encoding="utf-8",
    )

    writer = KnowledgeDualWriter(milvus_uri=temp_milvus_db)
    try:
        stats = await build_knowledge_base(
            kb_dir=str(kb_dir),
            clean=False,
            repair_only=False,
            session=sqlite_session,
            writer=writer,
        )
        assert stats["ingested_files"] == 1
        assert stats["total_chunks"] >= 2
        assert stats["pre_repaired"] == 0
        assert stats["post_repaired"] == 0

        # 验证 Milvus 中有记录
        assert writer.store.count() >= 2

        # 验证数据库记录状态全为 done
        stmt = select(KnowledgeChunk)
        result = await sqlite_session.execute(stmt)
        chunks = list(result.scalars().all())
        assert len(chunks) >= 2
        for c in chunks:
            assert c.vectorize_status == "done"
            assert c.vector_id == str(c.id)
    finally:
        writer.store.close()


@pytest.mark.asyncio
async def test_build_knowledge_base_clean_and_repair_only(sqlite_session, temp_milvus_db, tmp_path):
    """测试 build_knowledge_base 的 clean 模式与 repair_only 模式"""
    from scripts.build_knowledge_base import build_knowledge_base

    writer = KnowledgeDualWriter(milvus_uri=temp_milvus_db)
    try:
        # 先插入一个 pending 记录
        chunk = KnowledgeChunk(
            category="待修补",
            questions="修补问题",
            answer="修补内容",
            vectorize_status="pending",
        )
        sqlite_session.add(chunk)
        await sqlite_session.commit()

        # repair_only 模式
        stats = await build_knowledge_base(
            repair_only=True,
            session=sqlite_session,
            writer=writer,
        )
        assert stats["pre_repaired"] == 1
        assert stats["ingested_files"] == 0

        # 验证 chunk 已修复
        await sqlite_session.refresh(chunk)
        assert chunk.vectorize_status == "done"

        # clean 模式清空
        stats_clean = await build_knowledge_base(
            clean=True,
            repair_only=True,
            session=sqlite_session,
            writer=writer,
        )
        stmt = select(KnowledgeChunk)
        res = await sqlite_session.execute(stmt)
        assert len(res.scalars().all()) == 0
        assert writer.store.count() == 0
    finally:
        writer.store.close()

