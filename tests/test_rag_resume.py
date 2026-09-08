import os
import shutil
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from unittest.mock import patch, MagicMock

from app.db.session import Base
from app.models.knowledge import KnowledgeChunk
from app.services.rag.splitter import DocChunk
from app.services.rag.dual_writer import KnowledgeDualWriter
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
    db_file = tmp_path / "milvus_resume" / "resume_test.db"
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


@pytest.mark.asyncio
async def test_repair_pending_chunks_empty(sqlite_session, temp_milvus_db):
    """测试当数据库中无 pending 块时，repair_pending_chunks 返回 0 且无副作用"""
    writer = KnowledgeDualWriter(milvus_uri=temp_milvus_db)
    try:
        repaired = await writer.repair_pending_chunks(sqlite_session)
        assert repaired == 0
    finally:
        writer.store.close()


@pytest.mark.asyncio
async def test_repair_pending_chunks_interrupted_recovery(sqlite_session, temp_milvus_db):
    """测试模拟意外中断遗留的 pending 块被 100% 自动捡起、补全向量化并最终一致"""
    writer = KnowledgeDualWriter(milvus_uri=temp_milvus_db)
    try:
        # 人为制造 2 个因中断未能向量化的 pending 块
        chunk1 = KnowledgeChunk(
            category="发票说明",
            questions="如何申请专票？",
            answer="在开票信息中填写企业税号与开户行即可。",
            section_path="发票政策 > 专票",
            content_type="policy",
            is_key_clause=False,
            vectorize_status="pending",
            vector_id=None,
        )
        chunk2 = KnowledgeChunk(
            category="发票说明",
            questions="电子发票开具后多久能收到？",
            answer="开具后一般 1-3 个工作日发送到邮箱。",
            section_path="发票政策 > 时效",
            content_type="policy",
            is_key_clause=False,
            vectorize_status="pending",
            vector_id=None,
        )
        sqlite_session.add_all([chunk1, chunk2])
        await sqlite_session.flush()
        # 绑定前后指针
        chunk1.next_chunk_id = chunk2.id
        chunk2.prev_chunk_id = chunk1.id
        await sqlite_session.commit()

        c1_id = chunk1.id
        c2_id = chunk2.id

        # 验证此时状态为 pending
        assert chunk1.vectorize_status == "pending"
        assert chunk2.vectorize_status == "pending"
        assert writer.store.count() == 0

        # 执行自愈补偿
        repaired_count = await writer.repair_pending_chunks(sqlite_session)
        assert repaired_count == 2

        # 验证数据库状态更新为 done 且回填 vector_id
        await sqlite_session.refresh(chunk1)
        await sqlite_session.refresh(chunk2)

        assert chunk1.vectorize_status == "done"
        assert chunk1.vector_id == str(c1_id)
        assert chunk2.vectorize_status == "done"
        assert chunk2.vector_id == str(c2_id)

        # 验证 Milvus 成功写入了这 2 条记录
        assert writer.store.count() == 2

        # 再次执行自愈，应返回 0
        repaired_again = await writer.repair_pending_chunks(sqlite_session)
        assert repaired_again == 0
    finally:
        writer.store.close()


@pytest.mark.asyncio
async def test_repair_pending_chunks_mixed_status(sqlite_session, temp_milvus_db):
    """验证混合状态下只补偿 pending 块，不影响已有 done 块"""
    writer = KnowledgeDualWriter(milvus_uri=temp_milvus_db)
    try:
        # 1 个已有 done 块
        chunk_done = KnowledgeChunk(
            category="常规模板",
            questions="已有问题",
            answer="已有答案",
            vectorize_status="done",
            vector_id="999",
        )
        # 1 个 pending 块
        chunk_pending = KnowledgeChunk(
            category="常规模板",
            questions="遗留问题",
            answer="遗留答案",
            vectorize_status="pending",
            vector_id=None,
        )
        sqlite_session.add_all([chunk_done, chunk_pending])
        await sqlite_session.commit()
        await sqlite_session.refresh(chunk_done)
        await sqlite_session.refresh(chunk_pending)

        pending_id = chunk_pending.id

        repaired_count = await writer.repair_pending_chunks(sqlite_session)
        assert repaired_count == 1

        await sqlite_session.refresh(chunk_done)
        await sqlite_session.refresh(chunk_pending)

        # done 块保持原样
        assert chunk_done.vectorize_status == "done"
        assert chunk_done.vector_id == "999"

        # pending 块转为 done
        assert chunk_pending.vectorize_status == "done"
        assert chunk_pending.vector_id == str(pending_id)

        # Milvus 中写入了 pending 块
        assert writer.store.count() == 1
    finally:
        writer.store.close()


@pytest.mark.asyncio
async def test_write_chunks_failure_and_subsequent_repair(sqlite_session, temp_milvus_db):
    """测试在 write_chunks 过程中 Milvus 阶段偶发异常时：
    - MySQL 原文已被持久化且标记为 pending
    - 紧接着调用 repair_pending_chunks 能够顺利完成断点续跑补齐
    """
    writer = KnowledgeDualWriter(milvus_uri=temp_milvus_db)
    try:
        chunks = [
            DocChunk(
                category="异常测试",
                questions="断网怎么办？",
                answer="系统会自动保留 pending 并重试。",
            )
        ]

        # 模拟 Milvus upsert 抛出异常
        with patch.object(writer.store, "upsert", side_effect=RuntimeError("Milvus Connection Timeout")):
            with pytest.raises(RuntimeError, match="Milvus Connection Timeout"):
                await writer.write_chunks(sqlite_session, chunks)

        # 验证 MySQL 权威源已存在该条目，且状态为 pending
        stmt = select(KnowledgeChunk).where(KnowledgeChunk.category == "异常测试")
        result = await sqlite_session.execute(stmt)
        saved_chunk = result.scalar_one()
        assert saved_chunk.vectorize_status == "pending"
        assert saved_chunk.vector_id is None

        # 恢复正常的 Milvus，调用 repair_pending_chunks 完成断点续跑
        repaired = await writer.repair_pending_chunks(sqlite_session)
        assert repaired == 1

        await sqlite_session.refresh(saved_chunk)
        assert saved_chunk.vectorize_status == "done"
        assert saved_chunk.vector_id == str(saved_chunk.id)
        assert writer.store.count() == 1
    finally:
        writer.store.close()
