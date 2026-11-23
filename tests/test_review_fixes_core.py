"""TDD regression tests covering CORE-01..CORE-20, VER-05/06/07/08/10, and FE-01(5)."""

import asyncio
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db.session import Base, engine as global_engine, get_db
from app.models.conversation import Conversation
from app.models.faith_case import FaithCase, upsert_faith_case
from app.models.knowledge import KnowledgeChunk
from app.models.low_confidence import LowConfidenceQuestion
from app.models.message import Message
from app.models.review_queue import ReviewQueue, ReviewStatus
from app.models.summary import ConversationSummary
from app.models.tool_audit_log import ToolAuditLog
from app.models.topic_classification import TopicClassification
from main import app

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest_asyncio.fixture
async def async_sqlite_db():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=eng, class_=AsyncSession, expire_on_commit=False
    )

    async def override_get_db():
        async with session_factory() as s:
            yield s

    app.dependency_overrides[get_db] = override_get_db
    yield eng, session_factory
    app.dependency_overrides.pop(get_db, None)
    await eng.dispose()


# =============================================================================
# CORE-01: CORS, default host, optional admin auth, conversation IDOR
# =============================================================================

def test_core01_cors_and_default_host():
    assert getattr(settings, "app_host", None) == "127.0.0.1"
    client = TestClient(app)
    resp = client.get("/health", headers={"Origin": "https://evil.example"})
    assert resp.status_code == 200
    allow_origin = resp.headers.get("access-control-allow-origin")
    allow_creds = resp.headers.get("access-control-allow-credentials")
    assert not (allow_origin == "https://evil.example" and allow_creds == "true")
    assert not (allow_origin == "*" and allow_creds == "true")


@pytest.mark.asyncio
async def test_core01_admin_token_enforcement(async_sqlite_db, monkeypatch):
    _, session_factory = async_sqlite_db
    async with session_factory() as db:
        rq = ReviewQueue(normalized_question="测试鉴权", review_status=ReviewStatus.PENDING.value)
        db.add(rq)
        await db.commit()
        await db.refresh(rq)
        rq_id = rq.id

    monkeypatch.setattr(settings, "admin_api_token", "super-secret-token", raising=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Missing token -> 401
        r1 = await client.post("/api/jobs/start", json={"job_name": "eval-rag"})
        assert r1.status_code == 401

        r2 = await client.post(f"/api/review-queue/{rq_id}/reject")
        assert r2.status_code == 401

        # Wrong token -> 403
        r3 = await client.post(
            f"/api/review-queue/{rq_id}/reject",
            headers={"X-Admin-Token": "wrong-token"},
        )
        assert r3.status_code == 403

        # Valid X-Admin-Token -> 200
        r4 = await client.post(
            f"/api/review-queue/{rq_id}/reject",
            headers={"X-Admin-Token": "super-secret-token"},
        )
        assert r4.status_code == 200

        # Valid Bearer token -> allowed (404 because job_id does not exist, not 401/403)
        r5 = await client.post(
            "/api/jobs/nonexistent_job/stop",
            headers={"Authorization": "Bearer super-secret-token"},
        )
        assert r5.status_code == 404


@pytest.mark.asyncio
async def test_core01_conversation_idor_enforcement(async_sqlite_db):
    _, session_factory = async_sqlite_db
    async with session_factory() as db:
        conv = Conversation(user_id="alice", status="进行中")
        db.add(conv)
        await db.commit()
        await db.refresh(conv)
        db.add(Message(conversation_id=conv.id, role="user", content="alice secret"))
        await db.commit()
        cid = conv.id

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Wrong user via query param -> 403
        r_read_wrong = await client.get(f"/api/conversations/{cid}/messages?user_id=bob")
        assert r_read_wrong.status_code == 403

        # Wrong user via header -> 403
        r_del_wrong = await client.delete(
            f"/api/conversations/{cid}", headers={"X-User-Id": "bob"}
        )
        assert r_del_wrong.status_code == 403

        # Matching user -> 200
        r_read_ok = await client.get(f"/api/conversations/{cid}/messages?user_id=alice")
        assert r_read_ok.status_code == 200
        assert len(r_read_ok.json()) == 1

        # Backward compatible when omitted -> 200
        r_del_ok = await client.delete(f"/api/conversations/{cid}")
        assert r_del_ok.status_code == 200


# =============================================================================
# CORE-02 & VER-05: setup_docker_proxy & docker-compose security
# =============================================================================

def test_core02_setup_docker_proxy_and_docker_compose():
    proxy_src = (ROOT_DIR / "scripts" / "setup_docker_proxy.py").read_text(encoding="utf-8")
    assert "chmod\", \"666\"" not in proxy_src
    assert "chmod 666" not in proxy_src
    assert "returncode" in proxy_src

    compose_src = (ROOT_DIR / "docker-compose.yml").read_text(encoding="utf-8")
    assert "127.0.0.1:3306:3306" in compose_src
    assert "${MYSQL_ROOT_PASSWORD" in compose_src


# =============================================================================
# CORE-03: JobRegistry.get_command Makefile alignment & CUDA_PYTHON env
# =============================================================================

def test_core03_job_registry_makefile_and_cuda_python(tmp_path, monkeypatch):
    from app.core.jobs import JobRegistry, _resolve_cuda_python

    jobs_src = (ROOT_DIR / "app" / "core" / "jobs.py").read_text(encoding="utf-8")
    assert "D:/Anaconda3/envs/pytorch/python.exe" not in jobs_src

    fake_py = tmp_path / "cuda_python.exe"
    fake_py.write_text("", encoding="utf-8")
    monkeypatch.setenv("CUDA_PYTHON", str(fake_py))
    assert _resolve_cuda_python() == str(fake_py)

    with patch("shutil.which", return_value="/usr/bin/make"):
        # Present in Makefile -> make
        assert JobRegistry.get_command("classify-pool") == ["make", "classify-pool"]
        # Absent from Makefile -> Python recipe, never ["make", ...]
        for missing_target in [
            "eval-rag",
            "cost-analysis",
            "eval-pipeline",
            "calibrate-confidence",
            "flywheel-pipeline",
        ]:
            cmd = JobRegistry.get_command(missing_target)
            assert cmd[0] != "make", f"{missing_target} is not in Makefile"
            assert Path(cmd[1]).exists()


# =============================================================================
# CORE-05: ChatService multi-tool-call persistence & persist before SSE yield
# =============================================================================

@pytest.mark.asyncio
async def test_core05_multi_tool_calls_and_persist_before_yield(async_sqlite_db):
    from app.services.chat_service import ChatService

    _, session_factory = async_sqlite_db

    tc1 = {"id": "call_1", "name": "query_order", "args": {"order_id": "1001"}}
    tc2 = {"id": "call_2", "name": "query_logistics", "args": {"order_id": "1001"}}

    mock_engine = MagicMock()
    mock_engine.run = AsyncMock(
        return_value={
            "conversation_id": 1,
            "current_turn_tool_messages": [
                AIMessage(content="", tool_calls=[tc1, tc2]),
                ToolMessage(content='{"status": "paid"}', tool_call_id="call_1", name="query_order"),
                ToolMessage(content='{"status": "shipped"}', tool_call_id="call_2", name="query_logistics"),
            ],
            "response_text": "订单已发货。",
            "suggested_actions": [],
        }
    )

    service = ChatService(workflow_engine=mock_engine, use_workflow=True)

    # Disconnect immediately upon receiving final "text" event
    events = []
    async with session_factory() as db:
        gen = service.stream_chat(db=db, conversation_id=None, message="查订单和物流")
        async for ev in gen:
            events.append(ev)
            if ev.get("event_type") == "text":
                await gen.aclose()
                break

    async with session_factory() as db:
        rows = (
            await db.execute(select(Message).order_by(Message.id.asc()))
        ).scalars().all()

    roles = [r.role for r in rows]
    # Expected: user, ONE assistant(tool_calls=[tc1, tc2]), tool(call_1), tool(call_2), assistant(final)
    assert roles == ["user", "assistant", "tool", "tool", "assistant"]
    assert isinstance(rows[1].tool_calls, list)
    assert len(rows[1].tool_calls) == 2
    assert rows[4].content == "订单已发货。"


# =============================================================================
# CORE-06: Layer 1 atomic tool-call degradation & JSON-string tool_calls parsing
# =============================================================================

def test_core06_atomic_tool_call_degradation_and_str_tool_calls():
    from app.services.context.manager import (
        apply_layer1_degradation,
        format_layer1_messages,
        format_layer2_messages,
        partition_messages,
    )

    conv = Conversation(id=1, summary_upto_msg_id=None, layer1_from_msg_id=None)
    tc_json = json.dumps([{"id": "c1", "name": "query_order", "args": {"order_id": "1001"}}])

    msgs = [
        Message(id=1, role="user", content="请帮我查询订单详情一二三四五六七八九十"),
        Message(id=2, role="assistant", content="", tool_calls=tc_json),
        Message(id=3, role="tool", content="订单状态已发货一二三四五", tool_call_id="c1"),
        Message(id=4, role="assistant", content="您好，已发货。"),
    ]

    # Budget enough for [msg 3, msg 4] (~20 tokens) so naive pop(0) would stop after msg 2,
    # leaving orphan tool msg 3 at start of Layer 1.
    apply_layer1_degradation(conv, msgs, layer1_budget=25)
    _, _, l1 = partition_messages(
        msgs,
        summary_upto_msg_id=conv.summary_upto_msg_id,
        layer1_from_msg_id=conv.layer1_from_msg_id,
    )
    assert len(l1) > 0
    assert l1[0].role != "tool", "Layer 1 must never start with an orphan tool message"

    # Both formatters must parse JSON-string tool_calls cleanly
    f1 = format_layer1_messages([msgs[1]])
    f2 = format_layer2_messages([msgs[1]])
    assert isinstance(f1[0], AIMessage) and len(f1[0].tool_calls) == 1
    assert isinstance(f2[0], AIMessage) and len(f2[0].tool_calls) == 1


# =============================================================================
# CORE-07 & FE-01(5): Review queue approve atomicity, leak fix, 409, reject body
# =============================================================================

@pytest.mark.asyncio
async def test_core07_review_queue_approve_failure_duplicate_and_reject_reason(async_sqlite_db):
    _, session_factory = async_sqlite_db
    async with session_factory() as db:
        rq1 = ReviewQueue(normalized_question="问题A", review_status=ReviewStatus.PENDING.value)
        rq2 = ReviewQueue(normalized_question="问题B", review_status=ReviewStatus.PENDING.value)
        db.add_all([rq1, rq2])
        await db.commit()
        await db.refresh(rq1)
        await db.refresh(rq2)
        id1, id2 = rq1.id, rq2.id

    # 1. Simulate Stage 3 commit followed by Milvus failure -> status must stay PENDING & close() called
    async def flaky_write_chunks(session, chunks):
        await session.commit()
        raise RuntimeError("Milvus upsert failed")

    with patch("app.api.review_queue_routes.KnowledgeDualWriter") as MockWriter:
        inst = MockWriter.return_value
        inst.write_chunks = AsyncMock(side_effect=flaky_write_chunks)
        inst.close = MagicMock()

        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
        ) as client:
            resp = await client.post(
                f"/api/review-queue/{id1}/approve", json={"approved_answer": "标准答案"}
            )
            assert resp.status_code == 500
        inst.close.assert_called_once()

    async with session_factory() as db:
        item = await db.get(ReviewQueue, id1)
        assert item.review_status == ReviewStatus.PENDING.value

    # 2. Successful approve followed by duplicate approve -> 409
    with patch("app.api.review_queue_routes.KnowledgeDualWriter") as MockWriter:
        inst = MockWriter.return_value
        inst.write_chunks = AsyncMock(return_value=[])
        inst.close = MagicMock()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r_ok = await client.post(
                f"/api/review-queue/{id1}/approve", json={"approved_answer": "标准答案"}
            )
            assert r_ok.status_code == 200
            r_dup = await client.post(
                f"/api/review-queue/{id1}/approve", json={"approved_answer": "重复答案"}
            )
            assert r_dup.status_code == 409
            assert inst.write_chunks.await_count == 1
            assert inst.close.call_count == 1

    # 3. FE-01(5): Reject accepts optional reason JSON body
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r_rej = await client.post(
            f"/api/review-queue/{id2}/reject", json={"reason": "已有等价 FAQ"}
        )
        assert r_rej.status_code == 200
        body = r_rej.json()
        assert body["status"] == "rejected"
        assert body.get("reason") == "已有等价 FAQ"


# =============================================================================
# CORE-08: GET /api/kb/chunks NULL content_type & pagination bounds
# =============================================================================

@pytest.mark.asyncio
async def test_core08_kb_chunks_null_content_type_and_pagination(async_sqlite_db):
    _, session_factory = async_sqlite_db
    async with session_factory() as db:
        db.add(
            KnowledgeChunk(
                category="售后",
                questions="如何退货",
                answer="7天无理由",
                content_type=None,
            )
        )
        await db.commit()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r_ok = await client.get("/api/kb/chunks?page=1&page_size=20")
        assert r_ok.status_code == 200
        assert len(r_ok.json()["items"]) == 1
        assert r_ok.json()["items"][0]["content_type"] is None

        r_bad_page = await client.get("/api/kb/chunks?page=0")
        assert r_bad_page.status_code == 422

        r_bad_size = await client.get("/api/kb/chunks?page=1&page_size=500")
        assert r_bad_size.status_code == 422


# =============================================================================
# CORE-09: JobRunner non-blocking stop, lock release during terminate, bounded cap
# =============================================================================

@pytest.mark.asyncio
async def test_core09_job_runner_non_blocking_lock_and_retention():
    from app.core.jobs import JobRecord, JobRunner, stop_job, job_runner

    # 1. stop_job must offload blocking work via asyncio.to_thread (event loop stays responsive)
    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.02)
            ticks += 1

    def slow_stop(job_id: str):
        time.sleep(0.15)
        return {"job_id": job_id, "status": "stopped"}

    with patch.object(job_runner, "stop_job", side_effect=slow_stop):
        t = asyncio.create_task(ticker())
        res = await stop_job("job_fake")
        await t
    assert res["status"] == "stopped"
    assert ticks >= 3, "Event loop was blocked during stop_job()"

    # 2. _execute_job must NOT hold self._lock while calling _terminate_process_tree
    runner = JobRunner(max_retained_jobs=3)
    lock_was_free = False

    def fake_terminate(pid: int):
        nonlocal lock_was_free
        acquired = runner._lock.acquire(blocking=False)
        if acquired:
            lock_was_free = True
            runner._lock.release()

    mock_proc = MagicMock()
    mock_proc.pid = 424242

    def fake_popen(*args, **kwargs):
        # Simulate concurrent stop_job flipping status while Popen executes
        job.status = "stopped"
        return mock_proc

    job = JobRecord(job_id="j_race", job_name="eval-rag", status="running")
    runner._jobs[job.job_id] = job
    log_path = ROOT_DIR / "log" / "acceptance" / "test_race.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with patch.object(runner, "_terminate_process_tree", side_effect=fake_terminate):
            with patch("subprocess.Popen", side_effect=fake_popen):
                runner._execute_job(job.job_id, [sys.executable, "-c", "pass"], log_path)
        assert lock_was_free, "_lock must not be held during _terminate_process_tree"
    finally:
        log_path.unlink(missing_ok=True)

    # 3. Finished jobs pruned to max_retained_jobs
    runner2 = JobRunner(max_retained_jobs=2)
    for i in range(4):
        runner2._jobs[f"old_{i}"] = JobRecord(
            job_id=f"old_{i}", job_name="eval-rag", status="completed"
        )
    with patch("threading.Thread.start"):
        runner2.start_job("eval-rag")
    finished = [j for j in runner2._jobs.values() if j.status != "running"]
    assert len(finished) <= 2


# =============================================================================
# CORE-10: SummaryService task safety, session split, orphan guard, token cap
# =============================================================================

@pytest.mark.asyncio
async def test_core10_summary_service_safety_session_split_and_cap(async_sqlite_db):
    from app.services.context.budget import estimate_tokens
    from app.services.context.summary_service import SummaryService

    _, session_factory = async_sqlite_db
    service = SummaryService(session_factory=session_factory)

    # 1. If create_task fails (e.g. no running loop), conv_id must NOT remain wedged in _active_tasks
    with patch("asyncio.create_task", side_effect=RuntimeError("no running event loop")):
        with pytest.raises(RuntimeError):
            service.trigger_async_summary(conv_id=77777, from_msg_id=1, upto_msg_id=2)
    assert 77777 not in service._active_tasks

    # 2. persist_summary_segment must NOT insert orphan row when conversation was deleted
    async with session_factory() as session:
        orphan_res = await service.persist_summary_segment(
            session=session, conv_id=999999, from_id=1, to_id=2, content="孤儿摘要"
        )
        assert orphan_res is None
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(ConversationSummary).where(ConversationSummary.conversation_id == 999999)
            )
        ).scalars().all()
        assert rows == []

    # 3. execute_summary_task must NOT hold DB session open across LLM call
    async with session_factory() as db:
        conv = Conversation(user_id="sum_user", status="进行中")
        db.add(conv)
        await db.commit()
        await db.refresh(conv)
        m1 = Message(conversation_id=conv.id, role="user", content="订单1001催单")
        db.add(m1)
        await db.commit()
        await db.refresh(m1)
        conv_id, m1_id = conv.id, m1.id

    open_sessions = 0
    sessions_open_during_llm = None

    class TrackingSessionCtx:
        def __init__(self, inner):
            self._inner = inner

        async def __aenter__(self):
            nonlocal open_sessions
            open_sessions += 1
            return await self._inner.__aenter__()

        async def __aexit__(self, *args):
            nonlocal open_sessions
            try:
                return await self._inner.__aexit__(*args)
            finally:
                open_sessions -= 1

    tracking_service = SummaryService(
        session_factory=lambda: TrackingSessionCtx(session_factory())
    )

    async def fake_llm(messages):
        nonlocal sessions_open_during_llm
        sessions_open_during_llm = open_sessions
        resp = MagicMock()
        resp.content = "用户催促订单1001。"
        return resp

    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(side_effect=fake_llm)

    res = await tracking_service.execute_summary_task(
        conv_id=conv_id, from_msg_id=m1_id, upto_msg_id=m1_id, model=mock_model
    )
    assert res is not None
    assert sessions_open_during_llm == 0, "DB session was held open across LLM call"

    # 4. Bounded conv.summary projection within summary_max_tokens
    async with session_factory() as session:
        for i in range(10):
            await service.persist_summary_segment(
                session=session,
                conv_id=conv_id,
                from_id=10 + i * 2,
                to_id=11 + i * 2,
                content=f"第{i+2}段摘要：" + ("用户确认售后退货与换货规则并登记工单信息" * 3),
            )
    async with session_factory() as session:
        refreshed = await session.get(Conversation, conv_id)
        assert estimate_tokens(refreshed.summary) <= settings.summary_max_tokens


# =============================================================================
# CORE-11 & VER-07: Async after_sale extraction + None guard + leakage cache key
# =============================================================================

@pytest.mark.asyncio
async def test_core11_after_sale_async_and_none_guard():
    from app.schemas.after_sale import AfterSaleTicket
    from app.services.after_sale_service import (
        aextract_after_sale_ticket,
        extract_after_sale_ticket,
    )

    mock_model = MagicMock()
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(
        return_value=AfterSaleTicket(
            order_id="ORD123",
            issue_type="破损",
            expected_solution="退款",
            raw_description="orig",
        )
    )
    mock_model.with_structured_output.return_value = mock_structured

    res = await aextract_after_sale_ticket("商品破损要退款", model=mock_model)
    assert res.order_id == "ORD123"
    assert res.raw_description == "商品破损要退款"
    mock_structured.ainvoke.assert_awaited_once()

    # None guard in both sync and async extractors
    mock_structured.ainvoke = AsyncMock(return_value=None)
    mock_structured.invoke.return_value = None
    with pytest.raises(ValueError):
        await aextract_after_sale_ticket("破损", model=mock_model)
    with pytest.raises(ValueError):
        extract_after_sale_ticket("破损", model=mock_model)


def test_core11_acceptance_leakage_cache_keys_by_dir(tmp_path):
    from app.api.acceptance import get_cached_leakage_check

    d1 = tmp_path / "ds1"
    d2 = tmp_path / "ds2"
    d1.mkdir()
    d2.mkdir()

    # ds1: zero leakage; ds2: leaked sample
    for d, test_text in [(d1, "唯一测试句"), (d2, "训练句一")]:
        (d / "train.jsonl").write_text('{"text": "训练句一"}\n', encoding="utf-8")
        (d / "val.jsonl").write_text('{"text": "验证句一"}\n', encoding="utf-8")
        (d / "test.jsonl").write_text(f'{{"text": "{test_text}"}}\n', encoding="utf-8")
        for fn in ("train.jsonl", "val.jsonl", "test.jsonl"):
            os.utime(d / fn, (1700000000, 1700000000))

    rep1 = get_cached_leakage_check(d1)
    rep2 = get_cached_leakage_check(d2)
    assert rep1["passed"] is True
    assert rep2["passed"] is False


# =============================================================================
# CORE-12: DB engine pool_pre_ping=True
# =============================================================================

def test_core12_db_pool_pre_ping():
    assert global_engine.pool._pre_ping is True
    if getattr(global_engine.sync_engine.dialect, "driver", None) == "aiomysql":
        assert getattr(global_engine.sync_engine.dialect, "_send_false_to_ping", False) is True


# =============================================================================
# CORE-13: No duplicate /api/jobs routes & zero Duplicate Operation ID warnings
# =============================================================================

def test_core13_no_duplicate_jobs_routes_or_openapi_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Clear cached openapi schema so generation runs cleanly
        app.openapi_schema = None
        spec = app.openapi()

    dup_warnings = [w for w in caught if "Duplicate Operation ID" in str(w.message)]
    assert dup_warnings == [], f"Unexpected OpenAPI duplicate warnings: {dup_warnings}"

    logs_params = spec["paths"]["/api/jobs/{job_id}/logs"]["get"]["parameters"]
    tail_param = next(p for p in logs_params if p["name"] == "tail")
    # jobs.py caps tail at 5000 (not rag_eval_routes.py's legacy 2000)
    tail_schema = json.dumps(tail_param)
    assert "5000" in tail_schema and "2000" not in tail_schema


# =============================================================================
# CORE-14: /api/chat/feedback validation, 404, message_id, score=0.0
# =============================================================================

@pytest.mark.asyncio
async def test_core14_chat_feedback_validation_404_and_zero_score(async_sqlite_db):
    _, session_factory = async_sqlite_db
    async with session_factory() as db:
        conv = Conversation(user_id="u1", status="进行中")
        db.add(conv)
        await db.commit()
        await db.refresh(conv)
        m_u1 = Message(conversation_id=conv.id, role="user", content="第一个问题")
        m_a1 = Message(
            conversation_id=conv.id,
            role="assistant",
            content="第一个回答",
            tool_calls=[{"name": "query_faq", "args": {}}],
        )
        m_u2 = Message(conversation_id=conv.id, role="user", content="第二个问题")
        db.add_all([m_u1, m_a1, m_u2])
        db.add(
            ToolAuditLog(
                conversation_id=conv.id,
                tool_name="query_faq",
                tool_source="builtin",
                status="成功",
                result_summary=json.dumps(
                    {"hits": [{"text": "片段内容", "score": 0.0, "distance": 0.77, "section": "S1"}]}
                ),
            )
        )
        await db.commit()
        await db.refresh(m_a1)
        cid, msg1_id = conv.id, m_a1.id

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Invalid feedback_type -> 422
        r_bad = await client.post(
            "/api/chat/feedback", json={"conversation_id": cid, "feedback_type": "sideways"}
        )
        assert r_bad.status_code == 422

        # Missing conversation -> 404
        r_404 = await client.post(
            "/api/chat/feedback", json={"conversation_id": 999999, "feedback_type": "down"}
        )
        assert r_404.status_code == 404

        # Valid down feedback with message_id -> honors message_id & preserves score == 0.0
        r_ok = await client.post(
            "/api/chat/feedback",
            json={"conversation_id": cid, "message_id": msg1_id, "feedback_type": "down"},
        )
        assert r_ok.status_code == 200
        entry_id = r_ok.json()["entry_id"]

    async with session_factory() as db:
        entry = await db.get(LowConfidenceQuestion, entry_id)
        assert entry.raw_question == "第一个问题"
        assert entry.retrieved_chunks is not None
        assert entry.retrieved_chunks[0]["score"] == 0.0


# =============================================================================
# CORE-15: /api/kb/documents/manual filename validation, no orphan file, overwrite
# =============================================================================

@pytest.mark.asyncio
async def test_core15_kb_manual_document_validation_and_no_orphan_file(async_sqlite_db, tmp_path):
    with patch("app.api.routes.KB_DIR", tmp_path):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Invalid filenames -> 400
            for bad_name in ["../evil.md", ".md", "..", "CON.md"]:
                r = await client.post(
                    "/api/kb/documents/manual",
                    json={
                        "filename": bad_name,
                        "content": "# 标题\n## 节\n正文内容",
                        "save_file": True,
                        "sync_vector": False,
                    },
                )
                assert r.status_code == 400, f"Expected 400 for filename={bad_name!r}, got {r.status_code}"

            # Empty chunk split -> 400 and NO orphan file written
            with patch("app.api.routes.MarkdownSectionSplitter.split_markdown", return_value=[]):
                r_empty = await client.post(
                    "/api/kb/documents/manual",
                    json={
                        "filename": "orphan_test.md",
                        "content": "内容",
                        "save_file": True,
                        "sync_vector": False,
                    },
                )
                assert r_empty.status_code == 400
                assert not (tmp_path / "orphan_test.md").exists()

            # Existing file without overwrite=True -> 409
            (tmp_path / "existing.md").write_text("old", encoding="utf-8")
            r_dup = await client.post(
                "/api/kb/documents/manual",
                json={
                    "filename": "existing.md",
                    "content": "# 标题\n## 节\n新正文内容",
                    "save_file": True,
                    "sync_vector": False,
                },
            )
            assert r_dup.status_code == 409
            assert (tmp_path / "existing.md").read_text(encoding="utf-8") == "old"


# =============================================================================
# CORE-16 & CORE-20: FaithCase resolution validation, revert clear, UTC, IntegrityError retry
# =============================================================================

@pytest.mark.asyncio
async def test_core16_and_20_faith_case_resolve_and_upsert(async_sqlite_db):
    _, session_factory = async_sqlite_db
    async with session_factory() as db:
        case = await upsert_faith_case(
            db,
            eval_id="T01",
            bucket="A_policy",
            query="退货运费",
            answer="全免",
            reason="编造",
        )
        case_id = case.id

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Resolution > 300 chars -> 422
        r_long = await client.post(
            f"/api/rag-eval/faith-cases/{case_id}/resolve",
            json={"status": "已解决", "resolution": "x" * 301},
        )
        assert r_long.status_code == 422

        # Resolve -> sets UTC resolved_at
        r_res = await client.post(
            f"/api/rag-eval/faith-cases/{case_id}/resolve",
            json={"status": "已解决", "resolution": "已修正知识库"},
        )
        assert r_res.status_code == 200

        # Revert to 未解决 -> clears resolution per DDL spec
        r_rev = await client.post(
            f"/api/rag-eval/faith-cases/{case_id}/resolve",
            json={"status": "未解决", "resolution": ""},
        )
        assert r_rev.status_code == 200
        assert r_rev.json()["resolution"] is None

    async with session_factory() as db:
        refetched = await db.get(FaithCase, case_id)
        assert refetched.resolution is None
        assert refetched.resolved_at is not None
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        assert abs((now_utc - refetched.resolved_at).total_seconds()) < 60

    # Check ORM index alignment (Conversation.user_id & FAQ.category)
    assert Conversation.__table__.c.user_id.index is True
    from app.models.faq import FAQ
    assert FAQ.__table__.c.category.index is True


@pytest.mark.asyncio
async def test_core20_upsert_faith_case_integrity_error_retry(async_sqlite_db):
    _, session_factory = async_sqlite_db
    async with session_factory() as db:
        await upsert_faith_case(
            db, eval_id="RACE01", bucket="A_policy", query="q", answer="a1", reason="r1"
        )

    # Simulate concurrent race where first SELECT returns None, INSERT hits IntegrityError, retry updates row
    async with session_factory() as db:
        orig_execute = db.execute
        call_count = 0

        async def race_execute(stmt, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                empty = MagicMock()
                empty.scalar_one_or_none.return_value = None
                return empty
            return await orig_execute(stmt, *args, **kwargs)

        with patch.object(db, "execute", side_effect=race_execute):
            updated = await upsert_faith_case(
                db, eval_id="RACE01", bucket="A_policy", query="q", answer="a2", reason="r2"
            )
        assert updated.seen_count == 2
        assert updated.answer == "a2"


# =============================================================================
# CORE-17: Eliminate N+1 & enforce pagination bounds on conversations & review-queue
# =============================================================================

@pytest.mark.asyncio
async def test_core17_no_n_plus_1_and_pagination_bounds(async_sqlite_db):
    eng, session_factory = async_sqlite_db
    async with session_factory() as db:
        convs = [Conversation(user_id="bulk_user", status="进行中") for _ in range(12)]
        rqs = [ReviewQueue(normalized_question=f"问题{i}") for i in range(12)]
        db.add_all(convs + rqs)
        await db.commit()
        for c in convs:
            db.add(Message(conversation_id=c.id, role="user", content=f"首问{c.id}"))
        for rq in rqs:
            db.add(LowConfidenceQuestion(raw_question=f"原话{rq.id}", source="self_check", matched_review_id=rq.id))
        await db.commit()

    query_count = 0

    def _count_sql(conn, cursor, statement, parameters, context, executemany):
        nonlocal query_count
        query_count += 1

    event.listen(eng.sync_engine, "before_cursor_execute", _count_sql)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            query_count = 0
            r_conv = await client.get("/api/conversations?user_id=bulk_user&limit=10&offset=0")
            assert r_conv.status_code == 200
            assert len(r_conv.json()) == 10
            assert query_count <= 4, f"N+1 detected in list_conversations: {query_count} queries"

            query_count = 0
            r_rq = await client.get("/api/review-queue?limit=10&offset=0")
            assert r_rq.status_code == 200
            assert len(r_rq.json()["items"]) == 10
            assert query_count <= 4, f"N+1 detected in get_review_queue: {query_count} queries"

            # Pagination bounds
            assert (await client.get("/api/conversations?limit=500")).status_code == 422
            assert (await client.get("/api/conversations?offset=-1")).status_code == 422
            assert (await client.get("/api/review-queue?limit=500")).status_code == 422
            assert (await client.get("/api/review-queue?offset=-1")).status_code == 422
    finally:
        event.remove(eng.sync_engine, "before_cursor_execute", _count_sql)


# =============================================================================
# CORE-18: Scripts (verify_mysql, wsl_helper, start_mysql.ps1) & run.py engine_override
# =============================================================================

@pytest.mark.asyncio
async def test_core18_scripts_and_run_engine_override():
    from scripts import wsl_helper

    # wsl_helper dynamic path conversion & no hardcoded /mnt/d/PycharmProjects/AstroBot
    wsl_src = (ROOT_DIR / "scripts" / "wsl_helper.py").read_text(encoding="utf-8")
    assert "/mnt/d/PycharmProjects/AstroBot" not in wsl_src
    assert "atexit.register" in wsl_src
    assert wsl_helper._to_wsl_path(Path("C:/Work/AstroBot/docker-compose.yml")) == "/mnt/c/Work/AstroBot/docker-compose.yml"

    ps1_src = (ROOT_DIR / "scripts" / "start_mysql.ps1").read_text(encoding="utf-8")
    assert "/mnt/d/PycharmProjects/AstroBot" not in ps1_src

    # verify_mysql.py adds PROJECT_ROOT to sys.path (does not fail with ModuleNotFoundError)
    verify_src = (ROOT_DIR / "scripts" / "verify_mysql.py").read_text(encoding="utf-8")
    assert "sys.path" in verify_src

    # run.py passes engine_override=engine to init_ch07_db and init_ch08_db
    import run as run_mod

    mock_init_ch07 = AsyncMock(return_value=[])
    mock_init_ch08 = AsyncMock(return_value=[])
    mock_init_ch09 = AsyncMock(return_value=[])
    mock_init_ch10 = AsyncMock(return_value=[])
    mock_engine = MagicMock()
    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__.return_value = mock_conn
    mock_engine.dispose = AsyncMock()
    mock_session = AsyncMock()
    mock_res0 = MagicMock()
    mock_res0.scalar.return_value = 0
    mock_res1 = MagicMock()
    mock_res1.scalar.return_value = 5
    mock_session.execute.side_effect = [mock_res0, mock_res1]
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_store = MagicMock()
    mock_store.count.return_value = 5

    with (
        patch("scripts.wsl_helper.ensure_mysql_ready"),
        patch("app.db.session.engine", mock_engine),
        patch("scripts.seed_data.seed_all_data", new_callable=AsyncMock, return_value=0),
        patch("scripts.init_ch07_db.init_ch07_db", mock_init_ch07),
        patch("scripts.init_ch08_db.init_ch08_db", mock_init_ch08),
        patch("scripts.init_ch09_db.init_ch09_db", mock_init_ch09),
        patch("scripts.init_ch10_db.init_ch10_db", mock_init_ch10),
        patch("app.db.session.AsyncSessionLocal", return_value=mock_ctx),
        patch("app.services.rag.milvus_client.MilvusKnowledgeStore", return_value=mock_store),
        patch("scripts.wsl_helper.is_port_open", return_value=False),
    ):
        ok = await run_mod.check_storage_readiness(clean_kb=False)
        assert ok is True
        mock_init_ch07.assert_awaited_once_with(engine_override=mock_engine)
        mock_init_ch08.assert_awaited_once_with(engine_override=mock_engine)


# =============================================================================
# CORE-19: Idempotent MySQL multi-clause ALTER TABLE in init_ch07_db & init_ch09_db
# =============================================================================

@pytest.mark.asyncio
async def test_core19_mysql_partial_alter_continues_remaining_clauses():
    from scripts.init_ch09_db import init_ch09_db

    executed_sqls = []

    async def fake_execute(stmt):
        sql = str(stmt)
        executed_sqls.append(sql)
        if "retrieved_chunks" in sql and "ALTER TABLE" in sql.upper():
            raise RuntimeError("(1060, \"Duplicate column name 'retrieved_chunks'\")")
        return MagicMock()

    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock(side_effect=fake_execute)
    mock_engine = MagicMock()
    mock_engine.dialect.name = "mysql"
    mock_engine.url = "mysql+aiomysql://root@127.0.0.1/astro_bot"
    mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=None)

    await init_ch09_db(engine_override=mock_engine)
    alter_sqls = [s for s in executed_sqls if "ALTER TABLE" in s.upper()]
    assert any("matched_review_id" in s and "retrieved_chunks" not in s for s in alter_sqls)
    assert any("fk_lcq_review" in s for s in alter_sqls)


# =============================================================================
# CORE-20 & VER-08: create_manual_chunk, topic_distribution, SessionManager, hygiene
# =============================================================================

@pytest.mark.asyncio
async def test_core20_manual_chunk_error_topic_distribution_and_session_cap(async_sqlite_db):
    _, session_factory = async_sqlite_db

    # 1. create_manual_chunk must not double-wrap HTTPException
    with patch("app.api.routes.KnowledgeDualWriter") as MockWriter:
        inst = MockWriter.return_value
        inst.write_chunks = AsyncMock(return_value=[])
        inst.close = MagicMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/api/kb/chunks/manual",
                json={
                    "category": "售后",
                    "questions": "测试",
                    "answer": "测试答复",
                    "sync_vector": True,
                },
            )
            assert r.status_code == 500
            assert r.json()["detail"] == "保存知识块失败"

    # 2. get_topic_distribution keeps top_3_topics strictly within TAXONOMY_17
    async with session_factory() as db:
        lcq = LowConfidenceQuestion(raw_question="q", source="self_check")
        db.add(lcq)
        await db.commit()
        await db.refresh(lcq)
        db.add(TopicClassification(question_id=lcq.id, labels=["UNKNOWN_BOGUS_LABEL", "退换货"]))
        await db.commit()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r_td = await client.get("/api/topics/distribution")
        assert r_td.status_code == 200
        td = r_td.json()
        assert "UNKNOWN_BOGUS_LABEL" not in td["top_3_topics"]

    # 3. SessionManager evicts oldest sessions beyond max_sessions
    from app.services.session_manager import SessionManager

    sm = SessionManager(max_sessions=3)
    for i in range(5):
        sm.get_or_create_session(f"s{i}")
        sm.add_user_message(f"s{i}", f"msg{i}")
    assert len(sm._sessions) == 3
    assert sm.get_history("s0") == []
    assert len(sm.get_history("s4")) == 1


def test_ver08_gitignore_and_requirements_hygiene():
    gi = (ROOT_DIR / ".gitignore").read_text(encoding="utf-8")
    for entry in [".ruff_cache/", ".mypy_cache/", ".superpowers/"]:
        assert entry in gi

    req = (ROOT_DIR / "requirements.txt").read_text(encoding="utf-8").lower()
    for dep in ["pymilvus", "aiosqlite", "huggingface-hub", "jieba"]:
        assert dep in req

    ch03_ddl = (ROOT_DIR / "sql" / "ch03-ddl.sql").read_text(encoding="utf-8")
    assert "ENUM('pending','done','failed')" in ch03_ddl
    assert "ENUM('extracted','kept','discarded','ingested')" in ch03_ddl
