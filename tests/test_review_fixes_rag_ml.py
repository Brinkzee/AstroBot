"""Regression test suite for RAG, Eval, Flywheel, Observability, Classifier & Scripts audit fixes."""

import asyncio
import json
from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db.session import Base
from app.models.conversation import Conversation
from app.models.eval_run import EvalRun
from app.models.knowledge import KnowledgeChunk
from app.models.low_confidence import LowConfidenceQuestion
from app.models.message import Message
from app.models.review_queue import ReviewQueue
from app.models.staging import QAExtractionStaging
from app.models.topic_classification import TopicClassification


@pytest.fixture
async def async_sqlite_db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


# ==============================================================================
# RAG-01: Milvus filter expression escaping
# ==============================================================================

def test_rag01_milvus_category_filter_escaping():
    from app.services.rag.milvus_client import MilvusKnowledgeStore

    store = object.__new__(MilvusKnowledgeStore)
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    mock_client.search.return_value = [[]]
    mock_client.hybrid_search.return_value = [[]]
    store.client = mock_client
    store.uri = "./dummy.db"

    payload = '退换货" or id >= 0 or category == "hack\\test'

    store.search_dense([0.1] * 1024, category_filter=payload)
    dense_filter = mock_client.search.call_args.kwargs["filter"]
    assert '\\"' in dense_filter
    assert '\\\\' in dense_filter
    assert 'or id >= 0' not in dense_filter.split('\\"')[0]

    store.search_bm25("退货", category_filter=payload)
    bm25_filter = mock_client.search.call_args.kwargs["filter"]
    assert '\\"' in bm25_filter
    assert '\\\\' in bm25_filter

    store.hybrid_search([0.1] * 1024, "退货", category_filter=payload)
    reqs = mock_client.hybrid_search.call_args.kwargs["reqs"]
    for req in reqs:
        assert '\\"' in req.expr
        assert '\\\\' in req.expr


# ==============================================================================
# RAG-02 / CORE-04: Concurrent ContextVar isolation for last_result
# ==============================================================================

@pytest.mark.asyncio
async def test_rag02_advanced_retriever_last_result_isolated_across_tasks():
    from app.services.rag.advanced_retriever import AdvancedKnowledgeRetriever
    from app.services.rag.query_processor import QueryUnderstandingResult

    mock_qp = MagicMock()

    async def slow_aprocess(q):
        await asyncio.sleep(0.01 if q == "query_A" else 0.03)
        return QueryUnderstandingResult(q, q, [], q)

    mock_qp.aprocess = slow_aprocess

    mock_emb = MagicMock()
    mock_emb.aembed_query = AsyncMock(return_value=[0.1] * 1024)

    mock_store = MagicMock()
    mock_store.search_dense = MagicMock(
        side_effect=lambda **kw: [{"id": 1, "question": "Q", "answer": "A", "distance": 0.9}]
    )

    retriever = AdvancedKnowledgeRetriever(
        store=mock_store,
        embedding_client=mock_emb,
        query_processor=mock_qp,
        reranker_client=MagicMock(),
    )

    async def task_a():
        await retriever.retrieve_with_strategy("query_A", strategy="vector_only", top_k=1)
        await asyncio.sleep(0.05)
        return retriever.last_result.query_understanding.original_query

    async def task_b():
        await asyncio.sleep(0.01)
        await retriever.retrieve_with_strategy("query_B", strategy="vector_only", top_k=1)
        return retriever.last_result.query_understanding.original_query

    res_a, res_b = await asyncio.gather(task_a(), task_b())
    assert res_a == "query_A"
    assert res_b == "query_B"


# ==============================================================================
# RAG-03: Pre-reorder ranked_docs & plural questions field in metrics
# ==============================================================================

@pytest.mark.asyncio
async def test_rag03_hybrid_rerank_eval_uses_ranked_docs_and_plural_questions():
    from app.services.rag.advanced_retriever import AdvancedKnowledgeRetriever
    from app.services.rag.evaluator import RAGEvaluator, compute_retrieval_metrics
    from app.services.rag.query_processor import QueryUnderstandingResult

    # Test plural 'questions' field directly
    docs_plural = [
        {"section_path": "其他", "questions": "不相关问题", "category": "其他"},
        {"section_path": "其他", "questions": "PRO-X99充电规格说明", "category": "商品"},
    ]
    m = compute_retrieval_metrics(docs_plural, expect_section=["PRO-X99充电规格"])
    assert m["recall@3"] == 1.0
    assert m["mrr"] == 0.5

    # Test hybrid_rerank where ground-truth is #2 by rerank_score (moved to pos 10 by Lost-in-the-Middle)
    candidates = [
        {"id": i, "section_path": f"其他 > 干扰{i}", "questions": f"干扰{i}", "answer": f"A{i}"}
        for i in range(1, 11)
    ]
    candidates[1]["section_path"] = "售后政策 > 退货流程"
    candidates[1]["questions"] = "退货流程是什么"

    reranked = []
    for idx, c in enumerate(candidates):
        d = dict(c)
        d["rerank_score"] = round(1.0 - idx * 0.05, 4)
        reranked.append(d)

    mock_qp = MagicMock()
    mock_qp.aprocess = AsyncMock(return_value=QueryUnderstandingResult("退货", "退货", [], "退货"))
    mock_emb = MagicMock()
    mock_emb.aembed_query = AsyncMock(return_value=[0.1] * 1024)
    mock_store = MagicMock()
    mock_store.hybrid_search = MagicMock(return_value=candidates)
    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=reranked)

    retriever = AdvancedKnowledgeRetriever(
        store=mock_store,
        embedding_client=mock_emb,
        query_processor=mock_qp,
        reranker_client=mock_reranker,
    )
    res = await retriever.retrieve_with_strategy("退货", strategy="hybrid_rerank", top_k=10)
    # Lost in the middle moves #2 (id=2) to last slot in docs, but ranked_docs preserves rank #2
    assert res.docs[-1]["id"] == 2
    assert res.ranked_docs[1]["id"] == 2

    evaluator = RAGEvaluator(retriever=retriever, is_mock=False)
    sample = {
        "id": "A1",
        "bucket": "A_policy",
        "query": "退货流程",
        "expect_section": ["退货流程"],
        "should_refuse": False,
    }
    eval_res = await evaluator.evaluate_sample(sample, strategy="hybrid_rerank")
    assert eval_res["recall@3"] == 1.0
    assert eval_res["recall@5"] == 1.0
    assert eval_res["mrr"] == 0.5


# ==============================================================================
# RAG-04: Milvus hybrid_search one-sided empty, _format_hits entity strip, closed client guard
# ==============================================================================

def test_rag04_milvus_edge_cases():
    from app.services.rag.milvus_client import MilvusKnowledgeStore

    store = object.__new__(MilvusKnowledgeStore)
    store.uri = "./dummy.db"
    store.client = None

    with pytest.raises(RuntimeError):
        store.init_collection("knowledge")
    with pytest.raises(RuntimeError):
        store.upsert([{"id": 1, "vector": [0.1] * 1024}])

    # _format_hits must strip vector & sparse_vector from entity too
    raw = [{
        "id": 1,
        "distance": 0.9,
        "entity": {"category": "c", "vector": [0.1] * 1024, "sparse_vector": {1: 0.5}},
    }]
    formatted = MilvusKnowledgeStore._format_hits(raw)
    assert "vector" not in formatted[0]
    assert "sparse_vector" not in formatted[0]
    assert "vector" not in formatted[0]["entity"]
    assert "sparse_vector" not in formatted[0]["entity"]

    # hybrid_search delegates on one-sided empty input
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    store.client = mock_client
    store.search_bm25 = MagicMock(return_value=[{"id": 10}])
    store.search_dense = MagicMock(return_value=[{"id": 20}])

    assert store.hybrid_search([], "退货政策") == [{"id": 10}]
    store.search_bm25.assert_called_once()
    assert store.hybrid_search([0.1] * 1024, "   ") == [{"id": 20}]
    store.search_dense.assert_called_once()


# ==============================================================================
# RAG-05: Embedding timeout/validation/fallback & DualWriter safety + byte truncation
# ==============================================================================

@pytest.mark.asyncio
async def test_rag05_embedding_and_dual_writer_consistency(async_sqlite_db):
    from app.services.rag.dual_writer import KnowledgeDualWriter
    from app.services.rag.embedding import BGEEmbeddingClient
    from app.services.rag.splitter import DocChunk

    # 1. InferenceClient receives timeout
    mock_hf = MagicMock()
    with patch("huggingface_hub.InferenceClient", return_value=mock_hf) as mock_cls:
        client = BGEEmbeddingClient(token="hf_live", timeout=12.5, fallback_to_mock=False)
        assert mock_cls.call_args.kwargs.get("timeout") == 12.5

        # Ragged 3D-like token list response (varying seq_len)
        ragged = [
            [[1.0] * 1024, [1.0] * 1024],
            [[2.0] * 1024, [2.0] * 1024, [2.0] * 1024],
        ]
        parsed = client._process_hf_response(ragged, expected_count=2, dim=1024)
        assert len(parsed) == 2
        assert len(parsed[0]) == 1024

        # Wrong count raises ValueError
        with pytest.raises(ValueError):
            client._process_hf_response(np.ones((1, 1024)), expected_count=2, dim=1024)

        # Wrong dim raises ValueError
        with pytest.raises(ValueError):
            client._process_hf_response(np.ones((2, 512)), expected_count=2, dim=1024)

    # 2. DualWriter fails cleanly without marking 'done' when real embedding fails
    mock_hf_fail = MagicMock()
    mock_hf_fail.feature_extraction.side_effect = RuntimeError("HF 503")
    with patch("huggingface_hub.InferenceClient", return_value=mock_hf_fail):
        emb = BGEEmbeddingClient(token="hf_live", max_retries=1, retry_delay=0.001, fallback_to_mock=False)
        mock_store = MagicMock()
        writer = KnowledgeDualWriter(store=mock_store, embedding_client=emb)
        chunks = [DocChunk(category="分类", questions="Q", answer="A")]
        with pytest.raises(RuntimeError):
            await writer.write_chunks(async_sqlite_db, chunks)

        mock_store.upsert.assert_not_called()
        rows = list((await async_sqlite_db.execute(select(KnowledgeChunk))).scalars().all())
        assert len(rows) == 1
        assert rows[0].vectorize_status == "pending"

    # 3. DualWriter raises on vector count mismatch & truncates oversized UTF-8 VARCHARs
    mock_emb_short = MagicMock()
    mock_emb_short.aembed_documents = AsyncMock(return_value=[[0.1] * 1024])  # only 1 for 2 chunks
    mock_store2 = MagicMock()
    writer2 = KnowledgeDualWriter(store=mock_store2, embedding_client=mock_emb_short)
    with pytest.raises(ValueError):
        await writer2.write_chunks(
            async_sqlite_db,
            [DocChunk("c1", "q1", "a1"), DocChunk("c2", "q2", "a2")],
        )

    long_cat = "中" * 80  # 240 UTF-8 bytes > 128
    long_ans = "答" * 4000  # 12000 UTF-8 bytes > 8192
    mock_emb_ok = MagicMock()
    mock_emb_ok.aembed_documents = AsyncMock(return_value=[[0.1] * 1024])
    mock_store3 = MagicMock()
    mock_store3.upsert = MagicMock(return_value=1)
    writer3 = KnowledgeDualWriter(store=mock_store3, embedding_client=mock_emb_ok)
    await writer3.write_chunks(
        async_sqlite_db,
        [DocChunk(category=long_cat, questions="问" * 1000, answer=long_ans, section_path="路" * 300)],
    )
    upserted_rec = mock_store3.upsert.call_args[0][0][0]
    assert len(upserted_rec["category"].encode("utf-8")) <= 128
    assert len(upserted_rec["section_path"].encode("utf-8")) <= 512
    assert len(upserted_rec["questions"].encode("utf-8")) <= 2048
    assert len(upserted_rec["answer"].encode("utf-8")) <= 8192


# ==============================================================================
# RAG-06 & VER-06: Reranker score-scale consistency, max_candidates, None client guard
# ==============================================================================

@pytest.mark.asyncio
async def test_rag06_reranker_uniform_fallback_and_max_candidates():
    from app.services.rag.reranker import BGERerankerClient

    BGERerankerClient.reset_circuit_breaker()

    # None client guard
    r_none = BGERerankerClient(mock=True)
    with pytest.raises(RuntimeError):
        r_none._call_hf_api("q", ["doc"])

    # Batch 1 succeeds, batch 2 fails -> must re-score ALL uniformly on mock scale
    reranker = BGERerankerClient(token="hf_tok", max_retries=1, retry_delay=0.001, batch_size=2, mock=False)
    candidates = [
        {"id": 1, "answer": "完全无关的噪音文本甲"},
        {"id": 2, "answer": "完全无关的噪音文本乙"},
        {"id": 3, "answer": "退货政策说明7天无理由退货"},
        {"id": 4, "answer": "退货政策说明"},
    ]
    call_idx = 0

    def fake_api(query, batch):
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            return [0.99, 0.98]  # inflated cross-encoder scores on unrelated batch 1
        raise TimeoutError("batch 2 timeout")

    with patch.object(reranker, "_call_hf_api", side_effect=fake_api):
        res = await reranker.rerank("退货政策说明", candidates, top_k=4)

    # Under uniform mock scoring, id=4/3 (exact match) must rank ABOVE unrelated id=1/2 (not beaten by 0.99)
    assert res[0]["id"] in (3, 4)
    assert res[0]["rerank_score"] != 0.99
    BGERerankerClient.reset_circuit_breaker()

    # max_candidates threaded through async rerank()
    reranker_mock = BGERerankerClient(mock=True)
    many = [{"id": i, "answer": "噪音"} for i in range(1, 26)]
    many[24]["answer"] = "星光PRO-X99防水等级5ATM"
    out = await reranker_mock.rerank("星光PRO-X99防水等级", many, top_k=5, max_candidates=25)
    assert out[0]["id"] == 25


# ==============================================================================
# RAG-07: Generator citation text/score extraction, empty-citation DB guard, mid-stream error
# ==============================================================================

@pytest.mark.asyncio
async def test_rag07_generator_fixes():
    from app.services.rag.generator import RAGControlledGenerator
    from app.services.rag.reorder import build_citation_items

    # 1. check_sufficiency extracts question+answer from build_citation_items
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = MagicMock(content='{"useful": false, "reason": "证据不足"}')
    gen = RAGControlledGenerator(model=mock_model)

    citations = build_citation_items([
        {"id": 1, "section_path": "退货", "question": "退货时效", "answer": "7天无理由", "rerank_score": 0.42}
    ])
    captured = {}

    async def fake_record(**kw):
        captured.update(kw)

    with patch("app.services.rag.generator.record_low_confidence", side_effect=fake_record):
        res = await gen.check_sufficiency("支持货到付款吗", citations, db=MagicMock())
    assert res.useful is False
    assert captured["retrieved_chunks"][0]["text"] != ""
    assert "7天无理由" in captured["retrieved_chunks"][0]["text"]

    # 2. Empty citations DB error is caught instead of crashing
    with patch("app.services.rag.generator.record_low_confidence", side_effect=RuntimeError("DB down")):
        res_empty = await gen.check_sufficiency("未知问题", [], db=MagicMock())
    assert res_empty.useful is False

    # 3. Mid-stream exception does NOT append mock text after partial tokens
    class PartialFailLLM:
        async def astream(self, messages):
            yield MagicMock(content="部分真实回答")
            raise RuntimeError("stream reset")

    gen_stream = RAGControlledGenerator(stream_model=PartialFailLLM())
    chunks = [c async for c in gen_stream.astream_generate("退货", citations)]
    full = "".join(chunks)
    assert "部分真实回答" in full
    assert "根据官方服务规范为您解答：" not in full


# ==============================================================================
# RAG-08 & VER-06: Miner idempotent ingest, per-conv isolation, None content guard
# ==============================================================================

@pytest.mark.asyncio
async def test_rag08_miner_idempotent_ingest_and_error_isolation(async_sqlite_db):
    from app.services.rag.miner import DialogueKnowledgeMiner

    # 1. Idempotent ingest_kept_chunks
    async_sqlite_db.add(
        QAExtractionStaging(
            batch_no="B1", source_ref="c1", question="保修多久？", answer="一年质保。", status="kept"
        )
    )
    await async_sqlite_db.commit()

    mock_writer = MagicMock()
    mock_writer.write_chunks = AsyncMock(return_value=[MagicMock()])
    miner = DialogueKnowledgeMiner(llm=MagicMock())

    n1 = await miner.ingest_kept_chunks(async_sqlite_db, dual_writer=mock_writer, batch_no="B1")
    n2 = await miner.ingest_kept_chunks(async_sqlite_db, dual_writer=mock_writer, batch_no="B1")
    assert n1 == 1
    assert n2 == 0

    rows = list((await async_sqlite_db.execute(select(QAExtractionStaging))).scalars().all())
    assert rows[0].status == "ingested"

    # 2. Per-conversation isolation in extract_batch_dialogues
    c1 = Conversation(user_id="u1", status="已结束")
    c2 = Conversation(user_id="u2", status="已结束")
    async_sqlite_db.add_all([c1, c2])
    await async_sqlite_db.flush()
    async_sqlite_db.add_all([
        Message(conversation_id=c1.id, role="user", content="Q1"),
        Message(conversation_id=c2.id, role="user", content="Q2"),
    ])
    await async_sqlite_db.commit()

    calls = 0

    async def flaky_extract(session, dialogue_text, source_ref, batch_no):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("LLM error on conv 1")
        return [MagicMock()]

    with patch.object(miner, "extract_and_stage", side_effect=flaky_extract):
        staged = await miner.extract_batch_dialogues(async_sqlite_db, batch_size=10)
    assert staged == 1


# ==============================================================================
# RAG-09: AdvancedKnowledgeRetriever min_score on non-rerank strategies & top_k <= 0
# ==============================================================================

@pytest.mark.asyncio
async def test_rag09_advanced_retriever_min_score_and_top_k_zero():
    from app.services.rag.advanced_retriever import AdvancedKnowledgeRetriever
    from app.services.rag.query_processor import QueryUnderstandingResult

    mock_qp = MagicMock()
    mock_qp.aprocess = AsyncMock(return_value=QueryUnderstandingResult("退货", "退货", [], "退货"))
    mock_emb = MagicMock()
    mock_emb.aembed_query = AsyncMock(return_value=[0.1] * 1024)
    mock_store = MagicMock()
    mock_store.search_dense = MagicMock(return_value=[
        {"id": 1, "distance": 0.9, "question": "q1", "answer": "a1"},
        {"id": 2, "distance": 0.3, "question": "q2", "answer": "a2"},
    ])

    retriever = AdvancedKnowledgeRetriever(
        store=mock_store, embedding_client=mock_emb, query_processor=mock_qp, reranker_client=MagicMock()
    )

    res = await retriever.retrieve_with_strategy("退货", strategy="vector_only", top_k=5, min_score=0.8)
    assert [d["id"] for d in res.docs] == [1]

    res_zero = await retriever.retrieve_with_strategy("退货", strategy="vector_only", top_k=0)
    assert res_zero.docs == []


# ==============================================================================
# RAG-10: QueryProcessor running-loop error & Splitter multi-line FAQ + chunk_size guard
# ==============================================================================

@pytest.mark.asyncio
async def test_rag10_query_processor_and_splitter():
    from app.services.rag.query_processor import QueryProcessor
    from app.services.rag.splitter import MarkdownStructureSplitter, split_text_with_sentence_boundary

    # Async-only LLM raising inside a running event loop must fall back cleanly without RuntimeError
    class AsyncOnlyFailingLLM:
        async def ainvoke(self, messages):
            raise ValueError("upstream 500")

    qp = QueryProcessor(llm=AsyncOnlyFailingLLM(), max_retries=0)
    res = qp.process("退货政策")
    assert res.standard_query == "退货政策"

    # Splitter multi-line heading FAQ strips leading '答：'
    splitter = MarkdownStructureSplitter(chunk_size=500, overlap_size=50)
    chunks = splitter.split_text("### Q: 如何退货？\n答：第一行规则。\n第二行补充说明。")
    assert len(chunks) == 1
    assert not chunks[0].answer.startswith("答：")
    assert "第一行规则" in chunks[0].answer and "第二行补充说明" in chunks[0].answer

    with pytest.raises(ValueError):
        split_text_with_sentence_boundary("测试文本", chunk_size=0, overlap_size=0)


# ==============================================================================
# RAG-11: EvalPipelineService unwraps AdvancedRetrievalResult, excludes D_absent, boundary delta
# ==============================================================================

@pytest.mark.asyncio
async def test_rag11_eval_pipeline_real_retriever_and_deltas(async_sqlite_db, tmp_path):
    from app.services.eval.eval_pipeline import EvalPipelineService
    from app.services.rag.advanced_retriever import AdvancedRetrievalResult

    dataset = tmp_path / "eval.jsonl"
    rows = [
        {"id": "A1", "bucket": "A_policy", "query": "退货政策", "expect_section": ["退货"], "should_refuse": False},
        {"id": "D1", "bucket": "D_absent", "query": "卖火箭吗", "expect_section": [], "should_refuse": True},
    ]
    dataset.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    class StubAdvancedRetriever:
        async def retrieve_with_strategy(self, query, strategy="hybrid_rerank"):
            if "火箭" in query:
                return AdvancedRetrievalResult(docs=[], ranked_docs=[], strategy=strategy)
            doc = {"id": 1, "section_path": "售后 > 退货", "questions": "退货政策", "answer": "7天无理由"}
            return AdvancedRetrievalResult(docs=[doc], ranked_docs=[doc], strategy=strategy)

    service = EvalPipelineService()
    run1 = await service.run_eval_round(
        db=async_sqlite_db, dataset_path=dataset, retriever=StubAdvancedRetriever()
    )
    # D_absent must not depress recall@3/mrr from 1.0 to 0.5
    assert run1.metrics["recall_at_3"] == 1.0
    assert run1.metrics["mrr"] == 1.0

    # Create 3 runs with distinct metrics and verify limit=2 still computes delta for the 2nd run
    async_sqlite_db.add(EvalRun(triggered_by="手动", dataset_size=2, metrics={"mrr": 0.5}))
    await async_sqlite_db.flush()
    async_sqlite_db.add(EvalRun(triggered_by="手动", dataset_size=2, metrics={"mrr": 0.7}))
    await async_sqlite_db.flush()
    async_sqlite_db.add(EvalRun(triggered_by="手动", dataset_size=2, metrics={"mrr": 0.9}))
    await async_sqlite_db.commit()

    recent = await service.get_recent_runs_with_deltas(async_sqlite_db, limit=2)
    assert len(recent) == 2
    assert recent[0]["deltas"]["mrr"] == pytest.approx(0.2)
    assert recent[1]["deltas"]["mrr"] == pytest.approx(0.2)


# ==============================================================================
# RAG-12: calibrate_confidence_gate probes _call_hf_api & closes retriever
# ==============================================================================

@pytest.mark.asyncio
async def test_rag12_calibrate_probes_call_hf_api_and_closes_retriever(tmp_path):
    from scripts.calibrate_confidence_gate import _build_default_retriever, run_calibration_async

    mock_qp = MagicMock(is_mock=True, _llm=None)
    mock_emb = MagicMock(is_mock=True, client=None)
    mock_reranker = MagicMock(is_mock=False, client=MagicMock())
    mock_reranker._call_hf_api = MagicMock(return_value=[0.95])
    del mock_reranker._call_hf_rerank_batch

    with patch("app.services.rag.query_processor.QueryProcessor", return_value=mock_qp), \
         patch("app.services.rag.embedding.BGEEmbeddingClient", return_value=mock_emb), \
         patch("app.services.rag.reranker.BGERerankerClient", return_value=mock_reranker), \
         patch("app.services.rag.advanced_retriever.AdvancedKnowledgeRetriever") as mock_ret_cls:
        retriever, degraded = await _build_default_retriever(min_score=0.1, offline=False)
        mock_reranker._call_hf_api.assert_called_once()
        assert "reranker_client" not in degraded
        assert mock_reranker.is_mock is False

    # Ensure owned retriever is closed in run_calibration_async
    ds = tmp_path / "mini.jsonl"
    ds.write_text(
        json.dumps({"id": "A1", "query": "退货", "should_refuse": False}) + "\n"
        + json.dumps({"id": "D1", "query": "火星", "should_refuse": True}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.json"
    fake_ret = MagicMock()
    fake_ret.retrieve_with_strategy = AsyncMock(return_value=[{"rerank_score": 0.5}])
    fake_ret.close = MagicMock()
    with patch("scripts.calibrate_confidence_gate._build_default_retriever", new=AsyncMock(return_value=(fake_ret, []))):
        await run_calibration_async(dataset_path=str(ds), output_path=str(out), retriever=None)
    fake_ret.close.assert_called_once()


# ==============================================================================
# RAG-13: RAGEvaluator no fake mock fallback, unified answer_coverage, markdown parser
# ==============================================================================

@pytest.mark.asyncio
async def test_rag13_evaluator_integrity_and_md_parser():
    from app.services.rag.evaluator import (
        BucketMetrics,
        EvaluationReport,
        RAGEvaluator,
        StrategyMetrics,
        parse_evaluation_report_md,
        serialize_report_to_dict,
    )

    # 1. Live retriever exception must NOT fabricate perfect mock retrieval
    bad_retriever = MagicMock()
    bad_retriever.retrieve_with_strategy = AsyncMock(side_effect=RuntimeError("Milvus down"))
    ev = RAGEvaluator(retriever=bad_retriever, is_mock=False)
    sample = {"id": "A1", "bucket": "A_policy", "query": "退货", "expect_section": ["退货"], "should_refuse": False}
    res = await ev.evaluate_sample(sample, strategy="hybrid_rerank")
    assert res["recall@3"] == 0.0
    assert res["mrr"] == 0.0

    # 2. serialize_report_to_dict unifies answer_coverage between generation & full_table
    report = EvaluationReport(
        strategies={
            "vector_only": StrategyMetrics(
                strategy="vector_only",
                bucket_metrics={"A_policy": BucketMetrics("A_policy", 1, 0.5, 0.5, 0.5, 0.5, 0.8)},
                overall=BucketMetrics("overall", 1, 0.5, 0.5, 0.5, 0.5, 0.8),
            ),
            "hybrid_rerank": StrategyMetrics(
                strategy="hybrid_rerank",
                bucket_metrics={
                    "A_policy": BucketMetrics("A_policy", 1, 1.0, 1.0, 1.0, 0.7, 0.9),
                    "D_absent": BucketMetrics("D_absent", 1, refusal_rate=0.95),
                },
                overall=BucketMetrics("overall", 2, 1.0, 1.0, 1.0, 0.7, 0.9, refusal_rate=0.95),
            ),
        },
        total_samples=2,
    )
    serialized = serialize_report_to_dict(report)
    gen_cov = serialized["generation"]["answer_coverage"]
    for row in serialized["full_table"]:
        assert row["answer_coverage"] == gen_cov[row["strategy"]]

    # 3. parse_evaluation_report_md skips | :---: | dividers & picks hybrid_rerank refusal_rate
    md = """# AstroBot RAG Chapter 4 四策略对比评测报告

## 2. 平均倒数排名 (MRR) 对比
| 策略 | A_policy | B_model | C_colloquial | E_multi | Overall |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `hybrid_rerank` | 0.900 | 0.850 | 0.800 | 0.750 | **0.825** |
| `vector_only` | 0.700 | 0.600 | 0.650 | 0.500 | **0.612** |

### 忠实度 (Faithfulness) 与 D_absent 拒答率
| 策略 | 忠实度 (Faithfulness) | D_absent 拒答率 | 编造个案数 |
| --- | --- | --- | --- |
| `hybrid_rerank` | 0.950 | 0.980 | 1 |
| `vector_only` | 0.800 | 0.500 | 5 |
"""
    parsed = parse_evaluation_report_md(md)
    assert "---" not in parsed["retrieval"]["mrr"]
    assert ":---:" not in parsed["retrieval"]["mrr"]
    assert parsed["generation"]["out_of_scope_refusal_rate"] == pytest.approx(0.98)


# ==============================================================================
# RAG-14: FlywheelPipeline strict duplicate parsing & async error isolation
# ==============================================================================

@pytest.mark.asyncio
async def test_rag14_flywheel_strict_duplicate_and_isolation(async_sqlite_db):
    from app.services.flywheel.pipeline import FlywheelPipeline

    pipeline = FlywheelPipeline()
    cands = [ReviewQueue(id=1, normalized_question="退货政策"), ReviewQueue(id=2, normalized_question="运费标准")]

    # Verbose negative responses must NOT falsely match candidate 1 or candidates[0]
    for bad_reply in ["NONE, unlike ID 1", "Not a match for 1", "NO match; YES this is a new topic"]:
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=bad_reply)
        assert pipeline.find_duplicate_question("发票怎么开", cands, llm=llm) is None

    # Exact integer ID still matches
    llm_ok = MagicMock()
    llm_ok.invoke.return_value = MagicMock(content="2")
    assert pipeline.find_duplicate_question("运费多少", cands, llm=llm_ok) == 2

    # Per-item failure isolation + ainvoke support
    async_sqlite_db.add_all([
        LowConfidenceQuestion(raw_question="bad_q", source="self_check"),
        LowConfidenceQuestion(raw_question="good_q", source="self_check"),
    ])
    await async_sqlite_db.commit()

    class AsyncFlakyLLM:
        async def ainvoke(self, prompt):
            if "bad_q" in str(prompt):
                raise RuntimeError("transient error")
            if "标准FAQ" in str(prompt):
                return MagicMock(content="好问题\n\n好答案")
            return MagicMock(content="NONE")

    res = await pipeline.process_pending_questions(async_sqlite_db, batch_size=10, llm=AsyncFlakyLLM())
    assert res["new_created"] == 1
    assert res["processed_count"] == 1


# ==============================================================================
# RAG-15: ClassifierInferenceEngine explicit truncation + threshold schema parity
# ==============================================================================

def test_rag15_classifier_truncation_and_threshold_parsing(tmp_path):
    import os
    from tokenizers import Tokenizer
    from app.services.classifier_service.server import ClassifierInferenceEngine
    from scripts.evaluate_classifier import _resolve_threshold
    from scripts.scan_threshold_replay import replay_threshold_scan

    tok_src = Path(os.getenv("TOKENIZER_PATH", "data/ch10/onnx/tokenizer.json"))
    model_src = Path(os.getenv("ONNX_MODEL_PATH", "data/ch10/onnx/model.onnx"))
    thresh_src = Path(os.getenv("THRESHOLD_PATH", "data/ch10/onnx/threshold.json"))
    if tok_src.exists() and model_src.exists():
        # Create a tokenizer.json with truncation explicitly disabled
        tok_data = json.loads(tok_src.read_text(encoding="utf-8"))
        tok_data["truncation"] = None
        no_trunc_tok = tmp_path / "tokenizer_no_trunc.json"
        no_trunc_tok.write_text(json.dumps(tok_data, ensure_ascii=False), encoding="utf-8")

        eng = ClassifierInferenceEngine(
            model_path=model_src,
            tokenizer_path=no_trunc_tok,
            threshold_path=thresh_src,
        )
        eng.load()
        assert eng.tokenizer.truncation is not None
        # 600+ tokens must not crash ONNX Runtime (max_position_embeddings = 512)
        out = eng.classify(["退货" * 350])
        assert len(out) == 1
        assert len(out[0]["scores"]) == 17

    # Per-class threshold dict supported by evaluate_classifier helper
    per_class = {"退换货": 0.6, "物流": 0.4}
    resolved = _resolve_threshold(per_class, ["退换货", "物流"])
    assert isinstance(resolved, np.ndarray)
    assert resolved.tolist() == pytest.approx([0.6, 0.4])


# ==============================================================================
# RAG-16: classify_pool rejects length mismatch & handles IntegrityError
# ==============================================================================

@pytest.mark.asyncio
async def test_rag16_classify_pool_length_validation_and_integrity(async_sqlite_db):
    from scripts.classify_pool import run_classify_pool

    for i in range(3):
        async_sqlite_db.add(LowConfidenceQuestion(raw_question=f"Q{i}", source="self_check"))
    await async_sqlite_db.commit()

    # Length 1 for 3 inputs must raise RuntimeError (no silent 1-to-N broadcast)
    with patch(
        "scripts.classify_pool.call_classifier_service",
        new=AsyncMock(return_value=[{"labels": ["退换货"], "scores": {"退换货": 0.9}}]),
    ):
        with pytest.raises(RuntimeError, match="数量"):
            await run_classify_pool(db=async_sqlite_db, min_batch=1, force=True)

    rows = list((await async_sqlite_db.execute(select(TopicClassification))).scalars().all())
    assert len(rows) == 0

    with pytest.raises(ValueError):
        await run_classify_pool(db=async_sqlite_db, batch_size=0, force=True)


# ==============================================================================
# RAG-17 & VER-10: classifier_service Windows 0.0.0.0 probe, cwd, scheme & cmdline check
# ==============================================================================

def test_rag17_classifier_service_governance():
    from scripts.classifier_service import kill_process_tree, query_healthz

    captured_urls = []

    def fake_urlopen(req, timeout=2.0):
        captured_urls.append(req.full_url)
        cm = MagicMock()
        cm.__enter__.return_value.status = 200
        cm.__enter__.return_value.read.return_value = b'{"status": "ok"}'
        return cm

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        res = query_healthz("0.0.0.0", 8110)
    assert res == {"status": "ok"}
    assert captured_urls == ["http://127.0.0.1:8110/healthz"]

    # Expected cmdline guard refuses to kill mismatched process
    fake_proc = MagicMock()
    fake_proc.is_running.return_value = True
    fake_proc.status.return_value = "running"
    fake_proc.cmdline.return_value = ["notepad.exe"]
    with patch("psutil.pid_exists", return_value=True), patch("psutil.Process", return_value=fake_proc):
        killed = kill_process_tree(12345, expected_cmdline="classifier_service")
    assert killed is False
    fake_proc.terminate.assert_not_called()


# ==============================================================================
# RAG-18 & RAG-19: Classifier eval unknown labels, no-assert gates, data_prep & pos_weights
# ==============================================================================

def test_rag18_rag19_eval_data_prep_and_pos_weights(tmp_path):
    from scripts.data_prep_ch10 import clean_and_mask_text, stratified_split
    from scripts.evaluate_classifier import verify_error_accounting_balance
    from scripts.scan_threshold_replay import replay_threshold_scan
    from scripts.train_classifier import compute_pos_weights
    from app.services.classifier.taxonomy import check_dataset_leakage

    # verify_error_accounting_balance raises ValueError (not AssertionError)
    with pytest.raises(ValueError):
        verify_error_accounting_balance(
            {"under_tag_count": 1, "over_tag_count": 0, "displaced_count": 0}, total_fp_fn=5
        )

    # Empty candidates raises ValueError (not IndexError)
    tf = tmp_path / "t.json"
    tf.write_text('{"threshold": 0.45}', encoding="utf-8")
    vs = tmp_path / "v.json"
    vs.write_text(json.dumps({"samples": [{"text": "a", "true_labels": ["退换货"], "scores": {"退换货": 0.9}}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        replay_threshold_scan(val_scores_file=vs, threshold_file=tf, candidates=[])

    # clean_and_mask_text must NOT corrupt common customer phrases
    assert clean_and_mask_text("我是会员想看用户评价怎么样") == "我是会员想看用户评价怎么样"
    assert "[姓名]" in clean_and_mask_text("我叫张伟，手机13812345678")
    assert clean_and_mask_text("用户：用户") == "用户：[姓名]"

    # stratified_split deduplicates texts & keeps single-sample classes in train
    raw = [
        {"text": "重复句子", "labels": ["退换货"]},
        {"text": "重复句子", "labels": ["退换货"]},
        {"text": "唯一样本", "labels": ["评价"]},
    ] + [{"text": f"样本{i}", "labels": ["物流"]} for i in range(10)]
    tr, va, te = stratified_split(raw)
    leak = check_dataset_leakage(tr, va, te)
    assert leak["passed"] is True
    assert any(x["text"] == "唯一样本" for x in tr)

    # compute_pos_weights handles duplicate labels in a single record without TypeError
    w = compute_pos_weights([{"labels": ["退换货", "退换货", "退换货"]}], taxonomy=["退换货", "物流"])
    assert len(w) == 2
    assert float(w[0]) >= 1.0


# ==============================================================================
# RAG-20: CostAnalyticsService cost_percentage math, sorting, None tokens, Langfuse v4 api
# ==============================================================================

def test_rag20_cost_analytics_math_and_langfuse_v4(tmp_path):
    from app.services.observability.cost_analytics import CostAnalyticsService

    svc = CostAnalyticsService(prompt_cost_per_m=2.5, completion_cost_per_m=10.0)
    records = [
        {"intent": "prompt_heavy", "prompt_tokens": 1_000_000, "completion_tokens": 0},      # $2.50
        {"intent": "completion_heavy", "prompt_tokens": 0, "completion_tokens": 1_000_000},  # $10.00
        {"intent": "none_safe", "prompt_tokens": None, "completion_tokens": None},
    ]
    report = svc.aggregate_records(records)
    assert report["items"][0]["intent"] == "completion_heavy"
    assert report["items"][0]["cost_percentage"] == pytest.approx(80.0)
    assert report["items"][1]["intent"] == "prompt_heavy"
    assert report["items"][1]["cost_percentage"] == pytest.approx(20.0)
    assert report["generated_at"].endswith("Z") or "+" in report["generated_at"]

    # Live mode uses lf.api + total_pages + shutdown()
    fake_lf = MagicMock()
    del fake_lf.client  # Langfuse v4 has no .client
    trace_obj = MagicMock(
        id="t1", metadata={"intent": "refund"}, total_cost=0.05, observations=[]
    )
    meta_obj = MagicMock(total_pages=1)
    fake_lf.api.trace.list.return_value = MagicMock(data=[trace_obj], meta=meta_obj)
    fake_lf.api.observations.get_many.return_value = MagicMock(
        data=[MagicMock(usage_details={"input": 1000, "output": 500}, usage=None)]
    )

    out = tmp_path / "cost.json"
    with patch("langfuse.Langfuse", return_value=fake_lf):
        rep = svc.fetch_and_generate_report(output_file=out, mock=False)
    fake_lf.api.trace.list.assert_called_once_with(page=1)
    fake_lf.shutdown.assert_called_once()
    assert rep.get("is_mock", False) is False
    assert rep["total_requests"] == 1


# ==============================================================================
# RAG-21: LangfuseManager v4 update_current_trace & handler isolation
# ==============================================================================

def test_rag21_langfuse_manager_v4_trace_update_and_no_shared_handler_mutation():
    from app.services.observability.langfuse_service import LangfuseManager

    fake_client = MagicMock()
    LangfuseManager._client = fake_client
    LangfuseManager._handler = None

    with patch.object(settings, "langfuse_enabled", True), \
         patch.object(settings, "langfuse_public_key", "pk-test"), \
         patch.object(settings, "langfuse_secret_key", "sk-test"):
        LangfuseManager.update_current_trace(metadata={"intent": "售后"}, tags=["售后"])

    # Must invoke client span/trace update rather than silently discarding propagate_attributes CM
    assert fake_client.update_current_span.called or fake_client.update_current_trace.called
    LangfuseManager._client = None


# ==============================================================================
# RAG-22: Reindex text format alignment & status sync
# ==============================================================================

@pytest.mark.asyncio
async def test_rag22_reindex_format_and_status_sync(async_sqlite_db):
    from app.services.rag.dual_writer import compose_embedding_text
    from scripts.reindex_ch04_knowledge import compose_chunk_text, reindex_knowledge_chunks

    assert compose_chunk_text("退货", "怎么退", "7天无理由") == compose_embedding_text("退货", "怎么退", "7天无理由")

    chunk = KnowledgeChunk(
        category="退货", questions="怎么退", answer="7天无理由", vectorize_status="pending"
    )
    async_sqlite_db.add(chunk)
    await async_sqlite_db.commit()

    mock_store = MagicMock()
    mock_store.upsert.return_value = 1
    mock_emb = MagicMock()
    mock_emb.aembed_documents = AsyncMock(return_value=[[0.1] * 1024])

    stats = await reindex_knowledge_chunks(
        session=async_sqlite_db, store=mock_store, embedding_client=mock_emb
    )
    assert stats["indexed_chunks"] == 1
    await async_sqlite_db.refresh(chunk)
    assert chunk.vectorize_status == "done"
    assert chunk.vector_id == str(chunk.id)


@pytest.mark.asyncio
async def test_ver04_run_ch04_evaluation_does_not_mutate_tracked_report_under_pytest():
    import argparse
    from scripts.run_ch04_evaluation import run_evaluation

    tracked_report = Path("reports/ch04_evaluation_report.md")
    before_bytes = tracked_report.read_bytes() if tracked_report.exists() else None

    args = argparse.Namespace(
        strategies=["vector_only"],
        sample=1,
        mock=True,
        output=None,
        dataset="tests/data/eval_ch04.jsonl",
    )
    rc = await run_evaluation(args)
    assert rc == 0

    after_bytes = tracked_report.read_bytes() if tracked_report.exists() else None
    assert after_bytes == before_bytes
