"""Chapter 04 端到端综合业务验收测试 (四大验收标准系统化验证)

验收标准 1: 四策略（vector_only, bm25_only, hybrid, hybrid_rerank）对比评测引擎能成功跑通并产出包含数字的完整指标对象，Markdown 报告无异常/无 NaN
验收标准 2: 带有具体型号（PRO-X99）的专有名词提问，BM25 单路精准命中对应 chunk，且 hybrid_rerank 下该 chunk 置于最前列 Top-1
验收标准 3: 受控生成链路产出的文本包含形如 [1] 的引用角标，提取所有角标序号完全匹配 citations 快照中的 n，且元数据非空完整
验收标准 4: 超纲问题证据为空或自检不足时明确输出礼貌拒答语，截断硬编，且 low_confidence_questions 表成功持久化问题记录
"""

import math
import re
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.session import Base
from app.models import Conversation, Message
from app.models.low_confidence import LowConfidenceQuestion
from app.services.chat_service import ChatService
from app.services.rag.advanced_retriever import (
    AdvancedKnowledgeRetriever,
    AdvancedRetrievalResult,
)
from app.services.rag.evaluator import (
    EvaluationReport,
    RAGEvaluator,
    StrategyMetrics,
    render_markdown_report,
)
from app.services.rag.generator import (
    DEFAULT_REFUSAL_RESPONSE,
    RAGControlledGenerator,
    SelfCheckResult,
)
from app.services.rag.query_processor import QueryUnderstandingResult
from app.services.rag.reorder import build_citation_items, lost_in_the_middle_reorder


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
    """提供纯内存 SQLite 异步会话适配器，并在测试执行完毕后清理释放。"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield AsyncSessionAdapter(session)
    Base.metadata.drop_all(engine)
    engine.dispose()


# ==============================================================================
# 验收标准 1: 四策略对比报告跑出有效数字与完整 Markdown 矩阵渲染
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_criterion_1_comparative_eval_produces_valid_metrics():
    """验收标准 1:
    1. 验证四策略（vector_only, bm25_only, hybrid, hybrid_rerank）对比评测引擎成功跑通；
    2. 产出包含数字的完整指标对象（Recall@3, 5, 10, MRR, Refusal Rate 等均为非空有效浮点数）；
    3. Markdown 报告渲染器成功渲染出包含 4 策略表格且无 NaN/undefined/异常的完整报告。
    """
    evaluator = RAGEvaluator(is_mock=True)
    target_strategies = ["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]

    # 运行包含 5 个分桶（A_policy, B_model, C_colloquial, E_multi, D_absent）的对比评测
    report = await evaluator.run_comparative_eval(
        strategies=target_strategies,
        samples_per_bucket=1,
    )

    # 1. 验证评测结果报告顶层结构
    assert isinstance(report, EvaluationReport), "必须产出标准的 EvaluationReport 对象"
    assert len(report.strategies) == 4, "必须完整覆盖 4 种检索策略"
    assert report.total_samples > 0, "总样本数必须大于 0"

    # 2. 验证各个策略下的指标对象包含真实有效的非空浮点数
    expected_buckets = ["A_policy", "B_model", "C_colloquial", "E_multi", "D_absent"]

    for strat in target_strategies:
        assert strat in report.strategies, f"报告中必须包含策略 {strat}"
        strat_metrics: StrategyMetrics = report.strategies[strat]
        assert strat_metrics.strategy == strat

        # 检查各分桶指标
        for bucket in expected_buckets:
            assert bucket in strat_metrics.bucket_metrics, f"策略 {strat} 必须包含分桶 {bucket} 的指标"
            bm = strat_metrics.bucket_metrics[bucket]
            if bucket == "D_absent":
                # D_absent 专测拒答率
                assert isinstance(bm.refusal_rate, (int, float)), f"{strat} - {bucket} 的 refusal_rate 必须为数值"
                assert not math.isnan(bm.refusal_rate), f"{strat} - {bucket} 的 refusal_rate 不能为 NaN"
                assert 0.0 <= bm.refusal_rate <= 1.0
            else:
                # 正常分桶检验检索与忠实度指标
                for field_name in ["recall_at_3", "recall_at_5", "recall_at_10", "mrr", "faithfulness"]:
                    val = getattr(bm, field_name)
                    assert isinstance(val, (int, float)), f"{strat} - {bucket} 的 {field_name} 必须为数值"
                    assert not math.isnan(val), f"{strat} - {bucket} 的 {field_name} 不能为 NaN"
                    assert 0.0 <= val <= 1.0, f"{strat} - {bucket} 的 {field_name} 值需在 [0, 1] 区间"

        # 检查 Overall 全局聚合指标（涵盖检索 Recall, MRR, 忠实度与拒答率）
        assert strat_metrics.overall is not None, f"策略 {strat} 必须包含 overall 指标"
        for field_name in ["recall_at_3", "recall_at_5", "recall_at_10", "mrr", "refusal_rate", "faithfulness"]:
            val = getattr(strat_metrics.overall, field_name)
            assert isinstance(val, (int, float)), f"{strat} overall 的 {field_name} 必须为数值"
            assert not math.isnan(val), f"{strat} overall 的 {field_name} 不能为 NaN"
            assert 0.0 <= val <= 1.0, f"{strat} overall 的 {field_name} 值需在 [0, 1] 区间"

    # 3. 验证 Markdown 对比报告渲染
    md_report = render_markdown_report(report)
    assert isinstance(md_report, str) and len(md_report) > 0, "渲染出的 Markdown 报告不能为空"
    assert "# AstroBot RAG Chapter 4 四策略对比评测报告" in md_report

    # 验证包含 4 个策略名以及关键表头
    for strat in target_strategies:
        assert f"`{strat}`" in md_report, f"Markdown 报告中必须列出策略 `{strat}`"

    assert "Recall@3" in md_report
    assert "Recall@5" in md_report
    assert "Recall@10" in md_report
    assert "MRR" in md_report
    assert "Faithfulness" in md_report
    assert "拒答率" in md_report or "Refusal" in md_report

    # 严禁出现 NaN 或 undefined 异常
    lower_report = md_report.lower()
    assert "nan" not in lower_report, "Markdown 报告中不得出现 NaN 异常"
    assert "undefined" not in lower_report, "Markdown 报告中不得出现 undefined"
    assert "| none |" not in lower_report, "Markdown 报告中表格不得包含未处理的 None 单元格"


# ==============================================================================
# 验收标准 2: 带具体型号（PRO-X99）问题 BM25 命中并置顶于 Top-1
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_criterion_2_bm25_hits_specific_product_model():
    """验收标准 2:
    1. 构造带有明确型号（PRO-X99）的专有名词提问；
    2. 验证 BM25 单路（strategy='bm25_only'）能够精准召回该型号对应的 chunk；
    3. 验证在进阶策略 hybrid_rerank 下，该型号知识块被重排置于最前列（Top-1）。
    """
    model_query = "请问星光PRO-X99智能手表的无线充电规格是多少瓦？"

    # 构造测试知识块数据集：包含专有型号 PRO-X99 以及泛化干扰项
    pro_x99_chunk = {
        "id": 99,
        "chunk_id": 99,
        "category": "智能手表",
        "section_path": "商品规格 > PRO-X99智能手表",
        "question": "PRO-X99充电规格与续航说明是什么？",
        "answer": "星光PRO-X99智能手表支持15W无线磁吸快充与Type-C应急充电，内置450mAh电池，常规使用续航可达7天。",
        "chunk_text": "【类目】智能手表\n【标准问法】PRO-X99充电规格与续航说明是什么？\n【解答】星光PRO-X99智能手表支持15W无线磁吸快充与Type-C应急充电，内置450mAh电池，常规使用续航可达7天。",
    }
    generic_watch_chunk = {
        "id": 101,
        "chunk_id": 101,
        "category": "智能手表",
        "section_path": "商品规格 > 通用智能穿戴",
        "question": "智能穿戴通用充电规格是什么？",
        "answer": "普通智能手环及基础手表采用5V1A磁吸触点充电，请勿使用超过5V的快充头。",
        "chunk_text": "【类目】智能手表\n【标准问法】智能穿戴通用充电规格是什么？\n【解答】普通智能手环及基础手表采用5V1A磁吸触点充电，请勿使用超过5V的快充头。",
    }
    policy_chunk = {
        "id": 102,
        "chunk_id": 102,
        "category": "售后政策",
        "section_path": "服务规范 > 发货与退换货",
        "question": "智能手表支持7天无理由退换吗？",
        "answer": "手表类商品在未拆封未激活状态下支持7天无理由退货。",
        "chunk_text": "【类目】售后政策\n【标准问法】智能手表支持7天无理由退换吗？\n【解答】手表类商品在未拆封未激活状态下支持7天无理由退货。",
    }

    # 1. 模拟 Query 理解处理器
    mock_query_processor = MagicMock()
    mock_qu = QueryUnderstandingResult(
        original_query=model_query,
        standard_query="星光PRO-X99智能手表无线充电规格说明",
        expanded_keywords=["PRO-X99", "无线快充", "15W", "充电"],
        bm25_query="星光PRO-X99智能手表无线充电规格说明 PRO-X99 无线快充 15W 充电",
    )
    mock_query_processor.aprocess = AsyncMock(return_value=mock_qu)

    # 2. 模拟向量检索客户端与底层 Store
    mock_embedding = MagicMock()
    mock_embedding.aembed_query = AsyncMock(return_value=[0.1] * 1024)

    mock_store = MagicMock()
    # BM25 准确根据 "PRO-X99" 词元高分命中目标 chunk
    mock_store.search_bm25.return_value = [pro_x99_chunk, generic_watch_chunk]
    # Dense 向量由于语义泛化，可能将"通用充电"与"PRO-X99充电"并列召回
    mock_store.search_dense.return_value = [generic_watch_chunk, pro_x99_chunk]
    # 粗排 hybrid 召回
    mock_store.hybrid_search.return_value = [generic_watch_chunk, pro_x99_chunk, policy_chunk]

    # 3. 模拟 BGE-Reranker-v2-m3 语义精排打分
    mock_reranker = MagicMock()
    async def fake_rerank(query, candidates, top_k=10):
        # 命中 PRO-X99 专有型号的知识块赋予绝对高分 (0.98)，泛化干扰项分值较低 (0.35)
        scored = []
        for c in candidates:
            item = dict(c)
            if "PRO-X99" in str(item.get("section_path", "")) or "PRO-X99" in str(item.get("question", "")):
                item["rerank_score"] = 0.98
            else:
                item["rerank_score"] = 0.35
            scored.append(item)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]

    mock_reranker.rerank = AsyncMock(side_effect=fake_rerank)

    retriever = AdvancedKnowledgeRetriever(
        store=mock_store,
        embedding_client=mock_embedding,
        query_processor=mock_query_processor,
        reranker_client=mock_reranker,
    )

    # 验证步骤 A：BM25 单路（strategy="bm25_only"）精准召回 PRO-X99
    bm25_res = await retriever.retrieve_with_strategy(query=model_query, strategy="bm25_only")
    assert isinstance(bm25_res, AdvancedRetrievalResult)
    assert len(bm25_res.docs) >= 1
    bm25_chunk_ids = [d.get("chunk_id") for d in bm25_res.docs]
    assert 99 in bm25_chunk_ids, "BM25 单路检索必须成功召回 PRO-X99 对应的 chunk (id=99)"
    hit_chunk = next(d for d in bm25_res.docs if d.get("chunk_id") == 99)
    assert "PRO-X99" in hit_chunk["section_path"]
    assert "15W" in hit_chunk["answer"]

    # 验证步骤 B：进阶策略 hybrid_rerank 将该型号置于最前列 Top-1
    hybrid_res = await retriever.retrieve_with_strategy(query=model_query, strategy="hybrid_rerank")
    assert isinstance(hybrid_res, AdvancedRetrievalResult)
    assert len(hybrid_res.docs) >= 1
    top_1_doc = hybrid_res.docs[0]
    assert top_1_doc.get("chunk_id") == 99, f"hybrid_rerank 必须将 PRO-X99 精排置顶为 Top-1，实际 Top-1: {top_1_doc}"
    assert "PRO-X99" in top_1_doc.get("section_path", "")
    assert top_1_doc.get("rerank_score") == 0.98

    # 验证生成的 citations 首条对应 PRO-X99
    assert len(hybrid_res.citations) >= 1
    assert hybrid_res.citations[0]["n"] == 1
    assert hybrid_res.citations[0]["chunk_id"] == 99
    assert "PRO-X99" in hybrid_res.citations[0]["question"]


# ==============================================================================
# 验收标准 3: 答案引用编号 [n] 定位回原文并与 citations 完全匹配
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_criterion_3_citation_numbers_map_to_source_chunks(sqlite_session):
    """验收标准 3:
    1. 验证受控生成链路产出的文本包含形如 [1] 的引用角标；
    2. 提取所有引用角标序号，验证其与返回的 citations 快照中 n 完全对应；
    3. 验证 citations 中每一项的 section_path、chunk_id、question、answer 均非空完整。
    """
    db = sqlite_session

    citations_fixture = [
        {
            "n": 1,
            "chunk_id": 101,
            "section_path": "商品规格与售后 > PRO-X99智能手表退货规范",
            "question": "PRO-X99 支持退货吗？",
            "answer": "星光 PRO-X99 智能手表自签收之日起支持7天无理由退货，需保证商品未受损且配件完整。",
        },
        {
            "n": 2,
            "chunk_id": 102,
            "section_path": "售后政策 > 退款时效与资费",
            "question": "退货运费谁出？退款多久到账？",
            "answer": "质量问题由商家承担运费；平台审核退货入库后1-3个工作日原路退回，具体以发卡行为准。",
        },
    ]

    # 1. 模拟 ChatService 依赖：模型首轮决策调用 query_faq
    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="",
            tool_calls=[{
                "name": "query_faq",
                "args": {"keyword": "PRO-X99 退货退款规则"},
                "id": "call_faq_accept_3",
            }],
        )
    )
    mock_llm.bind_tools.return_value = mock_bound

    # 2. 模拟进阶检索器产出 citations
    mock_retriever = AsyncMock()
    mock_retriever.retrieve_with_strategy.return_value = MagicMock(
        citations=citations_fixture,
        docs=citations_fixture,
    )

    # 3. 模拟工具执行器，避免真实调用外部网络或真实数据库
    mock_executor = MagicMock()
    mock_executor.execute = AsyncMock(return_value={
        "name": "query_faq",
        "output": "【FAQ 检索成功】",
        "success": True,
        "tool_call_id": "call_faq_accept_3",
    })

    # 4. 模拟受控生成器产出带 [1] 与 [2] 角标的受控流式文本
    mock_generator = MagicMock()
    mock_generator.check_sufficiency = AsyncMock(
        return_value=SelfCheckResult(useful=True, reason="退货与退款证据充分", source="self_check")
    )

    async def fake_controlled_stream(query, citations, history=None):
        yield "星光 PRO-X99 智能手表支持自签收之日起7天无理由退货[1]。"
        yield "因质量问题产生的退货由商家承担运费[2]，"
        yield "平台审核入库后1-3个工作日原路退回款项[2]。"

    mock_generator.astream_generate = fake_controlled_stream

    chat_service = ChatService(
        model=mock_llm,
        executor=mock_executor,
        retriever=mock_retriever,
        rag_generator=mock_generator,
    )

    # 执行端到端会话流
    events = [
        e async for e in chat_service.stream_chat(
            db=db,
            conversation_id=None,
            message="请问 PRO-X99 支持退货吗？退款什么时候到账？",
        )
    ]

    # A. 验证下发了 citations 事件帧
    citations_events = [e for e in events if e.get("event_type") == "citations"]
    assert len(citations_events) == 1, "必须下发且仅下发一次 citations 事件帧"
    citations_data = citations_events[0]["citations"]
    assert len(citations_data) == 2, "快照中必须包含 2 条证据"

    # B. 验证 citations 快照字段完整性
    citations_by_n: Dict[int, Dict[str, Any]] = {}
    for item in citations_data:
        n = item.get("n")
        assert n is not None and isinstance(n, int), "证据项必须具有整型序号 n"
        citations_by_n[n] = item

        # section_path, chunk_id, question, answer 均非空完整
        assert item.get("chunk_id") is not None, f"证据 [{n}] 的 chunk_id 不能为空"
        assert item.get("section_path") and len(str(item["section_path"]).strip()) > 0, f"证据 [{n}] 的 section_path 不能为空"
        assert item.get("question") and len(str(item["question"]).strip()) > 0, f"证据 [{n}] 的 question 不能为空"
        assert item.get("answer") and len(str(item["answer"]).strip()) > 0, f"证据 [{n}] 的 answer 不能为空"

    # C. 验证生成文本包含形如 [1], [2] 的角标，且所有角标能定位回 citations 原文
    text_pieces = [e["content"] for e in events if e.get("event_type") == "text"]
    full_answer_text = "".join(text_pieces)
    assert len(full_answer_text) > 0, "生成的回答文本不能为空"

    # 提取所有 [n] 引用角标序号
    cite_matches = re.findall(r"\[(\d+)\]", full_answer_text)
    assert len(cite_matches) >= 2, f"回答文本必须包含多个引用角标，实际文本: {full_answer_text}"

    unique_cite_nums = set(int(m) for m in cite_matches)
    assert 1 in unique_cite_nums, "必须包含对证据 [1] 的引用"
    assert 2 in unique_cite_nums, "必须包含对证据 [2] 的引用"

    # 每个提取出的角标都必须在 citations 快照中找到对应原文
    for cite_num in unique_cite_nums:
        assert cite_num in citations_by_n, f"回答中引用的角标 [{cite_num}] 在 citations 快照中不存在"
        source_chunk = citations_by_n[cite_num]
        if cite_num == 1:
            assert "7天无理由退货" in source_chunk["answer"]
            assert "PRO-X99" in source_chunk["section_path"]
        elif cite_num == 2:
            assert "1-3个工作日" in source_chunk["answer"]


# ==============================================================================
# 验收标准 4: 超纲问题明确拒答、截断硬编且落库 low_confidence_questions 表
# ==============================================================================

@pytest.mark.asyncio
async def test_acceptance_criterion_4_out_of_scope_query_refusal_and_pool_logging(sqlite_session):
    """验收标准 4:
    1. 针对超纲问题，验证证据为空或自检不足时，系统输出礼貌拒答语，截断硬编，不下发 citations 事件；
    2. 在 SQLite 数据库中查询验证 low_confidence_questions 表成功持久化了该问题记录（raw_question, source, reason 完整无误）。
    """
    db = sqlite_session
    out_of_scope_query = "你们商城有没有宇宙飞船核动力引擎现货？能不能加急顺丰送到火星基地？"

    # 1. 模拟 ChatService 首次决策检索知识库
    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="",
            tool_calls=[{
                "name": "query_faq",
                "args": {"keyword": "宇宙飞船核动力引擎 火星基地"},
                "id": "call_faq_accept_4",
            }],
        )
    )
    mock_llm.bind_tools.return_value = mock_bound

    # 2. 模拟检索器对于知识库外问题返回空证据
    mock_retriever = AsyncMock()
    mock_retriever.retrieve_with_strategy.return_value = MagicMock(
        citations=[],
        docs=[],
    )

    # 3. 模拟工具执行器隔离网络与真实库
    mock_executor = MagicMock()
    mock_executor.execute = AsyncMock(return_value={
        "name": "query_faq",
        "output": "未找到相关知识",
        "success": True,
        "tool_call_id": "call_faq_accept_4",
    })

    # 4. 模拟受控生成器的两阶段自检：判定为 useful=False 并执行落库
    mock_generator = MagicMock()
    mock_generator.refusal_text = DEFAULT_REFUSAL_RESPONSE

    async def fake_check_sufficiency(query, citations, db=None, conversation_id=None):
        reason = "检索结果未命中任何相关知识库条目，且涉及商城经营范围之外的未知领域"
        if db is not None:
            # 真实写入 SQLite 数据库
            record = LowConfidenceQuestion(
                conversation_id=conversation_id,
                raw_question=query,
                source="retrieval_low_conf",
                reason=reason,
            )
            db.add(record)
            await db.commit()
            await db.refresh(record)
        return SelfCheckResult(useful=False, reason=reason, source="retrieval_low_conf")

    mock_generator.check_sufficiency = fake_check_sufficiency

    chat_service = ChatService(
        model=mock_llm,
        executor=mock_executor,
        retriever=mock_retriever,
        rag_generator=mock_generator,
    )

    # 执行超纲会话流
    events = [
        e async for e in chat_service.stream_chat(
            db=db,
            conversation_id=None,
            message=out_of_scope_query,
        )
    ]

    # A. 验证事件流：拒答时严禁下发 citations 事件帧
    event_types = [e.get("event_type") for e in events]
    assert "citations" not in event_types, "超纲拒答时严禁下发 citations 事件帧"
    assert "text" in event_types, "必须下发礼貌拒答文本"

    # B. 验证下发了礼貌拒答语，且严禁模型硬编超纲内容
    refusal_content = "".join(e["content"] for e in events if e.get("event_type") == "text")
    assert "非常抱歉" in refusal_content, "拒答语中必须包含礼貌致歉"
    assert "暂未收录" in refusal_content or "人工" in refusal_content, "拒答语中必须声明知识库未收录并引导转人工"
    assert "核动力" not in refusal_content, "系统严禁硬编外推超纲实体信息"
    assert "火星" not in refusal_content, "系统严禁硬编外推超纲实体信息"

    # C. 在 SQLite 数据库中直接查询验证 low_confidence_questions 表成功持久化记录
    stmt = select(LowConfidenceQuestion).where(
        LowConfidenceQuestion.raw_question == out_of_scope_query
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    assert len(records) == 1, "low_confidence_questions 表中必须且仅有一条入池记录"
    recorded_question = records[0]
    assert recorded_question.raw_question == out_of_scope_query
    assert recorded_question.source == "retrieval_low_conf"
    assert "未命中任何相关知识库条目" in recorded_question.reason
    assert recorded_question.conversation_id is not None
    assert recorded_question.created_at is not None
