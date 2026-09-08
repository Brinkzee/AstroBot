import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import Conversation, Message
from app.models.low_confidence import LowConfidenceQuestion
from app.prompts.rag_qa import (
    SELF_CHECK_SYSTEM_PROMPT,
    RAG_CONTROLLED_QA_SYSTEM_PROMPT,
    DEFAULT_REFUSAL_RESPONSE,
)
from app.services.rag.generator import (
    RAGControlledGenerator,
    SelfCheckResult,
)
from app.services.chat_service import ChatService


class FakeAsyncSession:
    """In-memory AsyncSession mock for testing low-confidence recording and chat flow."""

    def __init__(self):
        self.conversations = {}
        self.messages = []
        self.low_confidence_questions = []
        self._conv_id_counter = 1
        self._msg_id_counter = 1
        self._lcq_id_counter = 1

    def add(self, obj):
        if isinstance(obj, Conversation):
            if getattr(obj, "id", None) is None:
                obj.id = self._conv_id_counter
                self._conv_id_counter += 1
            self.conversations[obj.id] = obj
        elif isinstance(obj, Message):
            if getattr(obj, "id", None) is None:
                obj.id = self._msg_id_counter
                self._msg_id_counter += 1
            self.messages.append(obj)
        elif isinstance(obj, LowConfidenceQuestion):
            if getattr(obj, "id", None) is None:
                obj.id = self._lcq_id_counter
                self._lcq_id_counter += 1
            self.low_confidence_questions.append(obj)

    async def commit(self):
        pass

    async def refresh(self, obj):
        pass

    async def get(self, entity_cls, ident):
        if entity_cls is Conversation:
            return self.conversations.get(ident)
        return None

    async def execute(self, stmt):
        stmt_str = str(stmt)
        if "conversations" in stmt_str:
            target_id = None
            for criterion in getattr(stmt, "_where_criteria", ()):
                if hasattr(criterion, "right") and hasattr(criterion.right, "value"):
                    target_id = criterion.right.value
            conv = self.conversations.get(target_id)

            class MockConvResult:
                def scalar_one_or_none(self):
                    return conv

                def scalars(self):
                    class MockScalars:
                        def all(self):
                            return [conv] if conv else []
                    return MockScalars()

            return MockConvResult()

        elif "messages" in stmt_str:
            target_conv_id = None
            for criterion in getattr(stmt, "_where_criteria", ()):
                if hasattr(criterion, "right") and hasattr(criterion.right, "value"):
                    target_conv_id = criterion.right.value

            matched = [
                m for m in self.messages
                if target_conv_id is None or m.conversation_id == target_conv_id
            ]

            class MockMsgResult:
                def scalars(self):
                    class MockScalars:
                        def all(self):
                            return matched
                    return MockScalars()

            return MockMsgResult()

        elif "low_confidence_questions" in stmt_str:
            matched = list(self.low_confidence_questions)

            class MockLCQResult:
                def scalars(self):
                    class MockScalars:
                        def all(self):
                            return matched
                    return MockScalars()

            return MockLCQResult()

        class DefaultMockResult:
            def scalar_one_or_none(self):
                return None

            def scalars(self):
                class MockScalars:
                    def all(self):
                        return []
                return MockScalars()

        return DefaultMockResult()


# ----------------------------------------------------------------------
# 1. 提示词约束单测
# ----------------------------------------------------------------------

def test_rag_controlled_prompt_contains_negative_rules_and_citations():
    """验证受控生成 Prompt 包含严格负面知识红线与 [n] 引用规则"""
    assert "退款" in RAG_CONTROLLED_QA_SYSTEM_PROMPT
    assert "到账" in RAG_CONTROLLED_QA_SYSTEM_PROMPT or "发卡行" in RAG_CONTROLLED_QA_SYSTEM_PROMPT
    assert "赔付" in RAG_CONTROLLED_QA_SYSTEM_PROMPT or "补偿" in RAG_CONTROLLED_QA_SYSTEM_PROMPT
    assert "[n]" in RAG_CONTROLLED_QA_SYSTEM_PROMPT or "【证据" in RAG_CONTROLLED_QA_SYSTEM_PROMPT
    assert "严禁" in RAG_CONTROLLED_QA_SYSTEM_PROMPT or "禁止" in RAG_CONTROLLED_QA_SYSTEM_PROMPT


def test_self_check_prompt_structure():
    """验证自评提示词要求纯 JSON 输出 useful 与 reason"""
    assert "useful" in SELF_CHECK_SYSTEM_PROMPT
    assert "reason" in SELF_CHECK_SYSTEM_PROMPT
    assert "json" in SELF_CHECK_SYSTEM_PROMPT.lower()
    assert "true" in SELF_CHECK_SYSTEM_PROMPT.lower()
    assert "false" in SELF_CHECK_SYSTEM_PROMPT.lower()


# ----------------------------------------------------------------------
# 2. RAGControlledGenerator 自检 Phase 1 单测
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_self_check_empty_citations_retrieval_low_conf():
    """当知识证据为空时：直接判定 useful=False，来源为 retrieval_low_conf 并落库"""
    db = FakeAsyncSession()
    generator = RAGControlledGenerator()

    result = await generator.check_sufficiency(
        query="火星基地的重力加速度是多少？",
        citations=[],
        db=db,
        conversation_id=10,
    )

    assert isinstance(result, SelfCheckResult)
    assert result.useful is False
    assert result.source == "retrieval_low_conf"
    assert "证据" in result.reason or "空" in result.reason

    # 验证低置信度入池
    assert len(db.low_confidence_questions) == 1
    record = db.low_confidence_questions[0]
    assert record.raw_question == "火星基地的重力加速度是多少？"
    assert record.source == "retrieval_low_conf"
    assert record.conversation_id == 10
    assert record.reason == result.reason


@pytest.mark.asyncio
async def test_self_check_irrelevant_citations_self_check_refusal():
    """当知识证据不相关或超纲时：LLM 自评 useful=False，来源为 self_check 并落库"""
    db = FakeAsyncSession()

    # 模拟 LLM 返回自评不充分
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = MagicMock(
        content=json.dumps({
            "useful": False,
            "reason": "提供的证据为七天无理由退货条款，无法回答关于量子计算机散热的问题",
        })
    )

    generator = RAGControlledGenerator(model=mock_model)

    citations = [
        {
            "n": 1,
            "chunk_id": 1,
            "section_path": "售后政策 > 退货政策",
            "question": "退货政策是什么？",
            "answer": "支持7天无理由退货，商品需包装完好。",
        }
    ]

    result = await generator.check_sufficiency(
        query="量子计算机的超导芯片需要多少开尔文温度散热？",
        citations=citations,
        db=db,
        conversation_id=12,
    )

    assert result.useful is False
    assert result.source == "self_check"
    assert "量子计算机" in result.reason or "无法回答" in result.reason

    # 验证低置信度入池
    assert len(db.low_confidence_questions) == 1
    record = db.low_confidence_questions[0]
    assert record.raw_question == "量子计算机的超导芯片需要多少开尔文温度散热？"
    assert record.source == "self_check"
    assert record.conversation_id == 12


@pytest.mark.asyncio
async def test_self_check_sufficient_citations():
    """当知识证据充分时：LLM 自评 useful=True，不写入低置信度表"""
    db = FakeAsyncSession()

    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = MagicMock(
        content=json.dumps({
            "useful": True,
            "reason": "证据1明确说明了星光PRO-X99支持15W无线快充，信息完全充分",
        })
    )

    generator = RAGControlledGenerator(model=mock_model)

    citations = [
        {
            "n": 1,
            "chunk_id": 2,
            "section_path": "商品规格 > PRO-X99",
            "question": "PRO-X99充电规格是什么？",
            "answer": "星光PRO-X99支持15W无线磁吸快充与Type-C应急充电。",
        }
    ]

    result = await generator.check_sufficiency(
        query="星光PRO-X99手表支持无线充电吗？",
        citations=citations,
        db=db,
        conversation_id=15,
    )

    assert result.useful is True
    assert result.source == "self_check"
    # 不应该写入 low_confidence_questions
    assert len(db.low_confidence_questions) == 0


@pytest.mark.asyncio
async def test_self_check_fallback_mock_when_no_model():
    """在未配置模型或模型异常的 Mock 模式下，自评能平滑进行确定性判断"""
    db = FakeAsyncSession()
    generator = RAGControlledGenerator(model=None)

    # 包含无关超纲词，Mock 应判断为 False
    res_false = await generator.check_sufficiency(
        query="火星飞船怎么加油？",
        citations=[{"n": 1, "question": "退换货", "answer": "7天包退"}],
        db=db,
        conversation_id=20,
    )
    assert res_false.useful is False
    assert res_false.source == "self_check"

    # 包含高度重合词，Mock 应判断为 True
    res_true = await generator.check_sufficiency(
        query="退换货规则是什么？",
        citations=[{"n": 1, "question": "退换货规则", "answer": "7天包退，15天换货"}],
        db=db,
        conversation_id=21,
    )
    assert res_true.useful is True


# ----------------------------------------------------------------------
# 3. RAGControlledGenerator 受控流式生成 Phase 2 单测
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_astream_generate_mock_output_citations():
    """验证确定性 Mock 流式生成带 [n] 角标的回答"""
    generator = RAGControlledGenerator(stream_model=None)

    citations = [
        {
            "n": 1,
            "chunk_id": 5,
            "section_path": "售后政策 > 退货退款",
            "question": "商品退货运费谁出？",
            "answer": "质量问题商家承担运费，非质量问题由买家承担[1]。",
        },
        {
            "n": 2,
            "chunk_id": 6,
            "section_path": "售后政策 > 退款时效",
            "question": "退款什么时候到账？",
            "answer": "退款审核通过后1-3个工作日原路退回，具体到账时间以发卡银行为准[2]。",
        },
    ]

    chunks = []
    async for chunk in generator.astream_generate("退货运费谁出？退款什么时候到账？", citations):
        chunks.append(chunk)

    full_text = "".join(chunks)
    assert len(full_text) > 0
    # 验证答案包含引用角标 [1] 或 [2]
    assert "[1]" in full_text or "[2]" in full_text


@pytest.mark.asyncio
async def test_astream_generate_with_stream_model():
    """验证使用流式模型时的正确提示词注入与 chunk 输出"""
    mock_stream_model = MagicMock()

    async def mock_astream(messages):
        yield MagicMock(content="根据平台规定，质量问题由商家承担运费[1]。")
        yield MagicMock(content="退款时间以发卡行为准[2]。")

    mock_stream_model.astream = mock_astream

    generator = RAGControlledGenerator(stream_model=mock_stream_model)
    citations = [
        {"n": 1, "question": "运费谁出", "answer": "商家承担运费"},
        {"n": 2, "question": "退款时间", "answer": "以发卡行为准"},
    ]

    chunks = [c async for c in generator.astream_generate("运费谁出？", citations)]
    full_text = "".join(chunks)
    assert "质量问题由商家承担运费[1]" in full_text
    assert "以发卡行为准[2]" in full_text


# ----------------------------------------------------------------------
# 4. ChatService 结合 RAGControlledGenerator 编排单测
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_service_faq_stream_emits_citations_first():
    """验证 ChatService 针对知识库问答：首帧下发 citations 事件，随后流式下发带角标文本"""
    db = FakeAsyncSession()

    # 1. 模拟首轮决策为调用 query_faq
    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="",
            tool_calls=[{
                "name": "query_faq",
                "args": {"keyword": "PRO-X99防水吗"},
                "id": "call_faq_1",
            }],
        )
    )
    mock_llm.bind_tools.return_value = mock_bound

    # 2. 模拟进阶检索器返回 citations
    mock_retriever = AsyncMock()
    mock_retriever.retrieve_with_strategy.return_value = MagicMock(
        citations=[{
            "n": 1,
            "chunk_id": 101,
            "section_path": "商品规格 > PRO-X99",
            "question": "PRO-X99防水等级是多少？",
            "answer": "PRO-X99支持5ATM专业防水等级[1]。",
        }],
        docs=[{
            "chunk_id": 101,
            "question": "PRO-X99防水等级是多少？",
            "answer": "PRO-X99支持5ATM专业防水等级[1]。",
        }],
    )

    # 3. 模拟受控生成器
    mock_generator = MagicMock()
    mock_generator.check_sufficiency = AsyncMock(
        return_value=SelfCheckResult(useful=True, reason="证据充分", source="self_check")
    )

    async def fake_astream(query, citations, history=None):
        yield "星光PRO-X99手表支持5ATM专业防水[1]，"
        yield "可在游泳时佩戴[1]。"

    mock_generator.astream_generate = fake_astream

    service = ChatService(
        model=mock_llm,
        retriever=mock_retriever,
        rag_generator=mock_generator,
    )

    events = [e async for e in service.stream_chat(db, conversation_id=None, message="请问PRO-X99手表防水吗？")]

    # 检查事件类型顺序
    event_types = [e["event_type"] for e in events]
    assert "tool_start" in event_types
    assert "tool_end" in event_types
    assert "citations" in event_types
    assert "text" in event_types

    # 验证 citations 紧跟在 tool_end 之后，且位于所有 text 事件之前
    citations_idx = event_types.index("citations")
    first_text_idx = event_types.index("text")
    tool_end_idx = event_types.index("tool_end")

    assert tool_end_idx < citations_idx
    assert citations_idx < first_text_idx

    # 验证 citations 数据结构
    citations_event = events[citations_idx]
    assert len(citations_event["citations"]) == 1
    assert citations_event["citations"][0]["n"] == 1
    assert "5ATM" in citations_event["citations"][0]["answer"]

    # 验证生成的文本包含角标
    text_content = "".join(e["content"] for e in events if e["event_type"] == "text")
    assert "[1]" in text_content

    # 验证消息正常落盘
    assert len(db.messages) == 4
    assert db.messages[3].role == "assistant"
    assert "[1]" in db.messages[3].content


@pytest.mark.asyncio
async def test_chat_service_faq_refusal_when_self_check_fails():
    """验证知识库问答当自评为 useful=False 时：输出拒答语，不下发 citations，且记录 low_confidence"""
    db = FakeAsyncSession()

    mock_llm = MagicMock()
    mock_bound = MagicMock()
    mock_bound.ainvoke = AsyncMock(
        return_value=MagicMock(
            content="",
            tool_calls=[{
                "name": "query_faq",
                "args": {"keyword": "如何制造反物质飞船"},
                "id": "call_faq_2",
            }],
        )
    )
    mock_llm.bind_tools.return_value = mock_bound

    mock_retriever = AsyncMock()
    mock_retriever.retrieve_with_strategy.return_value = MagicMock(
        citations=[],
        docs=[],
    )

    mock_generator = MagicMock()
    refusal_msg = "非常抱歉，当前知识库中暂未收录相关信息，已为您登记至后台人工处理。"
    mock_generator.refusal_text = refusal_msg

    # 模拟自评并写入 db
    async def mock_check(query, citations, db=None, conversation_id=None):
        if db is not None:
            db.add(LowConfidenceQuestion(
                raw_question=query,
                source="retrieval_low_conf",
                conversation_id=conversation_id,
                reason="未检索到任何相关证据",
            ))
        return SelfCheckResult(useful=False, reason="未检索到任何相关证据", source="retrieval_low_conf")

    mock_generator.check_sufficiency = mock_check

    service = ChatService(
        model=mock_llm,
        retriever=mock_retriever,
        rag_generator=mock_generator,
    )

    events = [e async for e in service.stream_chat(db, conversation_id=None, message="如何制造反物质飞船？")]

    event_types = [e["event_type"] for e in events]
    assert "tool_start" in event_types
    assert "tool_end" in event_types
    # 拒答时不应该下发 citations 事件
    assert "citations" not in event_types

    # 下发了拒答文本
    text_content = "".join(e["content"] for e in events if e["event_type"] == "text")
    assert "非常抱歉，当前知识库中暂未收录相关信息" in text_content

    # 验证 low_confidence_questions 表中成功入池
    assert len(db.low_confidence_questions) == 1
    assert db.low_confidence_questions[0].raw_question == "如何制造反物质飞船？"
    assert db.low_confidence_questions[0].source == "retrieval_low_conf"
