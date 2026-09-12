import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.session import Base
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.summary import ConversationSummary
from app.prompts.summarizer import (
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT_TEMPLATE,
    format_summary_prompt,
)
from app.services.context.summary_service import (
    SummaryService,
    should_trigger_summary,
)


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
def sqlite_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def async_session_factory(sqlite_engine):
    def factory():
        sync_sess = Session(sqlite_engine, expire_on_commit=False)
        return AsyncSessionAdapter(sync_sess)
    return factory


class TestSummaryTriggerLogic:
    """测试基于 Token 估算的触发判定逻辑"""

    def test_should_trigger_when_over_budget(self):
        assert should_trigger_summary(layer2_tokens=1696, layer2_budget=1695) is True
        assert SummaryService.should_trigger_summary(layer2_tokens=2000, layer2_budget=1695) is True

    def test_should_not_trigger_when_within_budget(self):
        assert should_trigger_summary(layer2_tokens=1695, layer2_budget=1695) is False
        assert should_trigger_summary(layer2_tokens=500, layer2_budget=1695) is False
        assert should_trigger_summary(layer2_tokens=0, layer2_budget=1695) is False


class TestSummarizerPrompt:
    """测试摘要提炼 Prompt 规范与红线约束"""

    def test_system_prompt_fact_and_negative_rules(self):
        # 必须包含关键业务事实与诉求约束
        assert "商品型号" in SUMMARY_SYSTEM_PROMPT
        assert "订单号" in SUMMARY_SYSTEM_PROMPT or "手机号" in SUMMARY_SYSTEM_PROMPT
        assert "诉求" in SUMMARY_SYSTEM_PROMPT

        # 必须包含严禁编造的 Hard Gate
        assert "编造" in SUMMARY_SYSTEM_PROMPT or "真实性" in SUMMARY_SYSTEM_PROMPT

        # 必须包含剔除寒暄客套规范
        assert "寒暄" in SUMMARY_SYSTEM_PROMPT or "客套" in SUMMARY_SYSTEM_PROMPT or "您好" in SUMMARY_SYSTEM_PROMPT

        # 必须包含 50 ~ 200 字篇幅约束
        assert "50" in SUMMARY_SYSTEM_PROMPT and "200" in SUMMARY_SYSTEM_PROMPT

        # 必须包含对旧梗概的隔离约束（严禁重复输出或合并改写）
        assert "旧梗概" in SUMMARY_SYSTEM_PROMPT or "已有前情" in SUMMARY_SYSTEM_PROMPT or "背景" in SUMMARY_SYSTEM_PROMPT
        assert "严禁重复输出" in SUMMARY_SYSTEM_PROMPT or "严禁合并改写" in SUMMARY_SYSTEM_PROMPT

    def test_format_summary_prompt_with_and_without_existing_summary(self):
        prompt_with_bg = format_summary_prompt(
            dialogue_text="用户: 衣服降价了\n客服: 可以为您申请保价",
            existing_summary="用户购买了羽绒服，订单号SN123。",
        )
        assert "【已有前情背景" in prompt_with_bg
        assert "用户购买了羽绒服，订单号SN123。" in prompt_with_bg
        assert "用户: 衣服降价了" in prompt_with_bg

        prompt_without_bg = format_summary_prompt(
            dialogue_text="用户: 查一下物流\n客服: 正在运输中",
            existing_summary=None,
        )
        assert "暂无" in prompt_without_bg
        assert "用户: 查一下物流" in prompt_without_bg

    def test_format_dialogue_for_summary(self):
        from langchain_core.messages import ToolMessage
        from app.services.context.summary_service import format_dialogue_for_summary
        msgs = [
            Message(role="user", content="我想查询保修"),
            Message(role="assistant", content="请提供商品名称"),
            ToolMessage(content="已查询到保修卡", tool_call_id="c1", name="check_warranty"),
            Message(role="tool", content="订单保修一年", tool_call_id="c2"),
        ]
        text = format_dialogue_for_summary(msgs)
        assert "用户: 我想查询保修" in text
        assert "客服: 请提供商品名称" in text
        assert "工具(check_warranty): 已查询到保修卡" in text
        assert "工具: 订单保修一年" in text

    def test_summarize_dialogue_sync(self):
        from app.services.context.summary_service import summarize_dialogue_sync
        mock_model = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "用户询问保修期，客服查询确认保修一年。"
        mock_model.invoke.return_value = mock_resp

        res = summarize_dialogue_sync(
            dialogue_text="用户: 保修多久？\n客服: 一年",
            existing_summary="用户购入手机。",
            model=mock_model,
        )
        assert res == "用户询问保修期，客服查询确认保修一年。"
        assert mock_model.invoke.call_count == 1


class TestSummaryExecutionAndPersistence:
    """测试摘要任务执行、段落追加、seq 递增与 conversations 投影更新"""

    @pytest.mark.asyncio
    async def test_persist_summary_segment_multi_rounds(self, sqlite_engine, async_session_factory):
        # 1. 准备初始会话
        with Session(sqlite_engine, expire_on_commit=False) as init_sess:
            conv = Conversation(user_id="u_001", status="进行中")
            init_sess.add(conv)
            init_sess.commit()
            conv_id = conv.id

        service = SummaryService(session_factory=async_session_factory)

        # 2. 插入第 1 段摘要
        async with async_session_factory() as session:
            s1 = await service.persist_summary_segment(
                session=session,
                conv_id=conv_id,
                from_id=1,
                to_id=5,
                content="第1段事实：用户咨询极简羽绒服尺码，客服推荐L码。",
            )
            assert s1.seq == 1
            assert s1.from_msg_id == 1
            assert s1.upto_msg_id == 5

        # 校验数据库更新
        with Session(sqlite_engine) as verify_sess:
            refetched = verify_sess.get(Conversation, conv_id)
            assert refetched.summary_upto_msg_id == 5
            assert refetched.summary == "第1段事实：用户咨询极简羽绒服尺码，客服推荐L码。"

        # 3. 插入第 2 段摘要
        async with async_session_factory() as session:
            s2 = await service.persist_summary_segment(
                session=session,
                conv_id=conv_id,
                from_id=6,
                to_id=10,
                content="第2段事实：用户下单并提供订单号SN8899，要求尽快发货。",
            )
            assert s2.seq == 2
            assert s2.from_msg_id == 6
            assert s2.upto_msg_id == 10

        # 校验多段投影正确按顺序拼接
        with Session(sqlite_engine) as verify_sess:
            refetched = verify_sess.get(Conversation, conv_id)
            assert refetched.summary_upto_msg_id == 10
            expected_projection = (
                "第1段事实：用户咨询极简羽绒服尺码，客服推荐L码。\n\n"
                "第2段事实：用户下单并提供订单号SN8899，要求尽快发货。"
            )
            assert refetched.summary == expected_projection

            # 校验 conversation_summaries 表中的真实记录
            summaries = verify_sess.scalars(
                select(ConversationSummary).where(ConversationSummary.conversation_id == conv_id).order_by(ConversationSummary.seq.asc())
            ).all()
            assert len(summaries) == 2
            assert summaries[0].seq == 1
            assert summaries[1].seq == 2

    @pytest.mark.asyncio
    async def test_execute_summary_task_end_to_end(self, sqlite_engine, async_session_factory, caplog):
        # 准备会话与消息
        with Session(sqlite_engine, expire_on_commit=False) as init_sess:
            conv = Conversation(user_id="u_002", status="进行中")
            init_sess.add(conv)
            init_sess.commit()
            conv_id = conv.id

            m1 = Message(conversation_id=conv_id, role="user", content="我的手机号是13800000000，订单SN1001还没到")
            m2 = Message(conversation_id=conv_id, role="assistant", content="您好，为您核实物流已在中转站，预计明天派送。")
            init_sess.add_all([m1, m2])
            init_sess.commit()
            m1_id = m1.id
            m2_id = m2.id

        # Mock LLM
        mock_model = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "用户提供手机号13800000000，催促订单SN1001物流进度，客服告知已在中转站预计次日送达。"
        mock_model.ainvoke = AsyncMock(return_value=mock_resp)

        service = SummaryService(session_factory=async_session_factory)

        # 执行摘要任务
        with caplog.at_level("INFO"):
            result = await service.execute_summary_task(
                conv_id=conv_id,
                from_msg_id=m1_id,
                upto_msg_id=m2_id,
                model=mock_model,
            )

        assert result is not None
        assert result.seq == 1
        assert result.from_msg_id == m1_id
        assert result.upto_msg_id == m2_id

        # 校验数据库更新
        with Session(sqlite_engine) as verify_sess:
            refetched = verify_sess.get(Conversation, conv_id)
            assert refetched.summary_upto_msg_id == m2_id
            assert "13800000000" in refetched.summary
            assert "SN1001" in refetched.summary

        # 校验生命周期日志输出
        log_text = caplog.text
        assert "[summary start]" in log_text
        assert f"conv_id={conv_id}" in log_text
        assert "[summary done]" in log_text
        assert "耗时=" in log_text

    @pytest.mark.asyncio
    async def test_execute_summary_task_skips_when_empty_range(self, sqlite_engine, async_session_factory, caplog):
        with Session(sqlite_engine, expire_on_commit=False) as init_sess:
            conv = Conversation(user_id="u_002_empty", status="进行中")
            init_sess.add(conv)
            init_sess.commit()
            conv_id = conv.id

        service = SummaryService(session_factory=async_session_factory)
        with caplog.at_level("INFO"):
            result = await service.execute_summary_task(
                conv_id=conv_id,
                from_msg_id=100,
                upto_msg_id=200,
            )
        assert result is None
        assert "[summary skip]" in caplog.text


class TestConcurrencyLockAndLifecycle:
    """测试并发防重锁与生命周期日志"""

    @pytest.mark.asyncio
    async def test_concurrency_lock_skips_duplicate_task(self, async_session_factory, caplog):
        service = SummaryService(session_factory=async_session_factory)
        conv_id = 999

        # 手动将会话设为活跃中
        service._active_tasks.add(conv_id)

        with caplog.at_level("INFO"):
            task = service.trigger_async_summary(
                conv_id=conv_id,
                from_msg_id=1,
                upto_msg_id=5,
            )

        # 必须被跳过，返回 None
        assert task is None
        assert "[summary skip]" in caplog.text
        assert f"conv_id={conv_id}" in caplog.text

        # 清理
        service._active_tasks.discard(conv_id)

    @pytest.mark.asyncio
    async def test_trigger_async_summary_non_blocking(self, sqlite_engine, async_session_factory, caplog):
        with Session(sqlite_engine, expire_on_commit=False) as init_sess:
            conv = Conversation(user_id="u_003", status="进行中")
            init_sess.add(conv)
            init_sess.commit()
            conv_id = conv.id

            m1 = Message(conversation_id=conv_id, role="user", content="请问商品保修期是多久？")
            init_sess.add(m1)
            init_sess.commit()
            m1_id = m1.id

        mock_model = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "用户咨询商品保修期限。"
        mock_model.ainvoke = AsyncMock(return_value=mock_resp)

        service = SummaryService(session_factory=async_session_factory)

        with caplog.at_level("INFO"):
            # 派发非阻塞后台任务
            task = service.trigger_async_summary(
                conv_id=conv_id,
                from_msg_id=m1_id,
                upto_msg_id=m1_id,
                model=mock_model,
                layer2_tokens=1800,
                layer2_budget=1695,
            )

            assert task is not None
            assert isinstance(task, asyncio.Task)
            # 在任务执行期间，防重锁有效
            assert conv_id in service._active_tasks

            # 等待后台任务执行完成
            await task

            # 任务执行完成后，防重锁自动释放
            assert conv_id not in service._active_tasks

        assert "[summary trigger]" in caplog.text
        assert "[summary start]" in caplog.text
        assert "[summary done]" in caplog.text


class TestSummaryErrorHandling:
    """测试大模型调用或数据库异常时的容错与日志"""

    @pytest.mark.asyncio
    async def test_execute_summary_task_handles_llm_failure(self, sqlite_engine, async_session_factory, caplog):
        with Session(sqlite_engine, expire_on_commit=False) as init_sess:
            conv = Conversation(user_id="u_004", status="进行中")
            init_sess.add(conv)
            init_sess.commit()
            conv_id = conv.id

            m = Message(conversation_id=conv_id, role="user", content="我想退款")
            init_sess.add(m)
            init_sess.commit()
            m_id = m.id

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(side_effect=RuntimeError("LLM service unavailable"))

        service = SummaryService(session_factory=async_session_factory)

        with caplog.at_level("INFO"):
            result = await service.execute_summary_task(
                conv_id=conv_id,
                from_msg_id=m_id,
                upto_msg_id=m_id,
                model=mock_model,
            )

        assert result is None
        assert "[summary fail]" in caplog.text
        assert "LLM service unavailable" in caplog.text
        # 防重锁依然被安全释放
        assert conv_id not in service._active_tasks
