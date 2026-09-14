import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.services.context.logger import (
    DEFAULT_LOG_FILE,
    get_context_logger,
    log_history_context,
    log_model_context,
    log_summary_lifecycle,
)
from app.services.workflow.nodes.agent_node import main_agent_node
from app.services.workflow.nodes.pre_nodes import coreference_resolution_node
from app.services.workflow.state import create_initial_state


@pytest.fixture(autouse=True)
def setup_log_file():
    """Ensure log directory exists and reset log file before each test."""
    DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    yield


class TestContextLoggerUnit:
    """单元测试：验证 log/app.log 文件生成、UTF-8 编码与格式规范"""

    def test_log_file_creation_and_utf8(self):
        """测试 1: 验证 log/app.log 存在且支持 UTF-8 中文写入与即时行缓冲刷新"""
        logger = get_context_logger()
        test_msg = "[model_ctx] conv_id=9999 summary_len=0 window_msgs=1 estimated_tokens=10\n--- SUMMARY ---\n(none)\n--- SLIDING WINDOW ---\n[1] (user) 测试UTF-8中文写入"
        logger.info(test_msg)

        assert DEFAULT_LOG_FILE.exists()
        content = DEFAULT_LOG_FILE.read_text(encoding="utf-8")
        assert "测试UTF-8中文写入" in content
        assert "[model_ctx] conv_id=9999" in content

    def test_log_model_context_format_with_summary(self):
        """测试 2a: 验证 log_model_context 包含摘要时的完整格式与字段"""
        conv_id = 101
        summary = "用户咨询退货政策，需提供订单编号"
        messages = [
            HumanMessage(content="你好，我要退羽绒服"),
            AIMessage(content="请提供您的订单号"),
        ]

        formatted = log_model_context(
            conv_id=conv_id,
            summary=summary,
            window_messages=messages,
            estimated_tokens=42,
        )

        # 校验格式规范
        expected_header = f"[model_ctx] conv_id={conv_id} summary_len={len(summary)} window_msgs=2 estimated_tokens=42"
        assert expected_header in formatted
        assert "--- SUMMARY ---" in formatted
        assert summary in formatted
        assert "--- SLIDING WINDOW ---" in formatted
        assert "[1] (user) 你好，我要退羽绒服" in formatted
        assert "[2] (assistant) 请提供您的订单号" in formatted

        # 校验写入 log/app.log
        content = DEFAULT_LOG_FILE.read_text(encoding="utf-8")
        assert expected_header in content

    def test_log_model_context_format_without_summary(self):
        """测试 2b: 验证无摘要时显示 (none) 且自动估算 tokens"""
        conv_id = 102
        messages = [HumanMessage(content="查询物流")]

        formatted = log_model_context(
            conv_id=conv_id,
            summary=None,
            window_messages=messages,
        )

        assert f"[model_ctx] conv_id={conv_id} summary_len=0 window_msgs=1 estimated_tokens=" in formatted
        assert "--- SUMMARY ---\n(none)" in formatted
        assert "[1] (user) 查询物流" in formatted

    def test_log_history_context_format(self):
        """测试 3: 验证 log_history_context 单行摘要与历史窗口格式规范"""
        conv_id = 201
        summary_multiline = "用户询问羽绒服是否支持退款\n第二行额外信息"
        history_msgs = [
            HumanMessage(content="羽绒服到了"),
            AIMessage(content="您好，请问尺码合适吗？"),
        ]

        formatted = log_history_context(
            conv_id=conv_id,
            summary_line=summary_multiline,
            window_messages=history_msgs,
        )

        # 校验只取单行摘要，且带双引号
        expected_header = f'[history_ctx] conv_id={conv_id} summary_line="用户询问羽绒服是否支持退款" window_msgs=2'
        assert expected_header in formatted
        assert "--- HISTORY WINDOW ---" in formatted
        assert "用户: 羽绒服到了" in formatted
        assert "客服: 您好，请问尺码合适吗？" in formatted

        content = DEFAULT_LOG_FILE.read_text(encoding="utf-8")
        assert expected_header in content

    def test_log_summary_lifecycle_tags(self):
        """测试 4: 验证后台摘要生命周期全留痕日志"""
        log_summary_lifecycle("trigger", conv_id=301, detail="层2 约 1800 token > 预算 1695, range=(1..5)")
        log_summary_lifecycle("start", conv_id=301, detail="第1段 开始执行, msg_range=(1..5)")
        log_summary_lifecycle("done", conv_id=301, detail="第1段完成, 耗时=0.45s, 覆盖至 msg_id=5")
        log_summary_lifecycle("skip", conv_id=301, detail="原因=该会话已有正在运行的摘要任务")
        log_summary_lifecycle("fail", conv_id=301, detail="异常=LLM connection timeout")

        content = DEFAULT_LOG_FILE.read_text(encoding="utf-8")
        assert "[summary trigger] conv_id=301" in content
        assert "[summary start] conv_id=301" in content
        assert "[summary done] conv_id=301" in content
        assert "[summary skip] conv_id=301" in content
        assert "[summary fail] conv_id=301" in content


class TestWorkflowNodesObservabilityIntegration:
    """集成测试：验证工作流前置节点与主力 Agent 节点自动打点"""

    @pytest.mark.asyncio
    async def test_pre_nodes_always_logs_history_ctx_on_greeting(self):
        """测试 5a: 纯打招呼寒暄轮次（进入 chitchat，无大模型改写），必须每轮必打 [history_ctx]"""
        conv_id = 401
        state = create_initial_state(conv_id, "你好")

        # 执行指代消解前置节点
        result = await coreference_resolution_node(state)
        assert result["resolved_query"] == "你好"

        # 校验 log/app.log 中记录了 [history_ctx]
        content = DEFAULT_LOG_FILE.read_text(encoding="utf-8")
        match = re.search(rf'\[history_ctx\] conv_id={conv_id} summary_line=".*?" window_msgs=0', content)
        assert match is not None, f"未能从 log/app.log 找到 conv_id={conv_id} 的 [history_ctx]"

    @pytest.mark.asyncio
    async def test_pre_nodes_logs_history_ctx_with_existing_history(self):
        """测试 5b: 包含历史轮次时，前置节点正确记录 [history_ctx] 窗口"""
        conv_id = 402
        state = create_initial_state(conv_id, "它能退吗")
        state["summary"] = "用户咨询退货条件"
        state["messages"] = [
            HumanMessage(content="我想看下订单1001"),
            AIMessage(content="订单1001为极简羽绒服"),
            HumanMessage(content="它能退吗"),
        ]

        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="订单1001极简羽绒服支持退货吗"))

        result = await coreference_resolution_node(state, model=mock_model)
        assert "1001" in result["resolved_query"]

        content = DEFAULT_LOG_FILE.read_text(encoding="utf-8")
        assert f'[history_ctx] conv_id={conv_id} summary_line="用户咨询退货条件" window_msgs=2' in content
        assert "用户: 我想看下订单1001" in content

    @pytest.mark.asyncio
    async def test_agent_node_logs_model_ctx_before_llm_call(self):
        """测试 6: 主力 Agent 节点调模型前，记录 [model_ctx] 真实入参"""
        conv_id = 501
        state = create_initial_state(conv_id, "查询订单1001")
        state["summary"] = "历史摘要：用户常购服饰"
        state["messages"] = [
            HumanMessage(content="查询订单1001"),
        ]

        mock_model = MagicMock()
        mock_resp = AIMessage(content="订单1001正在派送中。")
        bound_model = MagicMock()
        bound_model.ainvoke = AsyncMock(return_value=mock_resp)
        mock_model.bind_tools.return_value = bound_model
        mock_model.ainvoke = bound_model.ainvoke

        result = await main_agent_node(state, model=mock_model, tools=[])
        assert "派送中" in result["response_text"]

        content = DEFAULT_LOG_FILE.read_text(encoding="utf-8")
        assert f"[model_ctx] conv_id={conv_id}" in content
        assert "--- SUMMARY ---\n历史摘要：用户常购服饰" in content
        assert "--- SLIDING WINDOW ---" in content
        assert "[1] (user) 查询订单1001" in content


class TestLogRetrievalGrepSimulation:
    """检索测试：模拟 grep model_ctx 与 grep history_ctx 检索能力"""

    def test_grep_model_ctx_and_history_ctx(self):
        """测试 7: 模拟 grep 命令直查 model_ctx 和 history_ctx"""
        conv_id = 601
        log_model_context(conv_id=conv_id, summary="测试摘要", window_messages=[HumanMessage(content="问答1")])
        log_history_context(conv_id=conv_id, summary_line="测试摘要单行", window_messages=[HumanMessage(content="问答1")])

        content = DEFAULT_LOG_FILE.read_text(encoding="utf-8")
        lines = content.splitlines()

        model_ctx_lines = [line for line in lines if "model_ctx" in line and f"conv_id={conv_id}" in line]
        history_ctx_lines = [line for line in lines if "history_ctx" in line and f"conv_id={conv_id}" in line]

        assert len(model_ctx_lines) >= 1
        assert len(history_ctx_lines) >= 1
        assert "summary_len=4" in model_ctx_lines[0]
        assert 'summary_line="测试摘要单行"' in history_ctx_lines[0]
