import logging
from unittest import mock
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import Settings
from app.services.context.budget import (
    ContextBudgetResult,
    calculate_context_budget,
    check_budget_on_startup,
    estimate_tokens,
)


class TestTokenEstimator:
    """测试中文与混合文本/消息的 Token 估算器"""

    def test_pure_chinese_characters(self):
        # 测试纯中文文本估算（20 个汉字严格得到 20 Token）
        chinese_text_20 = "一二三四五六七八九十一二三四五六七八九十"
        assert len(chinese_text_20) == 20
        assert estimate_tokens(chinese_text_20) == 20

    def test_chinese_with_fullwidth_punctuation(self):
        # 汉字 + 中文全角标点符号：严格 1 字符 = 1 Token
        text = "你好，世界！“测试”……《星舟》【客服】、"
        # '你', '好', '，', '世', '界', '！', '“', '测', '试', '”', '…', '…', '《', '星', '舟', '》', '【', '客', '服', '】', '、' -> 21 字符
        assert len(text) == 21
        assert estimate_tokens(text) == 21

    def test_pure_ascii_text(self):
        # 西文字符/数字：按约 3.5 字符 ≈ 1 Token 计算 (math.ceil(len(ascii_non_space) / 3.5))
        # "Hello world" 包含 10 个非空格 ASCII 字符 -> ceil(10 / 3.5) = 3
        text = "Hello world"
        assert estimate_tokens(text) == 3

        # "Order12345" 包含 10 个字符 -> ceil(10 / 3.5) = 3
        assert estimate_tokens("Order12345") == 3

    def test_mixed_chinese_and_english(self):
        # 4 个汉字 (4 tokens) + 6 个数字 (ceil(6 / 3.5) = 2 tokens) = 6 tokens
        mixed_text = "订单号是123456"
        assert estimate_tokens(mixed_text) == 6

    def test_empty_and_whitespace(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("   \n\t  ") == 0
        assert estimate_tokens(None) == 0

    def test_langchain_base_messages(self):
        human_msg = HumanMessage(content="我想查询我的订单状态")
        # 10 个汉字 -> 10 tokens
        assert estimate_tokens(human_msg) == 10

        ai_msg = AIMessage(
            content="好的，请稍候。",  # 7 个中文字符/标点 -> 7 tokens
            tool_calls=[{"name": "query_order", "args": {"order_id": "SN12345"}, "id": "call_1"}],
        )
        # 7 tokens (content) + tool_calls json tokens > 7
        tokens = estimate_tokens(ai_msg)
        assert tokens > 7

    def test_message_list_and_dict(self):
        messages = [
            HumanMessage(content="你好"),  # 2 tokens
            {"role": "assistant", "content": "您好！有什么可以帮您？"},  # 11 tokens
        ]
        assert estimate_tokens(messages) == 13

    def test_tool_message(self):
        tool_msg = ToolMessage(content='{"status": "已发货", "express_no": "SF123"}', tool_call_id="call_1")
        assert estimate_tokens(tool_msg) > 0


class TestConfigSettings:
    """测试 Settings 新增 12 项配置字段与默认值"""

    def test_new_context_settings_defaults(self):
        s = Settings(_env_file=None)
        assert s.model_context_window == 128000
        assert s.max_output_tokens == 2000
        assert s.max_user_input_tokens == 2000
        assert s.max_agent_steps == 3
        assert s.tool_result_max_tokens == 1200
        assert s.rerank_top_k == 5
        assert s.system_prompt_tokens == 1500
        assert s.doc_tokens_per_chunk == 500
        assert s.summary_max_tokens == 250
        assert s.safety_margin_tokens == 500
        assert s.target_history_turns == 20
        assert s.steady_turn_tokens == 500

    def test_settings_env_override(self):
        env = {
            "MODEL_CONTEXT_WINDOW": "18000",
            "MAX_AGENT_STEPS": "5",
            "RERANK_TOP_K": "3",
        }
        with mock.patch.dict("os.environ", env, clear=False):
            s = Settings(_env_file=None)
            assert s.model_context_window == 18000
            assert s.max_agent_steps == 5
            assert s.rerank_top_k == 3


class TestContextBudgetCalculator:
    """测试动态 Token 预算倒推算法器与启动自检"""

    def test_demo_configuration_18000_window(self):
        # 演示配置（18000 窗口）：验证固定开销 8750、峰值 3600、滑窗 5650、层 1 预算 3954、层 2 预算 1695
        demo_settings = Settings(
            _env_file=None,
            model_context_window=18000,
            max_output_tokens=2000,
            max_user_input_tokens=2000,
            max_agent_steps=3,
            tool_result_max_tokens=1200,
            rerank_top_k=5,
            system_prompt_tokens=1500,
            doc_tokens_per_chunk=500,
            summary_max_tokens=250,
            safety_margin_tokens=500,
            target_history_turns=20,
            steady_turn_tokens=500,
        )
        res = calculate_context_budget(demo_settings)
        assert isinstance(res, ContextBudgetResult)
        assert res.fixed_overhead == 8750
        assert res.peak_react_overhead == 3600
        assert res.window_available == 5650
        assert res.window_budget == 5650
        assert res.layer1_budget == 3954
        assert res.layer2_budget == 1695
        assert res.is_usable is True

    def test_default_configuration_128k_window(self):
        # 默认 128k 窗口：滑窗预算取稳态 10000，层 1 为 7000，层 2 为 3000
        default_settings = Settings(
            _env_file=None,
            model_context_window=128000,
        )
        res = calculate_context_budget(default_settings)
        assert res.fixed_overhead == 8750
        assert res.peak_react_overhead == 3600
        assert res.window_available == 115650
        assert res.window_budget == 10000
        assert res.layer1_budget == 7000
        assert res.layer2_budget == 3000
        assert res.is_usable is True

    def test_ultra_low_window_startup_check(self, caplog):
        # 超低窗口（例如 window=5000）：触发启动自检报警（is_usable=False，返回 False）
        low_settings = Settings(
            _env_file=None,
            model_context_window=5000,
        )
        res = calculate_context_budget(low_settings)
        assert res.window_available < 0
        assert res.is_usable is False
        assert res.window_budget == 0
        assert res.layer1_budget == 0
        assert res.layer2_budget == 0

        with caplog.at_level(logging.ERROR):
            ok = check_budget_on_startup(low_settings)
            assert ok is False
            assert "CRITICAL: 上下文预算不足，窗口无法容纳单轮稳态交互" in caplog.text

    def test_normal_startup_check(self):
        default_settings = Settings(_env_file=None)
        assert check_budget_on_startup(default_settings) is True
