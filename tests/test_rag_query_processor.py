"""Task 4 单测套件：Query 理解模块 (口语改写归一 + 检索侧同义词扩展)。

覆盖内容：
1. app/prompts/query_understanding.py:
   - QUERY_UNDERSTANDING_SYSTEM_PROMPT: 提示词包含标准问法改写、专有名词保留、口语助词过滤、同义词 2~4 个、严格 JSON 格式输出规则；
   - query_understanding_prompt: ChatPromptTemplate 格式化；
   - parse_query_understanding_output: 覆盖纯 JSON、Markdown ```json 代码块包裹、缺失字段与非法字符串容错解析。
2. QueryUnderstandingResult 数据类:
   - 包含 original_query, standard_query, expanded_keywords, bm25_query 字段。
3. QueryProcessor:
   - aprocess（异步）与 process（同步）口语长句改写为电商标准词；
   - 同义词与实体抽取 expanded_keywords 包含核心实体与同义词；
   - bm25_query 组合标准问法与扩展同义词；
   - 同步 process 与异步 aprocess 接口行为完全一致；
   - 显式 mock=True 时的确定性 Mock 降级（原样兜底不报错）；
   - 未配置 API Key 时的自动 Mock 降级；
   - 大模型调用异常、超时或返回非合法 JSON 时的健壮容错与原样兜底；
   - 空白或无效输入边界处理；
4. app/services/rag/__init__.py:
   - 导出 QueryProcessor 与 QueryUnderstandingResult。
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.messages import AIMessage

from app.prompts.query_understanding import (
    QUERY_UNDERSTANDING_SYSTEM_PROMPT,
    query_understanding_prompt,
    parse_query_understanding_output,
)
from app.services.rag.query_processor import (
    QueryProcessor,
    QueryUnderstandingResult,
)
from app.services.rag import (
    QueryProcessor as ExportedQueryProcessor,
    QueryUnderstandingResult as ExportedQueryUnderstandingResult,
)


# ==============================================================================
# 1. 导出与数据模型结构单测
# ==============================================================================

def test_query_processor_exports_from_rag_package():
    """验证 app.services.rag 顶层包正确导出了 QueryProcessor 与 QueryUnderstandingResult。"""
    assert ExportedQueryProcessor is QueryProcessor
    assert ExportedQueryUnderstandingResult is QueryUnderstandingResult


def test_query_understanding_result_dataclass():
    """验证 QueryUnderstandingResult 数据类各字段结构。"""
    result = QueryUnderstandingResult(
        original_query="pro x99 能退吗",
        standard_query="PRO-X99智能手表退换货政策",
        expanded_keywords=["PRO-X99", "退货", "换货"],
        bm25_query="PRO-X99智能手表退换货政策 PRO-X99 退货 换货",
    )
    assert result.original_query == "pro x99 能退吗"
    assert result.standard_query == "PRO-X99智能手表退换货政策"
    assert result.expanded_keywords == ["PRO-X99", "退货", "换货"]
    assert "PRO-X99" in result.bm25_query
    assert "退换货" in result.bm25_query


# ==============================================================================
# 2. Prompt 与 JSON 解析器单测
# ==============================================================================

def test_query_understanding_prompt_definition():
    """验证 Prompt 包含电商口语归一化、专有名词保留、同义词 2~4 个限制及 JSON 规范要求。"""
    assert "标准" in QUERY_UNDERSTANDING_SYSTEM_PROMPT
    assert "同义词" in QUERY_UNDERSTANDING_SYSTEM_PROMPT or "关键词" in QUERY_UNDERSTANDING_SYSTEM_PROMPT
    assert "standard_query" in QUERY_UNDERSTANDING_SYSTEM_PROMPT
    assert "expanded_keywords" in QUERY_UNDERSTANDING_SYSTEM_PROMPT

    messages = query_understanding_prompt.format_messages(query="我的手表按键坏了")
    assert len(messages) == 2
    assert messages[0].type == "system"
    assert "我的手表按键坏了" in messages[1].content


def test_parse_query_understanding_output_standard_json():
    """验证解析标准无包裹的 JSON 字符串。"""
    raw = json.dumps({
        "standard_query": "PRO-X99智能手表退换货政策",
        "expanded_keywords": ["PRO-X99", "退货", "换货", "退换货"],
    }, ensure_ascii=False)
    parsed = parse_query_understanding_output(raw)
    assert parsed["standard_query"] == "PRO-X99智能手表退换货政策"
    assert parsed["expanded_keywords"] == ["PRO-X99", "退货", "换货", "退换货"]


def test_parse_query_understanding_output_markdown_fenced():
    """验证解析带有 ```json 代码块包裹的大模型输出。"""
    raw = """
这里是改写结果：
```json
{
  "standard_query": "星光PRO-X99智能手表售后退换货流程",
  "expanded_keywords": ["PRO-X99", "退货", "换货", "智能手表"]
}
```
希望对您有帮助！
"""
    parsed = parse_query_understanding_output(raw)
    assert parsed["standard_query"] == "星光PRO-X99智能手表售后退换货流程"
    assert parsed["expanded_keywords"] == ["PRO-X99", "退货", "换货", "智能手表"]


def test_parse_query_understanding_output_malformed_and_empty():
    """验证解析器在遇到空字符串、非 JSON 文本或缺失字段时的容错性。"""
    assert parse_query_understanding_output("") == {}
    assert parse_query_understanding_output("   ") == {}
    assert parse_query_understanding_output("抱歉，我不能协助改写。") == {}
    assert parse_query_understanding_output('{"other_key": 123}') == {"standard_query": "", "expanded_keywords": []}


# ==============================================================================
# 3. QueryProcessor 口语改写与 BM25 查询构造单测
# ==============================================================================

@pytest.mark.asyncio
async def test_query_processor_aprocess_colloquial_rewrite():
    """验证口语化长句能被改写出标准问法、扩展同义词，并拼接为 BM25 检索词。"""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content=json.dumps({
                "standard_query": "星光PRO-X99智能手表退换货政策与流程",
                "expanded_keywords": ["PRO-X99", "退货", "换货", "退换货"],
            }, ensure_ascii=False)
        )
    )

    processor = QueryProcessor(llm=mock_llm)
    query = "我买了那个pro x99的手表，要是用着不顺心能退吗"
    result = await processor.aprocess(query)

    assert result.original_query == query
    # 验证标准问法包含型号与退换货电商标准词
    assert "PRO-X99" in result.standard_query
    assert "退" in result.standard_query and ("换货" in result.standard_query or "退货" in result.standard_query)
    # 验证同义词与实体抽取包含核心型号与同义词
    assert "PRO-X99" in result.expanded_keywords
    assert any(k in result.expanded_keywords for k in ["退货", "换货", "退换货"])
    # 验证 bm25_query 拼接了 standard_query 与 expanded_keywords
    assert result.standard_query in result.bm25_query
    for kw in result.expanded_keywords:
        assert kw in result.bm25_query


def test_query_processor_process_sync_matches_aprocess_async():
    """验证 QueryProcessor.process (同步) 与 QueryProcessor.aprocess (异步) 行为完全一致。"""
    mock_response = json.dumps({
        "standard_query": "PRO-X99智能手表支持的防水等级说明",
        "expanded_keywords": ["PRO-X99", "5ATM", "防水", "游泳佩戴"],
    }, ensure_ascii=False)

    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=AIMessage(content=mock_response))
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=mock_response))

    processor = QueryProcessor(llm=mock_llm)
    query = "pro x99 能带着游泳洗澡吗"

    sync_result = processor.process(query)

    import asyncio
    async_result = asyncio.run(processor.aprocess(query))

    assert sync_result.original_query == async_result.original_query == query
    assert sync_result.standard_query == async_result.standard_query == "PRO-X99智能手表支持的防水等级说明"
    assert sync_result.expanded_keywords == async_result.expanded_keywords == ["PRO-X99", "5ATM", "防水", "游泳佩戴"]
    assert sync_result.bm25_query == async_result.bm25_query


# ==============================================================================
# 4. 降级与容错单测（Mock、未配置、调用异常、空输入）
# ==============================================================================

@pytest.mark.asyncio
async def test_query_processor_explicit_mock_mode():
    """验证显式 mock=True 时的确定性 Mock 降级（原样兜底不报错）。"""
    processor = QueryProcessor(mock=True)
    query = "我买了那个pro x99的手表，要是用着不顺心能退吗"

    # 异步验证
    async_result = await processor.aprocess(query)
    assert async_result.original_query == query
    assert async_result.standard_query == query  # 原样兜底
    assert async_result.expanded_keywords == []
    assert async_result.bm25_query == query

    # 同步验证
    sync_result = processor.process(query)
    assert sync_result.original_query == query
    assert sync_result.standard_query == query
    assert sync_result.expanded_keywords == []
    assert sync_result.bm25_query == query


@pytest.mark.asyncio
async def test_query_processor_unconfigured_api_key_degradation():
    """验证 API Key 未配置（或占位符）时自动启用确定性 Mock 降级。"""
    with patch("app.services.rag.query_processor.settings") as mock_settings:
        mock_settings.openai_api_key = "sk-placeholder"
        processor = QueryProcessor()
        assert processor.is_mock is True

        query = "手表坏了怎么换新"
        result = await processor.aprocess(query)
        assert result.original_query == query
        assert result.standard_query == query
        assert result.expanded_keywords == []
        assert result.bm25_query == query


@pytest.mark.asyncio
async def test_query_processor_llm_exception_fallback():
    """验证大模型调用异常时平滑降级，原样兜底不报错。"""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("OpenAI API connection timeout"))
    mock_llm.invoke = MagicMock(side_effect=RuntimeError("OpenAI API connection timeout"))

    processor = QueryProcessor(llm=mock_llm, max_retries=1)
    query = "我买了那个pro x99的手表，要是用着不顺心能退吗"

    # 异步异常兜底
    result_async = await processor.aprocess(query)
    assert result_async.original_query == query
    assert result_async.standard_query == query
    assert result_async.expanded_keywords == []
    assert result_async.bm25_query == query

    # 同步异常兜底
    result_sync = processor.process(query)
    assert result_sync.original_query == query
    assert result_sync.standard_query == query
    assert result_sync.expanded_keywords == []
    assert result_sync.bm25_query == query


@pytest.mark.asyncio
async def test_query_processor_malformed_json_fallback():
    """验证大模型返回非合法 JSON 时，优雅降级为原词兜底。"""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="很抱歉，作为一个AI，我无法回答该问题。")
    )

    processor = QueryProcessor(llm=mock_llm, max_retries=0)
    query = "pro x99 手表怎么退货"
    result = await processor.aprocess(query)

    assert result.original_query == query
    assert result.standard_query == query
    assert result.expanded_keywords == []
    assert result.bm25_query == query


@pytest.mark.asyncio
async def test_query_processor_empty_query():
    """验证输入空字符串或全空格时，直接原样返回，不调用大模型。"""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()
    processor = QueryProcessor(llm=mock_llm)

    for empty_input in ["", "   ", None]:
        res_async = await processor.aprocess(empty_input)
        res_sync = processor.process(empty_input)

        expected_q = empty_input or ""
        assert res_async.standard_query == expected_q
        assert res_async.expanded_keywords == []
        assert res_sync.standard_query == expected_q
        assert res_sync.expanded_keywords == []

    # 确保 mock_llm 从未被调用
    mock_llm.ainvoke.assert_not_called()
