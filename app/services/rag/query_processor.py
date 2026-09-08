"""Query 理解模块：口语改写归一与检索侧同义词扩展。

特性：
1. 口语改写归一：将口语化长句（如“我买了那个pro x99的手表，要是用着不顺心能退吗”）改写为标准电商术语（如“星光PRO-X99智能手表退换货政策与流程”）；
2. 实体抽取与检索同义词扩展：抽取核心型号与 2~4 个精准同义词；
3. 构造组合检索词 bm25_query：无缝拼接 standard_query 与 expanded_keywords；
4. 双模接口：提供异步 aprocess 与同步 process；
5. 健壮容错与确定性 Mock 降级：未配置 Key、显式 mock=True 或大模型调用异常时，原样兜底不报错，确保 100% 可用性。
"""

import asyncio
from dataclasses import dataclass, field
import logging
import os
from typing import Any, List, Optional

from app.config import settings
from app.llm import get_chat_model
from app.prompts.query_understanding import (
    parse_query_understanding_output,
    query_understanding_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class QueryUnderstandingResult:
    """用户查询意图理解与改写结果 DTO。"""

    original_query: str
    standard_query: str
    expanded_keywords: List[str] = field(default_factory=list)
    bm25_query: str = ""


class QueryProcessor:
    """Query 理解与改写处理器。"""

    def __init__(
        self,
        llm: Optional[Any] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        mock: bool = False,
        max_retries: int = 1,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

        # 判断是否显式开启或通过环境变量开启 Mock 模式
        if (
            mock
            or os.getenv("MOCK_LLM") == "1"
            or os.getenv("ASTRO_MOCK_LLM") == "1"
            or os.getenv("ASTRO_MOCK_QUERY_PROCESSOR") == "1"
        ):
            self.is_mock = True
            self._llm = None
            logger.info("QueryProcessor: 显式开启 Mock 模式")
            return

        if llm is not None:
            self._llm = llm
            self.is_mock = False
        else:
            api_key = getattr(settings, "openai_api_key", None)
            if not api_key or api_key in ("sk-placeholder", "mock"):
                self.is_mock = True
                self._llm = None
                logger.info("QueryProcessor: 未检测到有效 OpenAI API Key，自动启用确定性 Mock 降级模式")
            else:
                self.is_mock = False
                try:
                    self._llm = get_chat_model(
                        streaming=False,
                        temperature=self.temperature,
                        model_name=self.model_name,
                    )
                except Exception as e:
                    logger.warning(f"QueryProcessor: 初始化 Chat 模型失败: {e}，降级为 Mock 模式")
                    self.is_mock = True
                    self._llm = None

    def _build_bm25_query(self, standard_query: str, expanded_keywords: List[str]) -> str:
        """拼接 standard_query 与 expanded_keywords 生成 BM25 稀疏检索查询词。"""
        clean_sq = standard_query.strip() if standard_query else ""
        clean_kws = [k.strip() for k in expanded_keywords if k and str(k).strip()]
        if not clean_kws:
            return clean_sq
        if not clean_sq:
            return " ".join(clean_kws)
        return f"{clean_sq} {' '.join(clean_kws)}"

    def _fallback_result(self, query: Optional[str]) -> QueryUnderstandingResult:
        """生成确定性原词兜底结果。"""
        q = query if query is not None else ""
        return QueryUnderstandingResult(
            original_query=q,
            standard_query=q,
            expanded_keywords=[],
            bm25_query=q,
        )

    async def aprocess(self, query: Optional[str]) -> QueryUnderstandingResult:
        """异步处理用户查询，执行口语改写归一与同义词扩展。"""
        if not query or not query.strip():
            return self._fallback_result(query)

        if self.is_mock or self._llm is None:
            return self._fallback_result(query)

        for attempt in range(self.max_retries + 1):
            try:
                messages = query_understanding_prompt.format_messages(query=query)
                if hasattr(self._llm, "ainvoke"):
                    resp = await self._llm.ainvoke(messages)
                elif hasattr(self._llm, "invoke"):
                    resp = self._llm.invoke(messages)
                elif callable(self._llm):
                    res = self._llm(messages)
                    if asyncio.iscoroutine(res):
                        resp = await res
                    else:
                        resp = res
                else:
                    raise ValueError(f"不支持的 LLM 客户端类型: {type(self._llm)}")

                content = resp.content if hasattr(resp, "content") else str(resp)
                parsed = parse_query_understanding_output(content)
                standard_query = parsed.get("standard_query")

                if not standard_query:
                    if attempt < self.max_retries:
                        continue
                    logger.warning(
                        f"QueryProcessor: 模型返回未能提取有效 standard_query: {content[:100]}"
                    )
                    return self._fallback_result(query)

                expanded_keywords = parsed.get("expanded_keywords") or []
                bm25_query = self._build_bm25_query(standard_query, expanded_keywords)
                return QueryUnderstandingResult(
                    original_query=query,
                    standard_query=standard_query,
                    expanded_keywords=expanded_keywords,
                    bm25_query=bm25_query,
                )
            except Exception as e:
                if attempt < self.max_retries:
                    continue
                logger.warning(f"QueryProcessor: aprocess 执行异常: {e}，优雅降级为原词兜底")
                return self._fallback_result(query)

        return self._fallback_result(query)

    def process(self, query: Optional[str]) -> QueryUnderstandingResult:
        """同步处理用户查询，与 aprocess 保持完全一致的改写与降级行为。"""
        if not query or not query.strip():
            return self._fallback_result(query)

        if self.is_mock or self._llm is None:
            return self._fallback_result(query)

        for attempt in range(self.max_retries + 1):
            try:
                messages = query_understanding_prompt.format_messages(query=query)
                if hasattr(self._llm, "invoke"):
                    resp = self._llm.invoke(messages)
                elif callable(self._llm):
                    resp = self._llm(messages)
                elif hasattr(self._llm, "ainvoke"):
                    import concurrent.futures
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                resp = executor.submit(
                                    asyncio.run, self._llm.ainvoke(messages)
                                ).result()
                        else:
                            resp = loop.run_until_complete(self._llm.ainvoke(messages))
                    except Exception:
                        resp = asyncio.run(self._llm.ainvoke(messages))
                else:
                    raise ValueError(f"不支持的同步 LLM 客户端类型: {type(self._llm)}")

                if asyncio.iscoroutine(resp):
                    import concurrent.futures
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                resp = executor.submit(asyncio.run, resp).result()
                        else:
                            resp = loop.run_until_complete(resp)
                    except Exception:
                        resp = asyncio.run(resp)

                content = resp.content if hasattr(resp, "content") else str(resp)
                parsed = parse_query_understanding_output(content)
                standard_query = parsed.get("standard_query")

                if not standard_query:
                    if attempt < self.max_retries:
                        continue
                    logger.warning(
                        f"QueryProcessor: 同步调用未能提取有效 standard_query: {content[:100]}"
                    )
                    return self._fallback_result(query)

                expanded_keywords = parsed.get("expanded_keywords") or []
                bm25_query = self._build_bm25_query(standard_query, expanded_keywords)
                return QueryUnderstandingResult(
                    original_query=query,
                    standard_query=standard_query,
                    expanded_keywords=expanded_keywords,
                    bm25_query=bm25_query,
                )
            except Exception as e:
                if attempt < self.max_retries:
                    continue
                logger.warning(f"QueryProcessor: process 执行异常: {e}，优雅降级为原词兜底")
                return self._fallback_result(query)

        return self._fallback_result(query)
