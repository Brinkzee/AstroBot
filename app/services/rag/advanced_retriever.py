"""进阶混合检索与重排服务流水线。

特性：
1. 整合 Query 理解 (QueryProcessor)、BGE-M3 Dense 语义向量 (BGEEmbeddingClient)、
   Milvus 2.5 原生 BM25 + Dense 混合检索 (MilvusKnowledgeStore) 与 BGE-Reranker-v2-m3 交叉编码精排 (BGERerankerClient)；
2. 支持 4 种检索召回策略切流：
   - 'vector_only': 单路 Dense 向量检索 (COSINE)
   - 'bm25_only': 单路 Milvus 原生 BM25 全文检索 (jieba)
   - 'hybrid': Dense 50 + BM25 50 并发召回并经 RRFRanker(k=60) 融合输出
   - 'hybrid_rerank': 双路召回 + RRF 融合 Top-50 -> BGE-Reranker 重排 Top-10 -> Lost in the Middle 首尾重排 -> citations 快照
3. 统一返回 AdvancedRetrievalResult 结构体；
4. 保持对既有 KnowledgeRetriever.retrieve / retrieve_faq_text / format_faq_hits 的 100% 接口向后兼容性。
"""

import asyncio
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.milvus_client import MilvusKnowledgeStore
from app.services.rag.query_processor import (
    QueryProcessor,
    QueryUnderstandingResult,
)
from app.services.rag.reorder import (
    build_citation_items,
    lost_in_the_middle_reorder,
)
from app.services.rag.reranker import BGERerankerClient

logger = logging.getLogger(__name__)


@dataclass
class AdvancedRetrievalResult:
    """进阶检索与重排结果统一 DTO。"""

    docs: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    strategy: str = "hybrid_rerank"
    query_understanding: Optional[QueryUnderstandingResult] = None


class AdvancedKnowledgeRetriever:
    """进阶混合检索与重排编排器。"""

    VALID_STRATEGIES = ("vector_only", "bm25_only", "hybrid", "hybrid_rerank")

    def __init__(
        self,
        store: Optional[MilvusKnowledgeStore] = None,
        embedding_client: Optional[BGEEmbeddingClient] = None,
        query_processor: Optional[QueryProcessor] = None,
        reranker_client: Optional[BGERerankerClient] = None,
        collection_name: str = "knowledge",
    ):
        self.store = store or MilvusKnowledgeStore()
        self.embedding_client = embedding_client or BGEEmbeddingClient()
        self.query_processor = query_processor or QueryProcessor()
        self.reranker_client = reranker_client or BGERerankerClient()
        self.collection_name = collection_name
        logger.info(
            f"AdvancedKnowledgeRetriever 已初始化 (collection={self.collection_name})"
        )

    async def retrieve_with_strategy(
        self,
        query: str,
        strategy: str = "hybrid_rerank",
        category_filter: Optional[str] = None,
        top_k: int = 10,
        top_k_per_route: int = 50,
    ) -> AdvancedRetrievalResult:
        """根据指定策略执行进阶检索召回与重排。

        Args:
            query: 用户原始提问或查询语句
            strategy: 检索策略，可选 'vector_only', 'bm25_only', 'hybrid', 'hybrid_rerank'
            category_filter: 可选的类目标量过滤条件
            top_k: 最终保留的最大结果条数，默认 10
            top_k_per_route: 粗排单路召回条目数，默认 50

        Returns:
            统一封装的 AdvancedRetrievalResult
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"不支持的检索策略: '{strategy}'，可选值为: {list(self.VALID_STRATEGIES)}"
            )

        if not query or not str(query).strip():
            return AdvancedRetrievalResult(
                docs=[],
                citations=[],
                strategy=strategy,
                query_understanding=None,
            )

        raw_query = str(query).strip()

        # 1. 意图理解与口语改写归一
        qu: QueryUnderstandingResult = await self.query_processor.aprocess(raw_query)
        dense_query_text = (
            qu.standard_query.strip() if qu and qu.standard_query else raw_query
        )
        bm25_query_text = (
            qu.bm25_query.strip() if qu and qu.bm25_query else raw_query
        )

        # 2. 根据策略切流调度
        if strategy == "vector_only":
            dense_vector = await self.embedding_client.aembed_query(dense_query_text)
            hits = await asyncio.to_thread(
                self.store.search_dense,
                query_vector=dense_vector,
                top_k=top_k,
                category_filter=category_filter,
                collection_name=self.collection_name,
            )
            docs = hits[:top_k]
            citations = build_citation_items(docs)
            return AdvancedRetrievalResult(
                docs=docs,
                citations=citations,
                strategy=strategy,
                query_understanding=qu,
            )

        if strategy == "bm25_only":
            hits = await asyncio.to_thread(
                self.store.search_bm25,
                query_text=bm25_query_text,
                top_k=top_k,
                category_filter=category_filter,
                collection_name=self.collection_name,
            )
            docs = hits[:top_k]
            citations = build_citation_items(docs)
            return AdvancedRetrievalResult(
                docs=docs,
                citations=citations,
                strategy=strategy,
                query_understanding=qu,
            )

        if strategy == "hybrid":
            dense_vector = await self.embedding_client.aembed_query(dense_query_text)
            hits = await asyncio.to_thread(
                self.store.hybrid_search,
                dense_vector=dense_vector,
                bm25_text=bm25_query_text,
                category_filter=category_filter,
                top_k_per_route=top_k_per_route,
                limit=top_k,
                collection_name=self.collection_name,
            )
            docs = hits[:top_k]
            citations = build_citation_items(docs)
            return AdvancedRetrievalResult(
                docs=docs,
                citations=citations,
                strategy=strategy,
                query_understanding=qu,
            )

        if strategy == "hybrid_rerank":
            # a. 向量化 + Milvus 原生 BM25 双路并发召回并经 RRF 融合
            dense_vector = await self.embedding_client.aembed_query(dense_query_text)
            candidates = await asyncio.to_thread(
                self.store.hybrid_search,
                dense_vector=dense_vector,
                bm25_text=bm25_query_text,
                category_filter=category_filter,
                top_k_per_route=top_k_per_route,
                limit=top_k_per_route,
                collection_name=self.collection_name,
            )

            if not candidates:
                return AdvancedRetrievalResult(
                    docs=[],
                    citations=[],
                    strategy=strategy,
                    query_understanding=qu,
                )

            # b. BGE-Reranker-v2-m3 交叉编码语义精排 Top-K
            reranked_docs = await self.reranker_client.rerank(
                query=dense_query_text,
                candidates=candidates,
                top_k=top_k,
            )

            # c. Lost in the Middle 首尾重排：[D1, D3, D5, D7, D9, D10, D8, D6, D4, D2]
            reordered_docs = lost_in_the_middle_reorder(reranked_docs)

            # d. 组装标准 1..K citations 快照
            citations = build_citation_items(reordered_docs)

            return AdvancedRetrievalResult(
                docs=reordered_docs,
                citations=citations,
                strategy=strategy,
                query_understanding=qu,
            )

        raise ValueError(f"未知检索策略: {strategy}")

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
        strategy: str = "hybrid_rerank",
        category_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """向后兼容的检索接口，默认采用 hybrid_rerank 策略，返回排布好的知识块列表。

        Args:
            query: 用户提问
            top_k: 返回候选知识条数，默认 3
            strategy: 检索策略，默认为 'hybrid_rerank'
            category_filter: 可选类目过滤
            **kwargs: 兼容旧版参数 (如 min_score, filter 等)

        Returns:
            检索并重排后的候选文档列表
        """
        result = await self.retrieve_with_strategy(
            query=query,
            strategy=strategy,
            category_filter=category_filter,
            top_k=top_k,
        )
        return result.docs[:top_k]

    def format_faq_hits(
        self, hits: List[Dict[str, Any]], keyword: str = ""
    ) -> str:
        """将检索到的知识块命中列表格式化为用户可读的 FAQ 规范文本。

        格式规范：
        1. 问：{questions的第一行或主问法}
           答：{answer}

        多条之间以空行 \\n\\n 分隔；若无命中则返回未找到提示。
        """
        if not hits:
            return f"未找到与【{keyword}】相关的常见问题解答。"

        lines: List[str] = []
        for idx, hit in enumerate(hits, 1):
            raw_q = hit.get("questions") or hit.get("question") or ""
            if isinstance(raw_q, list):
                q_main = str(raw_q[0]).strip() if raw_q else ""
            else:
                q_lines = [l.strip() for l in str(raw_q).splitlines() if l.strip()]
                q_main = q_lines[0] if q_lines else ""

            if not q_main:
                q_main = hit.get("section_path") or "常见问题"

            ans = str(hit.get("answer") or "").strip()
            lines.append(f"{idx}. 问：{q_main}\n   答：{ans}")

        return "\n\n".join(lines)

    async def retrieve_faq_text(
        self,
        keyword: str,
        top_k: int = 3,
        strategy: str = "hybrid_rerank",
        category_filter: Optional[str] = None,
    ) -> str:
        """根据关键词或问题直接检索并输出规范化 FAQ 文本。"""
        kw = str(keyword).strip() if keyword is not None else ""
        if not kw:
            return f"未找到与【{keyword}】相关的常见问题解答。"

        hits = await self.retrieve(
            query=kw,
            top_k=top_k,
            strategy=strategy,
            category_filter=category_filter,
        )
        return self.format_faq_hits(hits, keyword=keyword)
