"""在线密集向量语义检索器服务。

支持基于 BGE-M3 语义向量的 Milvus 近邻召回、余弦相似度阈值过滤、
标准问法提取与 FAQ 问答规范化格式输出。
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.milvus_client import MilvusKnowledgeStore

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """RAG 知识库在线语义检索器。
    
    负责将用户自然语言问题转换为 1024 维 Dense 语义向量，
    并在 Milvus 集合中执行基于余弦相似度的 Top-K 检索与多行 FAQ 规范化提取。
    """

    def __init__(
        self,
        store: Optional[MilvusKnowledgeStore] = None,
        embedding_client: Optional[BGEEmbeddingClient] = None,
        min_score: float = 0.35,
        collection_name: str = "knowledge",
    ):
        self.store = store or MilvusKnowledgeStore()
        self.embedding_client = embedding_client or BGEEmbeddingClient()
        self.min_score = min_score
        self.collection_name = collection_name
        logger.info(
            f"KnowledgeRetriever 已初始化 (min_score={self.min_score}, collection={self.collection_name})"
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
        filter: Optional[str] = None,
        category: Optional[str] = None,
        category_filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """执行自然语言问题的 Dense 向量余弦相似度近邻检索。
        
        Args:
            query: 用户提问文本
            top_k: 期望返回的最大候选知识条数，默认 3
            min_score: 最低余弦相似度阈值（若未指定则使用类默认值 min_score）
            filter: 标量过滤表达式 (可选)
            category: 分类过滤 (可选，与 category_filter 含义相同)
            category_filter: 分类过滤表达式 (可选)
            **kwargs: 额外参数透传
            
        Returns:
            满足相似度阈值的知识命中列表（按相似度降序排序）
        """
        if not query or not str(query).strip():
            return []

        query_text = str(query).strip()
        effective_min_score = self.min_score if min_score is None else min_score
        effective_category = category_filter or category or kwargs.get("category_filter") or kwargs.get("category")

        # 异步向量化 query 文本
        query_vector = await self.embedding_client.aembed_query(query_text)

        # 异步线程执行 Milvus COSINE 近邻搜索
        hits = await asyncio.to_thread(
            self.store.search,
            query_vector=query_vector,
            top_k=top_k,
            min_score=effective_min_score,
            filter=filter,
            category_filter=effective_category,
            collection_name=self.collection_name,
        )

        # 确保严格满足相似度阈值并按得分降序
        valid_hits = [
            h for h in hits if float(h.get("distance", 0.0)) >= effective_min_score
        ]
        valid_hits.sort(key=lambda x: float(x.get("distance", 0.0)), reverse=True)
        return valid_hits[:top_k]

    def format_faq_hits(self, hits: List[Dict[str, Any]], keyword: str = "") -> str:
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

    async def retrieve_faq_text(self, keyword: str, top_k: int = 3) -> str:
        """根据关键字检索相关 FAQ 并直接输出格式化回答文本。
        
        Args:
            keyword: 检索关键词或问题
            top_k: 最大返回条数
            
        Returns:
            排版整洁的带序号问答文本，若无结果返回标准兜底提示。
        """
        kw = str(keyword).strip() if keyword is not None else ""
        if not kw:
            return f"未找到与【{keyword}】相关的常见问题解答。"

        hits = await self.retrieve(kw, top_k=top_k)
        return self.format_faq_hits(hits, keyword=keyword)

    def close(self) -> None:
        """关闭检索器并释放底层 Milvus 与模型资源。"""
        if hasattr(self, "store") and self.store is not None:
            if hasattr(self.store, "close"):
                try:
                    self.store.close()
                except Exception:
                    pass
