"""RAG 检索增强生成服务模块。"""

from app.services.rag.advanced_retriever import (
    AdvancedKnowledgeRetriever,
    AdvancedRetrievalResult,
)
from app.services.rag.dual_writer import KnowledgeDualWriter, compose_embedding_text
from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.milvus_client import MilvusKnowledgeStore
from app.services.rag.generator import RAGControlledGenerator, SelfCheckResult
from app.services.rag.miner import DialogueKnowledgeMiner
from app.services.rag.query_processor import QueryProcessor, QueryUnderstandingResult
from app.services.rag.reorder import build_citation_items, lost_in_the_middle_reorder
from app.services.rag.reranker import BGERerankerClient
from app.services.rag.retriever import KnowledgeRetriever
from app.services.rag.splitter import DocChunk, MarkdownStructureSplitter

__all__ = [
    "AdvancedKnowledgeRetriever",
    "AdvancedRetrievalResult",
    "BGEEmbeddingClient",
    "BGERerankerClient",
    "MilvusKnowledgeStore",
    "MarkdownStructureSplitter",
    "DocChunk",
    "KnowledgeDualWriter",
    "compose_embedding_text",
    "DialogueKnowledgeMiner",
    "KnowledgeRetriever",
    "QueryProcessor",
    "QueryUnderstandingResult",
    "lost_in_the_middle_reorder",
    "build_citation_items",
    "RAGControlledGenerator",
    "SelfCheckResult",
]



