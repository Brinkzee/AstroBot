"""RAG 检索增强生成服务模块。"""

from app.services.rag.dual_writer import KnowledgeDualWriter, compose_embedding_text
from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.milvus_client import MilvusKnowledgeStore
from app.services.rag.miner import DialogueKnowledgeMiner
from app.services.rag.splitter import MarkdownStructureSplitter, DocChunk

__all__ = [
    "BGEEmbeddingClient",
    "MilvusKnowledgeStore",
    "MarkdownStructureSplitter",
    "DocChunk",
    "KnowledgeDualWriter",
    "compose_embedding_text",
    "DialogueKnowledgeMiner",
]


