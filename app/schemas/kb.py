from typing import List, Optional
from pydantic import BaseModel, Field


class ManualChunkCreateRequest(BaseModel):
    category: str = Field(..., min_length=1, description="业务分类，如'售后服务'、'物流运费'")
    questions: str = Field(..., min_length=1, description="标准问题或多问法，用换行或分号隔开")
    answer: str = Field(..., min_length=1, description="知识正文或解答")
    section_path: Optional[str] = Field(None, description="所属目录层级路径")
    content_type: str = Field(default="faq", description="知识类型：faq / policy / procedure / spec")
    is_key_clause: bool = Field(default=False, description="是否核心关键条款")
    sync_vector: bool = Field(default=True, description="是否立即执行 BGE-M3 密集向量化并同步写入 Milvus")


class KBSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="检索自然语言查询文本")
    top_k: int = Field(default=3, ge=1, le=10, description="返回近邻数")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="余弦相似度最低阈值")
    category: Optional[str] = Field(None, description="分类过滤")


class KBStatsResponse(BaseModel):
    total_chunks: int
    done_chunks: int
    pending_chunks: int
    failed_chunks: int
    milvus_count: int
    is_aligned: bool


class MaterialItem(BaseModel):
    filename: str
    file_path: str
    file_size_bytes: int
    modified_time: str
    estimated_chunks: int


class ChunkItem(BaseModel):
    id: int
    category: str
    questions: str
    answer: str
    section_path: Optional[str]
    content_type: str
    is_key_clause: bool
    vectorize_status: str
    vector_id: Optional[str]
    created_at: str


class ChunkListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[ChunkItem]


class KBSearchHit(BaseModel):
    id: int
    distance: float
    category: str
    questions: str
    answer: str
    section_path: Optional[str]
    content_type: str
    is_key_clause: bool


class KBSearchResponse(BaseModel):
    query: str
    top_k: int
    latency_ms: float
    total_hits: int
    hits: List[KBSearchHit]
    formatted_preview: str
