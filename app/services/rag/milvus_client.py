import logging
import os
from typing import Dict, List, Optional

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import settings

logger = logging.getLogger(__name__)


class MilvusKnowledgeStore:
    """基于 Milvus-Lite / Milvus 的 RAG 知识库向量与原生 BM25 混合存储管理器。
    
    支持 1024 维 Dense 语义向量与原生 BM25 稀疏向量的混合存储、按 MySQL chunk ID 幂等 Upsert、
    COSINE 相似度、BM25 稀疏检索、RRFRanker(k=60) 倒数排名融合检索、标量过滤及安全生命周期管理。
    """

    def __init__(self, uri: Optional[str] = None):
        if uri is None:
            uri = (
                getattr(settings, "MILVUS_URI", None)
                or getattr(settings, "milvus_uri", None)
                or "./data/milvus/astro_bot.db"
            )
        self.uri = uri

        # 若为本地文件路径，自动确保父目录存在
        if not (self.uri.startswith("http://") or self.uri.startswith("https://") or self.uri.startswith("tcp://")):
            abs_path = os.path.abspath(self.uri)
            parent_dir = os.path.dirname(abs_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        self.client: Optional[MilvusClient] = MilvusClient(uri=self.uri)
        logger.info(f"MilvusKnowledgeStore 已初始化，连接至: {self.uri}")

    def init_collection(self, collection_name: str = "knowledge", drop_existing: bool = False):
        """初始化知识库向量与 BM25 稀疏集合。
        
        若集合已存在且 drop_existing=False 则保持不变；
        若 drop_existing=True 则强制删除并重新创建。
        
        集合规范：
        - auto_id = False, enable_dynamic_field = True
        - id: INT64, primary=True (与 MySQL knowledge_chunks.id 对齐)
        - chunk_text: VARCHAR(65535), enable_analyzer=True, analyzer_params={"type": "jieba"}
        - sparse_vector: SPARSE_FLOAT_VECTOR (BM25 函数自动输出)
        - vector: FLOAT_VECTOR(dim=1024), metric_type=COSINE
        - category, section_path, questions, answer, content_type, is_key_clause 等标量元数据
        """
        if drop_existing and self.client.has_collection(collection_name):
            logger.info(f"drop_existing=True，正在删除已有集合: {collection_name}")
            self.client.drop_collection(collection_name)

        if not self.client.has_collection(collection_name):
            logger.info(f"正在创建 Milvus 2.5 混合检索集合: {collection_name} (dim=1024, BM25 jieba, COSINE)")
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(
                field_name="chunk_text",
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,
                analyzer_params={"type": "jieba"},
                nullable=True,
            )
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=128, nullable=True)
            schema.add_field(field_name="section_path", datatype=DataType.VARCHAR, max_length=512, nullable=True)
            schema.add_field(field_name="questions", datatype=DataType.VARCHAR, max_length=2048, nullable=True)
            schema.add_field(field_name="answer", datatype=DataType.VARCHAR, max_length=8192, nullable=True)
            schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=32, nullable=True)
            schema.add_field(field_name="is_key_clause", datatype=DataType.BOOL, nullable=True)

            bm25_fn = Function(
                name="chunk_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["chunk_text"],
                output_field_names=["sparse_vector"],
            )
            schema.add_function(bm25_fn)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
            # 兼容处理：在 milvus-lite 本地嵌入式模式下，sparse_vector 为二进制存储，
            # 若创建 AUTOINDEX 会触发 milvus_lite segment 重新从磁盘加载时的 FixedSizeList 类型校验崩溃；
            # 仅在非 sqlite 本地文件（独立 Milvus 服务）时添加 sparse_vector 索引。
            if not (self.uri and (self.uri.endswith(".db") or "./" in self.uri or "\\" in self.uri)):
                try:
                    index_params.add_index(field_name="sparse_vector", index_type="AUTOINDEX", metric_type="BM25")
                except Exception:
                    pass

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )

    def upsert(self, records: List[Dict], collection_name: str = "knowledge") -> int:
        """批量插入或按主键 ID 覆盖记录。
        
        每个 record 包含：
        id, vector, chunk_text, category, questions, answer, section_path, content_type, is_key_clause 等字段。
        若未提供 chunk_text，将自动基于 questions 和 answer 合成。
        
        返回写入条数。
        """
        if not records:
            return 0

        if not self.client.has_collection(collection_name):
            self.init_collection(collection_name)

        clean_records = []
        for r in records:
            rec = dict(r)
            if "chunk_text" not in rec or rec["chunk_text"] is None:
                q = rec.get("questions") or ""
                a = rec.get("answer") or ""
                cat = rec.get("category") or ""
                rec["chunk_text"] = f"【类目】{cat}\n【标准问法】{q}\n【解答】{a}".strip()
            clean_records.append(rec)

        res = self.client.upsert(collection_name=collection_name, data=clean_records)
        return int(res.get("upsert_count", len(clean_records)))

    @staticmethod
    def _format_hits(raw_hits: List[Dict], min_score: Optional[float] = None) -> List[Dict]:
        """格式化检索命中的原始结果，展开 entity 并去除大体积向量。"""
        formatted: List[Dict] = []
        for raw_hit in raw_hits:
            dist = float(raw_hit.get("distance", 0.0))
            if min_score is not None and dist < min_score:
                continue

            entity = raw_hit.get("entity") or {}
            item = dict(entity)
            item.pop("vector", None)
            item.pop("sparse_vector", None)
            item["id"] = raw_hit.get("id")
            item["distance"] = dist
            item["entity"] = entity
            formatted.append(item)
        return formatted

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 50,
        min_score: float = 0.0,
        filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        collection_name: str = "knowledge",
    ) -> List[Dict]:
        """单路 Dense 向量检索（COSINE 相似度）。
        
        专供四策略对比评测及密集语义召回。
        """
        if not self.client or not self.client.has_collection(collection_name):
            return []

        if not query_vector:
            return []

        try:
            self.client.load_collection(collection_name)
        except Exception as e:
            logger.debug(f"load_collection({collection_name}) 提示: {e}")

        expr_parts = []
        if filter:
            expr_parts.append(f"({filter})")
        if category_filter:
            expr_parts.append(f'(category == "{category_filter}")')
        final_filter = " and ".join(expr_parts) if expr_parts else ""

        raw_results = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="vector",
            limit=top_k,
            filter=final_filter,
            output_fields=["*"],
        )

        if not raw_results or len(raw_results) == 0:
            return []

        hits = self._format_hits(raw_results[0], min_score=min_score)
        hits.sort(key=lambda x: x["distance"], reverse=True)
        return hits[:top_k]

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        min_score: float = 0.0,
        filter: Optional[str] = None,
        collection_name: str = "knowledge",
    ) -> List[Dict]:
        """基于 COSINE 相似度的 Top-K 向量检索（向后兼容接口）。"""
        return self.search_dense(
            query_vector=query_vector,
            top_k=top_k,
            min_score=min_score,
            filter=filter,
            collection_name=collection_name,
        )

    def search_bm25(
        self,
        query_text: str,
        top_k: int = 50,
        filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        collection_name: str = "knowledge",
    ) -> List[Dict]:
        """单路 Milvus 原生 BM25 稀疏检索。
        
        基于 jieba 词元切分与 BM25 打分，专供四策略对比评测及精准型号/专有名词召回。
        """
        if not self.client or not self.client.has_collection(collection_name):
            return []

        if not query_text or not query_text.strip():
            return []

        try:
            self.client.load_collection(collection_name)
        except Exception as e:
            logger.debug(f"load_collection({collection_name}) 提示: {e}")

        expr_parts = []
        if filter:
            expr_parts.append(f"({filter})")
        if category_filter:
            expr_parts.append(f'(category == "{category_filter}")')
        final_filter = " and ".join(expr_parts) if expr_parts else ""

        raw_results = self.client.search(
            collection_name=collection_name,
            data=[query_text],
            anns_field="sparse_vector",
            limit=top_k,
            filter=final_filter,
            output_fields=["*"],
        )

        if not raw_results or len(raw_results) == 0:
            return []

        hits = self._format_hits(raw_results[0])
        hits.sort(key=lambda x: x["distance"], reverse=True)
        return hits[:top_k]

    def hybrid_search(
        self,
        dense_vector: List[float],
        bm25_text: str,
        category_filter: Optional[str] = None,
        top_k_per_route: int = 50,
        limit: int = 50,
        collection_name: str = "knowledge",
    ) -> List[Dict]:
        """基于 Dense + 原生 BM25 的双路召回与 RRFRanker(k=60) 混合检索。
        
        Args:
            dense_vector: 1024 维 Dense 语义向量
            bm25_text: 待检索的关键词或文本
            category_filter: 可选的类目标量过滤条件
            top_k_per_route: 单路检索候选条数，默认 50
            limit: RRF 融合截取总数，默认 50
            collection_name: 集合名称，默认 "knowledge"
        
        Returns:
            融合排序后的候选命中文档列表
        """
        if not self.client or not self.client.has_collection(collection_name):
            return []

        if not dense_vector and not bm25_text:
            return []

        try:
            self.client.load_collection(collection_name)
        except Exception as e:
            logger.debug(f"load_collection({collection_name}) 提示: {e}")

        filter_expr = f'category == "{category_filter}"' if category_filter else None

        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=top_k_per_route,
            expr=filter_expr,
        )

        bm25_req = AnnSearchRequest(
            data=[bm25_text],
            anns_field="sparse_vector",
            param={"metric_type": "BM25"},
            limit=top_k_per_route,
            expr=filter_expr,
        )

        raw_results = self.client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, bm25_req],
            ranker=RRFRanker(k=60),
            limit=limit,
            output_fields=["*"],
        )

        if not raw_results or len(raw_results) == 0:
            return []

        hits = self._format_hits(raw_results[0])
        return hits[:limit]

    def count(self, collection_name: str = "knowledge") -> int:
        """返回当前集合中的记录总数。"""
        if not self.client or not self.client.has_collection(collection_name):
            return 0

        try:
            self.client.load_collection(collection_name)
        except Exception:
            pass

        res = self.client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        if res and len(res) > 0:
            return int(res[0].get("count(*)", 0))
        return 0

    def close(self, release_server: bool = False):
        """关闭客户端连接并按需释放底层 Milvus-Lite 本地服务句柄。"""
        if self.client is not None:
            try:
                self.client.close()
            except Exception as e:
                logger.debug(f"关闭 MilvusClient 异常: {e}")
            self.client = None

        if release_server and self.uri and self.uri.endswith(".db"):
            try:
                from milvus_lite.server_manager import server_manager_instance
                server_manager_instance.release_server(self.uri)
            except Exception as e:
                logger.debug(f"释放 milvus_lite 本地 server 异常: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
