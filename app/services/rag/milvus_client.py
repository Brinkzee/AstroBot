import logging
import os
from typing import Dict, List, Optional

from pymilvus import MilvusClient

from app.config import settings

logger = logging.getLogger(__name__)


class MilvusKnowledgeStore:
    """基于 Milvus-Lite / Milvus 的 RAG 知识库向量存储管理器。
    
    支持 1024 维 Dense 语义向量的存储、按 MySQL chunk ID 幂等 Upsert、
    COSINE 相似度 Top-K 检索、标量过滤及安全生命周期管理。
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
        """初始化知识库向量集合。
        
        若集合已存在且 drop_existing=False 则保持不变；
        若 drop_existing=True 则强制删除并重新创建。
        
        集合规范：
        - dimension = 1024
        - primary_field_name = 'id', id_type = 'int', auto_id = False
        - vector_field_name = 'vector'
        - metric_type = 'COSINE'
        - enable_dynamic_field = True
        """
        if drop_existing and self.client.has_collection(collection_name):
            logger.info(f"drop_existing=True，正在删除已有集合: {collection_name}")
            self.client.drop_collection(collection_name)

        if not self.client.has_collection(collection_name):
            logger.info(f"正在创建集合: {collection_name} (dim=1024, metric=COSINE, id=INT64)")
            self.client.create_collection(
                collection_name=collection_name,
                dimension=1024,
                primary_field_name="id",
                id_type="int",
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
                enable_dynamic_field=True,
            )

    def upsert(self, records: List[Dict], collection_name: str = "knowledge") -> int:
        """批量插入或按主键 ID 覆盖记录。
        
        每个 record 预期包含：
        id, vector, category, questions, answer, chunk_text, content_type, is_key_clause 等字段。
        
        返回写入条数。
        """
        if not records:
            return 0

        if not self.client.has_collection(collection_name):
            self.init_collection(collection_name)

        res = self.client.upsert(collection_name=collection_name, data=records)
        return int(res.get("upsert_count", len(records)))

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        min_score: float = 0.0,
        filter: Optional[str] = None,
        collection_name: str = "knowledge",
    ) -> List[Dict]:
        """基于 COSINE 相似度的 Top-K 向量近邻检索。
        
        返回近邻列表，每个命中包含 id, distance, 以及 payload 字段。
        按相似度降序排序，过滤 distance >= min_score 的命中项。
        """
        if not self.client or not self.client.has_collection(collection_name):
            return []

        if not query_vector:
            return []

        # 确保集合处于已加载状态 (防止 released 状态导致 code=101 错误)
        try:
            self.client.load_collection(collection_name)
        except Exception as e:
            logger.debug(f"load_collection({collection_name}) 提示: {e}")

        raw_results = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
            filter=filter if filter else "",
            output_fields=["*"],
        )

        if not raw_results or len(raw_results) == 0:
            return []

        hits: List[Dict] = []
        for raw_hit in raw_results[0]:
            dist = float(raw_hit.get("distance", 0.0))
            if dist < min_score:
                continue

            entity = raw_hit.get("entity") or {}
            # 组装返回结构：展开 payload，排除庞大的原始 vector 字段，保留 entity 引用以最大化调用方兼容性
            item = dict(entity)
            item.pop("vector", None)
            item["id"] = raw_hit.get("id")
            item["distance"] = dist
            item["entity"] = entity
            hits.append(item)

        # 确保按相似度由高到低排序并截取 top_k
        hits.sort(key=lambda x: x["distance"], reverse=True)
        return hits[:top_k]

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
