# Task 3 Brief: BGE-M3 向量嵌入客户端与 Milvus-Lite 向量库管理器

## 1. 任务目标
实现 BGE-M3 Dense 向量嵌入客户端与基于 Milvus-Lite 的向量存储管理器，支撑 1024 维语义向量的批量生成、存储、按 ID 幂等 Upsert 与基于余弦相似度的 Top-K 检索。

## 2. 涉及文件
- Create: `app/services/rag/embedding.py`
- Create: `app/services/rag/milvus_client.py`
- Modify: `app/config.py` (新增 `MILVUS_URI: str = "./data/milvus/astro_bot.db"`)
- Test: `tests/test_rag_embedding_milvus.py`

## 3. 全局约束与技术规范
- 嵌入模型：`BAAI/bge-m3`，固定 1024 维 Dense 浮点向量；
- 嵌入驱动：优先读取配置 `settings.HUGGINGFACE_TOKEN` 初始化 `huggingface_hub.InferenceClient`，调用 `client.feature_extraction(text, model="BAAI/bge-m3")`；
- 向量数据库：`pymilvus.MilvusClient`，支持本地文件/目录 URI（自动确保父目录存在）；
- 集合名称：`knowledge`；
- 相似度度量：`COSINE`，主键 `id` 采用 `INT64` (`auto_id=False`)，数值严格与 MySQL chunk `id` 保持 1:1 对齐；
- 遵循 TDD：先编写失败单测 -> 验证失败 -> 编写实现 -> 验证全部通过。

## 4. 关键接口定义

### 4.1 `BGEEmbeddingClient` (`app/services/rag/embedding.py`)
- `__init__(token: Optional[str] = None, model: str = "BAAI/bge-m3")`
- `embed_query(text: str) -> List[float]`: 单句向量化，返回 1024 维 float 列表；
- `embed_documents(texts: List[str], batch_size: int = 16) -> List[List[float]]`: 批量文本向量化；
- `async aembed_query(text: str) -> List[float]` / `async aembed_documents(texts: List[str]) -> List[List[float]]`: 异步包装；
- 容错处理：网络请求遇偶发网络抖动时内置重试机制，若未配置 Token 或离线单测模式时提供优雅 Mock/兜底模式。

### 4.2 `MilvusKnowledgeStore` (`app/services/rag/milvus_client.py`)
- `__init__(uri: Optional[str] = None)`: 默认取 `settings.MILVUS_URI`，自动 `os.makedirs(os.path.dirname(uri), exist_ok=True)`；
- `init_collection(collection_name: str = "knowledge", drop_existing: bool = False)`:
  - 若集合不存在或 `drop_existing=True`，创建集合：
    - `dimension = 1024`
    - `primary_field_name = "id"`, `id_type = "int"`, `auto_id = False`
    - `vector_field_name = "vector"`
    - `metric_type = "COSINE"`
    - `enable_dynamic_field = True`
- `upsert(records: List[Dict], collection_name: str = "knowledge") -> int`:
  - 批量插入或按主键覆盖，每个 record 包含 `id`, `vector`, `category`, `questions`, `answer`, `chunk_text`, `content_type`, `is_key_clause`；
  - 返回写入条数；
- `search(query_vector: List[float], top_k: int = 3, min_score: float = 0.0, filter: Optional[str] = None, collection_name: str = "knowledge") -> List[Dict]`:
  - 返回近邻列表，每个命中包含 `id`, `distance` (相似度), 以及 payload 字段；按相似度降序过滤 `distance >= min_score`；
- `count(collection_name: str = "knowledge") -> int`: 返回当前集合记录数；
- `close()`: 关闭客户端。

## 5. 验收标准
- `pytest tests/test_rag_embedding_milvus.py -v` 100% 通过；
- 测试必须包含：BGE-M3 客户端向量维度校验（1024维）、Milvus-Lite 本地集合生命周期（创建、Upsert 幂等覆写、余弦近邻检索匹配、清理）；
- 全量回归 `pytest` 保持 100% 通过。
