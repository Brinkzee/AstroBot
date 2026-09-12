import os
import shutil
import pytest
from pymilvus import DataType
from milvus_lite.server_manager import server_manager_instance

from app.services.rag.milvus_client import MilvusKnowledgeStore


@pytest.fixture
def temp_bm25_milvus_path(tmp_path):
    """为 BM25 与混合检索单测提供独立的临时 Milvus-Lite db 路径，并在测试后安全清理。"""
    db_file = tmp_path / "milvus_bm25_test" / "test_rag_bm25.db"
    db_file_str = str(db_file)
    yield db_file_str

    try:
        server_manager_instance.release_server(db_file_str)
    except Exception:
        pass
    parent_dir = db_file.parent
    if parent_dir.exists():
        try:
            shutil.rmtree(parent_dir, ignore_errors=True)
        except Exception:
            pass


def test_milvus_init_collection_schema_and_functions(temp_bm25_milvus_path):
    """验证集合初始化成功包含 chunk_text (jieba分词), sparse_vector (BM25函数), vector (1024维)。"""
    store = MilvusKnowledgeStore(uri=temp_bm25_milvus_path)
    try:
        store.init_collection(collection_name="knowledge", drop_existing=True)
        assert store.client.has_collection("knowledge")

        # 检查集合描述与字段
        desc = store.client.describe_collection("knowledge")
        fields = {f["name"]: f for f in desc.get("fields", [])}

        # 验证必需字段
        assert "id" in fields
        assert fields["id"]["type"] == DataType.INT64
        assert fields["id"].get("is_primary") is True

        assert "chunk_text" in fields
        assert fields["chunk_text"]["type"] == DataType.VARCHAR

        assert "sparse_vector" in fields
        assert fields["sparse_vector"]["type"] == DataType.SPARSE_FLOAT_VECTOR

        assert "vector" in fields
        assert fields["vector"]["type"] == DataType.FLOAT_VECTOR
        assert fields["vector"]["params"]["dim"] == 1024

        assert "category" in fields
        assert "section_path" in fields
        assert "questions" in fields
        assert "answer" in fields

        # 验证 Function 挂载
        functions = desc.get("functions", [])
        assert len(functions) >= 1
        bm25_fn = next((fn for fn in functions if fn.get("name") == "chunk_bm25"), None)
        assert bm25_fn is not None
        assert bm25_fn.get("input_field_names") == ["chunk_text"]
        assert bm25_fn.get("output_field_names") == ["sparse_vector"]
    finally:
        store.close(release_server=True)


def test_milvus_search_bm25_model_keyword(temp_bm25_milvus_path):
    """验证插入包含型号（如 PRO-X99）测试数据后，search_bm25('PRO-X99') 成功高分精准命中。"""
    store = MilvusKnowledgeStore(uri=temp_bm25_milvus_path)
    try:
        store.init_collection(collection_name="knowledge", drop_existing=True)

        records = [
            {
                "id": 101,
                "vector": [0.0] * 1024,
                "category": "退货政策",
                "section_path": "售后政策 > 退货政策",
                "questions": "PRO-X99 手表支持退货吗？",
                "answer": "星光 PRO-X99 旗舰款智能手表支持7天无理由退货，需保证包装配件齐全。",
                "chunk_text": "【类目】退货政策\n【标准问法】PRO-X99 手表支持退货吗？\n【解答】星光 PRO-X99 旗舰款智能手表支持7天无理由退货，需保证包装配件齐全。",
                "content_type": "faq",
                "is_key_clause": True,
            },
            {
                "id": 102,
                "vector": [0.0] * 1024,
                "category": "物流政策",
                "section_path": "物流政策 > 发货时效",
                "questions": "商品什么时候发货？",
                "answer": "全场普通商品下单后48小时内顺丰或中通包邮发货。",
                "chunk_text": "【类目】物流政策\n【标准问法】商品什么时候发货？\n【解答】全场普通商品下单后48小时内顺丰或中通包邮发货。",
                "content_type": "faq",
                "is_key_clause": False,
            },
            {
                "id": 103,
                "vector": [0.0] * 1024,
                "category": "退货政策",
                "section_path": "售后政策 > 换货流程",
                "questions": "PRO-X88 型号换货标准？",
                "answer": "PRO-X88 型号支持15天质量问题免费换货。",
                "chunk_text": "【类目】退货政策\n【标准问法】PRO-X88 型号换货标准？\n【解答】PRO-X88 型号支持15天质量问题免费换货。",
                "content_type": "faq",
                "is_key_clause": False,
            },
        ]
        inserted = store.upsert(records, collection_name="knowledge")
        assert inserted == 3

        # BM25 检索 PRO-X99
        hits = store.search_bm25("PRO-X99", top_k=5, collection_name="knowledge")
        assert len(hits) >= 1
        top_hit = hits[0]
        assert top_hit["id"] == 101
        assert "PRO-X99" in top_hit["chunk_text"]
        assert top_hit["distance"] > 0.0
        assert top_hit["category"] == "退货政策"
        assert top_hit["section_path"] == "售后政策 > 退货政策"
    finally:
        store.close(release_server=True)


def test_milvus_search_dense_single_route(temp_bm25_milvus_path):
    """验证 search_dense 单路向量检索成功。"""
    store = MilvusKnowledgeStore(uri=temp_bm25_milvus_path)
    try:
        store.init_collection(collection_name="knowledge", drop_existing=True)

        vec1 = [0.0] * 1024
        vec1[0] = 1.0
        vec2 = [0.0] * 1024
        vec2[1] = 1.0

        records = [
            {
                "id": 201,
                "vector": vec1,
                "category": "质保政策",
                "section_path": "售后 > 保修",
                "questions": "整机保修多久？",
                "answer": "主机保修1年，电池保修6个月。",
                "chunk_text": "【类目】质保政策\n【标准问法】整机保修多久？\n【解答】主机保修1年，电池保修6个月。",
                "content_type": "policy",
                "is_key_clause": True,
            },
            {
                "id": 202,
                "vector": vec2,
                "category": "电池续航",
                "section_path": "参数 > 续航",
                "questions": "电池能用几天？",
                "answer": "正常续航14天，重度使用7天。",
                "chunk_text": "【类目】电池续航\n【标准问法】电池能用几天？\n【解答】正常续航14天，重度使用7天。",
                "content_type": "faq",
                "is_key_clause": False,
            },
        ]
        store.upsert(records, collection_name="knowledge")

        # 向量匹配 vec1
        hits = store.search_dense(vec1, top_k=2, collection_name="knowledge")
        assert len(hits) >= 1
        assert hits[0]["id"] == 201
        assert hits[0]["distance"] > 0.99
        assert hits[0]["category"] == "质保政策"
    finally:
        store.close(release_server=True)


def test_milvus_hybrid_search_with_rrf_and_filter(temp_bm25_milvus_path):
    """验证 hybrid_search 基于 RRFRanker(k=60) 融合输出，且 category_filter 标量过滤生效。"""
    store = MilvusKnowledgeStore(uri=temp_bm25_milvus_path)
    try:
        store.init_collection(collection_name="knowledge", drop_existing=True)

        # 准备语义正交向量
        vec_target = [0.0] * 1024
        vec_target[0] = 1.0

        vec_other = [0.0] * 1024
        vec_other[1] = 1.0

        records = [
            # 记录 1: 类别为退货政策，既有 PRO-X99 关键词，又有 vec_target 语义
            {
                "id": 301,
                "vector": vec_target,
                "category": "退货政策",
                "section_path": "售后 > 退货",
                "questions": "PRO-X99 旗舰款手表退货规则",
                "answer": "PRO-X99 支持7天无理由退货，包装完整配件齐全即可退货。",
                "chunk_text": "【类目】退货政策\n【标准问法】PRO-X99 旗舰款手表退货规则\n【解答】PRO-X99 支持7天无理由退货，包装完整配件齐全即可退货。",
                "content_type": "policy",
                "is_key_clause": True,
            },
            # 记录 2: 类别为维修服务，也有 PRO-X99 关键词，但向量不匹配
            {
                "id": 302,
                "vector": vec_other,
                "category": "维修服务",
                "section_path": "售后 > 维修",
                "questions": "PRO-X99 旗舰款手表进水能修吗",
                "answer": "PRO-X99 进水属于非保修范围，需自费更换主板。",
                "chunk_text": "【类目】维修服务\n【标准问法】PRO-X99 旗舰款手表进水能修吗\n【解答】PRO-X99 进水属于非保修范围，需自费更换主板。",
                "content_type": "policy",
                "is_key_clause": False,
            },
            # 记录 3: 类别为退货政策，没有 PRO-X99 关键词，但向量是 vec_target (语义匹配普通退货)
            {
                "id": 303,
                "vector": vec_target,
                "category": "退货政策",
                "section_path": "售后 > 退货",
                "questions": "通用商品退货流程",
                "answer": "全场通用商品支持无理由退货，申请后商家24小时内审核。",
                "chunk_text": "【类目】退货政策\n【标准问法】通用商品退货流程\n【解答】全场通用商品支持无理由退货，申请后商家24小时内审核。",
                "content_type": "policy",
                "is_key_clause": False,
            },
        ]
        store.upsert(records, collection_name="knowledge")

        # 1. 混合检索无过滤：PRO-X99 BM25 + vec_target Dense
        hybrid_hits = store.hybrid_search(
            dense_vector=vec_target,
            bm25_text="PRO-X99",
            category_filter=None,
            top_k_per_route=50,
            limit=50,
            collection_name="knowledge",
        )
        assert len(hybrid_hits) >= 2
        # id 301 既命中 Dense 又命中 BM25，RRF 融合后必排在第 1
        assert hybrid_hits[0]["id"] == 301
        assert hybrid_hits[0]["category"] == "退货政策"
        assert hybrid_hits[0]["chunk_text"] != ""
        # 验证返回结构
        for hit in hybrid_hits:
            assert "id" in hit
            assert "distance" in hit
            assert "entity" in hit
            assert "vector" not in hit  # 避免返回庞大向量

        # 2. 混合检索带标量过滤：category_filter="退货政策"
        filtered_hits = store.hybrid_search(
            dense_vector=vec_target,
            bm25_text="PRO-X99",
            category_filter="退货政策",
            top_k_per_route=50,
            limit=50,
            collection_name="knowledge",
        )
        assert len(filtered_hits) >= 1
        # 所有返回结果的 category 必须全部是 "退货政策"
        for hit in filtered_hits:
            assert hit["category"] == "退货政策"
        hit_ids = [h["id"] for h in filtered_hits]
        assert 301 in hit_ids
        assert 302 not in hit_ids  # 维修服务必须被过滤掉

        # 3. 混合检索带标量过滤：category_filter="维修服务"
        repair_hits = store.hybrid_search(
            dense_vector=vec_target,
            bm25_text="PRO-X99",
            category_filter="维修服务",
            top_k_per_route=50,
            limit=50,
            collection_name="knowledge",
        )
        assert len(repair_hits) == 1
        assert repair_hits[0]["id"] == 302
        assert repair_hits[0]["category"] == "维修服务"
    finally:
        store.close(release_server=True)


def test_milvus_search_backward_compatibility(temp_bm25_milvus_path):
    """验证现有的 search 方法保持向前兼容，行为与 search_dense 对齐。"""
    store = MilvusKnowledgeStore(uri=temp_bm25_milvus_path)
    try:
        store.init_collection(collection_name="knowledge", drop_existing=True)

        vec = [0.0] * 1024
        vec[0] = 1.0

        records = [
            {
                "id": 401,
                "vector": vec,
                "category": "发票问题",
                "questions": "可以开发票吗？",
                "answer": "支持开具电子普通发票。",
                "chunk_text": "【类目】发票问题\n【标准问法】可以开发票吗？\n【解答】支持开具电子普通发票。",
            }
        ]
        store.upsert(records, collection_name="knowledge")

        # 使用旧接口 search
        old_hits = store.search(vec, top_k=3, min_score=0.0, collection_name="knowledge")
        assert len(old_hits) == 1
        assert old_hits[0]["id"] == 401
        assert old_hits[0]["category"] == "发票问题"
        assert old_hits[0]["distance"] > 0.99
    finally:
        store.close(release_server=True)


def test_compose_chunk_text():
    """测试问答文本规范组装函数。"""
    from scripts.reindex_ch04_knowledge import compose_chunk_text
    text = compose_chunk_text("售后规则", "退货时效？", "7天无理由退货。")
    assert text == "【类目】售后规则\n【标准问法】退货时效？\n【解答】7天无理由退货。"


@pytest.mark.asyncio
async def test_reindex_knowledge_chunks_flow(temp_bm25_milvus_path):
    """测试全量数据重构灌库流程：Mock Session 读取数据并入库 Milvus。"""
    from unittest.mock import AsyncMock, MagicMock
    from app.models.knowledge import KnowledgeChunk
    from app.services.rag.embedding import BGEEmbeddingClient
    from scripts.reindex_ch04_knowledge import reindex_knowledge_chunks

    # 准备假数据
    fake_chunk_1 = KnowledgeChunk(
        id=501,
        category="维修保障",
        questions="PRO-X99 屏幕碎了怎么修？",
        answer="可联系售后寄修换屏。",
        section_path="售后 > 维修",
        content_type="policy",
        is_key_clause=True,
    )
    fake_chunk_2 = KnowledgeChunk(
        id=502,
        category="配送时效",
        questions="顺丰多久能到？",
        answer="核心城市次日达。",
        section_path="物流 > 配送",
        content_type="faq",
        is_key_clause=False,
    )

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [fake_chunk_1, fake_chunk_2]
    mock_session.execute.return_value = mock_result

    embedding_client = BGEEmbeddingClient(token=None)  # Mock 模式
    store = MilvusKnowledgeStore(uri=temp_bm25_milvus_path)

    try:
        stats = await reindex_knowledge_chunks(
            session=mock_session,
            store=store,
            embedding_client=embedding_client,
            collection_name="knowledge",
            batch_size=1,
        )

        assert stats["total_chunks"] == 2
        assert stats["indexed_chunks"] == 2
        assert store.count("knowledge") == 2

        # 验证 BM25 能够检索出来
        bm25_hits = store.search_bm25("PRO-X99", top_k=5)
        assert len(bm25_hits) >= 1
        assert bm25_hits[0]["id"] == 501
        assert "PRO-X99" in bm25_hits[0]["chunk_text"]
    finally:
        store.close(release_server=True)

