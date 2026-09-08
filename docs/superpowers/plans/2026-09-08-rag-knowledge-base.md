# RAG 知识库离线构建与向量语义检索升级实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 采用 Superpowers 模式为客服系统构建 RAG 知识库，完成结构感知文档切分、对话挖掘问答暂存去重、MySQL+Milvus 幂等双写与断点恢复，并将 `query_faq` 工具从关键词查表无缝升级为 BGE-M3 向量语义检索。

**Architecture:** 采用分层管道与领域服务架构。文档切分引擎纯内存解析 Markdown 标题栈与断句对齐重叠；对话挖掘管道分批提炼问答落暂存表并聚类去重；双写管理器编排 MySQL 权威源与 Milvus-Lite 向量集合并提供断点拾遗自愈；在线检索服务基于 BGE-M3 密集向量余弦近邻搜索无损替换 `query_faq` 内部实现。

**Tech Stack:** FastAPI, SQLAlchemy 2.0 Async, MySQL 8.0, Milvus-Lite (`pymilvus`), BGE-M3 (`huggingface_hub.InferenceClient`), LangChain `@tool`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-rag-knowledge-base-design.md`

## Global Constraints

- 嵌入模型固定为 `BAAI/bge-m3` (1024 维 Dense 语义向量)，通过 Hugging Face InferenceClient 调用，Token 读取自 `.env`。
- 向量库固定为 `Milvus-Lite`，集合名称 `knowledge`，距离度量方式 `COSINE`，主键数值与 MySQL `knowledge_chunks.id` 保持 1:1 对齐。
- MySQL 原文权威源表为 `knowledge_chunks` 与 `qa_extraction_staging`，严格遵循 `sql/ch03-ddl.sql` 结构定义。
- `query_faq` 工具的入参出参签名及 Docstring 契约保持 100% 绝对不变。
- 本章仅实现 Dense 向量单路检索，不做关键词召回、混合检索与重排。
- 过程留痕：在 `dev-notes/ch03.md` 中记录每阶段的关键原话、产出、纠偏和翻车返工，严禁收尾时一次性补齐。

---

### Task 1: ORM 数据模型与 DDL 初始化基础设施

**Files:**
- Create: `app/models/knowledge.py`
- Create: `app/models/staging.py`
- Modify: `app/models/__init__.py`
- Create: `scripts/init_ch03_db.py`
- Test: `tests/test_rag_models.py`

**Interfaces:**
- Consumes: `app.db.session.Base`, `sql/ch03-ddl.sql`
- Produces: `KnowledgeChunk`, `QAExtractionStaging` SQLAlchemy ORM 实体，支持 `id`, `category`, `questions`, `answer`, `section_path`, `content_type`, `is_key_clause`, `prev_chunk_id`, `next_chunk_id`, `vector_id`, `vectorize_status`, `batch_no`, `source_ref`, `question`, `answer`, `status`

- [ ] **Step 1: 编写数据模型失败测试**

```python
# tests/test_rag_models.py
import pytest
from sqlalchemy import select
from app.models.knowledge import KnowledgeChunk
from app.models.staging import QAExtractionStaging
from app.db.session import AsyncSessionLocal, Base, engine

@pytest.mark.asyncio
async def test_knowledge_chunk_and_staging_crud():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        chunk = KnowledgeChunk(
            category="售后政策 > 退货规范",
            questions="七天无理由退货规则\n退货条件是什么",
            answer="商品在签收7天内且保持完好可申请无理由退货。",
            section_path="服务指南 > 售后 > 退货",
            content_type="policy",
            is_key_clause=1,
            vectorize_status="pending"
        )
        session.add(chunk)
        staging = QAExtractionStaging(
            batch_no="BATCH_TEST_001",
            source_ref="conv_1001",
            question="邮费谁出？",
            answer="非质量问题退货由买家承担邮费。",
            status="extracted"
        )
        session.add(staging)
        await session.commit()
        await session.refresh(chunk)
        await session.refresh(staging)

        assert chunk.id is not None
        assert chunk.vectorize_status == "pending"
        assert staging.id is not None
        assert staging.status == "extracted"
```

- [ ] **Step 2: 运行测试验证失败**

运行：`pytest tests/test_rag_models.py -v`
预期：FAIL (ModuleNotFoundError: No module named 'app.models.knowledge')

- [ ] **Step 3: 编写数据模型与初始化脚本**

创建 `app/models/knowledge.py`，实现 `KnowledgeChunk`，兼顾 MySQL `BIGINT UNSIGNED` 与 SQLite 兼容性：
```python
from datetime import datetime
from typing import Optional
from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.mysql import BIGINT as MYSQL_BIGINT
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.session import Base

BigIntID = BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite")

class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BigIntID, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    content_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    prev_chunk_id: Mapped[Optional[int]] = mapped_column(BigIntID, ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True)
    next_chunk_id: Mapped[Optional[int]] = mapped_column(BigIntID, ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), nullable=True)
    vector_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
```
创建 `app/models/staging.py` 实现 `QAExtractionStaging`。更新 `app/models/__init__.py` 导出。
创建 `scripts/init_ch03_db.py` 执行 `sql/ch03-ddl.sql` 建表。

- [ ] **Step 4: 运行测试验证通过**

运行：`pytest tests/test_rag_models.py -v`
预期：PASS

- [ ] **Step 5: 提交代码与记录过程**

```bash
git add app/models/ tests/test_rag_models.py scripts/init_ch03_db.py
git commit -m "feat(models): add knowledge_chunks and qa_extraction_staging models"
```

---

### Task 2: 结构感知文档切分引擎

**Files:**
- Create: `app/services/rag/splitter.py`
- Test: `tests/test_rag_splitter.py`

**Interfaces:**
- Consumes: Markdown 文本或文件路径
- Produces: `DocChunk` 数据类（包含 `category`, `questions`, `answer`, `section_path`, `content_type`, `is_key_clause`, `order_index`），以及 `split_markdown(text: str) -> List[DocChunk]` 函数

- [ ] **Step 1: 编写切分引擎失败测试**

```python
# tests/test_rag_splitter.py
import pytest
from app.services.rag.splitter import MarkdownStructureSplitter

def test_markdown_hierarchy_splitting():
    md_content = """# 商城服务指南
## 退货与退款政策
### 七天无理由退货
在签收商品7天内，在不影响二次销售的前提下可申请无理由退货。
【特别说明】已激活的电子数码产品不支持无理由退货。

### 退货运费说明
| 责任方 | 运费承担 | 说明 |
|---|---|---|
| 买家原因 | 买家自理 | 个人喜好退换 |
| 质量问题 | 商家全额承担 | 需提供检测报告 |
| 偏远地区 | 双方协商 | 新疆西藏等特殊区域 |
"""
    splitter = MarkdownStructureSplitter(chunk_size=150, overlap_size=40)
    chunks = splitter.split_text(md_content)

    assert len(chunks) >= 2
    # 验证标题感知
    assert "退货与退款政策" in chunks[0].category
    assert "七天无理由退货" in chunks[0].questions
    # 验证特别说明识别为关键条款
    assert chunks[0].is_key_clause is True

def test_sentence_boundary_overlap_no_half_sentences():
    text = "# 规则\n## 细则\n第一句话非常明确完整。第二句话说明了退换货的截止时间与包裹寄回方式。第三句话强调了快递单号上传的必要性。"
    splitter = MarkdownStructureSplitter(chunk_size=45, overlap_size=20)
    chunks = splitter.split_text(text)

    # 验证重叠区域不留半截话，切分点必在句末标点（。！？）之后
    for chunk in chunks:
        assert not chunk.answer.startswith("化说明了")
        assert not chunk.answer.startswith("截止时间")

def test_table_splitting_preserves_header():
    table_md = """# 运费明细
## 资费标准
| 品类 | 基础运费 | 超重费 | 偏远补贴 | 备注说明 |
|---|---|---|---|---|
| 服饰鞋包 | 8元 | 2元/kg | 5元 | 满99包邮 |
| 家用电器 | 15元 | 5元/kg | 15元 | 大件送货上门 |
| 食品生鲜 | 12元 | 4元/kg | 不发货 | 冷链配送 |
| 电子数码 | 10元 | 3元/kg | 10元 | 顺丰特快 |
"""
    splitter = MarkdownStructureSplitter(chunk_size=90, overlap_size=0)
    chunks = splitter.split_text(table_md)

    assert len(chunks) > 1
    # 验证每一个切出来的表格 chunk 都完整保留了表头
    for chunk in chunks:
        assert "| 品类 | 基础运费 |" in chunk.answer
        assert "|---|---|" in chunk.answer
```

- [ ] **Step 2: 运行测试验证失败**

运行：`pytest tests/test_rag_splitter.py -v`
预期：FAIL (ModuleNotFoundError: No module named 'app.services.rag.splitter')

- [ ] **Step 3: 编写结构感知切分引擎实现**

在 `app/services/rag/splitter.py` 中实现：
1. `DocChunk` 数据类；
2. 栈式标题解析器维护当前祖先层级与小节标题；
3. `align_to_sentence_boundary(text, target_pos, window)`：在句号、感叹号、问号、分号处截断；
4. `split_table(table_lines, chunk_size)`：提取 Header + Divider，按行拆分数据行，并将 Header 广播复制到各 chunk 头部；
5. 关键条款匹配规则检测（`【重要提示】`、`【特别说明】`、`不支持` 等）。

- [ ] **Step 4: 运行测试验证通过**

运行：`pytest tests/test_rag_splitter.py -v`
预期：PASS

- [ ] **Step 5: 提交代码与记录过程**

```bash
git add app/services/rag/splitter.py tests/test_rag_splitter.py
git commit -m "feat(rag): implement structure-aware markdown splitter with sentence-aligned overlap and table header copying"
```

---

### Task 3: BGE-M3 向量嵌入客户端与 Milvus-Lite 向量库管理器

**Files:**
- Create: `app/services/rag/embedding.py`
- Create: `app/services/rag/milvus_client.py`
- Modify: `app/config.py`
- Test: `tests/test_rag_embedding_milvus.py`

**Interfaces:**
- Consumes: `.env` (`HUGGINGFACE_TOKEN`), `pymilvus.MilvusClient`
- Produces: `BGEEmbeddingClient.embed_query(str) -> List[float]`, `BGEEmbeddingClient.embed_documents(List[str]) -> List[List[float]]` (1024 维); `MilvusKnowledgeStore` (集合管理、upsert、近邻检索、删除)

- [ ] **Step 1: 编写嵌入客户端与 Milvus 客户端失败测试**

```python
# tests/test_rag_embedding_milvus.py
import pytest
import shutil
import os
from unittest.mock import patch, MagicMock
from app.services.rag.embedding import BGEEmbeddingClient
from app.services.rag.milvus_client import MilvusKnowledgeStore

def test_bge_embedding_client_dim():
    client = BGEEmbeddingClient()
    # 模拟或真实调用 feature extraction 返回 1024 维
    with patch.object(client._client, "feature_extraction", return_value=[0.1] * 1024):
        vec = client.embed_query("测试问题")
        assert len(vec) == 1024
        assert isinstance(vec[0], float)

def test_milvus_knowledge_store_crud():
    test_db = "./data/test_milvus.db"
    if os.path.exists(test_db):
        shutil.rmtree(test_db, ignore_errors=True)

    store = MilvusKnowledgeStore(uri=test_db)
    store.init_collection(drop_existing=True)

    test_records = [{
        "id": 101,
        "vector": [0.05] * 1024,
        "category": "运费规则",
        "questions": "邮费是多少",
        "answer": "满88元包邮，不满收取8元运费。",
        "content_type": "policy",
        "is_key_clause": False,
        "chunk_text": "分类：运费规则\n问题：邮费是多少\n内容：满88元包邮，不满收取8元运费。"
    }]
    store.upsert(test_records)

    hits = store.search(query_vector=[0.05] * 1024, top_k=1)
    assert len(hits) == 1
    assert hits[0]["id"] == 101
    assert hits[0]["category"] == "运费规则"

    store.close()
    shutil.rmtree(test_db, ignore_errors=True)
```

- [ ] **Step 2: 运行测试验证失败**

运行：`pytest tests/test_rag_embedding_milvus.py -v`
预期：FAIL (ModuleNotFoundError: No module named 'app.services.rag.embedding')

- [ ] **Step 3: 编写 BGE-M3 与 Milvus-Lite 管理器实现**

1. `app/config.py` 增加 `MILVUS_URI: str = "./data/milvus/astro_bot.db"`；
2. `app/services/rag/embedding.py`：使用 `huggingface_hub.InferenceClient(token=settings.HUGGINGFACE_TOKEN)`，调用 `feature_extraction(..., model="BAAI/bge-m3")`，支持重试、同步与异步包裹；
3. `app/services/rag/milvus_client.py`：基于 Context7 查阅的 `MilvusClient` 最佳实践实现集合创建（COSINE, 1024 维, FLAT 索引）、upsert、近邻检索及连接管理。

- [ ] **Step 4: 运行测试验证通过**

运行：`pytest tests/test_rag_embedding_milvus.py -v`
预期：PASS

- [ ] **Step 5: 提交代码与记录过程**

```bash
git add app/config.py app/services/rag/embedding.py app/services/rag/milvus_client.py tests/test_rag_embedding_milvus.py
git commit -m "feat(rag): add bge-m3 embedding client and milvus-lite store manager"
```

---

### Task 4: 双写落库与断点续跑管理器

**Files:**
- Create: `app/services/rag/dual_writer.py`
- Create: `scripts/build_knowledge_base.py`
- Test: `tests/test_rag_dual_writer.py`
- Test: `tests/test_rag_resume.py`

**Interfaces:**
- Consumes: `KnowledgeChunk`, `BGEEmbeddingClient`, `MilvusKnowledgeStore`
- Produces: `KnowledgeDualWriter.write_chunks(...)`, `KnowledgeDualWriter.repair_pending_chunks(...)`

- [ ] **Step 1: 编写双写与断点恢复失败测试**

```python
# tests/test_rag_resume.py
import pytest
from unittest.mock import patch
from app.models.knowledge import KnowledgeChunk
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.db.session import AsyncSessionLocal, Base, engine

@pytest.mark.asyncio
async def test_dual_write_and_resume_pending():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    writer = KnowledgeDualWriter(milvus_uri="./data/test_resume.db")
    writer.store.init_collection(drop_existing=True)

    with patch.object(writer.embedding_client, "embed_documents", return_value=[[0.02]*1024]):
        async with AsyncSessionLocal() as session:
            # 制造一个故意中断遗留的 pending 块
            broken_chunk = KnowledgeChunk(
                category="测试",
                questions="未完成向量化的问题",
                answer="未完成内容",
                vectorize_status="pending"
            )
            session.add(broken_chunk)
            await session.commit()
            await session.refresh(broken_chunk)
            broken_id = broken_chunk.id

        # 触发自愈补偿
        async with AsyncSessionLocal() as session:
            repaired_count = await writer.repair_pending_chunks(session)
            assert repaired_count >= 1

        # 验证状态被修正为 done 且已写入 Milvus
        async with AsyncSessionLocal() as session:
            refreshed = await session.get(KnowledgeChunk, broken_id)
            assert refreshed.vectorize_status == "done"
            assert refreshed.vector_id == str(broken_id)

    writer.store.close()
```

- [ ] **Step 2: 运行测试验证失败**

运行：`pytest tests/test_rag_resume.py -v`
预期：FAIL (ModuleNotFoundError: No module named 'app.services.rag.dual_writer')

- [ ] **Step 3: 编写双写落库与断点恢复实现**

在 `app/services/rag/dual_writer.py` 中实现：
1. 向量文本合成函数：`f"分类：{category}\n问题：{questions}\n内容：{answer}"`；
2. 四阶段事务控制：插入 MySQL `pending` $\to$ 获取 `id` 更新双向链表指针 $\to$ BGE-M3 向量化 $\to$ upsert 到 Milvus $\to$ 更新 MySQL 为 `done`；
3. `repair_pending_chunks(session)`：拉取所有 `vectorize_status == 'pending'` 的块重新向量化并 upsert 到 Milvus，回填 `vector_id`；
4. `scripts/build_knowledge_base.py` 离线构建脚本。

- [ ] **Step 4: 运行测试验证通过**

运行：`pytest tests/test_rag_resume.py tests/test_rag_dual_writer.py -v`
预期：PASS

- [ ] **Step 5: 提交代码与记录过程**

```bash
git add app/services/rag/dual_writer.py scripts/build_knowledge_base.py tests/test_rag_dual_writer.py tests/test_rag_resume.py
git commit -m "feat(rag): implement dual-write persistence with automatic pending chunk repair"
```

---

### Task 5: 历史客服对话知识挖掘管道

**Files:**
- Create: `app/prompts/qa_extraction.py`
- Create: `app/services/rag/miner.py`
- Create: `scripts/mine_dialogues.py`
- Test: `tests/test_rag_miner.py`

**Interfaces:**
- Consumes: MySQL `conversations`, `messages`, `qa_extraction_staging`, LLM API
- Produces: `DialogueKnowledgeMiner.extract_from_conversations(...)`, `DialogueKnowledgeMiner.deduplicate_and_ingest(...)`

- [ ] **Step 1: 编写对话挖掘与暂存去重失败测试**

```python
# tests/test_rag_miner.py
import pytest
from unittest.mock import AsyncMock, patch
from app.services.rag.miner import DialogueKnowledgeMiner
from app.models.staging import QAExtractionStaging
from app.db.session import AsyncSessionLocal, Base, engine

@pytest.mark.asyncio
async def test_dialogue_extraction_and_deduplication():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    miner = DialogueKnowledgeMiner()

    mock_llm_output = '[{"question": "退货邮费怎么算？", "answer": "非商品质量原因买家承担退货运费。"}, {"question": "寄回邮费谁出？", "answer": "个人原因退货由买方付邮费。"}]'
    with patch.object(miner, "_call_llm_for_qa", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_llm_output

        async with AsyncSessionLocal() as session:
            batch_no = "BATCH_TEST_MINER"
            inserted = await miner.extract_batch_dialogues(session, conversation_ids=[1, 2], batch_no=batch_no)
            assert inserted == 2

            # 执行去重
            kept_count, discarded_count = await miner.deduplicate_staging(session, batch_no=batch_no)
            assert kept_count >= 1
            assert discarded_count >= 1
```

- [ ] **Step 2: 运行测试验证失败**

运行：`pytest tests/test_rag_miner.py -v`
预期：FAIL (ModuleNotFoundError: No module named 'app.services.rag.miner')

- [ ] **Step 3: 编写对话抽取 Prompt 与挖掘去重服务**

1. `app/prompts/qa_extraction.py`：编写严谨的 Prompt 引导 LLM 过滤客套寒暄与隐私单号，提取结构化问答 JSON；
2. `app/services/rag/miner.py`：
   - `fetch_dialogue_batches`：从 `conversations` + `messages` 按会话分组提取问答上下文；
   - `extract_batch_dialogues`：写入 `qa_extraction_staging`（状态 `extracted`）；
   - `deduplicate_staging`：两阶段去重（精确文本哈希 + BGE-M3 余弦相似度 $\ge 0.92$ 聚类），将保留项置为 `kept`，重复项置为 `discarded`；
   - `ingest_kept_to_kb`：将 `kept` 记录转为 `KnowledgeChunk` 交由双写落库；
3. `scripts/mine_dialogues.py` 定时/离线任务执行入口。

- [ ] **Step 4: 运行测试验证通过**

运行：`pytest tests/test_rag_miner.py -v`
预期：PASS

- [ ] **Step 5: 提交代码与记录过程**

```bash
git add app/prompts/qa_extraction.py app/services/rag/miner.py scripts/mine_dialogues.py tests/test_rag_miner.py
git commit -m "feat(rag): add dialogue qa mining pipeline with staging and semantic deduplication"
```

---

### Task 6: 在线语义检索服务与 `query_faq` 工具升级

**Files:**
- Create: `app/services/rag/retriever.py`
- Modify: `app/tools/business_tools.py`
- Test: `tests/test_rag_retriever.py`
- Test: `tests/test_business_tools.py`

**Interfaces:**
- Consumes: `MilvusKnowledgeStore`, `BGEEmbeddingClient`
- Produces: `async def query_faq(keyword: str) -> str` (保持原有参数与返回契约，内部走 Milvus 向量检索)

- [ ] **Step 1: 编写检索器与 query_faq 升级失败测试**

```python
# tests/test_rag_retriever.py
import pytest
from unittest.mock import patch
from app.services.rag.retriever import KnowledgeRetriever
from app.tools.business_tools import query_faq

@pytest.mark.asyncio
async def test_retriever_dense_search():
    retriever = KnowledgeRetriever()
    mock_hits = [{
        "id": 1,
        "questions": "运费标准与偏远地区配送费说明",
        "answer": "全场满88元包邮；偏远地区加收15元运费。",
        "category": "平台规则 > 运费政策",
        "distance": 0.85
    }]
    with patch.object(retriever, "search", return_value=mock_hits):
        res = await retriever.retrieve_faq_text("邮费是多少")
        assert "运费标准与偏远地区配送费说明" in res
        assert "全场满88元包邮" in res
        assert "1. 问：" in res

@pytest.mark.asyncio
async def test_query_faq_contract_unchanged():
    # 验证工具签名与 Docstring
    assert query_faq.name == "query_faq"
    assert "keyword" in query_faq.args
```

- [ ] **Step 2: 运行测试验证失败**

运行：`pytest tests/test_rag_retriever.py -v`
预期：FAIL (ModuleNotFoundError: No module named 'app.services.rag.retriever')

- [ ] **Step 3: 编写语义检索器并升级 query_faq**

1. `app/services/rag/retriever.py`：
   - 接收查询文本，生成 Query 向量；
   - 访问 Milvus 执行 Top-3 余弦检索；
   - 阈值过滤（分值 $\ge 0.35$），组装为标号文本 `1. 问：...\n 答：...`；
   - 若空结果返回 `未找到与【{keyword}】相关的常见问题解答。`
2. `app/tools/business_tools.py`：修改 `query_faq` 函数体，调用 `KnowledgeRetriever`，保留原函数签名、参数及 Docstring。

- [ ] **Step 4: 运行测试验证通过**

运行：`pytest tests/test_rag_retriever.py tests/test_business_tools.py -v`
预期：PASS (全部业务工具单测包括升级后的 query_faq 全绿)

- [ ] **Step 5: 提交代码与记录过程**

```bash
git add app/services/rag/retriever.py app/tools/business_tools.py tests/test_rag_retriever.py tests/test_business_tools.py
git commit -m "feat(tools): upgrade query_faq implementation to dense vector semantic retrieval"
```

---

### Task 7: 知识库数据灌库与端到端双重验收验证

**Files:**
- Create: `data/kb/return_policy.md`
- Create: `data/kb/shipping_policy.md`
- Create: `data/kb/product_faq.md`
- Create: `scripts/verify_ch03_acceptance.py`
- Test: `tests/test_ch03_acceptance.py`
- Modify: `dev-notes/ch03.md`

**Interfaces:**
- Consumes: 全套 RAG 切分、挖掘、双写、检索与恢复组件
- Produces: 验收脚本与测试用例，100% 覆盖两项业务验收标准

- [ ] **Step 1: 编写双重验收标准端到端自动化测试**

```python
# tests/test_ch03_acceptance.py
import pytest
from app.tools.business_tools import query_faq
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.models.knowledge import KnowledgeChunk
from app.db.session import AsyncSessionLocal

@pytest.mark.asyncio
async def test_acceptance_criteria_1_semantic_rephrase():
    """验收标准 1:「邮费是多少」这类换说法的问题,现在能召回运费说明并答对"""
    # 提问“邮费是多少”（知识库中原文标题是“运费标准与偏远地区配送说明”）
    result = await query_faq.ainvoke({"keyword": "邮费是多少"})
    assert "运费" in result or "包邮" in result
    assert "未找到" not in result

@pytest.mark.asyncio
async def test_acceptance_criteria_2_resumed_pending_chunks():
    """验收标准 2:故意中断建库任务再重跑,漏向量化的块能被捡起补齐"""
    writer = KnowledgeDualWriter()
    async with AsyncSessionLocal() as session:
        chunk = KnowledgeChunk(
            category="测试中断",
            questions="故意漏掉的测试块",
            answer="断点测试内容",
            vectorize_status="pending"
        )
        session.add(chunk)
        await session.commit()
        await session.refresh(chunk)
        target_id = chunk.id

    # 模拟重跑断点修复
    async with AsyncSessionLocal() as session:
        repaired = await writer.repair_pending_chunks(session)
        assert repaired >= 1

    async with AsyncSessionLocal() as session:
        repaired_chunk = await session.get(KnowledgeChunk, target_id)
        assert repaired_chunk.vectorize_status == "done"
        assert repaired_chunk.vector_id == str(target_id)
```

- [ ] **Step 2: 运行测试验证**

运行：`pytest tests/test_ch03_acceptance.py -v`
预期：验证通过

- [ ] **Step 3: 运行完整建库与验证脚本**

编写并执行 `scripts/verify_ch03_acceptance.py`，输出包含：
1. 真实灌入 Markdown 知识文档；
2. 验证“邮费是多少”语义召回输出；
3. 验证断点模拟与自动自愈捡起补齐；
4. 全量运行 `pytest` 测试套件，确保 0 失败。

- [ ] **Step 4: 记录 dev-notes/ch03.md 并准备完结交付**

更新 `dev-notes/ch03.md`，记录各阶段执行、评审结论与返工记录。

- [ ] **Step 5: 提交代码**

```bash
git add data/kb/ scripts/verify_ch03_acceptance.py tests/test_ch03_acceptance.py dev-notes/ch03.md
git commit -m "test(acceptance): complete chapter 03 rag knowledge base and vector retrieval verification"
```
