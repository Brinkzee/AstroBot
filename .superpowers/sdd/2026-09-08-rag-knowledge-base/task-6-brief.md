# Task 6 Brief: 在线语义检索服务与 query_faq 工具升级

## 1. 任务目标
构建在线密集向量语义检索器 `KnowledgeRetriever`，并将客服系统的业务工具 `query_faq` 的内部实现从 SQL LIKE 查表升级为基于 BGE-M3 Dense 语义向量的 Milvus 近邻召回，同时保持外部调用契约（函数名、参数名、类型注解、Docstring、返回值字符串结构）100% 绝对不变。

## 2. 涉及文件
- Create: `app/services/rag/retriever.py`
- Modify: `app/tools/business_tools.py`
- Modify: `app/services/rag/__init__.py` (导出 `KnowledgeRetriever`)
- Test: `tests/test_rag_retriever.py`
- Modify: `tests/test_business_tools.py`

## 3. 全局约束与业务规范
- **契约零破坏原则 (Contract Immutability)**：
  - `@tool` 装饰器函数签名必须依然为 `async def query_faq(keyword: str) -> str`；
  - Docstring 保持完全一致；
  - 返回值格式必须保持与原有一致：
    ```text
    1. 问：{question}
       答：{answer}
    ```
    未命中时返回：`未找到与【{keyword}】相关的常见问题解答。`
- **检索逻辑升级**：
  - 弃用基于关键词的 `WHERE question LIKE %kw% OR answer LIKE %kw%`；
  - 接收用户输入的提问文本（如“邮费是多少”），调用 BGE-M3 提取 1024 维 Dense Query 向量；
  - 调用 Milvus `knowledge` 集合执行余弦相似度检索（`top_k = 3`）；
  - 最低相似度阈值过滤（默认 `min_score = 0.35`），过滤离题噪声；
  - 优雅保底：若 Milvus 尚未灌库或未启动时，可兼容回退原 FAQ 表以保证旧单测与无向量环境的平滑兼容。
- 遵循 TDD 流程。

## 4. 关键接口定义与实现要求

### 4.1 `KnowledgeRetriever` (`app/services/rag/retriever.py`)
- `__init__(store: Optional[MilvusKnowledgeStore] = None, embedding_client: Optional[BGEEmbeddingClient] = None, min_score: float = 0.35)`
- `async retrieve(query: str, top_k: int = 3) -> List[Dict]`:
  - 过滤空输入；
  - 异步向量化 `query`；
  - 到 Milvus 执行余弦相似度搜索；
  - 返回满足 `distance >= min_score` 的命中文档列表。
- `async retrieve_faq_text(keyword: str, top_k: int = 3) -> str`:
  - 调用 `retrieve(keyword, top_k=top_k)`；
  - 若无命中，返回 `f"未找到与【{keyword}】相关的常见问题解答。"`；
  - 若有命中，格式化为带序号的标准文本：
    ```text
    1. 问：{questions的第一行或主问法}
       答：{answer}
    ```
    多条以 `\n\n` 分隔。

### 4.2 升级 `app/tools/business_tools.py`
- 引入全局或按需单例 `KnowledgeRetriever`；
- 在 `query_faq` 函数体内调用 `await retriever.retrieve_faq_text(keyword, top_k=3)`；
- 保留 `query_faq` 的原始注释、参数名 `keyword: str` 与返回值类型 `-> str`。

## 5. 验收标准
- `pytest tests/test_rag_retriever.py tests/test_business_tools.py -v` 100% 通过；
- 验证包含：
  1. 检索器对余弦相似度 Top-K 结果的提取与格式化；
  2. 低分过滤与空输入兜底提示；
  3. `query_faq` 工具的 LangChain `@tool` 契约完整性（名字、参数模式、文档字符串保持不变）；
  4. 全量回归保持 100% 通过。
