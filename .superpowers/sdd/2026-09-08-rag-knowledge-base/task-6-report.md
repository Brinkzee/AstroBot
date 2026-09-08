# Task 6 Execution Report: 在线语义检索服务与 query_faq 工具升级

## 1. 任务概述 (Executive Summary)
本任务为 RAG 知识库实施计划的 Task 6。核心目标是构建在线密集向量语义检索器 `KnowledgeRetriever`，并将客服业务工具 `query_faq` 的内部实现从原有的 SQL `LIKE %kw%` 查表升级为基于 BGE-M3 Dense 语义向量的 Milvus 近邻召回，同时实现契约零破坏（函数名、参数名、类型注解、Docstring、返回值字符串结构 100% 不变），并在向量库无数据或未就绪时具备平滑回退 SQL 原始 FAQ 表的优雅保底能力。

## 2. 变更文件清单 (Files Modified & Created)
- `app/services/rag/retriever.py` (新增)：
  - 实现 `KnowledgeRetriever` 类；
  - 提供 `retrieve(query, top_k=3, min_score=0.35, filter=None)` 异步方法，处理空输入过滤、Query 异步向量化、Milvus 余弦相似度近邻检索及阈值过滤；
  - 提供 `format_faq_hits(hits, keyword)`，自动提取 `questions` 的首行主问法并格式化为标准带序号文本；
  - 提供 `retrieve_faq_text(keyword, top_k=3)` 便捷方法及空结果标准兜底提示。
- `app/services/rag/__init__.py` (修改)：
  - 导出 `KnowledgeRetriever`。
- `app/tools/business_tools.py` (修改)：
  - 增加 `get_retriever()` 单例获取函数；
  - 重构 `query_faq` 内部实现：优先尝试 `KnowledgeRetriever` 密集向量检索，命中即返回标准排版结果；
  - 若检索异常或向量库无命中数据，平滑兼容回退至原有 `AsyncSessionLocal()` SQL FAQ 查表逻辑。
- `tests/test_rag_retriever.py` (新增)：
  - 包含 9 个完整单测，涵盖默认参数初始化、空输入短路、余弦打分过滤、多行问法提取格式化、未找到兜底、临时 Milvus-Lite 端到端真实检索、LangChain `@tool` 契约不变性、优先向量检索及 SQL 优雅回退。
- `tests/test_business_tools.py` (修改)：
  - 保持原有旧 SQL 模拟单测 100% 绿灯；
  - 新增 `test_query_faq_dense_vector_retrieval` 验证业务工具接入 Dense 语义检索的调用路径。

## 3. TDD 执行过程 (TDD Cycle)
1. **RED 阶段**：
   - 编写 `tests/test_rag_retriever.py`；
   - 运行 `pytest tests/test_rag_retriever.py -v`，由于 `app.services.rag.retriever` 尚未实现，测试正确捕获 `ModuleNotFoundError: No module named 'app.services.rag.retriever'`，验证 RED 阶段有效。
2. **GREEN 阶段**：
   - 实现 `app/services/rag/retriever.py` 中的 `KnowledgeRetriever`；
   - 在 `app/services/rag/__init__.py` 中导出 `KnowledgeRetriever`；
   - 在 `app/tools/business_tools.py` 中升级 `query_faq` 内部逻辑并增加 `get_retriever()`；
   - 在 `tests/test_business_tools.py` 中增加向量检索测试用例；
   - 运行 `pytest tests/test_rag_retriever.py tests/test_business_tools.py -v`，全部 18 个测试通过。
3. **REFACTOR & 全量回归验证阶段**：
   - 优化代码异常捕获与日志记录；
   - 执行全量自动化测试套件：`pytest -v`；
   - 结果：全量 **105 passed, 72 warnings in 22.47s**，测试通过率 100%，无任何回归问题。

## 4. 契约与业务规则检验 (Contract & Business Compliance)
- **工具名称与签名**：`query_faq` 依然为 `@tool`，签名为 `async def query_faq(keyword: str) -> str`；
- **Docstring**：保持与原始注释完全一致；
- **格式化输出结构**：
  ```text
  1. 问：{questions的第一行或主问法}
     答：{answer}
  ```
  多条以 `\n\n` 分隔。未找到时返回：`未找到与【{keyword}】相关的常见问题解答。`
- **优雅保底机制**：当 Milvus-Lite 尚未灌库或未检索到相关内容时，自动平滑回退执行 MySQL/SQLite 的 FAQ 表查询。

## 5. Git 提交记录 (Commit)
- Commit Hash: `765dc855f725cfb4a41790b1efbf5f727f652412` (`765dc85`)
- Commit Message: `feat(tools): upgrade query_faq implementation to dense vector semantic retrieval`
- Staged Files:
  - `app/services/rag/__init__.py`
  - `app/services/rag/retriever.py`
  - `app/tools/business_tools.py`
  - `tests/test_business_tools.py`
  - `tests/test_rag_retriever.py`

## 6. 最终结论 (Conclusion)
Task 6 全部需求均已严格按 TDD 规范完成，所有测试通过，无遗留隐患。
