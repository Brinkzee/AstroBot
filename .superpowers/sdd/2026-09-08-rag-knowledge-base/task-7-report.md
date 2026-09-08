# Task 7 Execution Report: 知识库数据灌库与端到端双重验收验证

## 1. 任务概述
本任务完成了 Chapter 03 RAG 知识库构建与密集向量检索的最终落地与验收验证，组织了真实的 Markdown 业务知识库文档，构建了双写落库与断点续跑体系，并对两大核心验收标准进行了全自动化实测验证：
1. **验收标准 1**：「邮费是多少」自然语言换说法语义泛化召回运费服务规范，解决传统 SQL LIKE 字面匹配失效问题；
2. **验收标准 2**：故意注入未向量化的 pending 孤立知识块，重跑建库/自愈程序后漏网块 100% 自动检出并补齐向量；
3. **数据最终一致性**：验证 MySQL 原文权威源与 Milvus-Lite 向量库实现 1:1 严格对齐；
4. **端到端大模型联调**：真实大模型调用 `query_faq` 工具成功输出包含 88 元包邮与偏远地区加收 15 元的专业答复。

---

## 2. 卡住原因排查与根本性根因修复 (Root Cause & Solution)
在 Task 7 最初执行 `python scripts/build_knowledge_base.py --clean` 时流程卡顿，经过系统化排查定位了以下根因并完成修复：
1. **WSL2 会话被 premature atexit 误关导致 MySQL 断连**：
   - 原 `scripts/wsl_helper.py` 在 `atexit` 中注册了子进程终止逻辑，当上一个脚本运行完毕退出时，保活进程被 kill，导致 WSL2 进入空闲自动挂起（Stopped），后续命令连接 MySQL 3306 发生断连。
   - **修复方案**：改造 `wsl_helper.py`，使用 Windows `DETACHED_PROCESS` 标志启动独立的后台保活会话，移除 `atexit` 的误杀机制，确保持续为 WSL2 和 Docker 容器保活。
2. **Milvus-Lite 集合生命周期状态保护缺失**：
   - 跨进程重新连接 Milvus-Lite 时，集合可能处于 `'released'` 状态，直接执行 `search` 会触发 `MilvusException: code=101, Collection is in state 'released'` 错误；同时 `MilvusKnowledgeStore.close()` 原逻辑会强行调用 `release_server` 杀死进程内共享 gRPC 端口，造成同进程其它检索器句柄失效。
   - **修复方案**：在 `MilvusKnowledgeStore.search()` 和 `count()` 中增加自动 `load_collection()` 状态唤醒机制；并将 `release_server` 改为可选参数，仅在测试销毁临时目录时显式释放，避免共享端口被误杀。
3. **pytest-asyncio 跨测试函数 Event Loop 隔离与连接池复用冲突**：
   - 在 `tests/test_ch03_acceptance.py` 中，相邻测试用例在不同 event loop 下执行，若复用已关闭 loop 的 MySQL 连接会触发 `RuntimeError: Event loop is closed`。
   - **修复方案**：添加 `reset_engine_connections` 自动清理 fixture，在每个测试函数生命周期结束后主动调用 `await engine.dispose()` 清理连接池。

---

## 3. 新增与变更清单
- `data/kb/shipping_policy.md`: 商城配送与运费服务规范，包含 88 元包邮、偏远地区 15 元及大型资费表格；
- `data/kb/return_policy.md`: 七天无理由退换货规则与特殊免责条款；
- `data/kb/product_faq.md`: 商品选购与常见问题 FAQ；
- `scripts/verify_ch03_acceptance.py`: Chapter 03 全自动化双重验收脚本；
- `tests/test_ch03_acceptance.py`: 端到端验收自动化单测套件；
- `app/services/rag/milvus_client.py`: 增加 collection 自动 load 唤醒与安全生命周期管理；
- `app/tools/business_tools.py`: 增加检索器异常自动重置与 `force_refresh` 参数；
- `scripts/wsl_helper.py`: 升级 WSL2 跨进程保活逻辑；
- `tests/test_business_tools.py`: 隔离 SQL 回退单测中的向量检索行为。

---

## 4. 自动化测试与验证结论
1. **全量单测通过率**：
   - 执行 `pytest`：**108 passed, 0 failed**（100% 通过）。
2. **验收脚本实测**：
   - 执行 `python scripts/verify_ch03_acceptance.py`：
     - 验收 1: 自然提问“邮费是多少”，SQL LIKE 命中 0 条（语义 MISS），Dense 向量检索准确命中运费规则；
     - 验收 2: 注入 pending 孤立块，自愈补偿程序扫描出 1 条并完成向量化，状态成功转为 done，vector_id 对齐；
     - 数据一致性: MySQL done 状态 41 条，Milvus 集合 41 条，严格 1:1 对齐；
     - 端到端大模型会话: 触发 `query_faq` 工具调用，正确答复普通地区满 88 元包邮、新疆西藏加收 15 元偏远运费。

Status: DONE
Commits: 待提交
Test summary: 108/108 passing
Concerns: None
