# Task 7 Brief: 知识库数据灌库与端到端双重验收验证

## 1. 任务目标
组织真实业务 Markdown 知识库（包含运费规范大表格、退换货规范、关键免责条款与产品 FAQ），运行离线建库与双写落库，实现并自动化验证两大核心验收标准：
1. 「邮费是多少」换说法语义泛化召回运费说明并让客服答对；
2. 故意中断建库任务制造漏向量化数据，重跑后漏网块全部被捡起补齐（双写最终一致）。

## 2. 涉及文件
- Create: `data/kb/return_policy.md`
- Create: `data/kb/shipping_policy.md`
- Create: `data/kb/product_faq.md`
- Create: `scripts/verify_ch03_acceptance.py`
- Test: `tests/test_ch03_acceptance.py`
- Modify: `dev-notes/ch03.md`

## 3. 全局约束与验收标准要求
- **验收标准 1**：
  - 提问“邮费是多少”等自然语言同义表述（文档原文标题为“运费标准与偏远地区配送说明”）；
  - `query_faq` 通过 BGE-M3 Dense 语义近邻召回运费正文，不发生 SQL LIKE 的 MISS，模型结合召回内容正确答复运费政策。
- **验收标准 2**：
  - 人为注入/模拟建库过程意外中断（在写入 MySQL 后尚未向量化至 Milvus 时中断退出，留下 `vectorize_status='pending'` 的孤立块）；
  - 再次启动建库/自愈程序，验证遗留块被 100% 自动检出、重新生成向量并 upsert 到 Milvus，回填 `vector_id`，状态翻转为 `'done'`。
- **全量无退化**：全套单测套件通过率保持 100%。

## 4. 关键接口与功能要求

### 4.1 Markdown 知识文档规范 (`data/kb/`)
- `shipping_policy.md`：
  - 包含一级标题 `# 商城配送与运费服务规范`；
  - 包含二级标题 `## 运费标准与偏远地区配送说明`；
  - 包含明确的正文规则（“全场实付金额满88元普通快递包邮；不满88元收取基础运费8元；新疆、西藏等偏远地区每单加收15元偏远附加运费”）；
  - 包含大型资费表格（用于验证按行切分与表头复制）。
- `return_policy.md`：
  - 包含七天无理由退货规则、退款时效；
  - 包含关键免责条款 `【特别说明】已激活使用的数码产品与贴身衣物拆封后不支持无理由退换货`。
- `product_faq.md`：
  - 包含问答格式的商品常见咨询。

### 4.2 验收测试用例 (`tests/test_ch03_acceptance.py`)
- `test_acceptance_criteria_1_semantic_rephrase`: 验证“邮费是多少”精准命中运费说明；
- `test_acceptance_criteria_2_resumed_pending_chunks`: 验证人为制造 pending 块后通过自愈补齐为 done；
- `test_e2e_full_retrieval_and_answer`: 验证大模型通过 `query_faq` 检索知识库并回答运费。

### 4.3 验收脚本 (`scripts/verify_ch03_acceptance.py`)
- 自动化执行：
  1. 确保 MySQL 与表结构就绪；
  2. 离线全量构建 `data/kb/` 知识库入库；
  3. 执行换说法语义检索验收打印输出；
  4. 执行人为中断制造 pending 并重跑补齐验收打印输出；
  5. 统计知识块总数与向量库总数对比确认 1:1 对齐。

## 5. 验收标准
- `pytest tests/test_ch03_acceptance.py -v` 100% 通过；
- `python scripts/verify_ch03_acceptance.py` 成功执行并清晰输出验收结论；
- 全量回归 `pytest` 保持 100% 通过。
