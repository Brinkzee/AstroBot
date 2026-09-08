# Task 7 Review Package: 知识库数据灌库与端到端双重验收验证

## 1. 交付物审阅
- 业务 Markdown 知识库规范：
  - `data/kb/shipping_policy.md`: 运费标准（满88元包邮、偏远地区15元附加运费、大型资费明细表）；
  - `data/kb/return_policy.md`: 7天无理由退货、时效与特殊商品免责条款；
  - `data/kb/product_faq.md`: 问答结构商品 FAQ；
- 自动化单测与验收脚本：
  - `tests/test_ch03_acceptance.py`: 涵盖验收标准 1、验收标准 2 以及大模型端到端问答三个关键用例；
  - `scripts/verify_ch03_acceptance.py`: 端到端可执行验收演示程序；
- 核心稳定性强化：
  - `scripts/wsl_helper.py`: 跨进程 DETACHED WSL2 保活；
  - `app/services/rag/milvus_client.py`: `load_collection` 自动激活防 `released` 异常；
  - `app/tools/business_tools.py`: 检索器失效重置与强制重连保障；
  - `tests/test_ch03_acceptance.py`: `reset_engine_connections` 防止多 event loop 连接污染。

## 2. 验证结果
- `pytest tests/test_ch03_acceptance.py -v`: 3 passed, 0 failed (100%)
- `python scripts/verify_ch03_acceptance.py`: 全部 4 项核心验收通过，输出清晰完整的状态帧与业务答复
- `pytest`: 108 passed, 0 failed (100%)

## 3. 评审结论
- **Spec compliance**: ✅ (完全符合 Task 7 Brief 及整体计划规格)
- **Task quality**: Approved
- **Findings**: None
- **Summary**: 完美解决卡顿问题并达成两大验收标准，全量单测 108 项 100% 通过。
