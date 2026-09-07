# Task 2 Report: SQLAlchemy ORM 数据模型与 FAQ 测试数据灌入

## 1. 任务概述
- **任务编号与名称**: Task 2: SQLAlchemy ORM 数据模型与 FAQ 测试数据灌入
- **基础提交 (Base Commit)**: `98e06daf49fb9ce1736a089b9a8a4df1c6904d73`
- **生成提交 (New Commit)**: `b0546f2d682630b8624db25e2e58f7e6c2ed37d0`
- **提交信息**: `feat(models): add orm entities and faq seed script aligned with ch02-ddl.sql`

## 2. 接口与产出物
- `app.models.Conversation`: 映射 `conversations` 表，主键 `id` (BIGINT UNSIGNED)，支持多方言适配与级联关系；
- `app.models.Message`: 映射 `messages` 表，主键 `id`，外键 `conversation_id` 关联会话，`role` 对齐 OpenAI Chat 协议 (`user`, `assistant`, `tool`)，`tool_calls` 为 JSON 结构；
- `app.models.FAQ`: 映射 `faq` 表，包含 `question`, `answer`, `category`，为 `query_faq` 工具检索数据源；
- `app.models.Ticket`: 映射 `tickets` 表，主键 `ticket_no` (VARCHAR 32)，外键 `conversation_id` 关联会话；
- `app.models.__init__.py`: 统一导出 4 大 ORM 实体；
- `scripts.seed_data`:
  - `SEED_FAQS`: 预置「退货政策说明」、「运费标准与包邮政策」、「发票开具说明」三大知识点；
  - `seed_all_data(session: AsyncSession) -> int`: 具备幂等性的 FAQ 异步数据灌入函数，返回新插入记录数；
  - `main()`: 具备独立运行入口，支持 `python scripts/seed_data.py` 直接执行。
- `tests/test_models.py`: 覆盖模型实例属性、表元数据与外键检查、SQLite 内存环境 CRUD 验证、种子数据结构与幂等性异步校验。

## 3. TDD 执行过程

### 3.1 Step 1 & 2: RED 阶段
创建 `tests/test_models.py`，执行 `pytest tests/test_models.py -v`。
**测试失败输出 (RED)**:
```
=================================== ERRORS ====================================
____________________ ERROR collecting tests/test_models.py ____________________
ImportError while importing test module 'D:\PycharmProjects\AstroBot\tests\test_models.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
D:\Anaconda3\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_models.py:6: in <module>
    from app.models import Conversation, Message, FAQ, Ticket
E   ModuleNotFoundError: No module named 'app.models'
=========================== short test summary info ===========================
ERROR tests/test_models.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.50s ===============================
```

### 3.2 Step 3: 实现阶段
1. 创建 `app/models/conversation.py`：定义 `Conversation` 模型，对齐 `conversations` 表。
2. 创建 `app/models/message.py`：定义 `Message` 模型，包含 `tool_calls` (JSON) 与 `tool_call_id`。
3. 创建 `app/models/faq.py`：定义 `FAQ` 模型，对齐 `faq` 表。
4. 创建 `app/models/ticket.py`：定义 `Ticket` 模型，主键使用业务工单号 `ticket_no`。
5. 创建 `app/models/__init__.py`：导出 4 大模型实体。
6. 创建 `scripts/seed_data.py`：实现 `SEED_FAQS` 定义与幂等入库函数 `seed_all_data`。

### 3.3 Step 4: GREEN 阶段
执行模块测试与全量回归测试：
- `pytest tests/test_models.py -v`: 5 passed in 0.46s
- `pytest -v`: 26 passed, 1 warning in 2.98s (全量测试套件 100% 通过，无回归错误)

## 4. 代码审查与 Diff 自检
- **DDL 规范对齐**:
  - 表名与字段名严格匹配 `sql/ch02-ddl.sql`；
  - 主键与外键类型统一使用 `BigInteger().with_variant(MYSQL_BIGINT(unsigned=True), "mysql").with_variant(Integer, "sqlite")`，既确保在生产 MySQL 环境下符合 `BIGINT UNSIGNED` 的 DDL 约束，又保证了在轻量测试环境下 SQLite 自增与外键的无缝兼容；
  - `Enum` 定义使用命名枚举并明确合法取值范围。
- **幂等性与可用性**:
  - `seed_all_data` 在插入前先按 `question` 查询数据库，重复执行不会产生脏数据；
  - `scripts/seed_data.py` 自动检测并挂载 `PROJECT_ROOT` 到 `sys.path`，支持开箱即用。
- **提交内容**:
  - `app/models/__init__.py` (created)
  - `app/models/conversation.py` (created)
  - `app/models/message.py` (created)
  - `app/models/faq.py` (created)
  - `app/models/ticket.py` (created)
  - `scripts/seed_data.py` (created)
  - `tests/test_models.py` (created)

## 5. 潜在问题与注意事项 (Concerns)
- 无代码或设计层面的阻塞性隐患。
- 在真实运行 `python scripts/seed_data.py` 时需要确保 MySQL 8.0 实例已通过 Docker 启动，并且已执行初始化 DDL。
