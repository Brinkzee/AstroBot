# 客服三层会话上下文管理与动态预算实施计划 (Context Management Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建客服系统三层会话上下文管理机制（层 1 原文、层 2 半压缩、层 3 分段摘要），支持模型窗口动态预算倒推与中文折算校准、后台异步非阻塞事实摘要、固定顺序前缀缓存保护拼装、全链路可观测日志（`model_ctx` / `history_ctx`），以及前端多会话侧栏与历史回载。

**Architecture:** 
采用独立 ContextManager 与后台守护任务双轨设计：LangGraph State 维系全量真实会话流水与快照落盘；调模型前通过两边界锚点（`summary_upto_msg_id` 与 `layer1_from_msg_id`）现算精简视图并挂载证据包，保证 System 前缀绝对稳定；层 2 超出预算在后台派发独立异步协程提炼事实梗概追加至 `conversation_summaries`，全程不阻塞用户当前轮流式输出。

**Tech Stack:** Python 3.10+, FastAPI, SQLAlchemy (Asyncio + MySQL), LangGraph State (`add_messages` reducer + checkpoint), LangChain `trim_messages`, Uvicorn, pytest.

**Spec:** [`docs/superpowers/specs/2026-09-12-context-management-design.md`](file:///d:/PycharmProjects/AstroBot/docs/superpowers/specs/2026-09-12-context-management-design.md)

## Global Constraints

- **Python & Environment**: Windows PowerShell, Python 3.10+, utf-8 encoding for files and logs.
- **Database Schema**: 严格对齐 `sql/ch07-ddl.sql`，给 `conversations` 加 `summary`, `summary_upto_msg_id`, `layer1_from_msg_id`；新建 `conversation_summaries`（`id`, `conversation_id`, `seq`, `from_msg_id`, `upto_msg_id`, `content`, `created_at`）。
- **Token Math**: 演示配置 `MODEL_CONTEXT_WINDOW=18000` 严格倒推：Fixed=8750, Peak=3600, 滑窗=5650, 层 1=3954, 层 2=1695。
- **Chinese Token Ratio**: 中文汉字与全角字符严格执行 **1 字符 = 1 Token**。
- **Context Assembly Order**: System(人设+工具) -> Layer 2(半压缩) -> Layer 1(原文) -> User(当前提问) -> Attached Block(摘要投影 + RAG证据)。严禁梗概单占 System。
- **Logging Observable**: `log/app.log` 必须支持 `grep model_ctx` 与 `grep history_ctx`。

---

### Task 1: 数据库模型演进与 DDL 迁移脚本 (`conversations` & `conversation_summaries`)

**Files:**
- Modify: `app/models/conversation.py`
- Create: `app/models/summary.py`
- Modify: `app/models/__init__.py`
- Create: `scripts/init_ch07_db.py`
- Test: `tests/test_ch07_models.py`

**Interfaces:**
- Produces:
  - `Conversation.summary: Mapped[Optional[str]]`
  - `Conversation.summary_upto_msg_id: Mapped[Optional[int]]`
  - `Conversation.layer1_from_msg_id: Mapped[Optional[int]]`
  - `ConversationSummary(Base)` with `id, conversation_id, seq, from_msg_id, upto_msg_id, content, created_at`
  - `init_ch07_db(engine_override=None) -> List[str]`

- [ ] **Step 1: Write the failing test**
  编写 `tests/test_ch07_models.py`，验证 `Conversation` 新字段定义、`ConversationSummary` 模型属性与 `init_ch07_db` 迁移解析。

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_ch07_models.py -v`
  Expected: FAIL with `ImportError` or `AttributeError`.

- [ ] **Step 3: Write minimal implementation**
  - 在 `app/models/conversation.py` 添加 `summary`, `summary_upto_msg_id`, `layer1_from_msg_id` 列；
  - 创建 `app/models/summary.py` 定义 `ConversationSummary`；
  - 在 `app/models/__init__.py` 导出新模型；
  - 编写 `scripts/init_ch07_db.py` 执行 `sql/ch07-ddl.sql` 并保障幂等。

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_ch07_models.py -v`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add app/models/conversation.py app/models/summary.py app/models/__init__.py scripts/init_ch07_db.py tests/test_ch07_models.py
  git commit -m "feat(db): implement ch07 context management models and migration script"
  ```

---

### Task 2: 配置项扩展与动态 Token 预算倒推算法器 (`app/config.py` & `app/services/context/budget.py`)

**Files:**
- Modify: `app/config.py`
- Create: `app/services/context/budget.py`
- Test: `tests/test_context_budget.py`

**Interfaces:**
- Consumes: `Settings` 配置
- Produces:
  - `estimate_tokens(text_or_messages: Any) -> int`: 严格执行中文 1 字符 = 1 Token，西文 ~3.5 字符 ≈ 1 Token
  - `calculate_context_budget(settings: Optional[Settings] = None) -> ContextBudgetResult`:
    输出包含 `fixed_overhead`, `peak_react_overhead`, `window_available`, `window_budget`, `layer1_budget`, `layer2_budget`
  - `check_budget_on_startup(settings: Optional[Settings] = None) -> bool`: 检查是否支持单轮，不符时记录 CRITICAL 报警

- [ ] **Step 1: Write the failing test**
  编写 `tests/test_context_budget.py`：
  - 测试中文折算（汉字数与 Token 1:1 严格对齐）；
  - 测试演示配置（18000 窗口）：验证固定开销 8750、峰值 3600、滑窗 5650、层 1 预算 3954、层 2 预算 1695；
  - 测试默认 128k 窗口：验证滑窗预算取稳态 10000；
  - 测试超低窗口触发自检报警。

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_context_budget.py -v`
  Expected: FAIL with `ModuleNotFoundError` or `AttributeError`.

- [ ] **Step 3: Write minimal implementation**
  - 在 `app/config.py` 中增加 `model_context_window`, `max_output_tokens`, `max_user_input_tokens`, `max_agent_steps`, `tool_result_max_tokens`, `rerank_top_k`, `system_prompt_tokens`, `doc_tokens_per_chunk`, `summary_max_tokens`, `safety_margin_tokens`, `target_history_turns`, `steady_turn_tokens` 字段；
  - 创建 `app/services/context/budget.py` 实现 `estimate_tokens`, `calculate_context_budget`, `check_budget_on_startup`。

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_context_budget.py -v`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add app/config.py app/services/context/budget.py tests/test_context_budget.py
  git commit -m "feat(context): implement dynamic token budget calculator and chinese token estimator"
  ```

---

### Task 3: 三层上下文管理器与边界推进状态机 (`app/services/context/manager.py`)

**Files:**
- Create: `app/services/context/manager.py`
- Test: `tests/test_context_manager.py`

**Interfaces:**
- Consumes: `Conversation`, `Message`, `calculate_context_budget`, `estimate_tokens`
- Produces:
  - `ContextManager`:
    - `partition_messages(messages, summary_upto_msg_id, layer1_from_msg_id) -> (layer3, layer2, layer1)`
    - `apply_layer1_degradation(conv, messages, budget) -> Optional[int]`: 层 1 超预算时推移 `layer1_from_msg_id`
    - `format_layer2_messages(messages) -> List[BaseMessage]`: 客服截短 60 字，工具替换为单行
    - `format_layer1_messages(messages) -> List[BaseMessage]`: 原文原样保留
    - `build_model_messages(system_prompt, layer2_msgs, layer1_msgs, current_query, summary, retrieved_docs) -> List[BaseMessage]`: 前缀缓存友好装配
    - `build_history_context_text(layer2_msgs, layer1_msgs, summary) -> (str, str)`: 输出用于意图消解的精简背景与滑窗

- [ ] **Step 1: Write the failing test**
  编写 `tests/test_context_manager.py`：
  - 测试空历史与初态分配（全在层 1）；
  - 测试层 1 超标降级：锚点 `layer1_from_msg_id` 步进推进；
  - 测试层 2 格式化：客服答复超 60 字截短 + `...`，工具消息替换为单行语义标识；
  - 测试消息装配顺序：System 干净置顶，层 2 接着层 1，当前提问之后单接证据块，杜绝单独占用 System。

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_context_manager.py -v`
  Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**
  实现 `app/services/context/manager.py`，支持分区、格式化、层 1 降级推进与防缓存击穿装配。

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_context_manager.py -v`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add app/services/context/manager.py tests/test_context_manager.py
  git commit -m "feat(context): implement three-tier context manager and degradation state machine"
  ```

---

### Task 4: 后台异步分段摘要引擎与事实提炼 Prompt (`app/services/context/summary_service.py`)

**Files:**
- Create: `app/prompts/summarizer.py`
- Create: `app/services/context/summary_service.py`
- Test: `tests/test_context_summary_service.py`

**Interfaces:**
- Consumes: `ConversationSummary`, `Conversation`, `estimate_tokens`
- Produces:
  - `SummaryService`:
    - `should_trigger_summary(layer2_tokens, layer2_budget) -> bool`
    - `trigger_async_summary(conv_id, from_id, to_id, model=None)`: 非阻塞启动独立协程，带防重并发锁
    - `summarize_dialogue_sync(dialogue_text, existing_summary=None, model=None) -> str`: 执行事实提炼
    - `persist_summary_segment(db, conv_id, from_id, to_id, content) -> ConversationSummary`: 落盘与投影更新

- [ ] **Step 1: Write the failing test**
  编写 `tests/test_context_summary_service.py`：
  - 触发判定测试：基于 Token 估算判定，而不是消息条数；
  - 摘要提炼 Prompt 验证：无事实不胡编、剔除寒暄、字数 50~200；
  - 异步分段追加测试：`seq` 自增，旧梗概只作背景不重写，`conversations.summary` 投影正确拼接；
  - 并发防重锁机制测试：同一会话重复触发记录 `[summary skip]`。

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_context_summary_service.py -v`
  Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**
  - 在 `app/prompts/summarizer.py` 定义严格事实提炼 Prompt；
  - 在 `app/services/context/summary_service.py` 实现 `SummaryService`、异步非阻塞任务派发与数据库更新。

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_context_summary_service.py -v`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add app/prompts/summarizer.py app/services/context/summary_service.py tests/test_context_summary_service.py
  git commit -m "feat(context): implement async background summarization engine with fact-only prompt"
  ```

---

### Task 5: 上下文可观测日志系统与前置节点集成 (`log/app.log` & `pre_nodes.py` & `agent_node.py`)

**Files:**
- Create: `app/services/context/logger.py`
- Modify: `app/services/workflow/nodes/pre_nodes.py`
- Modify: `app/services/workflow/nodes/agent_node.py`
- Test: `tests/test_context_observability.py`

**Interfaces:**
- Produces:
  - `log_model_context(conv_id, summary, window_msgs, estimated_tokens)`: 格式化输出 `[model_ctx]` 到 `log/app.log`
  - `log_history_context(conv_id, summary_line, window_msgs)`: 格式化输出 `[history_ctx]` 到 `log/app.log`
  - `log_summary_lifecycle(...)`: 规范输出 `[summary trigger]`, `[summary start]`, `[summary done]`, `[summary skip]`, `[summary fail]`

- [ ] **Step 1: Write the failing test**
  编写 `tests/test_context_observability.py`：
  - 验证 `log/app.log` 生成与 UTF-8 格式；
  - 验证 `model_ctx` 包含摘要全文、滑窗逐条消息及 Token 估算；
  - 验证 `history_ctx` 每轮必打（包含闲聊与投诉轮）；
  - 验证 grep 语法能够准确命中关键字段。

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_context_observability.py -v`
  Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
  - 实现 `app/services/context/logger.py` 并配置独立 FileHandler（写入 `log/app.log`）；
  - 在 `pre_nodes.py`（指代消解后）输出 `[history_ctx]`；
  - 在 `agent_node.py`（调用模型前）输出 `[model_ctx]`。

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_context_observability.py -v`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add app/services/context/logger.py app/services/workflow/nodes/pre_nodes.py app/services/workflow/nodes/agent_node.py tests/test_context_observability.py
  git commit -m "feat(observability): implement model_ctx and history_ctx logging in log/app.log"
  ```

---

### Task 6: 工作流调度全链路整合 (`ChatService` & `WorkflowEngine` & `State`)

**Files:**
- Modify: `app/services/workflow/state.py`
- Modify: `app/services/workflow/engine.py`
- Modify: `app/services/chat_service.py`
- Test: `tests/test_chat_service_context.py`

**Interfaces:**
- Consumes: `ContextManager`, `SummaryService`, `calculate_context_budget`
- Produces:
  - `ChatService.stream_chat`: 接入三层切分、层 1 降级、非阻塞异步摘要与双轨 State 持久化

- [ ] **Step 1: Write the failing test**
  编写 `tests/test_chat_service_context.py`：
  - 验证多轮对话中消息进 messages 表；
  - 验证层 1 降级推移 `conversations.layer1_from_msg_id`；
  - 验证调模型时传入的是裁过的 `build_model_messages`，而 State 维护完整流水；
  - 验证异步摘要在后台静默运行不卡死流式输出。

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_chat_service_context.py -v`
  Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
  - 在 `app/services/chat_service.py` 中初始化 `ContextManager` 和 `SummaryService`；
  - 在每轮执行工作流前，执行层 1 降级检查，装配带前情摘要的上下文；
  - 流式完成后检查层 2 Token，满足阈值异步调度摘要任务。

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_chat_service_context.py -v`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add app/services/workflow/state.py app/services/workflow/engine.py app/services/chat_service.py tests/test_chat_service_context.py
  git commit -m "feat(workflow): integrate three-tier context management into chat service and workflow engine"
  ```

---

### Task 7: 多会话后端 REST 只读接口 (`GET /api/conversations` & `GET /api/conversations/{id}/messages`)

**Files:**
- Create: `app/schemas/conversation.py`
- Modify: `app/api/routes.py`
- Test: `tests/test_api_conversations.py`

**Interfaces:**
- Produces:
  - `GET /api/conversations`: 返回会话列表（按时间倒序、首问预览、`has_summary` 标签）
  - `GET /api/conversations/{id}/messages`: 返回指定会话全量时间正序原始消息列表

- [ ] **Step 1: Write the failing test**
  编写 `tests/test_api_conversations.py`：
  - 测试查询用户会话列表：字段完整性、排序正确性、`has_summary` 正确标记；
  - 测试按会话 ID 查询历史消息：包含 user/assistant/tool 消息，支持完整回溯；
  - 测试非法 ID 或空会话的边界返回。

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_api_conversations.py -v`
  Expected: FAIL with 404 or 405.

- [ ] **Step 3: Write minimal implementation**
  - 在 `app/schemas/conversation.py` 中定义 Pydantic Schema（`ConversationListItem`, `ConversationMessageItem`）；
  - 在 `app/api/routes.py` 中实现两个 GET 端点。

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_api_conversations.py -v`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add app/schemas/conversation.py app/api/routes.py tests/test_api_conversations.py
  git commit -m "feat(api): implement read-only conversation list and message history endpoints"
  ```

---

### Task 8: 前端多会话侧栏交互与历史回载 (`app/static/index.html`)

**Files:**
- Modify: `app/static/index.html`
- Test: `tests/test_frontend_routes.py` (验证静态页面与 API 联通性)

**Interfaces:**
- Consumes: `/api/conversations`, `/api/conversations/{id}/messages`, `/api/chat/stream`
- Produces:
  - 响应式多会话侧栏 UI；
  - 「+ 新对话」重置与历史会话切换回载能力；
  - 首问预览与 `[已摘要]` 醒目标签；
  - 加载失败静默降级（不弹窗、不阻断主聊天）。

- [ ] **Step 1: Write the failing test / verification**
  在 `tests/test_frontend_routes.py` 中验证 `index.html` 包含侧栏 DOM 元素结构（`.session-sidebar`, `.btn-new-chat`, `.session-list`）及两个 API 路由可访性。

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_frontend_routes.py -v`
  Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
  在 `app/static/index.html` 中加入侧栏 HTML 结构、美化 CSS 样式，并编写加载会话列表、点击会话拉取回显消息、点击新对话重置状态的前端 JS 逻辑，加入异常静默降级处理。

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_frontend_routes.py -v`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add app/static/index.html tests/test_frontend_routes.py
  git commit -m "feat(ui): add multi-session sidebar with history loading and summary badges"
  ```

---

### Task 9: 启动脚本与环境自愈集成 (`run.py`)

**Files:**
- Modify: `run.py`
- Test: `tests/test_run_integration.py`

**Interfaces:**
- Consumes: `init_ch07_db`, `check_budget_on_startup`
- Produces:
  - 自动执行 `init_ch07_db` 校验并迁移表结构；
  - 执行 `check_budget_on_startup`，若上下文预算不可用立即报警。

- [ ] **Step 1: Write the failing test**
  编写 `tests/test_run_integration.py`，测试 `run.py` 的存储检查调用了 `init_ch07_db` 并校验了 Token 预算自检。

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_run_integration.py -v`
  Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**
  在 `run.py` 的 `check_storage_readiness` 中引入 `init_ch07_db` 执行 DDL 迁移，并在启动前调用 `check_budget_on_startup()`。

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_run_integration.py -v`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add run.py tests/test_run_integration.py
  git commit -m "feat(startup): integrate ch07 database migrations and budget startup self-check"
  ```

---

### Task 10: 综合端到端全链路验收测试与全量回归套件 (`tests/test_ch07_acceptance.py`)

**Files:**
- Create: `tests/test_ch07_acceptance.py`
- Modify: `dev-notes/ch07.md`

**Interfaces:**
- 全面覆盖用户 5 大验收标准：
  1. 连续聊 20 轮以上，Token 不爆、系统不崩；
  2. 演示配置（18000 窗口）：滑窗 5650、层 1 3954、层 2 1695，完整日志级联（层 1 降级 -> 层 2 触发摘要 -> 摘要生成第 N 段 -> 问最初订单能根据梗概准确答出）；
  3. 默认 128k 窗口下聊 20 轮：不触发任何降级与摘要（装得下就不压）；
  4. 后台摘要非阻塞验证与 `log/app.log` grep `model_ctx` / `history_ctx` 日志留痕；
  5. 前端侧栏多会话开辟、切回与消息完整回载验证。

- [ ] **Step 1: Write the failing acceptance test**
  编写 `tests/test_ch07_acceptance.py`，完整组织上述 5 组验收用例。

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_ch07_acceptance.py -v`
  Expected: 模拟环境验证并发现薄弱环节。

- [ ] **Step 3: Fix & polish implementation**
  针对验收测试中的边界细节（如多段梗概投影拼接格式、Token 边界严格判断）进行打磨优化，确保全部 PASS。

- [ ] **Step 4: Run full regression test suite**
  Run: `pytest -v` (包含全量 250+ 测试用例)
  Expected: 100% PASS.

- [ ] **Step 5: Commit & Record Dev Notes**
  ```bash
  git add tests/test_ch07_acceptance.py dev-notes/ch07.md
  git commit -m "test(acceptance): add comprehensive ch07 acceptance suite and pass all verifications"
  ```
