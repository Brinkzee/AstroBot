# 即插即用工具体系与 MCP 协议集成实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 AstroBot 客服系统的工具层升级为支持统一注册、JSON Schema 参数校验、读写权限隔离、超时分类重试、无外键审计留痕、双独立进程 MCP Server 接入（Streamable HTTP）以及 LangGraph Interrupt 工单人机确认流的即插即用工具系统。

**Architecture:** 采用“动态统一注册中心 + 契约化执行网关 + 原生 Interrupt 交互”架构。统一 `ToolRegistry` 纳管内置工具与 `MultiServerMCPClient` 动态现问现拿；`ToolExecutor` 统筹 Schema 校验、权限门禁、网络重试与中文序列化；`tool_audit_logs` 独立事务全量留痕；工单发起通过 LangGraph 原生 `interrupt` 下发前端预览卡片，通过 `POST /api/chat/resume` 恢复执行。

**Tech Stack:** FastAPI, SQLAlchemy 2.0 (Async), MySQL 8.0 / SQLite (双方言), LangChain 0.3, LangGraph (interrupt / Command resume), `mcp` Python SDK (FastMCP, Streamable HTTP), `langchain-mcp-adapters` (`MultiServerMCPClient`), Pydantic v2, pytest / pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-14-plug-and-play-tools-design.md`

## Global Constraints

- **DDL 规范**：严格对齐 `sql/ch08-ddl.sql`，`tool_audit_logs` 不挂外键，`conversation_id`、`tool_name`、`status` 建普通索引，全库统一 ENGINE=InnoDB、CHARSET=utf8mb4；
- **MCP 协议与传输**：Server 端必须使用官方 `mcp` SDK 的 `FastMCP` 并走 `streamable-http` 传输；Client 端必须使用 `langchain-mcp-adapters` 的 `MultiServerMCPClient`；
- **权限安全**：外部 MCP 工具零信任，默认一律只读；系统唯一写操作 `create_ticket` 必须满足客户明确诉求且经过前端人机确认；
- **弹性与重试**：业务空结果不重试；写操作恒不自动重试；仅瞬时网络抖动重试 1 次；错误分诊为参数不合法、查询落空、真故障；JSON 序列化 `ensure_ascii=False`；
- **审计隔离**：审计写入失败绝不反向阻拦工具执行或影响用户答复；
- **向后兼容**：第 5 章投诉流程中前端原有的快捷建单按钮保持原样，直接调用 `POST /api/tickets` 接口；
- **TDD 规则**：除前端 Vibe Coding 外，所有功能严格先编写失败测试，再实现业务代码，全程验证后再提交。

---

## Task Decomposition

### Task 1: 审计数据模型与 DDL 迁移脚本 (`tool_audit_logs` & `init_ch08_db.py`)

**Files:**
- Create: `app/models/tool_audit_log.py`
- Modify: `app/models/__init__.py`
- Create: `scripts/init_ch08_db.py`
- Test: `tests/test_ch08_models.py`

**Interfaces:**
- Produces: `ToolAuditLog` ORM 模型，包含 `id`, `conversation_id`, `tool_call_id`, `tool_name`, `tool_source`, `mcp_server`, `arguments`, `result_summary`, `status`, `error_message`, `retry_count`, `duration_ms`, `created_at`。
- Produces: `init_ch08_db()` 幂等执行 `sql/ch08-ddl.sql`。

- [ ] **Step 1: Write the failing model & migration test**

```python
# tests/test_ch08_models.py
import pytest
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.tool_audit_log import ToolAuditLog
from scripts.init_ch08_db import init_ch08_db

@pytest.mark.asyncio
async def test_init_ch08_db_and_tool_audit_log_crud():
    await init_ch08_db()
    async with AsyncSessionLocal() as session:
        log = ToolAuditLog(
            conversation_id=99901,
            tool_call_id="call_test_001",
            tool_name="query_order",
            tool_source="builtin",
            arguments={"order_id": "1001"},
            result_summary="已发货",
            status="成功",
            retry_count=0,
            duration_ms=45,
        )
        session.add(log)
        await session.commit()
        await session.refresh(log)
        assert log.id is not None
        assert log.status == "成功"
        assert log.arguments == {"order_id": "1001"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ch08_models.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'app.models.tool_audit_log')

- [ ] **Step 3: Implement `ToolAuditLog` model and `init_ch08_db.py`**

创建 `app/models/tool_audit_log.py`，并在 `scripts/init_ch08_db.py` 中实现双方言（MySQL & SQLite）幂等表结构创建。导出至 `app/models/__init__.py`。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ch08_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/models/tool_audit_log.py app/models/__init__.py scripts/init_ch08_db.py tests/test_ch08_models.py
git commit -m "feat(db): implement tool_audit_logs model and ch08 ddl migration script"
```

---

### Task 2: 双独立进程业务 MCP Server 实现 (`LogisticsServer` & `AftersaleServer`)

**Files:**
- Create: `scripts/mcp_logistics_server.py`
- Create: `scripts/mcp_aftersale_server.py`
- Test: `tests/test_mcp_servers.py`

**Interfaces:**
- Produces: 物流 Server 运行在 8001 端口，暴露 `query_logistics(order_id: str)`。
- Produces: 售后 Server 运行在 8002 端口，暴露 `check_warranty(order_id: str, product_name: str = "")` 与 `query_return_progress(return_id_or_order_id: str)`，支持动态注入新工具。

- [ ] **Step 1: Write the failing MCP Server unit & integration test**

```python
# tests/test_mcp_servers.py
import pytest
from scripts.mcp_logistics_server import create_logistics_server
from scripts.mcp_aftersale_server import create_aftersale_server

@pytest.mark.asyncio
async def test_logistics_mcp_server_tools():
    server = create_logistics_server()
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    assert "query_logistics" in tool_names

@pytest.mark.asyncio
async def test_aftersale_mcp_server_tools():
    server = create_aftersale_server()
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    assert "check_warranty" in tool_names
    assert "query_return_progress" in tool_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_servers.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'scripts.mcp_logistics_server')

- [ ] **Step 3: Implement `mcp_logistics_server.py` and `mcp_aftersale_server.py`**

使用官方 `mcp` 的 `FastMCP` 实现两个 Server，封装工厂函数 `create_logistics_server()` / `create_aftersale_server()` 并支持 `__main__` 命令行运行（使用 `transport="streamable-http"`）。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_servers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/mcp_logistics_server.py scripts/mcp_aftersale_server.py tests/test_mcp_servers.py
git commit -m "feat(mcp): implement standalone logistics and aftersale fastmcp servers"
```

---

### Task 3: 统一工具注册中心与 MCP 客户端动态发现 (`app/tools/registry.py`)

**Files:**
- Modify: `app/config.py`
- Modify: `app/tools/registry.py`
- Modify: `app/tools/__init__.py`
- Test: `tests/test_tool_registry.py`

**Interfaces:**
- Consumes: `MultiServerMCPClient` from `langchain_mcp_adapters.client`.
- Produces: `ToolRegistry.get_all_tools()` 异步方法，合并内置工具与动态 MCP 工具，打上 `tool_source` 与 `mcp_server` 元数据。下线原内置 `query_logistics`。

- [ ] **Step 1: Write the failing registry test**

```python
# tests/test_tool_registry.py
import pytest
from app.tools.registry import ToolRegistry

@pytest.mark.asyncio
async def test_tool_registry_builtin_and_dynamic_mcp_tools():
    registry = ToolRegistry()
    tools = await registry.get_all_tools()
    tool_names = [t.name for t in tools]
    # 原内置 query_logistics 下线，保留 order, product, faq, ticket
    assert "query_order" in tool_names
    assert "create_ticket" in tool_names
    assert "query_product" in tool_names
    assert "query_faq" in tool_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_registry.py -v`
Expected: FAIL (get_all_tools is not an async method or still contains old 5 tools synchronously)

- [ ] **Step 3: Implement dynamic discovery in `app/tools/registry.py`**

配置 `MultiServerMCPClient`，实现 `async get_all_tools()`，并在单个 MCP Server 无法连接时优雅降级并记录警告日志。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/tools/registry.py app/tools/__init__.py tests/test_tool_registry.py
git commit -m "feat(tools): implement unified dynamic tool registry with multiserver mcp discovery"
```

---

### Task 4: 参数 Schema 校验器与执行分诊/重试引擎 (`app/tools/executor.py`)

**Files:**
- Modify: `app/tools/executor.py`
- Test: `tests/test_tool_executor_ch08.py`

**Interfaces:**
- Produces: `ToolExecutor.execute(tool_call, context)`：
  - JSON Schema 参数校验，校验失败回灌结构化说明，标记「校验拦下」；
  - 仅瞬时网络抖动重试 1 次，业务空结果不重试，写操作默认恒不重试；
  - 结果脱敏并保障 `ensure_ascii=False`。

- [ ] **Step 1: Write the failing executor test**

```python
# tests/test_tool_executor_ch08.py
import pytest
from app.tools.executor import ToolExecutor

@pytest.mark.asyncio
async def test_schema_validation_failure_feedback():
    executor = ToolExecutor()
    # 缺少 order_id 必填参数
    res = await executor.execute({"name": "query_order", "args": {}, "id": "c1"})
    assert res["success"] is False
    assert res["status"] == "校验拦下"
    assert "必填" in res["output"] or "校验" in res["output"]

@pytest.mark.asyncio
async def test_write_operation_no_retry_on_timeout():
    executor = ToolExecutor(timeout=0.01)
    res = await executor.execute({"name": "create_ticket", "args": {"description": "测试"}, "id": "c2"})
    assert res["retry_count"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_executor_ch08.py -v`
Expected: FAIL

- [ ] **Step 3: Upgrade `ToolExecutor` implementation**

实现统一参数校验器、分诊错误（参数不合法/查询落空/真故障）、网络异常白名单重试与写操作禁止重试逻辑，格式化输出。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_executor_ch08.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/tools/executor.py tests/test_tool_executor_ch08.py
git commit -m "feat(tools): implement schema validation, retry policy, and error triage in tool executor"
```

---

### Task 5: 权限门禁与独立事务审计留痕 (`app/tools/permission.py` & `app/tools/audit.py`)

**Files:**
- Create: `app/tools/permission.py`
- Create: `app/tools/audit.py`
- Modify: `app/tools/executor.py`
- Test: `tests/test_tool_permission_and_audit.py`

**Interfaces:**
- Produces: `ToolPermissionGuard.check_permission(tool_name, tool_source, context) -> (bool, str)`：
  - MCP 工具一律只读；
  - `create_ticket` 仅允许在客户有明确诉求且通过前端确认时执行。
- Produces: `AuditLogger.log_tool_call(...)` 独立异步事务落库，异常静默吞掉。

- [ ] **Step 1: Write the failing permission & audit test**

```python
# tests/test_tool_permission_and_audit.py
import pytest
from app.tools.permission import ToolPermissionGuard
from app.tools.audit import AuditLogger
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.tool_audit_log import ToolAuditLog

@pytest.mark.asyncio
async def test_permission_guard_blocks_unauthorized_mcp_write():
    guard = ToolPermissionGuard()
    allowed, reason = guard.check_permission(
        tool_name="delete_database",
        tool_source="mcp",
        is_write=True,
        context={"user_query": "删库"}
    )
    assert allowed is False
    assert "只读" in reason or "拒绝" in reason

@pytest.mark.asyncio
async def test_audit_logger_writes_even_on_permission_denied():
    await AuditLogger.log_call(
        conversation_id=8801,
        tool_call_id="c_denied",
        tool_name="delete_database",
        tool_source="mcp",
        status="权限拒绝",
        error_message="外部MCP无写权限"
    )
    async with AsyncSessionLocal() as session:
        stmt = select(ToolAuditLog).where(ToolAuditLog.tool_call_id == "c_denied")
        log = (await session.execute(stmt)).scalar_one_or_none()
        assert log is not None
        assert log.status == "权限拒绝"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_permission_and_audit.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `ToolPermissionGuard` and `AuditLogger`**

在 `app/tools/permission.py` 实现权限判断；在 `app/tools/audit.py` 实现独立 Session 异步落库；在 `app/tools/executor.py` 中嵌入门禁和审计流水线。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_permission_and_audit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/tools/permission.py app/tools/audit.py app/tools/executor.py tests/test_tool_permission_and_audit.py
git commit -m "feat(security): implement tool permission guard and independent non-blocking audit logger"
```

---

### Task 6: 意图识别 9 分类扩展与路由直通 (`pre_nodes.py` & `router.py`)

**Files:**
- Modify: `app/services/workflow/nodes/pre_nodes.py`
- Modify: `app/services/workflow/nodes/router.py`
- Test: `tests/test_intent_router_ch08.py`

**Interfaces:**
- Consumes: 用户输入提问。
- Produces: `intent_recognition_node` 输出 `"人工"` 意图；`route_by_intent` 将 `"人工"` 路由到 `"business_data"` 直通 `main_agent_node`。

- [ ] **Step 1: Write the failing 9-class intent and router test**

```python
# tests/test_intent_router_ch08.py
import pytest
from app.services.workflow.nodes.router import route_by_intent
from app.services.workflow.nodes.pre_nodes import VALID_INTENTS

def test_valid_intents_contains_human_agent():
    assert "人工" in VALID_INTENTS

def test_route_human_agent_intent_to_business_data():
    state = {"intent": "人工"}
    assert route_by_intent(state) == "business_data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_intent_router_ch08.py -v`
Expected: FAIL (assert '人工' in VALID_INTENTS fails)

- [ ] **Step 3: Update `pre_nodes.py` prompt and `router.py`**

在 `pre_nodes.py` 中将 `VALID_INTENTS` 扩充 `"人工"`，并在 Prompt 增加定义与 Few-shot；在 `router.py` 中将 `"人工"` 映射到 `"business_data"`。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_intent_router_ch08.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/workflow/nodes/pre_nodes.py app/services/workflow/nodes/router.py tests/test_intent_router_ch08.py
git commit -m "feat(router): add '人工' intent classification and route to business_data"
```

---

### Task 7: LangGraph Interrupt 工单人机确认与 Resume 流式接口 (`agent_node.py` & `routes.py`)

**Files:**
- Modify: `app/prompts/customer_service.py`
- Modify: `app/services/workflow/nodes/agent_node.py`
- Modify: `app/services/chat_service.py`
- Modify: `app/api/routes.py`
- Test: `tests/test_interrupt_resume_flow.py`

**Interfaces:**
- Produces: `main_agent_node` 遇到 `create_ticket` 且未确认时触发 `interrupt({"event_type": "ticket_preview", ...})`。
- Produces: `POST /api/chat/resume` 接口，支持 `Command(resume="confirm" | "cancel")`。

- [ ] **Step 1: Write the failing interrupt and resume flow test**

```python
# tests/test_interrupt_resume_flow.py
import pytest
from app.services.workflow.engine import WorkflowEngine
from langgraph.types import Command

@pytest.mark.asyncio
async def test_create_ticket_triggers_interrupt_and_resumes():
    engine = WorkflowEngine()
    conv_id = 99911
    # 模拟触发 interrupt
    # 验证 Command(resume="confirm") 放行并创建工单
    # 验证 Command(resume="cancel") 拒绝并记录权限拒绝审计
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interrupt_resume_flow.py -v`
Expected: FAIL

- [ ] **Step 3: Implement LangGraph `interrupt`, SSE `ticket_preview`, and `POST /api/chat/resume`**

在 `agent_node` 中调用 `interrupt`；在 `chat_service` 处理流式事件分发；在 `routes.py` 增加 `POST /api/chat/resume` 端点。更新 `customer_service.py` 提示词对缺失信息进行追问。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interrupt_resume_flow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/prompts/customer_service.py app/services/workflow/nodes/agent_node.py app/services/chat_service.py app/api/routes.py tests/test_interrupt_resume_flow.py
git commit -m "feat(workflow): implement langgraph interrupt for ticket creation and resume api"
```

---

### Task 8: 前端工单预览卡片交互 (Vibe Coding)

**Files:**
- Modify: `app/static/index.html`
- Test: `tests/test_frontend_ticket_card.py`

**Interfaces:**
- Consumes: SSE `ticket_preview` 事件（包含 `ticket_type`, `description`, `conversation_id`）。
- Produces: 动态渲染工单预览卡片，包含【确认提交】和【取消】按钮，点击调用 `POST /api/chat/resume`。

- [ ] **Step 1: Write the failing frontend contract test**

```python
# tests/test_frontend_ticket_card.py
import pytest
from bs4 import BeautifulSoup

def test_frontend_ticket_preview_card_elements():
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    assert "ticket_preview" in html
    assert "/api/chat/resume" in html
    assert "renderTicketPreviewCard" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frontend_ticket_card.py -v`
Expected: FAIL

- [ ] **Step 3: Implement frontend ticket preview card and resume fetch logic**

在 `app/static/index.html` 中增加工单预览卡片样式与事件处理函数，实现流式回载与按钮状态防重置锁定。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frontend_ticket_card.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/static/index.html tests/test_frontend_ticket_card.py
git commit -m "feat(ui): implement ticket preview card and resume interaction in frontend"
```

---

### Task 9: 启动脚本与环境就绪集成 (`run.py`)

**Files:**
- Modify: `run.py`
- Test: `tests/test_run_ch08.py`

**Interfaces:**
- Produces: `run.py` 启动时自动执行 `init_ch08_db()`，探测 MCP Server 连接状态，打印工具注册中心与 MCP 拓扑。

- [ ] **Step 1: Write the failing run.py readiness test**

```python
# tests/test_run_ch08.py
import pytest
from unittest.mock import patch, AsyncMock
from run import check_storage_readiness

@pytest.mark.asyncio
async def test_run_executes_ch08_db_init():
    with patch("scripts.init_ch08_db.init_ch08_db", new=AsyncMock()) as mock_init:
        await check_storage_readiness()
        mock_init.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_ch08.py -v`
Expected: FAIL

- [ ] **Step 3: Integrate Ch08 readiness into `run.py`**

在 `run.py` 的存储与环境检查中集成 `init_ch08_db()`，并在仪表盘中打印已发现工具数与 MCP Server 状态。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run_ch08.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add run.py tests/test_run_ch08.py
git commit -m "feat(startup): integrate ch08 ddl migration and mcp readiness into run.py"
```

---

### Task 10: 综合端到端全链路验收测试与全量回归套件 (`tests/test_ch08_acceptance.py`)

**Files:**
- Create: `tests/test_ch08_acceptance.py`
- Test: 全量验收 6 大用例与历史 397+ 项测试回归

**Interfaces:**
- Produces: 完整覆盖用户定义的 6 大验收标准：
  1. 纯注册新工具零改动核心代码可用；
  2. 双 MCP Server 独立进程运行与物流轨迹 MCP 查询；
  3. MCP Server 新增工具重启后客服系统免重启感知；
  4. 追问补齐问题 -> 预览卡片 -> 确认提交 -> tickets 落库并返回工单号；
  5. 预览卡片 -> 取消 -> 工单未建，`tool_audit_logs` 状态为「权限拒绝」；
  6. 读超时重试与审计，写超时不自动重试（重试次数为 0）。

- [ ] **Step 1: Write the 6 comprehensive acceptance tests**

在 `tests/test_ch08_acceptance.py` 中编写对齐 6 大验收标准的真实端到端测试。

- [ ] **Step 2: Run acceptance tests to verify they all pass**

Run: `pytest tests/test_ch08_acceptance.py -v`
Expected: 6 passed

- [ ] **Step 3: Run full repository regression test**

Run: `pytest -q`
Expected: 100% PASS, 0 failures

- [ ] **Step 4: Commit**

```bash
git add tests/test_ch08_acceptance.py
git commit -m "test(acceptance): add comprehensive e2e acceptance suite for ch08 plug-and-play tools"
```
