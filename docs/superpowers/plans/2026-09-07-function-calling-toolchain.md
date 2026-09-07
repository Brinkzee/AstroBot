# Function Calling 工具链与单轮收敛 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 FastAPI + SQLAlchemy Async + MySQL + LangChain @tool 搭建 Function Calling 工具链，实现模型自主调工具、状态帧推送、单轮收敛流式回答与会话数据落盘。

**Architecture:** 采用编排式单轮流式收敛架构（Orchestrated Single-Turn Pipeline）。API 层接收请求后通过 `chat_service` 驱动状态机：先识别意图，若触发工具调用则落盘并推 `tool_start` 帧，经 `ToolExecutor` 异步执行后推 `tool_end` 帧并将结果落盘，最后回灌模型通过 `astream` 逐 Token 吐出最终解答。

**Tech Stack:** FastAPI, SQLAlchemy 2.0 (Async), aiomysql, MySQL 8.0 (WSL2 Docker), LangChain (@tool, bind_tools, ChatOpenAI), Pydantic v2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-07-function-calling-toolchain-design.md`

## Global Constraints
- Python 版本为 3.12，包管理使用 pip
- 数据库驱动使用 `aiomysql`，连接池基于 `create_async_engine`
- 数据表严格对应 `sql/ch02-ddl.sql` 定义的四张表：`conversations`, `messages`, `faq`, `tickets`
- 工具定义严格使用 LangChain 官方 `@tool` 装饰器
- 工具执行超时时间固定为 5.0 秒，至多重试 1 次
- 严格单轮工具调用收敛，不做多轮 Agent Loop，不做向量检索或 RAG
- 聊天页面前端修改按照工作要求采用 Vibe Coding 模式快速交付

---

### Task 1: 依赖更新与异步数据库基础设施

**Files:**
- Modify: `requirements.txt`
- Modify: `app/config.py`
- Create: `docker-compose.yml`
- Create: `scripts/start_mysql.ps1`
- Create: `app/db/__init__.py`
- Create: `app/db/session.py`
- Test: `tests/test_db_session.py`

**Interfaces:**
- Produces:
  - `app.config.settings.database_url: str`
  - `app.db.session.engine`: SQLAlchemy `AsyncEngine`
  - `app.db.session.AsyncSessionLocal`: `async_sessionmaker[AsyncSession]`
  - `app.db.session.get_db() -> AsyncGenerator[AsyncSession, None]`
  - `app.db.session.Base`: `DeclarativeBase`

- [ ] **Step 1: 编写失败的数据库基础设施单元测试**

```python
# tests/test_db_session.py
import pytest
from sqlalchemy import text
from app.db.session import engine, AsyncSessionLocal, get_db

@pytest.mark.asyncio
async def test_async_session_maker():
    assert AsyncSessionLocal is not None
    async with AsyncSessionLocal() as session:
        assert session is not None

@pytest.mark.asyncio
async def test_get_db_generator():
    gen = get_db()
    session = await anext(gen)
    assert session is not None
    await gen.aclose()
```

- [ ] **Step 2: 运行测试验证其失败**
Run: `pytest tests/test_db_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.db'`

- [ ] **Step 3: 安装依赖并编写数据库配置与 Session 模块**
更新 `requirements.txt` 添加 `aiomysql>=0.2.0`, `cryptography>=43.0.0`。
在 `app/config.py` 增加 `database_url`。
创建 `docker-compose.yml` 与 `scripts/start_mysql.ps1`。
实现 `app/db/session.py`。

- [ ] **Step 4: 运行测试验证其通过**
Run: `pytest tests/test_db_session.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**
```bash
git add requirements.txt app/config.py docker-compose.yml scripts/start_mysql.ps1 app/db/ tests/test_db_session.py
git commit -m "feat(db): add async sqlalchemy session infrastructure and mysql docker config"
```

---

### Task 2: SQLAlchemy ORM 数据模型与 FAQ 测试数据灌入

**Files:**
- Create: `app/models/__init__.py`
- Create: `app/models/conversation.py`
- Create: `app/models/message.py`
- Create: `app/models/faq.py`
- Create: `app/models/ticket.py`
- Create: `scripts/seed_data.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `app.db.session.Base`, `app.db.session.AsyncSessionLocal`
- Produces:
  - `Conversation(id, user_id, status, created_at, updated_at)`
  - `Message(id, conversation_id, role, content, tool_calls, tool_call_id, created_at)`
  - `FAQ(id, question, answer, category, created_at, updated_at)`
  - `Ticket(ticket_no, conversation_id, description, ticket_type, status, created_at)`
  - `scripts.seed_data.seed_all_data(session: AsyncSession)`

- [ ] **Step 1: 编写数据模型与 Seed 验证的失败单测**

```python
# tests/test_models.py
import pytest
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.faq import FAQ
from app.models.ticket import Ticket

def test_model_attributes():
    conv = Conversation(user_id="u1", status="进行中")
    assert conv.user_id == "u1"
    msg = Message(conversation_id=1, role="user", content="hello")
    assert msg.role == "user"
    faq = FAQ(question="退货政策", answer="7天无理由", category="售后")
    assert faq.category == "售后"
    ticket = Ticket(ticket_no="T123", conversation_id=1, description="退款", ticket_type="售后")
    assert ticket.ticket_no == "T123"
```

- [ ] **Step 2: 运行测试验证其失败**
Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models'`

- [ ] **Step 3: 编写 ORM 模型与 Seed 灌测脚本**
实现 `app/models/` 下的各实体模型，映射 `sql/ch02-ddl.sql`。
编写 `scripts/seed_data.py`，预置「退货政策说明」（包含“退货政策”）和「运费标准与包邮政策」（包含“运费”）。

- [ ] **Step 4: 运行测试验证其通过**
Run: `pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**
```bash
git add app/models/ scripts/seed_data.py tests/test_models.py
git commit -m "feat(models): add orm entities and faq seed script aligned with ch02-ddl.sql"
```

---

### Task 3: LangChain 五大业务工具实现

**Files:**
- Create: `app/tools/__init__.py`
- Create: `app/tools/business_tools.py`
- Test: `tests/test_business_tools.py`

**Interfaces:**
- Produces:
  - `query_order(order_id: str) -> str`
  - `query_product(product_id_or_name: str) -> str`
  - `query_logistics(order_id: str) -> str`
  - `query_faq(keyword: str, db: Optional[AsyncSession] = None) -> str`
  - `create_ticket(conversation_id: int, description: str, ticket_type: str = "售后", db: Optional[AsyncSession] = None) -> str`

- [ ] **Step 1: 编写工具调用失败测试**

```python
# tests/test_business_tools.py
import pytest
from app.tools.business_tools import query_order, query_product, query_logistics, query_faq, create_ticket

def test_query_order():
    res = query_order.invoke({"order_id": "1001"})
    assert "1001" in res
    assert "订单状态" in res

def test_query_product():
    res = query_product.invoke({"product_id_or_name": "羽绒服"})
    assert "羽绒服" in res
    assert "库存" in res

def test_query_logistics():
    res = query_logistics.invoke({"order_id": "1001"})
    assert "1001" in res
    assert "物流" in res or "快递" in res
```

- [ ] **Step 2: 运行测试验证其失败**
Run: `pytest tests/test_business_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.tools'`

- [ ] **Step 3: 使用 LangChain @tool 实现业务工具**
实现 `query_order`, `query_product`, `query_logistics`, `query_faq`, `create_ticket`。

- [ ] **Step 4: 运行测试验证其通过**
Run: `pytest tests/test_business_tools.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**
```bash
git add app/tools/business_tools.py tests/test_business_tools.py
git commit -m "feat(tools): implement five business tools using langchain @tool"
```

---

### Task 4: 工具基础设施 (Registry, Validation, Executor 与异常回灌)

**Files:**
- Create: `app/tools/registry.py`
- Create: `app/tools/executor.py`
- Test: `tests/test_tool_executor.py`

**Interfaces:**
- Consumes: `app.tools.business_tools`
- Produces:
  - `ToolRegistry`: `register()`, `get_tool(name)`, `get_all_tools()`
  - `ToolExecutor`: `execute(tool_call: dict, db: Optional[AsyncSession] = None) -> dict`
  - 规范化错误回灌 JSON 格式

- [ ] **Step 1: 编写注册中心与执行器异常容错单测**

```python
# tests/test_tool_executor.py
import pytest
from app.tools.registry import default_tool_registry
from app.tools.executor import ToolExecutor

def test_tool_registry_contains_all_tools():
    tools = default_tool_registry.get_all_tools()
    tool_names = [t.name for t in tools]
    assert "query_order" in tool_names
    assert "query_logistics" in tool_names
    assert "query_product" in tool_names
    assert "query_faq" in tool_names
    assert "create_ticket" in tool_names

@pytest.mark.asyncio
async def test_tool_executor_success():
    executor = ToolExecutor(default_tool_registry)
    result = await executor.execute({"name": "query_order", "args": {"order_id": "1001"}, "id": "call_1"})
    assert result["success"] is True
    assert "1001" in result["output"]

@pytest.mark.asyncio
async def test_tool_executor_invalid_tool():
    executor = ToolExecutor(default_tool_registry)
    result = await executor.execute({"name": "non_existent", "args": {}, "id": "call_2"})
    assert result["success"] is False
    assert "未找到工具" in result["error"]
```

- [ ] **Step 2: 运行测试验证其失败**
Run: `pytest tests/test_tool_executor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 ToolRegistry 与具备超时重试及异常捕获的 ToolExecutor**
实现参数 Schema 校验、`asyncio.wait_for` 超时、重试与结构化回灌封装。

- [ ] **Step 4: 运行测试验证其通过**
Run: `pytest tests/test_tool_executor.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**
```bash
git add app/tools/registry.py app/tools/executor.py tests/test_tool_executor.py
git commit -m "feat(tools): add tool registry and executor with validation, timeout and error recovery"
```

---

### Task 5: 单轮收敛编排服务与会话落盘 (ChatService)

**Files:**
- Create: `app/services/chat_service.py`
- Test: `tests/test_chat_service.py`

**Interfaces:**
- Consumes: `app.db.session.AsyncSessionLocal`, `app.models`, `app.tools.executor.ToolExecutor`, `app.llm.get_chat_model`
- Produces:
  - `ChatService.stream_chat(db: AsyncSession, conversation_id: Optional[int], message: str) -> AsyncGenerator[dict, None]`

- [ ] **Step 1: 编写单轮编排状态机的单元测试 (Mock 模型)**

```python
# tests/test_chat_service.py
import pytest
from unittest.mock import AsyncMock, patch
from app.services.chat_service import ChatService

@pytest.mark.asyncio
async def test_stream_chat_pure_text(db_session):
    # 测试无工具调用时的纯对话分支
    service = ChatService()
    # 模拟纯文本回复，验证落库与事件帧
```

- [ ] **Step 2: 运行测试验证其失败**
Run: `pytest tests/test_chat_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.chat_service'`

- [ ] **Step 3: 实现具备单轮收敛与数据落盘的 ChatService**
实现会话外壳与消息流水的读写；实现 `tool_start`、`tool_end`、`text` 事件流式推送；实现单轮调用即收敛。

- [ ] **Step 4: 运行测试验证其通过**
Run: `pytest tests/test_chat_service.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**
```bash
git add app/services/chat_service.py tests/test_chat_service.py
git commit -m "feat(services): implement single-turn convergence chat service with db persistence"
```

---

### Task 6: FastAPI SSE 路由对接与集成测试

**Files:**
- Modify: `app/api/routes.py`
- Modify: `app/schemas/chat.py`
- Test: `tests/test_api_chat_stream.py`

**Interfaces:**
- Consumes: `app.services.chat_service.ChatService`, `app.db.session.get_db`
- Produces:
  - `POST /api/chat/stream` 支持 `conversation_id` / `session_id`，流式吐出统一 JSON 帧

- [ ] **Step 1: 编写 API 接口集成测试**
测试流式接口输出的 JSON 事件序列，验证 `tool_start`, `tool_end`, `text`, `[DONE]` 协议。

- [ ] **Step 2: 运行测试验证其失败**
Run: `pytest tests/test_api_chat_stream.py -v`
Expected: FAIL

- [ ] **Step 3: 改造 `app/api/routes.py` 对接 `ChatService`**
修改路由逻辑，统一输出 `data: {"event_type": ..., ...}\n\n`。

- [ ] **Step 4: 运行测试验证其通过**
Run: `pytest tests/test_api_chat_stream.py -v`
Expected: PASS

- [ ] **Step 5: 提交代码**
```bash
git add app/api/routes.py app/schemas/chat.py tests/test_api_chat_stream.py
git commit -m "feat(api): upgrade sse endpoint with tool calling event stream"
```

---

### Task 7: 客服聊天 Web 页面前端改造 (Vibe Coding)

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- 前端 JS 监听 SSE 事件：
  - `event_type === "tool_start"`: 渲染加载态徽章 `[⚡ 正在调用工具: xxx ...]`
  - `event_type === "tool_end"`: 徽章平滑切换为完成态小胶囊 `[📦 已调用: xxx]`
  - `event_type === "text"`: 流式追加正文文本

- [ ] **Step 1: 修改 index.html 升级事件解析与气泡徽章动效**
- [ ] **Step 2: 在浏览器启动服务进行人工交互校验**
- [ ] **Step 3: 提交代码**
```bash
git add app/static/index.html
git commit -m "feat(ui): add tool trajectory badge and event-type sse stream rendering"
```

---

### Task 8: 全面验证与三大验收标准交付

**Files:**
- Modify: `dev-notes/ch02.md`
- Create: `walkthrough.md`

- [ ] **Step 1: 启动 MySQL 容器并执行建表与灌测脚本**
- [ ] **Step 2: 运行全量单元测试套件 `pytest -v`**
- [ ] **Step 3: 浏览器实测验收标准 1（订单 1001 物流查询与徽章）**
- [ ] **Step 4: 浏览器实测验收标准 2（退货政策 FAQ 命中）**
- [ ] **Step 5: 浏览器实测验收标准 3（邮费是多少 关键词漏召回验证）**
- [ ] **Step 6: 记录阶段手记与完结交付说明**
```bash
git add dev-notes/ch02.md
git commit -m "docs(dev-notes): record completion of all ch02 tasks and acceptance verification"
```
