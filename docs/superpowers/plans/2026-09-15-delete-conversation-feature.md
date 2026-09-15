# 删除对话历史功能实现计划 (Delete Conversation Feature Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 AstroBot 提供会话删除能力，包含后端 `DELETE /api/conversations/{id}` 级联删除 API 与前端 Web 聊天界面侧边栏悬浮垃圾桶删除按钮及确认交互。

**Architecture:** 
- 后端在 `app/api/routes.py` 暴露 `DELETE /api/conversations/{id}` 路由，基于 SQLAlchemy 异步事务处理外键级联关系（安全清理 `messages`, `conversation_summaries`, `tickets`, `tool_audit_logs` 并解绑 `low_confidence_questions`，最终删除 `conversations` 主记录）；
- 前端在 `app/static/index.html` 侧边栏的每个会话项上提供悬浮/常显的精美垃圾桶图标，阻止事件冒泡并带有二次确认对话框，删除后自动刷新会话列表，若删除的是当前选中会话则自动重置为空白欢迎界面。

**Tech Stack:** FastAPI, SQLAlchemy 2.0 (AsyncIO), HTML5 / Modern CSS, JavaScript (Vanilla ES6+), Pytest / pytest-asyncio.

**Spec:** 用户明确要求“怎么删除没用的对话历史”，并确认通过方案为系统添加后端接口与前端侧边栏删除按钮。

## Global Constraints

- 遵循严格 TDD（RED -> GREEN -> REFACTOR），业务代码编写前必须先运行失败测试；
- 数据库操作必须使用同一个 `AsyncSession` 事务原子提交，失败回滚；
- 前端交互严禁事件冒泡（`event.stopPropagation()`），防止误触发打开会话；
- 删除当前正在查看的会话时，必须安全清理主聊天界面恢复至初始欢迎态，避免页面残留已删除会话的脏状态；
- 全库既有 479 项测试 100% 保持通过，零回归。

---

### Task 1: 后端 `DELETE /api/conversations/{id}` 级联删除接口

**Files:**
- Create: `tests/test_api_conversations_delete.py`
- Modify: `app/api/routes.py:200-245`

**Interfaces:**
- Produces: `DELETE /api/conversations/{id}` -> `{"success": bool, "message": str, "conversation_id": int}`
- Error Response: 404 if conversation not found

- [ ] **Step 1: 编写失败的单元测试 `tests/test_api_conversations_delete.py`**

```python
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select
from main import app
from app.db.session import AsyncSessionLocal
from app.models import Conversation, Message, Ticket, ConversationSummary
from app.models.tool_audit_log import ToolAuditLog
from app.models.low_confidence import LowConfidenceQuestion

@pytest.mark.asyncio
async def test_delete_conversation_not_found():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.delete("/api/conversations/99999999")
        assert res.status_code == 404
        assert "不存在" in res.json()["detail"]

@pytest.mark.asyncio
async def test_delete_conversation_cascade_success():
    async with AsyncSessionLocal() as session:
        # 创建测试会话及关联数据
        conv = Conversation(user_id="test_user", status="进行中")
        session.add(conv)
        await session.flush()
        conv_id = conv.id

        msg = Message(conversation_id=conv_id, role="user", content="测试消息")
        summary = ConversationSummary(conversation_id=conv_id, seq=1, from_msg_id=1, upto_msg_id=1, content="测试摘要")
        ticket = Ticket(ticket_no=f"T_TEST_{conv_id}", conversation_id=conv_id, description="测试工单", ticket_type="售后")
        audit = ToolAuditLog(conversation_id=conv_id, tool_name="query_order", tool_source="builtin", status="成功")
        lcq = LowConfidenceQuestion(conversation_id=conv_id, raw_question="低置信测试", source="self_check")

        session.add_all([msg, summary, ticket, audit, lcq])
        await session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.delete(f"/api/conversations/{conv_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["conversation_id"] == conv_id

    # 验证数据库中级联清理完全
    async with AsyncSessionLocal() as session:
        c = await session.get(Conversation, conv_id)
        assert c is None

        m_res = await session.execute(select(Message).where(Message.conversation_id == conv_id))
        assert len(m_res.scalars().all()) == 0

        s_res = await session.execute(select(ConversationSummary).where(ConversationSummary.conversation_id == conv_id))
        assert len(s_res.scalars().all()) == 0

        t_res = await session.execute(select(Ticket).where(Ticket.conversation_id == conv_id))
        assert len(t_res.scalars().all()) == 0

        a_res = await session.execute(select(ToolAuditLog).where(ToolAuditLog.conversation_id == conv_id))
        assert len(a_res.scalars().all()) == 0

        # lcq 应该解除关联 (conversation_id 为 None)
        l_res = await session.execute(select(LowConfidenceQuestion).where(LowConfidenceQuestion.raw_question == "低置信测试"))
        item = l_res.scalar_one_or_none()
        if item:
            assert item.conversation_id is None
```

- [ ] **Step 2: 运行测试验证失败 (RED)**
Run: `pytest tests/test_api_conversations_delete.py -v`
Expected: FAIL (405 Method Not Allowed 或 404 Route Not Found)

- [ ] **Step 3: 在 `app/api/routes.py` 中实现 `DELETE /api/conversations/{id}` 接口**
- 在 `app/api/routes.py` 中实现接口：
  - 导入 `delete`, `update`
  - 检查会话是否存在；若不存在返回 404；
  - 级联处理 `low_confidence_questions` (set conversation_id = None)；
  - 清理 `messages`, `conversation_summaries`, `tickets`, `tool_audit_logs`；
  - 删除 `Conversation` 并 `await db.commit()`；
  - 返回成功 JSON。

- [ ] **Step 4: 运行测试验证通过 (GREEN)**
Run: `pytest tests/test_api_conversations_delete.py -v`
Expected: PASS

- [ ] **Step 5: 提交后端变更**
```bash
git add app/api/routes.py tests/test_api_conversations_delete.py
git commit -m "feat(api): implement DELETE /api/conversations/{id} with cascade cleanup"
```

---

### Task 2: 前端侧边栏删除按钮与交互设计

**Files:**
- Create: `tests/test_frontend_conversation_delete.py`
- Modify: `app/static/index.html:1480-1550`

**Interfaces:**
- Consumes: `DELETE /api/conversations/{id}`
- UI Elements: `.session-delete-btn`, `deleteSession(event, conversationId)`

- [ ] **Step 1: 编写前端契约测试 `tests/test_frontend_conversation_delete.py`**
- 验证 `index.html` 中包含 `.session-delete-btn` 样式规则；
- 验证包含 `deleteSession` 函数定义；
- 验证包含 `DELETE` 方法的 `fetch('/api/conversations/'`；
- 验证包含 `stopPropagation`；
- 验证包含 `confirm` 拦截。

- [ ] **Step 2: 运行测试验证失败 (RED)**
Run: `pytest tests/test_frontend_conversation_delete.py -v`
Expected: FAIL

- [ ] **Step 3: 修改 `app/static/index.html` 添加样式与交互**
- 添加 CSS 样式：
  ```css
  .session-delete-btn {
    opacity: 0;
    transition: opacity 0.2s ease, color 0.2s ease, transform 0.1s ease;
    background: transparent;
    border: none;
    color: var(--text-muted, #94a3b8);
    cursor: pointer;
    padding: 2px 4px;
    border-radius: 4px;
    font-size: 13px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }
  .session-item:hover .session-delete-btn {
    opacity: 1;
  }
  .session-delete-btn:hover {
    color: #ef4444;
    transform: scale(1.15);
    background: rgba(239, 68, 68, 0.1);
  }
  ```
- 更新 `loadSessions()` 渲染模板，在 `.session-item-header` 中插入删除按钮；
- 实现 `deleteSession(event, conversationId)` 函数：
  - `event.stopPropagation();`
  - `if (!confirm(...)) return;`
  - `await fetch('/api/conversations/' + conversationId, { method: 'DELETE' });`
  - 若 `conversationId === currentConversationId`：重置界面为初始态；
  - `await loadSessions();`

- [ ] **Step 4: 运行前端契约测试验证通过 (GREEN)**
Run: `pytest tests/test_frontend_conversation_delete.py -v`
Expected: PASS

- [ ] **Step 5: 提交前端变更**
```bash
git add app/static/index.html tests/test_frontend_conversation_delete.py
git commit -m "feat(ui): add session delete button and interaction in sidebar"
```

---

### Task 3: 整体集成验证与文档更新

**Files:**
- Modify: `dev-notes/ch08.md`

- [ ] **Step 1: 运行全套相关测试与全量回归**
Run: `pytest tests/test_api_conversations_delete.py tests/test_frontend_conversation_delete.py tests/test_run_startup_optimization.py tests/test_ch08_acceptance.py -v`
Expected: All PASS

- [ ] **Step 2: 记录 `dev-notes/ch08.md` 阶段 14**
- [ ] **Step 3: 提交文档**
```bash
git add dev-notes/ch08.md
git commit -m "docs(dev-notes): record phase 14 conversation delete feature"
```
