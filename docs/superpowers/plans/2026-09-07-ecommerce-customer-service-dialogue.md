# 电商智能客服系统纯对话模块实施计划 (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 FastAPI + LangChain 构建电商智能客服纯对话系统，跑通多轮 SSE 流式对话、客服 Prompt 约束、多轮历史 Token 预算裁剪以及售后诉求结构化提取。

**Architecture:** 统一使用 OpenAI 协议对接上游模型，服务端内存维护 `SessionManager` 记录多轮会话并使用 LangChain 官方 `trim_messages` 进行滑动窗口 token 预算裁剪；FastAPI 对外暴露 SSE 流式端点与结构化售后提取端点。

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, LangChain 0.3, langchain-openai, tiktoken, pydantic-settings, pytest, pytest-asyncio, httpx.

**Spec:** `docs/superpowers/specs/2026-09-07-ecommerce-customer-service-dialogue-design.md`

## Global Constraints
- 严格遵循 TDD 循环（RED -> GREEN -> REFACTOR）；纯 Prompt 或评估类任务以标准样例评估集替代单测。
- 过程留痕：在 `dev-notes/ch01.md` 追记各阶段四要素（用户原话、关键产出、拒绝纠偏、翻车与返工）。
- 模型网关必须通用，通过 `.env` 中的 `OPENAI_API_BASE`/`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL_NAME` 配置，支持任意 OpenAI 兼容协议。
- 代码符合 KISS 原则，严禁过度工程化。

---

### Task 1: 配置管理与环境配置 (`app/config.py`)

**Files:**
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `.env.example`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` 类与全局单例 `settings`
  - `openai_api_base: Optional[str]`
  - `openai_api_key: str`
  - `openai_model_name: str` (default: "gpt-4o-mini")
  - `openai_temperature: float` (default: 0.7)
  - `max_context_tokens: int` (default: 2000)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import os
from unittest import mock
import pytest

def test_settings_load_defaults():
    from app.config import Settings
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "mock-key"}, clear=True):
        s = Settings()
        assert s.openai_api_key == "mock-key"
        assert s.openai_model_name == "gpt-4o-mini"
        assert s.openai_temperature == 0.7
        assert s.max_context_tokens == 2000

def test_settings_custom_env():
    from app.config import Settings
    custom_env = {
        "OPENAI_API_KEY": "custom-key",
        "OPENAI_BASE_URL": "https://api.deepseek.com/v1",
        "OPENAI_MODEL_NAME": "deepseek-chat",
        "OPENAI_TEMPERATURE": "0.2",
        "MAX_CONTEXT_TOKENS": "3000",
    }
    with mock.patch.dict(os.environ, custom_env, clear=True):
        s = Settings()
        assert s.openai_api_key == "custom-key"
        assert s.openai_base_url == "https://api.deepseek.com/v1"
        assert s.openai_model_name == "deepseek-chat"
        assert s.openai_temperature == 0.2
        assert s.max_context_tokens == 3000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/config.py
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    openai_api_key: str = "sk-placeholder"
    openai_base_url: Optional[str] = None
    openai_api_base: Optional[str] = None  # 别名兼容
    openai_model_name: str = "gpt-4o-mini"
    openai_temperature: float = 0.7
    max_context_tokens: int = 2000

    @property
    def effective_base_url(self) -> Optional[str]:
        return self.openai_base_url or self.openai_api_base

settings = Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 5: Record dev notes & commit**

Commit message: `feat(config): add settings management with pydantic-settings`

---

### Task 2: 客服 Prompt 模板与评估设计 (`app/prompts/customer_service.py`)

**Files:**
- Create: `app/prompts/__init__.py`
- Create: `app/prompts/customer_service.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Produces: 
  - `CUSTOMER_SERVICE_SYSTEM_PROMPT: str`
  - `customer_service_prompt: ChatPromptTemplate`
    - Input variables: `history: list[BaseMessage]`, `input: str`

- [ ] **Step 1: Write evaluation test cases for prompt formatting & constraints**

```python
# tests/test_prompts.py
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.prompts.customer_service import customer_service_prompt, CUSTOMER_SERVICE_SYSTEM_PROMPT

def test_system_prompt_contains_core_constraints():
    assert "客服" in CUSTOMER_SERVICE_SYSTEM_PROMPT
    assert "礼貌" in CUSTOMER_SERVICE_SYSTEM_PROMPT or "专业" in CUSTOMER_SERVICE_SYSTEM_PROMPT
    # 红线约束：不私自承诺违规赔偿
    assert "承诺" in CUSTOMER_SERVICE_SYSTEM_PROMPT or "权责" in CUSTOMER_SERVICE_SYSTEM_PROMPT

def test_prompt_template_formatting():
    history = [
        HumanMessage(content="你好"),
        AIMessage(content="您好！我是星光优选智能客服小星，请问有什么可以帮您？")
    ]
    formatted = customer_service_prompt.format_messages(history=history, input="我想查一下物流")
    
    assert len(formatted) == 4
    assert isinstance(formatted[0], SystemMessage)
    assert formatted[1].content == "你好"
    assert formatted[2].content == "您好！我是星光优选智能客服小星，请问有什么可以帮您？"
    assert formatted[3].content == "我想查一下物流"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompts.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'app.prompts'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/prompts/customer_service.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CUSTOMER_SERVICE_SYSTEM_PROMPT = """你是由“星光优选”电商平台研发的专业智能客服助手，名字叫“小星”。
你的职责是为顾客提供热情、专业、高效的售前咨询与售后服务支持。

【服务准则与行为约束】
1. 态度热情礼貌：始终使用“您好”、“请问有什么可以帮您”、“非常抱歉给您带来不便”等礼貌敬语。
2. 聚焦电商业务：仅回答与商城商品、订单、物流、退换货、优惠活动等电商相关的咨询，坚决拒绝回答政治、暴力、违规及与平台服务无关的提问。
3. 严格遵守售后权责红线：
   - 涉及商品破损、质量问题，先安抚客户情绪，并礼貌引导客户提供订单号及商品照片凭证；
   - 严禁未经系统核实私自向客户承诺具体的额外现金赔付或违规补偿；
   - 退款与换货需告知客户标准平台流程（如需在订单详情页提交申请）。
4. 表达简练明了：回答逻辑清晰，重点突出，单次回复避免冗长废话。
"""

customer_service_prompt = ChatPromptTemplate.from_messages([
    ("system", CUSTOMER_SERVICE_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompts.py -v`  
Expected: PASS

- [ ] **Step 5: Record dev notes & commit**

Commit message: `feat(prompts): add customer service prompt template with behavioral constraints`

---

### Task 3: 多轮会话管理与 Token 裁剪 (`app/services/session_manager.py`)

**Files:**
- Create: `app/services/__init__.py`
- Create: `app/services/session_manager.py`
- Test: `tests/test_session_manager.py`

**Interfaces:**
- Produces: `SessionManager` 类与全局单例 `session_manager`
  - `get_or_create_session(session_id: Optional[str]) -> str`
  - `add_user_message(session_id: str, content: str)`
  - `add_ai_message(session_id: str, content: str)`
  - `get_history(session_id: str) -> list[BaseMessage]`
  - `get_trimmed_history(session_id: str, max_tokens: int) -> list[BaseMessage]`
  - `clear_session(session_id: str)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_manager.py
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from app.services.session_manager import SessionManager

def test_session_lifecycle():
    sm = SessionManager()
    sid = sm.get_or_create_session()
    assert sid is not None
    assert sm.get_history(sid) == []

    sm.add_user_message(sid, "你好")
    sm.add_ai_message(sid, "您好！")
    history = sm.get_history(sid)
    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)

def test_session_trimming_under_token_budget():
    sm = SessionManager()
    sid = sm.get_or_create_session("test-session")
    
    # 添加多轮对话
    for i in range(10):
        sm.add_user_message(sid, f"用户提问 {i}: " + "这是一段较长的测试文本用于消耗token" * 10)
        sm.add_ai_message(sid, f"客服回复 {i}: " + "这是一段较长的客服回复内容" * 10)

    # 当预算非常小时，裁剪应只保留最近的消息，并且首条非 system 消息必须为 HumanMessage
    trimmed = sm.get_trimmed_history(sid, max_tokens=150)
    assert len(trimmed) < 20
    assert len(trimmed) > 0
    assert isinstance(trimmed[0], HumanMessage)
    assert "用户提问" in trimmed[0].content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_session_manager.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.session_manager'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/session_manager.py
import uuid
from typing import Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, trim_messages
import tiktoken

def count_tokens_fallback(messages: List[BaseMessage]) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None
    total = 0
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if enc:
            total += len(enc.encode(content)) + 4
        else:
            total += len(content) // 2 + 4
    return total

class SessionManager:
    def __init__(self):
        self._sessions: dict[str, List[BaseMessage]] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        if not session_id or not session_id.strip():
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return session_id

    def get_history(self, session_id: str) -> List[BaseMessage]:
        return list(self._sessions.get(session_id, []))

    def add_user_message(self, session_id: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(HumanMessage(content=content))

    def add_ai_message(self, session_id: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(AIMessage(content=content))

    def get_trimmed_history(self, session_id: str, max_tokens: int) -> List[BaseMessage]:
        raw_history = self.get_history(session_id)
        if not raw_history:
            return []
        
        # 使用官方 trim_messages，滑动窗口策略保留最近轮次，保证起始为 human
        trimmed = trim_messages(
            raw_history,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=count_tokens_fallback,
            start_on="human",
            include_system=True
        )
        return list(trimmed)

    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

session_manager = SessionManager()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_session_manager.py -v`  
Expected: PASS

- [ ] **Step 5: Record dev notes & commit**

Commit message: `feat(services): implement session manager with langchain trim_messages`

---

### Task 4: 统一模型工厂与售后结构化提取 (`app/llm.py`, `app/schemas/after_sale.py`, `app/services/after_sale_service.py`)

**Files:**
- Create: `app/llm.py`
- Create: `app/schemas/__init__.py`
- Create: `app/schemas/after_sale.py`
- Create: `app/services/after_sale_service.py`
- Test: `tests/test_after_sale.py`

**Interfaces:**
- Produces: 
  - `get_chat_model(streaming: bool = False) -> ChatOpenAI`
  - `AfterSaleTicket(BaseModel)`
  - `extract_after_sale_ticket(description: str, model=None) -> AfterSaleTicket`

- [ ] **Step 1: Write the failing test with schema validation & mock extraction**

```python
# tests/test_after_sale.py
from unittest.mock import MagicMock
from app.schemas.after_sale import AfterSaleTicket
from app.services.after_sale_service import extract_after_sale_ticket

def test_after_sale_ticket_schema():
    ticket = AfterSaleTicket(
        order_id="TB123456",
        issue_type="质量问题",
        expected_solution="退款",
        raw_description="衣服起球很严重，我想退款"
    )
    assert ticket.order_id == "TB123456"
    assert ticket.issue_type == "质量问题"
    assert ticket.expected_solution == "退款"

def test_extract_after_sale_ticket_with_mock_model():
    mock_model = MagicMock()
    mock_structured = MagicMock()
    expected_ticket = AfterSaleTicket(
        order_id="ORD999",
        issue_type="破损",
        expected_solution="换货",
        raw_description="鞋子开胶了"
    )
    mock_structured.invoke.return_value = expected_ticket
    mock_model.with_structured_output.return_value = mock_structured

    result = extract_after_sale_ticket("订单ORD999鞋子开胶了要换货", model=mock_model)
    assert result.order_id == "ORD999"
    assert result.issue_type == "破损"
    assert result.expected_solution == "换货"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_after_sale.py -v`  
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# app/schemas/after_sale.py
from typing import Optional
from pydantic import BaseModel, Field

class AfterSaleTicket(BaseModel):
    order_id: Optional[str] = Field(
        default=None,
        description="从用户描述中提取到的订单号或交易单号。如果用户未提及订单号，则为 null。"
    )
    issue_type: str = Field(
        description="售后问题的类型。例如：商品质量问题/破损、少件漏发、错发商品、物流延迟/未送达、七天无理由退货、价格保护、商品咨询等。"
    )
    expected_solution: str = Field(
        description="用户期望的解决方式。例如：仅退款、退货退款、换货、补发商品、维修、催促物流、赔偿补偿等。若未明确提及可根据语境合理归纳。"
    )
    raw_description: str = Field(
        description="用户提供的原始售后问题描述全文。"
    )

class AfterSaleExtractRequest(BaseModel):
    description: str = Field(..., min_length=1, description="用户售后问题描述")
```

```python
# app/llm.py
from langchain_openai import ChatOpenAI
from app.config import settings

def get_chat_model(streaming: bool = False) -> ChatOpenAI:
    kwargs = {
        "model": settings.openai_model_name,
        "api_key": settings.openai_api_key,
        "temperature": settings.openai_temperature,
        "streaming": streaming,
    }
    if settings.effective_base_url:
        kwargs["base_url"] = settings.effective_base_url
    return ChatOpenAI(**kwargs)
```

```python
# app/services/after_sale_service.py
from typing import Optional
from langchain_openai import ChatOpenAI
from app.schemas.after_sale import AfterSaleTicket
from app.llm import get_chat_model

AFTER_SALE_SYSTEM_PROMPT = """你是一位专业的电商售后工单分析专员。
你的任务是从用户口述或输入的售后投诉描述中，精确提取结构化关键字段。
请仔细甄别订单号、售后核心问题类型以及用户期望的解决方案。若用户未给出订单号，务必将其置为 null。
原始描述字段原样保留。
"""

def extract_after_sale_ticket(description: str, model: Optional[ChatOpenAI] = None) -> AfterSaleTicket:
    if model is None:
        model = get_chat_model(streaming=False)
    structured_model = model.with_structured_output(AfterSaleTicket)
    messages = [
        ("system", AFTER_SALE_SYSTEM_PROMPT),
        ("human", description)
    ]
    result = structured_model.invoke(messages)
    if isinstance(result, dict):
        result = AfterSaleTicket(**result)
    result.raw_description = description
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_after_sale.py -v`  
Expected: PASS

- [ ] **Step 5: Record dev notes & commit**

Commit message: `feat(after_sale): implement structured output for customer tickets`

---

### Task 5: FastAPI 接口与 SSE 流式推送 (`app/schemas/chat.py`, `app/api/routes.py`, `main.py`)

**Files:**
- Create: `app/schemas/chat.py`
- Create: `app/api/__init__.py`
- Create: `app/api/routes.py`
- Create: `main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces:
  - `POST /api/chat/stream` -> SSE 流式响应
  - `POST /api/after-sale/extract` -> JSON 响应
  - `GET /health` -> `{ "status": "ok" }`

- [ ] **Step 1: Write API failing test using FastAPI TestClient**

```python
# tests/test_api.py
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest
from main import app
from app.schemas.after_sale import AfterSaleTicket

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

@patch("app.api.routes.extract_after_sale_ticket")
def test_extract_endpoint(mock_extract):
    mock_extract.return_value = AfterSaleTicket(
        order_id="TB123",
        issue_type="尺码不合",
        expected_solution="换货",
        raw_description="鞋子买小了"
    )
    res = client.post("/api/after-sale/extract", json={"description": "鞋子买小了想换大一号，单号TB123"})
    assert res.status_code == 200
    data = res.json()
    assert data["order_id"] == "TB123"
    assert data["issue_type"] == "尺码不合"

@patch("app.api.routes.get_chat_model")
def test_chat_stream_endpoint(mock_get_model):
    async def fake_astream(prompt):
        class Chunk:
            content = "您好！"
        yield Chunk()
        class Chunk2:
            content = "请问有什么帮您？"
        yield Chunk2()

    mock_llm = MagicMock()
    mock_llm.astream = fake_astream
    mock_get_model.return_value = mock_llm

    res = client.post("/api/chat/stream", json={"message": "你好", "session_id": "s1"})
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    text = res.text
    assert "data: " in text
    assert "[DONE]" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/schemas/chat.py
from typing import Optional
from pydantic import BaseModel, Field

class ChatStreamRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="会话唯一标识，为空则自动生成")
    message: str = Field(..., min_length=1, description="用户提问内容")
```

```python
# app/api/routes.py
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatStreamRequest
from app.schemas.after_sale import AfterSaleExtractRequest, AfterSaleTicket
from app.services.session_manager import session_manager
from app.services.after_sale_service import extract_after_sale_ticket
from app.prompts.customer_service import customer_service_prompt
from app.llm import get_chat_model
from app.config import settings

router = APIRouter(prefix="/api")

@router.post("/chat/stream")
async def chat_stream(request: ChatStreamRequest):
    session_id = session_manager.get_or_create_session(request.session_id)
    history = session_manager.get_trimmed_history(session_id, max_tokens=settings.max_context_tokens)

    # 格式化 Prompt
    formatted_messages = customer_service_prompt.format_messages(
        history=history,
        input=request.message
    )

    # 先把用户当次提问记录入历史
    session_manager.add_user_message(session_id, request.message)

    llm = get_chat_model(streaming=True)

    async def event_generator():
        collected_content = []
        try:
            async for chunk in llm.astream(formatted_messages):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    collected_content.append(content)
                    payload = json.dumps({"session_id": session_id, "content": content}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
            
            # 流结束后，将完整回复追加到会话历史
            full_response = "".join(collected_content)
            if full_response:
                session_manager.add_ai_message(session_id, full_response)
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/after-sale/extract", response_model=AfterSaleTicket)
async def extract_ticket(request: AfterSaleExtractRequest):
    try:
        ticket = extract_after_sale_ticket(request.description)
        return ticket
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提取失败: {str(e)}")
```

```python
# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="AstroBot - 电商智能客服系统",
    description="支持多轮 SSE 流式对话、客服角色约束与售后工单结构化提取",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`  
Expected: PASS

- [ ] **Step 5: Record dev notes & commit**

Commit message: `feat(api): add SSE stream chat endpoint and after-sale extract endpoint`

---

### Task 6: 综合联调与端到端验收 (E2E Verification)

**Files:**
- Create: `.env` (配置或提供本地/模拟端点配置)
- Update: `dev-notes/ch01.md`
- Create: `walkthrough.md`

- [ ] **Step 1: 运行全量单元测试**
  Run: `pytest -v`
  Expected: 全部 PASS。

- [ ] **Step 2: 启动 Uvicorn 后台服务并验证验收标准**
  - **验收标准 1 (SSE 流式输出)**:
    ```bash
    curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
      -H "Content-Type: application/json" \
      -d "{\"message\": \"你好，我想问一下你们这买羽绒服保修多久？\"}"
    ```
    预期：收到逐 token 的 `data: {"session_id": "...", "content": "..."}` 及结尾 `data: [DONE]`。
  - **验收标准 2 (两轮对话接住上下文)**:
    第一轮：问“我刚刚看了你们那件黑色羽绒服”；
    第二轮带上同一 `session_id` 问：“它有内胆吗？”，检验模型能否结合上一轮的黑色羽绒服上下文准确作答。
  - **验收标准 3 (售后描述结构化提取)**:
    ```bash
    curl -X POST http://127.0.0.1:8000/api/after-sale/extract \
      -H "Content-Type: application/json" \
      -d "{\"description\": \"我上周买的羽绒服订单号是 TB20240901，拉链卡住了拉不上，我想直接换一件新的\"}"
    ```
    预期：返回包含 `order_id="TB20240901"`, `issue_type`, `expected_solution="换货"` 的合规 JSON。

- [ ] **Step 3: 完善留痕与交付文档**
  - 更新 `dev-notes/ch01.md` 记录各阶段实况。
  - 创建脑区 `walkthrough.md` 呈现端到端验证结果与命令。
