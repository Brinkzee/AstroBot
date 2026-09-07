# 电商智能客服系统 - 纯对话跑通设计规范 (Design Spec)

**日期**: 2026-09-07  
**状态**: Approved  
**作者**: Antigravity & User  

---

## 1. 目标与范围 (Goals & Scope)

构建电商智能客服系统的纯对话基础能力，统一对接上游 OpenAI 兼容协议大模型。

### 本章范围 (In Scope)
1. **统一模型网关**：基于 `langchain_openai.ChatOpenAI`，读取 `.env` 中的 `OPENAI_API_BASE` / `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL_NAME`、`OPENAI_TEMPERATURE`，兼容 DeepSeek、OpenAI、Claude、Ollama、SiliconFlow 等。
2. **客服 Prompt 模板化**：使用 `ChatPromptTemplate` 管理，包含客服角色定位（温暖、专业、遵守电商规则、不虚假承诺）与安全行为红线。
3. **多轮会话与 Token 裁剪**：
   - 采用服务端轻量内存会话管理（`SessionManager`），根据 `session_id` 维护消息流。
   - 采用 LangChain 官方 `trim_messages` 工具，结合 `strategy="last"`、`start_on="human"`、`include_system=True` 和 token 预算控制，确保历史长对话不爆 token 且语法合法。
4. **SSE 流式对话接口**：FastAPI 提供 `POST /api/chat/stream`，逐 token 产出 SSE 事件。
5. **售后信息结构化提取**：FastAPI 提供 `POST /api/after-sale/extract`，基于 `with_structured_output` 提取订单号、诉求类型、期望方案等。

### 本章不做 (Out of Scope)
- 外部工具调用 (Tool Calling / Function Calling)
- 复杂 Agent 循环 (ReAct / LangGraph 等)
- 前端 Web 界面（以 curl 与自动化测试作为验收标准）

---

## 2. 系统接口定义 (API Specifications)

### 2.1 流式对话接口
- **URL**: `POST /api/chat/stream`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "session_id": "可选字符串, 若为空服务端自动生成 uuid4",
    "message": "用户输入文本"
  }
  ```
- **Response**: `text/event-stream`
  - 正常事件格式：
    ```text
    data: {"session_id": "xxx", "content": "您"}

    data: {"session_id": "xxx", "content": "好"}
    ```
  - 结束标记：
    ```text
    data: [DONE]
    ```

### 2.2 售后诉求结构化提取接口
- **URL**: `POST /api/after-sale/extract`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "description": "我上周买的羽绒服订单号是 TB20240901，拉链卡住了拉不上，我想直接换一件新的"
  }
  ```
- **Response**: `application/json`
  ```json
  {
    "order_id": "TB20240901",
    "issue_type": "商品质量问题/破损",
    "expected_solution": "换货",
    "raw_description": "我上周买的羽绒服订单号是 TB20240901，拉链卡住了拉不上，我想直接换一件新的"
  }
  ```

---

## 3. 核心模块与架构设计 (Architecture & Components)

### 3.1 配置模块 (`app/config.py`)
- 使用 `pydantic-settings` 定义 `Settings`：
  - `openai_api_base`: Optional[str]
  - `openai_api_key`: str
  - `openai_model_name`: str (default: `gpt-4o-mini`)
  - `openai_temperature`: float (default: `0.7`)
  - `max_context_tokens`: int (default: `2000`)
  - `system_prompt`: str (系统提示词默认配置或引用)

### 3.2 模型适配工厂 (`app/llm.py`)
- 提供 `get_chat_model(streaming=False) -> ChatOpenAI`。
- 针对 streaming=True 返回具备流式支持的实例。

### 3.3 Prompt 管理 (`app/prompts/customer_service.py`)
- `SYSTEM_PROMPT`: 严格定义“星光优选智能客服·小星”的人设、服务礼貌用语、红线约束（如涉及质量问题安抚情绪，主动索要凭证与订单号；涉及退款需合规引导；不擅自承诺赔偿金额等）。
- `CUSTOMER_SERVICE_PROMPT`: `ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), MessagesPlaceholder(variable_name="history"), ("human", "{input}")])`。

### 3.4 会话与裁剪管理 (`app/services/session_manager.py`)
- `SessionManager`:
  - 内存字典存储 `_sessions: dict[str, list[BaseMessage]]`。
  - `get_or_create_session(session_id: Optional[str]) -> str`。
  - `get_history(session_id: str) -> list[BaseMessage]`。
  - `add_user_message(session_id: str, content: str)`。
  - `add_ai_message(session_id: str, content: str)`。
  - `get_trimmed_history(session_id: str, max_tokens: int) -> list[BaseMessage]`：
    内部调用 `langchain_core.messages.trim_messages`，参数设定：
    - `strategy="last"`
    - `start_on="human"`
    - `include_system=True`
    - `token_counter=tiktoken_counter` (或根据 model 计算)

### 3.5 售后提取服务 (`app/services/after_sale_service.py`)
- 数据模型 `AfterSaleTicket(BaseModel)`：
  - `order_id`: Optional[str] = Field(description="提取到的订单编号，若未提及则为 None")
  - `issue_type`: str = Field(description="售后类型，如：质量问题、少件漏发、商品错发、物流延迟、七天无理由退货、其他")
  - `expected_solution`: str = Field(description="用户期望的解决方式，如：退款、换货、补发、维修、催件、赔付")
  - `raw_description`: str = Field(description="用户原始输入的售后问题描述")
- 基于 `model.with_structured_output(AfterSaleTicket)` 执行提取。

### 3.6 FastAPI 应用与路由 (`app/api/routes.py`, `main.py`)
- 对话流式推送通过异步生成器 `async for chunk in chain.astream(...)` 实现，包裹为 `StreamingResponse`。
- 提取接口直接返回 Pydantic 序列化对象。

---

## 4. 验证与测试策略 (Verification & TDD)
1. **单测套件 (pytest)**:
   - `test_config.py`: 配置读取测试。
   - `test_prompts.py`: 模板格式化与占位符填充测试。
   - `test_session_manager.py`: 多轮消息记录、Token 超限裁剪逻辑测试（Mock/tiktoken）。
   - `test_after_sale.py`: 提取数据结构测试及评估测试用例（提供标准售后文本断言）。
   - `test_api.py`: FastAPI TestClient 校验流式响应与提取端点。
2. **端到端 curl 验收**:
   - 验证单轮流式。
   - 验证两轮对话连续性（携带 session_id）。
   - 验证售后提取输出合法 JSON。
