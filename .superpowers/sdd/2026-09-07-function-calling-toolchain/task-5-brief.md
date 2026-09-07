# Task 5 Brief: 单轮收敛编排服务与会话落盘 (ChatService)

## Files
- Create: `app/services/chat_service.py`
- Test: `tests/test_chat_service.py`

## Interfaces
- Consumes:
  - `app.models`: `Conversation`, `Message`
  - `app.tools.registry.default_tool_registry`, `app.tools.registry.ToolRegistry`
  - `app.tools.executor.default_tool_executor`, `app.tools.executor.ToolExecutor`
  - `app.llm.get_chat_model`
  - `app.prompts.customer_service.customer_service_prompt`
  - `app.db.session.AsyncSession`
- Produces:
  - `ChatService` in `app/services/chat_service.py`:
    - `__init__(registry: ToolRegistry = default_tool_registry, executor: ToolExecutor = default_tool_executor)`
    - `async def get_or_create_conversation(self, db: AsyncSession, conversation_id: Optional[int] = None, user_id: str = "default_user") -> Conversation`
    - `async def save_message(self, db: AsyncSession, conversation_id: int, role: str, content: Optional[str] = None, tool_calls: Optional[list] = None, tool_call_id: Optional[str] = None) -> Message`
    - `async def load_conversation_messages(self, db: AsyncSession, conversation_id: int) -> list[BaseMessage]`
    - `async def stream_chat(self, db: AsyncSession, conversation_id: Optional[int], message: str) -> AsyncGenerator[dict, None]`
      - Yields dictionaries representing SSE events:
        - `{"event_type": "tool_start", "conversation_id": int, "tool_name": str, "tool_label": str, "args": dict}`
        - `{"event_type": "tool_end", "conversation_id": int, "tool_name": str, "success": bool}`
        - `{"event_type": "text", "conversation_id": int, "content": str}`
        - `{"event_type": "error", "conversation_id": Optional[int], "error": str}` (if fatal unhandled error occurs)
    - Tool labels mapping:
      `{"query_order": "查询订单", "query_product": "查询商品", "query_logistics": "查询物流", "query_faq": "查询常见问题", "create_ticket": "创建人工工单"}`

## Exact Requirements
1. **Single-turn convergence (严格单轮收敛)**:
   - Model is called once with `bind_tools`.
   - If `resp.tool_calls` is empty:
     - Pure chat branch. Stream response chunks, record assistant message to DB.
   - If `resp.tool_calls` has calls:
     - Take `tool_call = resp.tool_calls[0]`.
     - Record assistant message with `tool_calls` to DB.
     - Yield `tool_start` event.
     - Execute tool via `self.executor.execute(tool_call)`.
     - Yield `tool_end` event.
     - Record tool message with `tool_call_id` to DB.
     - Feed back to model via `stream_llm.astream(...)`.
     - Yield `text` chunks.
     - Record final assistant message to DB.
     - DO NOT call tools again (single-turn convergence).

2. **Step 1 (TDD RED)**:
   - Create `tests/test_chat_service.py`:
     - Test `get_or_create_conversation` with new and existing IDs.
     - Test `save_message` saves `user`, `assistant`, `tool` roles correctly.
     - Test `stream_chat` in pure text scenario (Mock LLM returns text with no tool_calls): yields `text` events and saves messages to DB.
     - Test `stream_chat` in tool call scenario (Mock LLM first returns `tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call_1"}]`, second call streams tokens): yields `tool_start`, `tool_end`, `text` events, and saves 4 messages (user, assistant tool_call, tool, final assistant).
   - Run `pytest tests/test_chat_service.py -v` and record RED failure.

3. **Step 3 (Implement)**:
   - Implement `app/services/chat_service.py`.

4. **Step 4 (TDD GREEN)**:
   - Run `pytest tests/test_chat_service.py -v` and full suite `pytest -v`.

5. **Step 5 (Commit)**:
   - Commit: `git add app/services/chat_service.py tests/test_chat_service.py`
   - Commit message: `feat(services): implement single-turn convergence chat service with db persistence`
