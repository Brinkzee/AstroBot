# Task 6 Brief: FastAPI SSE 路由对接与集成测试

## Files
- Modify: `app/schemas/chat.py`
- Modify: `app/api/routes.py`
- Create: `tests/test_api_chat_stream.py`

## Interfaces
- Consumes:
  - `app.services.chat_service.default_chat_service` (or `ChatService`)
  - `app.db.session.get_db`
  - `app.schemas.chat.ChatStreamRequest`
- Produces:
  - Upgraded `ChatStreamRequest` in `app/schemas/chat.py`:
    - `conversation_id: Optional[int] = None`
    - `session_id: Optional[str] = None` (backward compatibility alias)
    - `message: str`
    - helper property `effective_conversation_id -> Optional[int]` that extracts integer from `conversation_id` or `session_id` if numeric
  - Upgraded `POST /api/chat/stream` in `app/api/routes.py`:
    - Depends on `db: AsyncSession = Depends(get_db)`
    - Iterates over `chat_service.stream_chat(db, request.effective_conversation_id, request.message)`
    - Formats each event dict as `data: {"event_type": ..., "conversation_id": ..., ...}\n\n`
    - Ends with `data: [DONE]\n\n`
    - Returns `StreamingResponse` with `media_type="text/event-stream"` and headers:
      `{"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}`
    - Keep `/api/after-sale/extract` and any other existing routes intact!

## Exact Requirements
1. **Global Constraints**:
   - All events emitted must match the JSON event specification: `tool_start`, `tool_end`, `text`, `error`.
   - Streaming must terminate with `data: [DONE]\n\n`.
   - Ensure existing tests in `tests/test_api.py` still pass or are updated gracefully if they expected the old format.
2. **Step 1 (TDD RED)**:
   - Create `tests/test_api_chat_stream.py`:
     - Test `POST /api/chat/stream` with mocked `chat_service.stream_chat` emitting `tool_start`, `tool_end`, `text` events.
     - Verify response status code is 200.
     - Verify headers include `text/event-stream`.
     - Verify SSE content chunks parse into the expected events and conclude with `data: [DONE]`.
     - Test backward compatibility: request passing `session_id="1"` or `conversation_id=1`.
   - Run `pytest tests/test_api_chat_stream.py -v` and verify failure before changes.
3. **Step 3 (Implement)**:
   - Update `app/schemas/chat.py`.
   - Update `app/api/routes.py`.
4. **Step 4 (TDD GREEN)**:
   - Run `pytest tests/test_api_chat_stream.py -v` and full suite `pytest -v`.
   - Fix any regression in `tests/test_api.py` (e.g. updating test assertions to match the new `event_type` payload).
5. **Step 5 (Commit)**:
   - Commit: `git add app/schemas/chat.py app/api/routes.py tests/test_api_chat_stream.py tests/test_api.py`
   - Commit message: `feat(api): upgrade sse endpoint with tool calling event stream`
