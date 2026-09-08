# Task 6 Report: FastAPI SSE 路由对接与集成测试

## 1. 任务概述
- **任务目标**：将 `POST /api/chat/stream` 路由与 `ChatService` 及 `get_db` 依赖注入对接，输出符合统一规约的 JSON SSE 事件流（`tool_start`, `tool_end`, `text`, `error`）并以 `data: [DONE]\n\n` 终止，同时升级请求模型 `ChatStreamRequest` 兼容历史入参。
- **关联基线 Commit**：`cfc4c1444c0850a7c8458e10081c65cf752a5dd5`
- **本次提交 Commit**：`66898aa5582f3efc6eb0d1487f54c9c22e519e99`
- **提交信息**：`feat(api): upgrade sse endpoint with tool calling event stream`

## 2. 变更文件列表
- `app/schemas/chat.py`：新增 `conversation_id: Optional[int]`，保留 `session_id: Optional[str]` 作为兼容别名，实现 `effective_conversation_id` 属性提取有效整数 ID。
- `app/api/routes.py`：升级 `POST /api/chat/stream`，注入 `db: AsyncSession = Depends(get_db)`，调用 `chat_service.stream_chat`，将事件序列格式化为标准 SSE 帧，保留 `/api/after-sale/extract`。
- `tests/test_api_chat_stream.py`：新增测试套件，全面覆盖 Schema 参数解析、SSE 事件帧格式与内容校验、别名向后兼容性、流式异常处理。
- `tests/test_api.py`：同步改造既有 API 路由测试，将对旧版原型模型的 mock 平滑迁移为对 `chat_service.stream_chat` 与新版 SSE 帧协议的断言。

## 3. TDD 执行过程记录
### 3.1 RED 阶段
编写 `tests/test_api_chat_stream.py` 后执行：
```bash
pytest tests/test_api_chat_stream.py -v
```
**结果**：4 failed
- `test_chat_stream_request_schema`：未实现 `conversation_id` 与 `effective_conversation_id` 属性导致校验失败；
- `test_chat_stream_sse_tool_and_text_events` 等 3 项：`AttributeError: module 'app.api.routes' has no attribute 'chat_service'`，确认测试真实有效且能准确拦截未实现的功能。

### 3.2 GREEN 阶段
1. 升级 `app/schemas/chat.py` 中的 `ChatStreamRequest`，实现 `effective_conversation_id` 提取逻辑。
2. 重构 `app/api/routes.py` 中的 `chat_stream`，消费 `ChatService` 实例并规范化 SSE 帧及 `[DONE]` 终止符。
3. 针对 `tests/test_api.py` 中的既有测试，更新其 mock 对象与断言格式。

### 3.3 验证结果
1. 单模块测试：
```bash
pytest tests/test_api_chat_stream.py -v
# 4 passed in 1.25s
```
2. 全量测试套件：
```bash
pytest -v
# 54 passed in 1.67s
```

## 4. 接口与核心逻辑
### 4.1 `ChatStreamRequest`
```python
class ChatStreamRequest(BaseModel):
    conversation_id: Optional[int] = Field(default=None, description="会话ID（自增主键）")
    session_id: Optional[str] = Field(default=None, description="会话ID，兼容旧版参数别名")
    message: str = Field(..., min_length=1, description="用户本次输入的内容")

    @property
    def effective_conversation_id(self) -> Optional[int]:
        if self.conversation_id is not None:
            return self.conversation_id
        if self.session_id is not None:
            s = str(self.session_id).strip()
            if s.isdigit():
                return int(s)
        return None
```

### 4.2 `POST /api/chat/stream` 响应协议
- 响应头包含：
  - `Content-Type: text/event-stream`
  - `Cache-Control: no-cache`
  - `Connection: keep-alive`
  - `X-Accel-Buffering: no`
- 输出格式：
  - 每条事件：`data: {"event_type": ..., ...}\n\n`
  - 终止帧：`data: [DONE]\n\n`

## 5. 潜在关注点 / 后续注意事项
- 前端 Web 聊天界面（Task 7）需要根据新的 SSE 帧协议适配渲染 `tool_start`、`tool_end` 的工具卡片动效与折叠展示。
