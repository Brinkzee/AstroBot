# Task 5 Report: 单轮收敛编排服务与会话落盘 (ChatService)

## 1. 任务概述
- **任务编号与名称**: Task 5: 单轮收敛编排服务与会话落盘 (ChatService)
- **基础提交 (Base Commit)**: `049b4be4d6995830bd64ad656050846c1a014ef1`
- **生成提交 (New Commit)**: `cfc4c1444c0850a7c8458e10081c65cf752a5dd5`
- **提交信息**: `feat(services): implement single-turn convergence chat service with db persistence`

## 2. 接口与产出物
- `app/services/chat_service.py`:
  1. `TOOL_LABELS`:
     - 规范定义五大业务工具中文显示名映射：
       - `query_order`: `"查询订单"`
       - `query_product`: `"查询商品"`
       - `query_logistics`: `"查询物流"`
       - `query_faq`: `"查询常见问题"`
       - `create_ticket`: `"创建人工工单"`
  2. `ChatService`:
     - `__init__(registry=default_tool_registry, executor=default_tool_executor, model=None, stream_model=None)`: 支持工具注册表与执行器注入，同时允许依赖注入测试模型；
     - `async get_or_create_conversation(db: AsyncSession, conversation_id: Optional[int] = None, user_id: str = "default_user") -> Conversation`: 会话外壳的创建与检索，持久化至 MySQL `conversations` 表；
     - `async save_message(db: AsyncSession, conversation_id: int, role: str, content: Optional[str] = None, tool_calls: Optional[list] = None, tool_call_id: Optional[str] = None) -> Message`: 消息流水的原子落盘，支持 `user`, `assistant`, `tool` 各角色；
     - `async load_conversation_messages(db: AsyncSession, conversation_id: int) -> list[BaseMessage]`: 会话历史回溯还原为 LangChain `HumanMessage`, `AIMessage`, `ToolMessage` 序列；
     - `async stream_chat(db: AsyncSession, conversation_id: Optional[int], message: str) -> AsyncGenerator[dict, None]`: 单轮流式收敛状态机，严格单轮收敛，分发 SSE 字典帧：
       - `tool_start`: `{"event_type": "tool_start", "conversation_id": int, "tool_name": str, "tool_label": str, "args": dict}`
       - `tool_end`: `{"event_type": "tool_end", "conversation_id": int, "tool_name": str, "success": bool}`
       - `text`: `{"event_type": "text", "conversation_id": int, "content": str}`
       - `error`: `{"event_type": "error", "conversation_id": Optional[int], "error": str}` (未捕获异常降级保护)
- `app/tools/executor.py`:
  - 导出单例 `default_tool_executor = ToolExecutor(registry=default_tool_registry)`，对齐接口规范。
- `tests/test_chat_service.py`:
  - 编写了 7 项独立单元测试：
    - `test_tool_labels_mapping`: 校验 5 大工具中文标签映射；
    - `test_get_or_create_conversation`: 校验新会话生成与旧会话检索；
    - `test_save_message`: 校验全部 4 种角色消息的数据库落盘；
    - `test_load_conversation_messages`: 校验消息流水向 LangChain BaseMessage 还原；
    - `test_stream_chat_pure_text`: 校验纯文本场景不触发工具调用，仅吐出 text 帧并落盘 2 条消息；
    - `test_stream_chat_tool_call_convergence`: 校验工具调用场景首轮判定、tool_start/tool_end 事件流、回灌模型流式输出与精确保存 4 条消息的严格单轮收敛；
    - `test_stream_chat_fatal_error_handling`: 校验未捕获异常时的 error 帧降级发射。

## 3. TDD 执行过程

### 3.1 Step 1 & 2: RED 阶段
编写测试文件 `tests/test_chat_service.py`，执行 `pytest tests/test_chat_service.py -v`。
**测试失败输出 (RED)**:
```
=================================== ERRORS ====================================
_________________ ERROR collecting tests/test_chat_service.py _________________
ImportError while importing test module 'D:\PycharmProjects\AstroBot\tests\test_chat_service.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
D:\Anaconda3\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_chat_service.py:7: in <module>
    from app.services.chat_service import ChatService, TOOL_LABELS
E   ModuleNotFoundError: No module named 'app.services.chat_service'
=========================== short test summary info ===========================
ERROR tests/test_chat_service.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.66s ===============================
```

### 3.2 Step 3: 实现阶段
1. 在 `app/services/chat_service.py` 实现 `ChatService`；
2. 实现基于 `db.get` 与 `select` 的会话读写、消息流水落库与 LangChain 消息对象转换；
3. 构建两阶段编排状态机：
   - 第一阶段：装配提示词与绑定五大业务工具的大模型客户端，判定是否触发 `tool_calls`；
   - 分支 A (无工具)：直接发射 `text` 帧，落盘助手回复；
   - 分支 B (触发工具)：落盘申请单，发射 `tool_start`，执行 `ToolExecutor.execute`，发射 `tool_end`，落盘工具结果，将 `AIMessage(tool_calls)` 与 `ToolMessage` 回灌至 `stream_llm.astream`，流式发射 `text` 帧并落盘最终助手回复，完成严格单轮收敛；
4. 在 `app/tools/executor.py` 导出 `default_tool_executor` 单例。

### 3.3 Step 4: GREEN 阶段
执行模块测试与全量回归测试：
- `pytest tests/test_chat_service.py -v`: 7 passed in 1.04s
- `pytest -v`: 50 passed, 1 warning in 1.56s (全量 50 项测试套件 100% 通过，无回归)

## 4. 代码审查与 Diff 自检
- **单轮收敛严谨性**: 状态机在回灌模型流式输出后直接落盘退出，不再次进入工具检索，完全杜绝无限循环 Agent Loop；
- **会话持久化与数据表对齐**: 严格匹配 `conversations` 与 `messages` 表定义（`role`, `content`, `tool_calls`, `tool_call_id`）；
- **容错防御**: `stream_chat` 内置最外层 `try...except` 捕获致命错误并输出标准 `error` 帧，同时 `tool_call["args"]` 支持 JSON 字符串自动反序列化与 `create_ticket` 的 `conversation_id` 自动补充；
- **提交范围**:
  - `app/services/chat_service.py` (新增)
  - `app/tools/executor.py` (补充默认单例)
  - `tests/test_chat_service.py` (新增)

## 5. 潜在问题与注意事项 (Concerns)
- 无阻塞性缺陷。
- 在后续 Task 6 (FastAPI SSE 路由改造) 中，FastAPI 路由可以直接依赖注入 `get_db` 获取 `AsyncSession`，并将 `chat_service.stream_chat(...)` 生成的事件字典序列化为 `f"data: {json.dumps(event, ensure_ascii=False)}\n\n"` 流式写入客户端，并在流结束时发送 `data: [DONE]\n\n`。
