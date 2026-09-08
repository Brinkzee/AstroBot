# Task 4 Brief: 工具基础设施 (Registry, Validation, Executor 与异常回灌)

## Files
- Create: `app/tools/registry.py`
- Create: `app/tools/executor.py`
- Test: `tests/test_tool_executor.py`

## Interfaces
- Consumes:
  - `app.tools.business_tools`: `query_order`, `query_product`, `query_logistics`, `query_faq`, `create_ticket`
- Produces:
  - `ToolRegistry`:
    - `register(tool: BaseTool)`
    - `get_tool(name: str) -> Optional[BaseTool]`
    - `get_all_tools() -> list[BaseTool]`
    - `default_tool_registry`: pre-registered with all 5 business tools
  - `ToolExecutor`:
    - `__init__(registry: ToolRegistry = default_tool_registry, timeout: float = 5.0, max_retries: int = 1)`
    - `async execute(tool_call: dict) -> dict`:
      - `tool_call`: `{"name": str, "args": dict, "id": str}`
      - Schema validation: validates `args` against `tool.args_schema` (handles Pydantic ValidationError)
      - Timeout & Retry: uses `asyncio.wait_for(..., timeout=self.timeout)`, retries up to `max_retries` on Exception
      - Structure of return value:
        ```python
        {
            "success": bool,
            "output": str,          # Text to be put into ToolMessage content
            "tool_name": str,
            "tool_call_id": str,
            "error": Optional[str], # None if success, else error description
        }
        ```
      - On failure (timeout, validation error, tool exception, or unknown tool): catches error, does NOT re-raise, but returns `success=False` with `output=f"工具 [{tool_name}] 调用失败: {error_detail}，请结合此情况向用户做解释并提供帮助"` so it can be gracefully fed back to the model as a `ToolMessage`!

## Exact Requirements
1. **Global Constraints**:
   - Timeout default 5.0 seconds
   - Max retry default 1
   - LangChain `@tool` tools can be either sync or async (e.g. `query_order` is sync, `query_faq` is async). `ToolExecutor` must handle both: call `ainvoke` if tool supports coroutine / async, or `invoke` in threadpool if sync, or LangChain's native `tool.ainvoke(args)` which automatically handles both!
2. **Step 1 (TDD RED)**:
   - Create `tests/test_tool_executor.py`:
     - Test `default_tool_registry` has all 5 tools
     - Test successful execution of `query_order` via `executor.execute`
     - Test unknown tool returns `success=False` with descriptive error
     - Test schema validation failure (e.g. invalid args) returns `success=False`
     - Test timeout handling: mock a slow tool exceeding timeout returns `success=False` with timeout error
     - Test retry behavior: mock a tool failing once then succeeding on retry
   - Run `pytest tests/test_tool_executor.py -v` and record RED failure.
3. **Step 3 (Implement)**:
   - Implement `app/tools/registry.py` and `app/tools/executor.py`.
4. **Step 4 (TDD GREEN)**:
   - Run `pytest tests/test_tool_executor.py -v` and full suite `pytest -v`.
5. **Step 5 (Commit)**:
   - Commit: `git add app/tools/registry.py app/tools/executor.py tests/test_tool_executor.py`
   - Commit message: `feat(tools): add tool registry and executor with validation, timeout and error recovery`
