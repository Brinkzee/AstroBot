# Task 4 Report: 工具基础设施 (Registry, Validation, Executor 与异常回灌)

## 1. 任务概述
- **任务编号与名称**: Task 4: 工具基础设施 (Registry, Validation, Executor 与异常回灌)
- **基础提交 (Base Commit)**: `8442aef0dc9134a837c0911196395fae13fa7b75`
- **生成提交 (New Commit)**: `049b4be4d6995830bd64ad656050846c1a014ef1`
- **提交信息**: `feat(tools): add tool registry and executor with validation, timeout and error recovery`

## 2. 接口与产出物
- `app/tools/registry.py`:
  1. `ToolRegistry`:
     - `register(tool: BaseTool) -> None`: 注册 BaseTool 工具实例；
     - `get_tool(name: str) -> Optional[BaseTool]`: 按工具名检索工具实例；
     - `get_all_tools() -> list[BaseTool]`: 获取全部已注册工具清单；
     - 支持 `__len__` 与 `__contains__` 快捷运算；
  2. `default_tool_registry`: 预先注册全部 5 个业务工具 (`query_order`, `query_product`, `query_logistics`, `query_faq`, `create_ticket`) 的默认单例注册中心。
- `app/tools/executor.py`:
  1. `ToolExecutor`:
     - `__init__(registry: Optional[ToolRegistry] = None, timeout: float = 5.0, max_retries: int = 1)`: 支持自定义注册中心、超时控制 (默认 5.0s) 与最大重试次数 (默认 1 次)；
     - `async execute(tool_call: dict, db: Optional[Any] = None, **kwargs) -> dict`: 具备多项企业级能力：
       - **参数格式化与 JSON 兼容**: 自动解析并反序列化字符串格式的参数，防御性校验字典结构；
       - **工具发现**: 未注册工具优雅拦截，不抛出异常；
       - **Schema 校验**: 严格校验 `args` 与工具 `tool.args_schema`（基于 Pydantic），捕获 `ValidationError` 并格式化；
       - **超时控制**: 基于 `asyncio.wait_for(..., timeout=self.timeout)`；
       - **瞬时抖动退避重试**: 针对网络与 I/O 抖动进行至多 `max_retries` 次重试；
       - **异常优雅回灌封装**: 出现任何错误均不向外层抛出，统一返回标准化结构：
         ```python
         {
             "success": False,
             "output": f"工具 [{tool_name}] 调用失败: {error_detail}，请结合此情况向用户做解释并提供帮助",
             "tool_name": tool_name,
             "tool_call_id": tool_call_id,
             "error": error_detail,
         }
         ```
- `tests/test_tool_executor.py`:
  - 9 项独立单元测试，完整覆盖工具注册检索、默认注册表、正常执行、未知工具、Schema 校验失败、JSON 字符串参数解析、超时控制拦截、重试恢复成功及重试耗尽失败。

## 3. TDD 执行过程

### 3.1 Step 1 & 2: RED 阶段
编写测试文件 `tests/test_tool_executor.py`，执行 `pytest tests/test_tool_executor.py -v`。
**测试失败输出 (RED)**:
```
=================================== ERRORS ====================================
________________ ERROR collecting tests/test_tool_executor.py _________________
ImportError while importing test module 'D:\PycharmProjects\AstroBot\tests\test_tool_executor.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
D:\Anaconda3\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_tool_executor.py:6: in <module>
    from app.tools.registry import ToolRegistry, default_tool_registry
E   ModuleNotFoundError: No module named 'app.tools.registry'
=========================== short test summary info ===========================
ERROR tests/test_tool_executor.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.83s ===============================
```

### 3.2 Step 3: 实现阶段
1. 创建 `app/tools/registry.py`，定义 `ToolRegistry` 并初始化预装五大业务工具的 `default_tool_registry`；
2. 创建 `app/tools/executor.py`，实现 `ToolExecutor`：
   - 适配 LangChain 工具的 `ainvoke` 与同步工具在线程池运行机制；
   - 引入 Pydantic `ValidationError` 强类型校验；
   - 引入 `asyncio.wait_for` 超时截断与指数/退避重试循环；
   - 编写 `_build_failure_response` 统一格式化失败回灌文本。

### 3.3 Step 4: GREEN 阶段
执行模块测试与全量回归测试：
- `pytest tests/test_tool_executor.py -v`: 9 passed in 0.90s
- `pytest -v`: 43 passed, 1 warning in 2.41s / 10.00s (全量 43 项测试套件 100% 通过，无回归)

## 4. 代码审查与 Diff 自检
- **架构一致性**:
  - `ToolRegistry` 与 `ToolExecutor` 严格按照设计规约提供接口，`default_tool_registry` 保证下游 `ChatService`（Task 5）开箱即用；
  - 返回字典结构完全匹配设计文档第 4.2 节关于异常回灌的要求；
  - 超时与重试逻辑不阻断异步主事件循环，退避使用 `asyncio.sleep`；
- **提交内容**:
  - `app/tools/registry.py` (created)
  - `app/tools/executor.py` (created)
  - `tests/test_tool_executor.py` (created)

## 5. 潜在问题与注意事项 (Concerns)
- 无阻塞性缺陷。
- 在后续 Task 5 (ChatService 编排服务) 中，`ToolExecutor` 返回的 `output` 字符串应直接写入 `ToolMessage(content=result["output"], tool_call_id=result["tool_call_id"])` 并落盘至数据库消息表，再回灌给大模型以完成单轮状态机收敛。
