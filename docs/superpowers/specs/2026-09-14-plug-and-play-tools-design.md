# 设计规范：客服系统即插即用工具体系与 MCP 协议集成

## 1. 概述与设计目标

本规范针对 AstroBot 智能客服系统的工具层进行全面升级，从固定硬编码的内置函数重构为支持动态发现、即插即用的现代化工具体系。系统全面支持：
1. **统一工具注册中心 (`ToolRegistry`)**：同时纳管内置业务工具与外部 MCP 工具，提供统一的工具名、描述与 JSON Schema 参数契约；实现 MCP 工具“现问现拿”动态发现，支持服务端加工具零重启感知；
2. **严格参数校验器 (`validate_tool_arguments`)**：工具执行前统一执行 JSON Schema 校验，拦截必填缺失、类型错误与越界取值，将校验错误回灌给大模型以引导追问；
3. **零信任读写权限门禁 (`ToolPermissionGuard`)**：严格区分只读与写操作。外部 MCP 工具默认一律只读；系统唯一写操作 `create_ticket` 设置“客户明确诉求”与“前端人机确认”双重门禁，无确认直接拒绝；
4. **统一执行引擎 (`ToolExecutor`)**：统一管控超时、指数退避重试（仅针对瞬时网络抖动，业务空结果不重试，写操作恒不自动重试）、三类错误分诊（参数不合法、查询落空、真故障）及中文不转义脱敏格式化；
5. **无外键高可用审计 (`tool_audit_logs`)**：对齐 `sql/ch08-ddl.sql`，全量记录每次调用的工具名、来源、入参、结果摘要、状态、错误信息、重试次数及耗时；独立事务落库，审计失败绝不反向阻碍主流程；
6. **双独立业务 MCP Server 接入**：基于官方 `mcp` Python SDK（Streamable HTTP 传输）自建物流（端口 8001）与售后（端口 8002）独立进程；客户端使用 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 接入；内置 `query_logistics` 平滑下线；
7. **LangGraph Interrupt 工单人机确认流**：客户明确建单且信息完整后触发 `create_ticket`，执行引擎利用 LangGraph 原生 `interrupt` 挂起并向前端推送工单预览卡片；前端提供「确认提交」与「取消」，经 `POST /api/chat/resume` 唤醒图恢复执行。

---

## 2. 总体架构与数据流

```mermaid
sequenceDiagram
    autonumber
    actor User as 用户 / 前端浏览器
    participant ChatAPI as FastAPI (Chat / Resume API)
    participant Engine as WorkflowEngine (LangGraph)
    participant Agent as main_agent_node
    participant Registry as ToolRegistry (Builtin + MultiServerMCPClient)
    participant MCPServer as 业务 MCP Server (物流 8001 / 售后 8002)
    participant Executor as ToolExecutor (校验 + 权限 + 重试 + 格式化)
    participant Audit as tool_audit_logs (独立事务审计)
    participant DB as MySQL 业务表 (tickets / conversations)

    User->>ChatAPI: 发送聊天请求 POST /api/chat/stream
    ChatAPI->>Engine: ainvoke / astream(initial_state)
    Engine->>Agent: 调度主力 ReAct Agent
    Agent->>Registry: get_all_tools() 动态获取工具列表
    Registry->>MCPServer: Streamable HTTP 现问现拿拉取工具定义
    MCPServer-->>Registry: 返回最新工具 Schema
    Registry-->>Agent: 合并内置与 MCP 工具清单 (bind_tools)
    Agent->>Agent: 大模型决策发起 tool_call

    rect rgb(240, 248, 255)
    Note over Agent,Executor: 工具统一执行与门禁校验阶段
    Agent->>Executor: execute(tool_call, context)
    Executor->>Executor: 1. JSON Schema 参数校验 (若失败落「校验拦下」回灌模型)
    Executor->>Executor: 2. 权限判定 (MCP写操作拦截; create_ticket 检查明确诉求)
    alt 正常只读或已确认写操作
        Executor->>MCPServer: 调用工具 (超时控制 5s + 网络抖动重试)
        MCPServer-->>Executor: 返回原始结果
        Executor->>Executor: 结果脱敏与中文不转义序列化
        Executor->>Audit: 异步写入审计记录 (状态: 成功/失败/超时)
        Executor-->>Agent: 回灌格式化 ToolMessage
    else 遇到未确认的 create_ticket
        Executor-->>Agent: 触发 LangGraph interrupt(ticket_preview)
        Agent-->>Engine: 状态持久化并挂起
        Engine-->>ChatAPI: 派发 SSE event: ticket_preview
        ChatAPI-->>User: 渲染工单预览卡片 (【确认提交】/【取消】)
    end
    end

    opt 用户在卡片上操作
        User->>ChatAPI: POST /api/chat/resume { conversation_id, action: "confirm" | "cancel" }
        ChatAPI->>Engine: astream(Command(resume=action))
        alt action == "confirm"
            Engine->>Executor: 凭证核验通过，物理执行 create_ticket
            Executor->>DB: 插入 tickets 表
            Executor->>Audit: 审计记录: 状态「成功」
            Executor-->>Agent: 返回 ticket_no
            Agent-->>User: SSE 流式推送回复: 已为您创建工单【T...】
        else action == "cancel"
            Engine->>Executor: 用户主动取消
            Executor->>Audit: 审计记录: 状态「权限拒绝」
            Executor-->>Agent: 回灌取消提示
            Agent-->>User: SSE 流式推送回复: 已为您取消工单创建
        end
    end
```

---

## 3. 详细模块设计

### 3.1 工具注册中心与 MCP 接入 (`app/tools/registry.py`)

1. **内置工具池**：
   - 包含：`query_order`、`query_product`、`query_faq`、`create_ticket`；
   - 彻底移除内置 `query_logistics`，由物流 MCP 接管，防止同名冲突。
2. **MCP Client 集成**：
   - 引入 `langchain_mcp_adapters.client.MultiServerMCPClient`；
   - 在 `app/config.py` 配置：
     - `MCP_LOGISTICS_SERVER_URL`: 默认 `http://127.0.0.1:8001/mcp`
     - `MCP_AFTERSALE_SERVER_URL`: 默认 `http://127.0.0.1:8002/mcp`
     - `MCP_CLIENT_TIMEOUT`: 默认 `5.0` 秒
   - 传输配置：
     ```python
     connections = {
         "logistics": {
             "transport": "streamable_http",
             "url": settings.MCP_LOGISTICS_SERVER_URL,
         },
         "aftersale": {
             "transport": "streamable_http",
             "url": settings.MCP_AFTERSALE_SERVER_URL,
         }
     }
     ```
3. **“现问现拿”动态发现 (`get_all_tools`)**：
   - Agent 每轮对话准备工具时，调用 `await registry.get_all_tools()`；
   - 并行拉取 `mcp_client.get_tools()`，为每个 MCP 工具动态注入元数据标记：
     - `_tool_source = "mcp"`
     - `_mcp_server = "logistics"` 或 `"aftersale"`
   - 容灾与降级：若任一 MCP Server 连不上或响应超时，记录 Warning 日志并剔除该 Server 工具，主干客服系统与其他 Server 工具保持正常服务。

### 3.2 独立进程 MCP Server 实现 (`scripts/`)

1. **物流 MCP Server (`scripts/mcp_logistics_server.py`)**：
   - 基于 `FastMCP("LogisticsServer")`，端口 `8001`，路径 `/mcp`；
   - 工具：`query_logistics(order_id: str)`：返回快递公司、运单号、当前状态与轨迹列表。
2. **售后 MCP Server (`scripts/mcp_aftersale_server.py`)**：
   - 基于 `FastMCP("AftersaleServer")`，端口 `8002`，路径 `/mcp`；
   - 工具 1：`check_warranty(order_id: str, product_name: str = "")`：返回在保状态、质保截止时间与保障范围；
   - 工具 2：`query_return_progress(return_id_or_order_id: str)`：返回退货退款审核状态与物流进度。
3. **动态更新验证工具**：
   - 售后 Server 预留测试用动态工具挂载能力（如 `evaluate_recycling_value`），用于验收标准 3 的热重启探测。

### 3.3 参数 Schema 校验与统一执行引擎 (`app/tools/executor.py`)

1. **参数 Schema 校验 (`validate_tool_arguments`)**：
   - 提取工具关联的 `args_schema`，利用 Pydantic `model_validate` 或 JSON Schema 校验器进行严格审查；
   - 拦截范围：必填参数缺失、数据类型不符、值域越界；
   - 拦截行为：捕获校验异常，返回结构化回灌字典：
     ```json
     {
       "success": false,
       "error_type": "VALIDATION_FAILED",
       "error_message": "参数校验未通过: [order_id] 必填项缺失，请向用户询问补充",
       "output": "工具 [query_order] 调用失败: 参数校验未通过: [order_id] 必填项缺失，请向用户追问以补齐必要信息。"
     }
     ```
   - 审计状态直接记录为「校验拦下」。
2. **超时与分类重试治理**：
   - 超时控制：统一使用 `asyncio.wait_for(..., timeout=self.timeout)`，超时设定默认 5.0 秒；
   - 重试白名单：仅限网络瞬时故障（如 `asyncio.TimeoutError`、`ConnectionResetError`、`ConnectError` 等读操作物理网络异常），最多重试 1 次，退避间隔 0.2 秒；
   - 坚决不重试策略：
     - **业务空结果**（如订单号未查到数据）：视为正常返回，重试次数恒为 0；
     - **写操作**（`create_ticket`）：任何超时或报错绝对不自动重试，避免重放脏数据，重试次数恒为 0；
3. **三类错误分诊**：
   - **参数不合法** -> 拦截回灌，引导模型追问；
   - **查询落空** -> 如实返回空结果提示（如“未查到相关退货进度”），由模型组织安抚语言；
   - **真故障** -> 超时或服务端抛错，向模型如实回灌系统临时故障信息，便于向用户说明。
4. **结果中文化与格式化**：
   - 剔除无用调试信息，内部状态枚举翻译为人话（如 `DELIVERING` -> `正在派送中`）；
   - JSON 序列化一律采用 `json.dumps(..., ensure_ascii=False)`。

### 3.4 权限门禁体系 (`app/tools/permission.py`)

1. **分类与零信任**：
   - 工具类型：`READ_ONLY` vs `WRITE`；
   - 本系统仅 `create_ticket` 属于 `WRITE`；
   - 外部 MCP 工具一律不可信：无论 MCP Server 描述为何，在客服系统内统一判定为只读；若有非白名单写操作调用，直接拦截。
2. **`create_ticket` 发起门禁**：
   - 前置校验：用户当前轮提问或上游意图必须明确具备建单/人工/投诉诉求；若模型自作主张发起调用，执行引擎直接判定权限不合规拦截，落「权限拒绝」审计。
3. **人机确认执行门禁**：
   - 未携带前端确认凭证的首次调用，执行引擎截断物理落库，向工作流返回需要人机确认的挂起状态；
   - 只有通过 Resume 放行后，才真正执行工单创建。

### 3.5 无外键审计日志 (`app/models/tool_audit_log.py` & `app/tools/audit.py`)

1. **ORM 映射**：
   - 严格映射 `sql/ch08-ddl.sql` 定义的 `tool_audit_logs` 表；
   - 字段对齐：`id`, `conversation_id`, `tool_call_id`, `tool_name`, `tool_source`, `mcp_server`, `arguments`, `result_summary`, `status`, `error_message`, `retry_count`, `duration_ms`, `created_at`。
   - `conversation_id` 无外键约束，仅普通索引。
2. **异步非阻塞落盘**：
   - 每次工具调用结算（包括校验拦下、权限拒绝、超时、成功、失败）生成一条审计记录；
   - 使用独立异步 Session 进行落库，包裹全局异常隔离：
     ```python
     try:
         async with AsyncSessionLocal() as session:
             session.add(audit_log)
             await session.commit()
     except Exception as e:
         logger.error(f"写入工具审计日志失败，静默降级: {e}")
     ```
   - 审计写入失败绝不向外抛出异常，绝不阻塞用户回复。

### 3.6 LangGraph Interrupt 工单流与前端配套

1. **模型端信息完整性要求 (`app/prompts/customer_service.py`)**：
   - Prompt 明确要求：当用户诉求为建工单但未说明具体问题时，Agent 必须主动追问“请问您具体遇到了什么问题？”，**严禁自行编造问题描述**；
   - 只有收集齐问题描述（`description`）后，模型才生成 `create_ticket` 调令。
2. **工作流中断机制 (`main_agent_node`)**：
   - 在 ReAct 循环中，若检测到调用 `create_ticket` 且非已确认上下文：
     ```python
     user_action = interrupt({
         "event_type": "ticket_preview",
         "ticket_type": args.get("ticket_type", "售后"),
         "description": args.get("description", ""),
         "conversation_id": conv_id,
         "tool_call_id": call_id,
     })
     ```
   - 工作流持久化 Checkpoint 并挂起，`chat_service.stream_chat` 向前端输出 SSE 事件 `ticket_preview`。
3. **前端工单预览卡片交互 (`app/static/index.html`)**：
   - 前端接收 `ticket_preview`，在消息流内渲染工单卡片：
     - 展示工单类型、问题描述详情；
     - 包含【确认提交】与【取消】两个按钮。
   - 用户点击响应：
     - 点击【确认提交】：向 `POST /api/chat/resume` 提交 `{ conversation_id, action: "confirm" }`；
     - 点击【取消】：向 `POST /api/chat/resume` 提交 `{ conversation_id, action: "cancel" }`。
4. **后端恢复执行接口 (`POST /api/chat/resume`)**：
   - 接收 `ResumeRequest(conversation_id: int, action: str)`；
   - 恢复图执行：`engine.graph.astream(Command(resume=action), config=...)`；
   - 若 `action == "confirm"`：
     - 放行执行 `create_ticket`，写入 `tickets` 表，审计记录为「成功」；
     - Agent 输出最终回复（含工单编号，如 `T2026...`），流式返回前端；
   - 若 `action == "cancel"`：
     - 拒绝执行，不写入 `tickets` 表，审计记录为「权限拒绝」；
     - Agent 输出友好取消确认回复流式返回前端。
5. **既有投诉按钮兼容**：
   - 第 5 章投诉流程中前端原有的快捷建单按钮保持原样，直接调用 `POST /api/tickets` 接口建单，不作变动。

### 3.7 意图识别 9 分类扩展与路由直通 (`app/services/workflow/nodes/pre_nodes.py` & `router.py`)

1. **意图分类体系演进为 9 分类**：
   - `VALID_INTENTS` 由原 7 个业务意图扩展为 8 个业务意图（加“其他”共 9 类）：`{"物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "人工"}`；
   - **「人工」意图定义**：
     > “人工：明确要求转人工、建工单或者找客服专员跟进的（无论是否有具体问题，只要提到[转人工/建工单]诉求就归为此类意图）”
   - **Few-shot 边界样例补充**：
     - 用户：“帮我转人工客服” -> `{"intent": "人工", "confidence": 0.99, "reason": "明确要求转接人工客服"}`
     - 用户：“羽绒服拉链坏了，帮我建个工单” -> `{"intent": "人工", "confidence": 0.98, "reason": "明确要求建立工单"}`
     - 用户：“帮我建个工单” -> `{"intent": "人工", "confidence": 0.99, "reason": "提到建工单诉求归为人工意图"}`
2. **路由出口映射 (`route_by_intent`)**：
   - 「人工」意图的路由出口定为 **`"business_data"`**（直通 `main_agent_node`）；
   - 规则：`elif intent in ("物流", "订单", "人工"): return "business_data"`；
   - 保证客户无论是单纯说“帮我转人工/帮我建单”，还是附带了问题描述，均能直接进入主力 ReAct Agent，由 Agent 进行信息完整性自检与主动追问，避免被误分流到知识检索或闲聊。

---

## 4. 验收标准与测试方案

### 4.1 六大核心验收用例规划

| 序号 | 验收标准 | 验证方式 |
| :--- | :--- | :--- |
| **AC-1** | 新写简单工具只做注册，Agent 即可在对话中用上 | 在注册中心通过 `register` 添加测试工具，Agent 立即能在对话中调用该工具获得回答 |
| **AC-2** | 两个独立 MCP Server 独立进程运行，问物流通过 MCP 获取 | 启动 Logistics 与 Aftersale Server，发起物流查询，确认数据来自 MCP 并正常回答 |
| **AC-3** | MCP Server 新增工具并重启 Server，客服端代码不动无需重启即可感知并调用 | 在售后 Server 侧动态添加新工具并重启该 Server，客服端直接提问，Agent 能实时调用新工具 |
| **AC-4** | 问建单未讲明问题，Agent 追问补齐；前端卡片点「确认提交」，tickets 表落库且回复带工单号 | 用户说“帮我建个工单”，Agent 主动追问；用户补充描述后下发预览卡片；点确认后落库 `tickets` 表并返回 `T...` |
| **AC-5** | 工单预览点「取消」，工单未建，`tool_audit_logs` 状态为「权限拒绝」 | 走到工单预览卡片后点取消，确认 `tickets` 无新增，`tool_audit_logs` 状态为「权限拒绝」 |
| **AC-6** | 读操作超时触发重试与降级，写操作超时不自动重试（重试次数为 0） | 人为注入延迟测试读操作超时（触发重试、记录耗时与超时状态）；测试写操作超时，确认重试次数恒为 0 |

---

## 5. 依赖变更清单

需要扩充项目环境与 `requirements.txt`：
- `mcp>=1.3.0` (官方 Model Context Protocol SDK)
- `langchain-mcp-adapters>=0.1.0` (LangChain 官方多 Server MCP 客户端适配器)
- `jsonschema>=4.20.0` (统一 Schema 参数校验器)
