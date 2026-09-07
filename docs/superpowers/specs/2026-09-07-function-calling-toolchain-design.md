# 系统设计规范 (Design Spec) - 第二章：Function Calling 工具链与单轮收敛

## 1. 背景与目标
在第一章跑通纯对话与 SSE 基础流式输出的基础上，本章为 AstroBot 智能客服系统装上“查数据”与“执行业务动作”的能力。
系统利用 OpenAI Function Calling 协议，由大模型自主决策是否调用业务工具；工具链深度长在客服聊天主链路中，实现「用户提问 → 模型决策调工具 → 执行工具 → 状态下发 → 结果回灌收敛 → 流式组织回复」的单轮完整闭环。

---

## 2. 核心架构与技术选型

- **API 框架**：FastAPI (异步路由，原生流式 `StreamingResponse`)
- **数据持久化与 ORM**：SQLAlchemy 2.0 Async (`create_async_engine`, `async_sessionmaker`, `AsyncSession` + `aiomysql`)
- **底层数据库**：MySQL 8.0 (运行于本地 WSL2 Docker 容器，端口 `3306` 自动转发至 `127.0.0.1:3306`，字符集 `utf8mb4`)
- **工具体系**：LangChain 官方 `@tool` 装饰器 + Pydantic `args_schema` 强校验
- **模型集成**：LangChain `ChatOpenAI.bind_tools(...)`，统一 OpenAI 兼容协议
- **交互协议**：Server-Sent Events (SSE)，统一 JSON 事件分发 (`tool_start`, `tool_end`, `text`, `done`, `error`)

---

## 3. 数据层与表结构规范 (对齐 `sql/ch02-ddl.sql`)

### 3.1 实体模型定义 (`app/models/`)

1. **`conversations` 表 (会话外壳)**
   - `id`: BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY
   - `user_id`: VARCHAR(64) NOT NULL (默认 `'default_user'`)
   - `status`: ENUM('进行中', '已转人工', '已结束') NOT NULL DEFAULT '进行中'
   - `created_at`: DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
   - `updated_at`: DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP

2. **`messages` 表 (消息流水)**
   - `id`: BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY
   - `conversation_id`: BIGINT UNSIGNED NOT NULL (外键关联 `conversations.id`)
   - `role`: ENUM('user', 'assistant', 'tool') NOT NULL
   - `content`: TEXT NULL (assistant 发起工具调用时可为 NULL)
   - `tool_calls`: JSON NULL (存储模型返回的工具调用申请单列表)
   - `tool_call_id`: VARCHAR(64) NULL (对应 tool 消息回灌给模型的调用唯一标识)
   - `created_at`: DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP

3. **`faq` 表 (常见问答数据源)**
   - `id`: BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY
   - `question`: VARCHAR(512) NOT NULL
   - `answer`: TEXT NOT NULL
   - `category`: VARCHAR(64) NOT NULL
   - `created_at`: DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
   - `updated_at`: DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP

4. **`tickets` 表 (人工工单记录)**
   - `ticket_no`: VARCHAR(32) PRIMARY KEY (如 `T202609071008`)
   - `conversation_id`: BIGINT UNSIGNED NOT NULL (外键关联 `conversations.id`)
   - `description`: TEXT NOT NULL
   - `ticket_type`: ENUM('售后', '投诉', '咨询') NOT NULL
   - `status`: ENUM('待处理', '已处理') NOT NULL DEFAULT '待处理'
   - `created_at`: DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP

---

## 4. 业务工具集与工具基础设施规范

### 4.1 五大业务工具 (`app/tools/business_tools.py`)

1. **`query_order`**
   - 目标：查询订单状态与详情
   - 参数：`order_id: str`
   - 实现：内部 Mock 生成订单数据（包含商品、实付金额、状态、下单时间）
2. **`query_product`**
   - 目标：查询商品基本信息与库存
   - 参数：`product_id_or_name: str`
   - 实现：内部 Mock 生成商品数据（名称、价格、库存状态、核心参数）
3. **`query_logistics`**
   - 目标：查询订单物流轨迹与承运商
   - 参数：`order_id: str`
   - 实现：内部 Mock 生成最新配送节点、物流状态与快递公司
4. **`query_faq`**
   - 目标：按关键词检索常见问答库
   - 参数：`keyword: str`
   - 实现：通过异步 SQLAlchemy 执行 `SELECT * FROM faq WHERE question LIKE :kw OR answer LIKE :kw LIMIT 3`。
5. **`create_ticket`**
   - 目标：转人工或复杂诉求创建人工处理工单
   - 参数：`conversation_id: int`, `description: str`, `ticket_type: Literal['售后', '投诉', '咨询'] = '售后'`
   - 实现：生成工单号并写入 `tickets` 表，返回工单号与受理结果

### 4.2 工具基础设施 (`app/tools/`)
- **`ToolRegistry`**：负责工具的发现、聚合、按名查找，以及生成模型绑定的工具清单。
- **`validate_tool_args`**：基于工具的 `args_schema` 进行参数强校验，校验失败拦截并抛出类型友好的格式化异常。
- **`ToolExecutor`**：
  - **超时控制**：默认 5 秒（`asyncio.wait_for`）。
  - **异常重试**：针对网络与 I/O 抖动进行至多 1 次退避重试。
  - **优雅回灌**：工具执行出现异常或超时不抛给用户，而是捕获后构造结构化错误结果包装成 `ToolMessage` 回灌给模型，引导模型向用户作出得体的解答。

---

## 5. 单轮收敛编排与 SSE 流式交互

### 5.1 单轮状态机流水线 (`app/services/chat_service.py`)
1. **消息接收**：校验/创建 `conversation_id`，将用户提问落盘至 `messages(role='user')`。
2. **第一阶段意图识别**：调用绑定了工具集的 `llm_with_tools`。
   - **分支 A (无需工具)**：直接流式吐出文本帧，输出完毕后落盘 `messages(role='assistant')`。
   - **分支 B (触发工具)**：
     1. 落盘 `messages(role='assistant', tool_calls=...)`；
     2. 发射 SSE `tool_start` 帧；
     3. 异步执行工具（带校验、超时与重试）；
     4. 发射 SSE `tool_end` 帧；
     5. 工具结果落盘至 `messages(role='tool', tool_call_id=...)`；
     6. 回灌模型：将工具执行结果作为上下文传入模型，调用 `llm.astream(...)` 逐 Token 吐出最终回答；
     7. 流式结束，最终回复文本落盘至 `messages(role='assistant')`。
3. **流终止**：发射 `data: [DONE]\n\n`。

### 5.2 SSE 事件结构定义
- `tool_start`: `{"event_type": "tool_start", "conversation_id": 1, "tool_name": "query_logistics", "tool_label": "查询物流", "args": {"order_id": "1001"}}`
- `tool_end`: `{"event_type": "tool_end", "conversation_id": 1, "tool_name": "query_logistics", "success": true}`
- `text`: `{"event_type": "text", "conversation_id": 1, "content": "..."}`
- `error`: `{"event_type": "error", "conversation_id": 1, "error": "..."}`

---

## 6. 前端改造与 Vibe Coding 规范
- 页面保留 `app/static/index.html` 优雅视觉设计。
- 收到 `tool_start`：在当前气泡上方渲染微动效加载徽章 `[⚡ 正在调用工具: 查询物流...]`。
- 收到 `tool_end`：徽章平滑切换为收敛态小胶囊 `[📦 已调用: query_logistics]`。
- 收到 `text`：正文内容自徽章下方正常平滑逐字打印。

---

## 7. 验收标准与验证方案

1. **验收 1（物流查询工具触发）**：
   - 输入：「订单 1001 的物流到哪了」
   - 结果：气泡出现工具徽章 `[query_logistics]`，且模型按生成的物流节点作答。
2. **验收 2（FAQ 精确命中）**：
   - 输入：「退货政策是什么」
   - 结果：命中 FAQ 库中的「退货政策说明」，模型输出正确的退货细则。
3. **验收 3（FAQ 关键词漏召回验证）**：
   - 输入：「邮费是多少」
   - 结果：FAQ 库中只有「运费」，SQL `LIKE '%邮费%'` 查无结果，模型优雅告知或提示转人工（漏召回为预期特性，为后续 RAG 升级提供对比依据）。
