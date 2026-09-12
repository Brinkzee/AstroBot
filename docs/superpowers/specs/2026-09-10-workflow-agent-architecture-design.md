# 架构设计规范 (Design Spec) - 第五章：Workflow 确定性编排与生产级 Agent 架构

## 1. 目标与背景

当前智能客服系统在前四章中完成了 Function Calling 工具链（Ch02）、基础 RAG（Ch03）与混合检索重排评估体系（Ch04）。然而，原有的单轮工具调用与意图处理仍偏离生产级架构的高可控性要求：当面临复杂对话场景时，单纯依赖大模型自由发挥容易出现路由漂移、滥用工具或幻觉拒答。

本章致力于将客服系统重塑为**“以 Workflow 确定性编排做骨架，主力 Agent 作为核心节点”**的生产级架构：
1. **祛魅热身**：先脱离任何第三方 Agent 框架，用最简原生 Python 手写裸 Agent 反馈循环，看清 Agent 的核心本质就是「LLM 决策 -> 工具调用 -> 结果喂回 -> 自发收敛」；
2. **确定性图骨架**：基于 LangGraph 编排确定性工作流，贯穿指代消解、意图识别、四出口路由分流、知识预检索、置信度闸门、主力 Agent 与审计日志；
3. **分流规则与置信度守卫**：将 7 类意图硬编码路由至 4 个出口（闲聊、投诉、知识类、业务数据类）。知识类问题进 Agent 之前必须先过置信度闸门，防止流式输出中途反悔；弱证据直接拦截并沉淀至低置信度问题池；
4. **主力 Agent ReAct 闭环**：支持简单问题 1 步收敛、复杂复合问题（如先查订单再查物流）多步工具自适应调用，并具备 5 步硬截断与 Token 消耗保护；
5. **解耦建议交互**：投诉及人工建议不进后端自动执行，前端独立渲染「转人工」与「建工单」两枚胶囊按钮，前者前端模拟小猫迎宾，后者通过独立接口入库 `tickets` 表，两者互不绑定。

---

## 2. 系统总体架构与拓扑

系统基于 LangGraph `StateGraph` 构建确定性主拓扑，配合 `MemorySaver` 实现基于 `conversation_id` 的状态持久化，拓扑流转如下：

```
                           [START]
                              │
                              ▼
                [coreference_resolution_node]
                (最简实现: resolved_query 透传)
                              │
                              ▼
                   [intent_recognition_node]
                (单次 Prompt 判成 7 类, 输出 JSON)
                              │
                              ▼
                    <route_by_intent_edge>
        ┌─────────────────────┼─────────────────────┬─────────────────────┐
        │                     │                     │                     │
        ▼                     ▼                     ▼                     ▼
 [出口 1: 闲聊]         [出口 2: 投诉]         [出口 3: 知识类]       [出口 4: 业务数据类]
 (意图: 闲聊)           (意图: 投诉)           (意图: 商品咨询/退款退货) (意图: 物流/订单/售后)
        │                     │                     │                     │
        ▼                     ▼                     ▼                     │
 [chitchat_node]       [complaint_node]   [knowledge_retrieval_node]      │
 (返回固定亲和话术,   (输出安抚话术, 推荐   (复用 Ch04 进阶检索器)           │
  0 LLM Token 消耗)   双按钮, 不进 Agent)           │                     │
        │                     │                     ▼                     │
        │                     │             <confidence_gate>             │
        │                     │             │               │             │
        │                     │    (得分<0.35/空)      (得分>=0.35)       │
        │                     │             │               │             │
        │                     │             ▼               └───────┐     │
        │                     │  [knowledge_fallback_node]          │     │
        │                     │  (输出兜底话术, 记录至              │     │
        │                     │   low_confidence_questions)         │     │
        │                     │             │                       │     │
        │                     │             │                       ▼     ▼
        │                     │             │              [main_agent_node]
        │                     │             │             (ReAct 工具循环: 简单1步,
        │                     │             │              复合多步, 5步截断/Token统计)
        │                     │             │                       │
        └─────────────────────┼─────────────┴───────────────────────┘
                              ▼
                        [logging_node]
                   (审计记录、耗时统计与落盘)
                              │
                              ▼
                            [END]
```

---

## 3. 详细模块设计

### 3.1 祛魅热身：原生裸 Agent 循环 (`scripts/bare_agent_loop.py`)
- **设计原理**：脱离 LangChain / LangGraph 等封装库，纯靠基础 Python 构造：
  - 消息维护列表 `messages = [SystemMessage(...), HumanMessage(query)]`；
  - 核心 `while step < max_steps` 循环：
    - `bound_llm.ainvoke(messages)` 获取响应；
    - 若 `not resp.tool_calls`：收敛跳出，返回最终答案；
    - 若包含 `tool_calls`：遍历执行各工具函数，生成 `ToolMessage(content=..., tool_call_id=...)` 追加至 `messages`，继续下一轮；
  - 步数超限防护（`max_steps=5`）：触发强制总结。
- **验证机制**：配套 `tests/test_bare_agent_loop.py`，单测覆盖单步收敛与多步调用，直观印证 Agent 循环本质。

### 3.2 全局 State 契约与 Checkpointer (`app/services/workflow/state.py`)
采用 LangGraph 原生 `TypedDict` 定义统一 State：
```python
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentWorkflowState(TypedDict):
    conversation_id: int
    user_id: str
    input_query: str                         # 用户当轮原始提问
    resolved_query: str                      # 指代消解后 Query
    intent: Optional[str]                    # 7类意图
    intent_reason: Optional[str]             # 意图分析原因
    retrieved_docs: List[Dict[str, Any]]     # 知识库召回 chunks
    confidence_passed: Optional[bool]        # 置信度闸门放行标记
    messages: Annotated[List[BaseMessage], add_messages] # 对话及 ReAct 工具轨迹
    response_text: str                       # 面向用户的最终文本
    suggested_actions: List[str]             # 建议前端按钮列表 ["transfer_agent", "create_ticket"]
    token_usage: Dict[str, int]              # prompt_tokens, completion_tokens, total_tokens
    steps_taken: int                         # ReAct 步数
    status: str                              # success | fallback | complaint | chitchat | error
```
- **Checkpointer 持久化机制**：
  - LangGraph 运行时使用 `MemorySaver`（配置 `thread_id=str(conversation_id)`），跨轮自动延续消息状态；
  - 业务落盘：单轮工作流结束时，将用户提问与 Assistant 最终回复同步写入 MySQL/SQLite `messages` 表，确保前端刷新页面能读取完整历史。

### 3.3 前置确定性节点与分流路由
1. **指代消解节点 (`coreference_node`)**：最简实现，`resolved_query = input_query` 直接透传；
2. **意图识别节点 (`intent_node`)**：单次轻量 Prompt 约束输出严格 JSON：
   - 分类范围：`物流`、`订单`、`商品咨询`、`退款退货`、`售后`、`投诉`、`闲聊`；
   - 提取格式：`{"intent": "...", "reason": "..."}`，异常时默认降级为 `业务数据类-售后`；
3. **确定性分流规则 (`route_by_intent`)**：
   - `闲聊` ➔ `chitchat_node`：输出预设固定亲和话术，不调用 LLM；
   - `投诉` ➔ `complaint_node`：输出诚恳安抚话术，设置 `suggested_actions = ["transfer_agent", "create_ticket"]`，阻断后续 Agent 链路；
   - `商品咨询` / `退款退货` ➔ `knowledge_retrieval_node`；
   - `物流` / `订单` / `售后` ➔ 直通 `main_agent_node`。

### 3.4 知识检索与置信度闸门
1. **知识检索节点 (`knowledge_retrieval_node`)**：
   - 复用第 4 章 `AdvancedKnowledgeRetriever` 混合检索与重排服务；
   - 检索 `resolved_query`，将命中的 Top-K 结果及其相似度打分注入 `state["retrieved_docs"]`；
2. **置信度闸门 (`confidence_gate`)**：
   - 判定标准：若 `len(hits) == 0` 或 `max(hit.score) < 0.35`，判定为证据弱，导向 `knowledge_fallback_node`；否则放行进入 `main_agent_node`；
   - 闸门前置意义：在流式生成开始前提前裁决，彻底规避回答中途反悔；
   - 兜底节点处理：
     - 生成标准兜底话术：“抱歉，针对您咨询的问题，当前知识库中暂未收录相关规范，建议您换个表述或联系人工客服。”；
     - 记录至 `low_confidence_questions` 表（记录字段包括：`query`, `conversation_id`, `entrance='workflow_confidence_gate'`, `reason='max_score < 0.35'`），为数据飞轮预留沉淀；
     - 附带建议按钮 `suggested_actions = ["transfer_agent", "create_ticket"]`。

### 3.5 主力 ReAct Agent 节点 (`main_agent_node`)
1. **工具集绑定**：复用第 2 章五大工具（`query_order`, `query_product`, `query_logistics`, `query_faq`, `create_ticket`）；
2. **知识融合机制**：
   - 若来自知识类分支且通过闸门，将 `retrieved_docs` 格式化后注入 Agent 的 System 上下文；
   - **退款退货复合场景支持**：对于退款退货问题，Agent 先检索并掌握退款时效规则（如“签收7天内可申请”），同时保留自主调用 `query_order` 的能力，核对订单实际签收时间后给出严密回答；
3. **多步推理与收敛控制**：
   - 简单场景（如“订单 1001 的物流到哪了”）：单次调用 `query_logistics` 后直接归纳回答，1 步收敛；
   - 复合场景（如“我买的羽绒服物流到哪了”）：先查订单确定商品所属订单编号，再查物流轨迹，多步（>=2 步）收敛；
   - 参数缺失（如“帮我查物流”未给单号）：LLM 识别参数缺失，直接输出反问提示，不触发工具执行；
   - 运行守卫：`max_steps = 5` 防死循环，累加记录 `token_usage`。

### 3.6 建议交互与独立工单 API
1. **建议下发**：当分流为投诉或触发兜底时，SSE 协议发射 `actions` 事件下发建议选项 `["transfer_agent", "create_ticket"]`；
2. **独立工单 API (`POST /api/tickets`)**：
   - 请求体：`{"conversation_id": int, "description": str, "ticket_type": str}`；
   - 响应体：`{"ticket_no": str, "conversation_id": int, "status": str}`；
   - 内部直接调用第 2 章 `create_ticket` 工具函数写入数据库 `tickets` 表，不产生模型费用；
3. **前端呈现与交互 (`index.html`)**：
   - 气泡下方独立渲染 `[转人工]` 与 `[建工单]`；
   - 点击 `[转人工]`：前端即时展示系统提示“已转接人工客服”，并模拟蹦出客服小猫问候语“您好，我是客服小猫，请问有什么可以帮您的 🐱”；
   - 点击 `[建工单]`：异步请求 `POST /api/tickets`，成功后渲染绿色卡片“工单已创建，单号【T...】”，按钮禁用；
   - 互不绑定与忽略：两者完全解耦，用户若不点击、直接在聊天框继续提问，后续对话照常流转。

---

## 4. 验证计划与验收标准映射

| 验收标准项 | 验证方案与断言设计 | 自动化测试用例 |
| :--- | :--- | :--- |
| **标准 1：政策类检索强制走闸** | 提问“退货政策是什么”，断言 State 轨迹包含 `knowledge_retrieval_node` 与 `confidence_gate` | `test_ch05_acceptance.py::test_acceptance_policy_mandatory_retrieval` |
| **标准 2：业务数据自主调工具** | 提问“订单 1001 的物流到哪了”，断言直通 Agent 且触发 `query_logistics` 工具调用并收敛 | `test_ch05_acceptance.py::test_acceptance_logistics_agent_tool_calling` |
| **标准 3：投诉建议与动作解耦** | 提问“我要投诉”，断言返回安抚语并携带建议按钮；分别测试转人工前端模拟与工单 API 落库 | `test_ch05_acceptance.py::test_acceptance_complaint_decoupled_actions` |
| **标准 4：闲聊固定话术** | 提问“你好”，断言返回固定话术，断言 LLM Token 消耗为 0 | `test_ch05_acceptance.py::test_acceptance_chitchat_fixed_response` |
| **标准 5：复杂链式多步 ReAct** | 提问复合型问题，断言 Agent 循环步数 `steps_taken >= 2` | `test_ch05_acceptance.py::test_acceptance_complex_multistep_react` |

---

## 5. Spec 自检结论
- [x] **无占位符扫描**：全文字无“TBD/TODO”等未定义内容；
- [x] **内部一致性**：State 结构、图节点命名与接口契约全链路吻合；
- [x] **范围受控性**：严格聚焦于 Workflow 编排、ReAct 节点与前端解耦交互，指代消解与意图识别保持最简实现；
- [x] **歧义消除**：明确转人工为纯前端模拟、建工单走专属 REST API、双轨持久化机制。
