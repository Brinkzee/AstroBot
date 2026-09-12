# 工作流核心分流器与退款确定性子流程实施计划 (Ch06 Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将客服工作流中的占位分流器全面重构成正式生产版，实现多轮指代消解与口语归一化、意图识别 Prompt 四件套与置信度级联、核心场景检索侧 Query 扩写与多路召回、确定性退款/售后子流程、订单槽位中断挂起与卡片回填、以及前端退款表单交互闭环。

**Architecture:** 基于 LangGraph 的确定性状态图编排，前置指代消解（原样透传保护）与 8 分类意图识别 Prompt 四件套；分流路由将「退款退货/售后」接入确定性子流程（缺单时挂起下发 `order_selector` 事件，命中订单时预取真实数据并并发扩写检索政策）；主力 Agent 专注退款资格裁决并联动前端订单卡片与退款单表单。

**Tech Stack:** Python 3.12, LangGraph, LangChain Core, FastAPI, SQLAlchemy, SSE (EventSource), HTML5/Tailwind/Vanilla JS.

**Spec:** `docs/superpowers/specs/2026-09-11-router-refinement-design.md`

## Global Constraints
- 无新增外部重型组件，延用现有系统上游；
- 意图识别走 LLM Prompt 路线，不训分类模型/BERT；
- 意图识别 Prompt 严格遵循四件套：8 分类枚举、强 JSON 输出（含 `confidence`）、边界 few-shot 样例、「其他」兜底类；
- Query 扩写仅对「退款退货」与「售后」触发，强制输出 `{"queries": [...]}`，纯检索侧现查现用，知识库物理存储保持单份；
- 缺订单号时严禁模型盲目猜测，执行阶段弹出订单卡片让用户点选回填；退款原因不追问，提交退款单时从固定类目下拉自选；
- 前端部分按 Vibe Coding 方式快速交付，后端严格遵循 TDD（非单测类使用评估样例集）；
- 每阶段在 `dev-notes/ch06.md` 实时追记四要素，严禁收尾一次性补记。

---

### Task 1: 扩展工作流 State 状态与初始化配置

**Files:**
- Modify: `app/services/workflow/state.py`
- Test: `tests/test_workflow_state.py`

**Interfaces:**
- Consumes: Existing `AgentWorkflowState` definition
- Produces: Updated `AgentWorkflowState` with `confidence: Optional[float]`, `order_id: Optional[str]`, `order_data: Optional[Dict[str, Any]]`, `suggested_orders: Optional[List[Dict[str, Any]]]`, and `create_initial_state`.

- [ ] **Step 1: Write the failing test**
Update `tests/test_workflow_state.py` to assert new fields (`confidence`, `order_id`, `order_data`, `suggested_orders`) in `create_initial_state`.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_workflow_state.py -v`

- [ ] **Step 3: Implement state extensions**
In `app/services/workflow/state.py`, add the fields to `AgentWorkflowState` TypedDict and initialize them in `create_initial_state`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_workflow_state.py -v`

- [ ] **Step 5: Commit**
```bash
git add app/services/workflow/state.py tests/test_workflow_state.py
git commit -m "feat(workflow): extend AgentWorkflowState with order slots and confidence"
```

---

### Task 2: 指代消解与口语归一化合并节点 (含透传保护)

**Files:**
- Modify: `app/services/workflow/nodes/pre_nodes.py`
- Test: `tests/test_workflow_coref_rewrite.py`

**Interfaces:**
- Consumes: `state.input_query`, `state.messages`
- Produces: `coreference_rewrite_node(state, model=None) -> {"resolved_query": str}`

- [ ] **Step 1: Write the failing test**
Create `tests/test_workflow_coref_rewrite.py` testing:
1. 代词消解：「它能退吗」结合历史消息「订单1001极简保暖羽绒服」补全为独立完整问题；
2. 口语归一化：「货走到哪了老铁」规范化为标准业务提问；
3. 原样透传保护（Hard Gate）：「查询订单1001」或「你好」无模糊指代提问必须原样返回。

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_workflow_coref_rewrite.py -v`

- [ ] **Step 3: Implement coreference_rewrite_node**
In `app/services/workflow/nodes/pre_nodes.py`:
- Construct `COREFERENCE_REWRITE_SYSTEM_PROMPT` specifying pronoun resolution, colloquial normalization, and strict pass-through rules;
- Format past conversation messages (last 4-6 turns) into prompt context;
- Call LLM `ainvoke` and return `{"resolved_query": resolved_text}`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_workflow_coref_rewrite.py -v`

- [ ] **Step 5: Commit**
```bash
git add app/services/workflow/nodes/pre_nodes.py tests/test_workflow_coref_rewrite.py
git commit -m "feat(workflow): implement coreference rewrite node with passthrough protection"
```

---

### Task 3: 意图识别 Prompt 四件套与置信度级联降级

**Files:**
- Modify: `app/services/workflow/nodes/pre_nodes.py`
- Test: `tests/test_workflow_intent_four_elements.py`

**Interfaces:**
- Consumes: `state.resolved_query`, `state.input_query`
- Produces: `intent_recognition_node(state, model=None, small_model=None) -> {"intent": str, "confidence": float, "intent_reason": str}`

- [ ] **Step 1: Write the failing test**
Create `tests/test_workflow_intent_four_elements.py` testing:
1. 8 分类（7 业务 + 1 「其他」）选择题 Prompt 与标准 JSON 解析（包含 `intent` 与 `confidence`）；
2. 边界 Few-shot 验证（退款 vs 售后，催发货 vs 催退款）；
3. 离群怪问题（如「今天北京天气怎么样」）稳定归入「其他」；
4. 双模型级联降级测试：小模型低置信度（< 0.85）时自动级联主力大模型。

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_workflow_intent_four_elements.py -v`

- [ ] **Step 3: Implement intent prompt four-piece kit & cascade**
In `app/services/workflow/nodes/pre_nodes.py`:
- Define `INTENT_FOUR_ELEMENTS_PROMPT` containing the 8 enumerated choices, few-shot examples, and strict JSON format requirement;
- Add `other_fallback_node` for friendly deflection when intent is "其他";
- Implement dual-model cascade in `intent_recognition_node`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_workflow_intent_four_elements.py -v`

- [ ] **Step 5: Commit**
```bash
git add app/services/workflow/nodes/pre_nodes.py tests/test_workflow_intent_four_elements.py
git commit -m "feat(workflow): implement intent recognition prompt four elements with confidence cascade"
```

---

### Task 4: 确定性退款子流程之订单槽位提取与缺单挂起

**Files:**
- Create: `app/services/workflow/nodes/refund_nodes.py`
- Test: `tests/test_workflow_refund_nodes.py`

**Interfaces:**
- Consumes: `state.resolved_query`, `state.input_query`, `state.messages`
- Produces:
  - `refund_order_check_node(state) -> Dict[str, Any]` (sets `order_id`, `order_data` if found; sets `status="need_order_selection"` and `suggested_orders` if missing)
  - `emit_order_selector_node(state) -> Dict[str, Any]` (prepares user message prompt and orders list)

- [ ] **Step 1: Write the failing test**
Create `tests/test_workflow_refund_nodes.py` testing:
1. 提问中包含订单号「订单1001可以退吗」时，成功提取 `order_id="1001"` 并预取真实订单数据；
2. 提问中未包含订单号「这个能退货吗」时，未匹配到订单号，返回 `status="need_order_selection"` 并填充 Mock 候选订单卡片数据。

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_workflow_refund_nodes.py -v`

- [ ] **Step 3: Implement refund_order_check_node**
In `app/services/workflow/nodes/refund_nodes.py`:
- Regex + context extractor for order numbers;
- If matched, load from `MOCK_ORDERS` into `order_data`;
- If missing, load candidate list from `MOCK_ORDERS`, set `status="need_order_selection"` and `response_text="为您查询退款政策前，请先选择您需要咨询的订单："`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_workflow_refund_nodes.py -v`

- [ ] **Step 5: Commit**
```bash
git add app/services/workflow/nodes/refund_nodes.py tests/test_workflow_refund_nodes.py
git commit -m "feat(workflow): implement refund order check and slot interruption node"
```

---

### Task 5: 核心场景检索侧 Query 扩写与多路政策召回

**Files:**
- Modify: `app/services/workflow/nodes/refund_nodes.py`
- Test: `tests/test_workflow_query_expansion.py`

**Interfaces:**
- Consumes: `state.resolved_query`, `state.order_data`, `retriever`
- Produces: `refund_expansion_retrieval_node(state, model=None, retriever=None) -> {"retrieved_docs": List[Dict[str, Any]]}`

- [ ] **Step 1: Write the failing test**
Create `tests/test_workflow_query_expansion.py` testing:
1. Query 扩写模型输出强 JSON `{"queries": [...]}` 校验；
2. 并发检索多路 queries，对相同文档按内容去重，保留最高分并按 score 排序截断；
3. 模型输出异常时的降级保护（回退至单 query 检索）。

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_workflow_query_expansion.py -v`

- [ ] **Step 3: Implement refund_expansion_retrieval_node**
In `app/services/workflow/nodes/refund_nodes.py`:
- Prompt generating 2-3 focused search queries for refund/after-sales;
- `asyncio.gather` for parallel retrieval via `retriever.retrieve_with_strategy` or `retrieve`;
- Deduplication and score ranking into `retrieved_docs`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_workflow_query_expansion.py -v`

- [ ] **Step 5: Commit**
```bash
git add app/services/workflow/nodes/refund_nodes.py tests/test_workflow_query_expansion.py
git commit -m "feat(workflow): implement query expansion and multi-path policy retrieval"
```

---

### Task 6: 退款子流程与主力 Agent 专注裁决集成

**Files:**
- Modify: `app/services/workflow/nodes/agent_node.py`
- Test: `tests/test_workflow_agent_refund.py`

**Interfaces:**
- Consumes: `state.order_data`, `state.retrieved_docs`, `state.intent`
- Produces: `main_agent_node` updated with refund-specialized system instructions and `suggested_actions: ["apply_refund"]` when eligible.

- [ ] **Step 1: Write the failing test**
Create `tests/test_workflow_agent_refund.py` testing:
1. 传入已发货且符合政策的订单与退货政策，Agent 输出裁决支持退货，并且包含 `suggested_actions: ["apply_refund"]`；
2. Agent 不再发起对 `query_order` 的冗余工具调用。

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_workflow_agent_refund.py -v`

- [ ] **Step 3: Implement refund context injection in main_agent_node**
In `app/services/workflow/nodes/agent_node.py`:
- Check if `order_data` is present in state;
- Inject `【当前订单真实状态】` and `【官方政策条款】` into system prompt;
- Instruct agent to focus solely on judging refund eligibility without tool loops;
- Append `apply_refund` to `suggested_actions` if eligible.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_workflow_agent_refund.py -v`

- [ ] **Step 5: Commit**
```bash
git add app/services/workflow/nodes/agent_node.py tests/test_workflow_agent_refund.py
git commit -m "feat(workflow): specialize main agent for refund adjudication and action trigger"
```

---

### Task 7: 组装正式版分流路由器与全图拓扑

**Files:**
- Modify: `app/services/workflow/nodes/router.py`
- Modify: `app/services/workflow/nodes/__init__.py`
- Modify: `app/services/workflow/engine.py`
- Test: `tests/test_workflow_engine_ch06.py`

**Interfaces:**
- Consumes: All updated and new workflow nodes
- Produces: `build_workflow_graph()`, `WorkflowEngine.run()` supporting the new 8-intent routing and deterministic refund subflow.

- [ ] **Step 1: Write the failing test**
Create `tests/test_workflow_engine_ch06.py` testing:
1. 意图分流条件边：闲聊、投诉、其他、商品咨询、物流/订单、退款退货/售后分别流向正确出口；
2. 退款缺单时直通 `order_selector -> logging -> END`；
3. 退款有单时流经 `refund_order_check -> refund_expansion_retrieval -> main_agent -> logging -> END`。

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_workflow_engine_ch06.py -v`

- [ ] **Step 3: Update router and assemble workflow graph**
In `app/services/workflow/nodes/router.py` and `app/services/workflow/engine.py`:
- Update `route_by_intent` to map 8 intents;
- Add `route_refund_slot` condition edge;
- Register nodes: `coreference` (using new rewrite), `intent`, `chitchat`, `complaint`, `other_fallback`, `knowledge_retrieval`, `refund_order_check`, `refund_expansion_retrieval`, `main_agent`, `logging`;
- Wire conditional edges per architecture spec.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_workflow_engine_ch06.py -v`

- [ ] **Step 5: Commit**
```bash
git add app/services/workflow/nodes/router.py app/services/workflow/nodes/__init__.py app/services/workflow/engine.py tests/test_workflow_engine_ch06.py
git commit -m "feat(workflow): assemble refined router and deterministic refund subflow graph"
```

---

### Task 8: ChatService 与 SSE 流式事件协议串联适配

**Files:**
- Modify: `app/services/chat_service.py`
- Test: `tests/test_api_chat_stream_ch06.py`

**Interfaces:**
- Consumes: `WorkflowEngine.run` final_state
- Produces: `stream_chat` yielding `order_selector`, `actions`, `text` SSE events.

- [ ] **Step 1: Write the failing test**
Create `tests/test_api_chat_stream_ch06.py` asserting that when `status == "need_order_selection"`, `stream_chat` yields an event with `event_type == "order_selector"` and valid orders list.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_api_chat_stream_ch06.py -v`

- [ ] **Step 3: Implement SSE event emission in ChatService**
In `app/services/chat_service.py`:
- When `final_state.get("status") == "need_order_selection"`:
  - Yield `{"event_type": "order_selector", "conversation_id": conv_id, "orders": final_state.get("suggested_orders", [])}`;
- Ensure `response_text` and `suggested_actions` are emitted cleanly.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_api_chat_stream_ch06.py -v`

- [ ] **Step 5: Commit**
```bash
git add app/services/chat_service.py tests/test_api_chat_stream_ch06.py
git commit -m "feat(chat): wire order selector and refund action events into SSE stream"
```

---

### Task 9: 前端 Vibe Coding 交互配套（订单选择卡片与退款单表单）

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes: SSE `order_selector` event, `actions` event (`apply_refund`), `POST /api/tickets`
- Produces:
  1. In-stream clickable order cards that auto-send `退款订单: {order_id}`;
  2. Modal refund form with prefilled order number and dropdown of fixed refund categories.

- [ ] **Step 1: Implement Order Selector Cards in Chat Stream**
Add CSS and JS handler for `order_selector` event:
- Render responsive order cards with order_id, status badge, amount, product name;
- Click card triggers disable state and sends user message `退款订单: 1001`.

- [ ] **Step 2: Implement Refund Application Modal Form**
Add Modal markup and JS handler for `apply_refund` action button:
- Show order_id, product;
- Dropdown select for refund reasons (`7天无理由退货`, `商品质量问题/破损`, `少件/漏发/错发`, `尺码/规格不合`, `物流停滞想退款`);
- Submit button calling `POST /api/tickets` with `ticket_type="退款退货"`.

- [ ] **Step 3: Commit**
```bash
git add app/static/index.html
git commit -m "feat(frontend): implement order selector cards and refund application modal"
```

---

### Task 10: 综合端到端验收套件与全量回归验证

**Files:**
- Create: `tests/test_ch06_acceptance.py`
- Modify: `dev-notes/ch06.md`

**Interfaces:**
- Consumes: Full system API & WorkflowEngine
- Produces: Acceptance test verifying all 4 criteria from user prompt.

- [ ] **Step 1: Write acceptance test cases**
In `tests/test_ch06_acceptance.py`:
1. 多轮用例：问物流 -> 要退款（无单号弹选择器） -> 选择订单1001 -> 问「它能退吗」（消解补全、政策检索、主力 Agent 裁决） -> 聊回物流；
2. 8 类意图 JSON 稳定解析，离群怪问题判定为「其他」；
3. 退款子流程强制触发 Query 扩写与政策检索；
4. 全流程端到端检验。

- [ ] **Step 2: Run acceptance tests & full regression**
Run: `pytest tests/test_ch06_acceptance.py -v`
Run: `pytest tests/ -v`

- [ ] **Step 3: Commit**
```bash
git add tests/test_ch06_acceptance.py
git commit -m "test(acceptance): add comprehensive chapter 6 acceptance test suite"
```
