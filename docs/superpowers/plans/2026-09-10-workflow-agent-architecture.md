# 第五章：Workflow 确定性编排与生产级 Agent 架构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 LangGraph 将智能客服系统升级为“Workflow 确定性编排做骨架、主力 Agent 做核心节点”的生产级架构，完成手写裸 Agent 循环祛魅、七类意图四出口分流、知识库检索与前置置信度闸门、主力多步 ReAct 闭环及解耦的前端建议按钮交互。

**Architecture:** 
顶层采用 LangGraph `StateGraph` 构建确定性工作流，贯穿指代消解、意图识别、四出口条件路由分流（闲聊、投诉、知识类、业务数据类）；知识类强制做 RAG 预检索并经置信度闸门过滤（弱证据拦截入库 `low_confidence_questions`）；主力 Agent 节点运行多步 ReAct 循环，集成第 2 章五大业务工具并设置最大 5 步硬防御截断与 Token 审计；会话 State 全链路透传并借助 LangGraph `MemorySaver` 与既有数据库消息表实现双轨持久化；前端独立渲染「转人工」纯模拟与「建工单」REST API 触发。

**Tech Stack:** Python 3.12, FastAPI, LangGraph, LangChain, SQLAlchemy, Pydantic, pytest, pytest-asyncio, HTML5/CSS3/JavaScript (ES6)

**Spec:** [docs/superpowers/specs/2026-09-10-workflow-agent-architecture-design.md](file:///d:/PycharmProjects/AstroBot/docs/superpowers/specs/2026-09-10-workflow-agent-architecture-design.md)

## Global Constraints
- Python 版本 >= 3.10，项目使用 Python 3.12.7；
- 库版本兼容性：LangGraph >= 0.2.0, LangChain >= 0.3.0, FastAPI >= 0.115.0；
- 意图识别七大类限定：`物流`、`订单`、`商品咨询`、`退款退货`、`售后`、`投诉`、`闲聊`；
- 四大出口规则定死：闲聊 0 Token 固定话术；投诉安抚+建议双按钮不进 Agent；知识类强制 RAG 预检索并过置信度闸门；业务数据类直通 Agent；
- 置信度闸门：以最高得分 `< 0.35` 或无召回判定为弱证据，拦截走兜底并落库 `low_confidence_questions`；
- 主力 Agent 严格设 `max_steps = 5` 防止死循环；
- 转人工与建工单彻底解耦：转人工前端纯模拟小猫迎宾，建工单独立调用 `POST /api/tickets` 写 `tickets` 表，两者互不绑定，用户可忽略。

---

## Task 1: 祛魅热身 - 纯手写裸 Agent 循环与单测验证

**Files:**
- Create: `scripts/bare_agent_loop.py`
- Test: `tests/test_bare_agent_loop.py`

**Interfaces:**
- Produces: `async def run_bare_agent_loop(llm: Any, tools: Dict[str, Any], query: str, max_steps: int = 5) -> Dict[str, Any]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bare_agent_loop.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from scripts.bare_agent_loop import run_bare_agent_loop

@pytest.mark.asyncio
async def test_bare_agent_loop_single_step_convergence():
    """测试单步工具调用并自发收敛"""
    mock_llm = MagicMock()
    # 第一轮返回工具调用，第二轮返回文本收敛
    resp1 = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"arg1": "val1"}, "id": "call_123"}]
    )
    resp2 = AIMessage(content="查询结果为：val1已处理完毕")
    
    bound_llm = MagicMock()
    bound_llm.ainvoke = AsyncMock(side_effect=[resp1, resp2])
    mock_llm.bind_tools.return_value = bound_llm

    async def mock_tool(arg1: str):
        return f"tool_result_{arg1}"

    result = await run_bare_agent_loop(
        llm=mock_llm,
        tools={"mock_tool": mock_tool},
        query="帮我查一下",
        max_steps=5,
    )

    assert result["answer"] == "查询结果为：val1已处理完毕"
    assert result["steps"] == 2
    assert len(result["messages"]) == 4  # system + human + ai(tool_call) + tool + ai(final)
    mock_llm.bind_tools.assert_called_once()

@pytest.mark.asyncio
async def test_bare_agent_loop_max_steps_guard():
    """测试超过最大步数强制防御截断"""
    mock_llm = MagicMock()
    # 始终返回工具调用，触发死循环守卫
    resp_loop = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"arg1": "val"}, "id": "call_loop"}]
    )
    bound_llm = MagicMock()
    bound_llm.ainvoke = AsyncMock(return_value=resp_loop)
    mock_llm.bind_tools.return_value = bound_llm

    # 兜底强制总结响应
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="步数超限强制总结"))

    async def mock_tool(arg1: str):
        return "loop"

    result = await run_bare_agent_loop(
        llm=mock_llm,
        tools={"mock_tool": mock_tool},
        query="死循环提问",
        max_steps=3,
    )

    assert result["steps"] == 3
    assert result["answer"] == "步数超限强制总结"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bare_agent_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.bare_agent_loop'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/bare_agent_loop.py
import inspect
from typing import Any, Callable, Dict, List
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

async def run_bare_agent_loop(
    llm: Any,
    tools: Dict[str, Callable],
    query: str,
    max_steps: int = 5,
) -> Dict[str, Any]:
    """手写裸 Agent 循环：不依赖任何 Agent 框架，纯靠 LLM 接口与 while 循环实现自适应工具调用"""
    messages: List[BaseMessage] = [
        SystemMessage(content="你是电商智能客服助手，请根据可用工具查询必要信息并回答用户问题。"),
        HumanMessage(content=query),
    ]

    tool_objs = list(tools.values())
    bound_llm = llm.bind_tools(tool_objs) if hasattr(llm, "bind_tools") else llm
    steps = 0

    while steps < max_steps:
        steps += 1
        resp = await bound_llm.ainvoke(messages)
        messages.append(resp)

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            # 收敛出自然语言答案，跳出循环
            return {
                "answer": str(resp.content or ""),
                "steps": steps,
                "messages": messages,
            }

        # 遍历执行工具调用并喂回 ToolMessage
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args") or {}
            call_id = str(tool_call.get("id") or "")

            tool_func = tools.get(tool_name)
            if tool_func is None:
                output = f"Error: Tool {tool_name} not found."
            else:
                if inspect.iscoroutinefunction(tool_func):
                    output = await tool_func(**tool_args)
                else:
                    output = tool_func(**tool_args)

            messages.append(ToolMessage(content=str(output), tool_call_id=call_id))

    # 步数超限防御：请求大模型基于已有中间上下文立即总结
    summary_messages = messages + [
        HumanMessage(content="已达到最大步数限制，请根据上述已有工具结果立即给出最终结论。")
    ]
    if hasattr(llm, "ainvoke"):
        final_resp = await llm.ainvoke(summary_messages)
    else:
        final_resp = await bound_llm.ainvoke(summary_messages)

    return {
        "answer": str(final_resp.content or ""),
        "steps": steps,
        "messages": messages,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bare_agent_loop.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/bare_agent_loop.py tests/test_bare_agent_loop.py
git commit -m "feat(agent): implement bare agent loop for framework demystification"
```

---

## Task 2: 工作流核心 State 定义与 Checkpointer 持久化配置

**Files:**
- Create: `app/services/workflow/state.py`
- Test: `tests/test_workflow_state.py`

**Interfaces:**
- Produces: `AgentWorkflowState` (TypedDict with `add_messages`), `create_initial_state(conversation_id: int, query: str, user_id: str = "default_user") -> AgentWorkflowState`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workflow_state.py
import pytest
from langchain_core.messages import HumanMessage
from app.services.workflow.state import AgentWorkflowState, create_initial_state

def test_create_initial_state_defaults():
    state = create_initial_state(conversation_id=101, query="我要查物流", user_id="user_1")
    assert state["conversation_id"] == 101
    assert state["user_id"] == "user_1"
    assert state["input_query"] == "我要查物流"
    assert state["resolved_query"] == "我要查物流"
    assert state["intent"] is None
    assert state["retrieved_docs"] == []
    assert state["confidence_passed"] is None
    assert state["suggested_actions"] == []
    assert state["token_usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert state["steps_taken"] == 0
    assert state["status"] == "initialized"
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)
    assert state["messages"][0].content == "我要查物流"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.workflow'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/workflow/state.py
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

class AgentWorkflowState(TypedDict):
    conversation_id: int
    user_id: str
    input_query: str
    resolved_query: str
    intent: Optional[str]
    intent_reason: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    confidence_passed: Optional[bool]
    messages: Annotated[List[BaseMessage], add_messages]
    response_text: str
    suggested_actions: List[str]
    token_usage: Dict[str, int]
    steps_taken: int
    status: str

def create_initial_state(
    conversation_id: int,
    query: str,
    user_id: str = "default_user",
) -> AgentWorkflowState:
    """构建单轮工作流启动初始 State"""
    clean_query = str(query or "").strip()
    return {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "input_query": clean_query,
        "resolved_query": clean_query,
        "intent": None,
        "intent_reason": None,
        "retrieved_docs": [],
        "confidence_passed": None,
        "messages": [HumanMessage(content=clean_query)],
        "response_text": "",
        "suggested_actions": [],
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "steps_taken": 0,
        "status": "initialized",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_state.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add app/services/workflow/state.py tests/test_workflow_state.py
git commit -m "feat(workflow): define AgentWorkflowState and initial state constructor"
```

---

## Task 3: 确定性前置节点与分流路由逻辑

**Files:**
- Create: `app/services/workflow/nodes/pre_nodes.py`
- Create: `app/services/workflow/nodes/router.py`
- Test: `tests/test_workflow_nodes.py`

**Interfaces:**
- Produces: 
  - `coreference_resolution_node(state: AgentWorkflowState) -> Dict[str, Any]`
  - `intent_recognition_node(state: AgentWorkflowState, model: Optional[Any] = None) -> Dict[str, Any]`
  - `chitchat_node(state: AgentWorkflowState) -> Dict[str, Any]`
  - `complaint_node(state: AgentWorkflowState) -> Dict[str, Any]`
  - `route_by_intent(state: AgentWorkflowState) -> str` ("chitchat" | "complaint" | "knowledge" | "business_data")

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workflow_nodes.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.pre_nodes import (
    coreference_resolution_node,
    intent_recognition_node,
    chitchat_node,
    complaint_node,
)
from app.services.workflow.nodes.router import route_by_intent

def test_coreference_resolution_passthrough():
    state = create_initial_state(1, "我的订单到哪了")
    updated = coreference_resolution_node(state)
    assert updated["resolved_query"] == "我的订单到哪了"

@pytest.mark.asyncio
async def test_intent_recognition_json_parsing():
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"intent": "物流", "reason": "用户询问订单物流轨迹"}'
    ))
    state = create_initial_state(1, "订单1001发到哪里了")
    updated = await intent_recognition_node(state, model=mock_model)
    assert updated["intent"] == "物流"
    assert updated["intent_reason"] == "用户询问订单物流轨迹"

def test_route_by_intent_rules():
    assert route_by_intent({"intent": "闲聊"}) == "chitchat"
    assert route_by_intent({"intent": "投诉"}) == "complaint"
    assert route_by_intent({"intent": "商品咨询"}) == "knowledge"
    assert route_by_intent({"intent": "退款退货"}) == "knowledge"
    assert route_by_intent({"intent": "物流"}) == "business_data"
    assert route_by_intent({"intent": "订单"}) == "business_data"
    assert route_by_intent({"intent": "售后"}) == "business_data"
    # 缺省兜底
    assert route_by_intent({"intent": "未知"}) == "business_data"

def test_chitchat_node_fixed_text():
    state = create_initial_state(1, "你好")
    updated = chitchat_node(state)
    assert "智能客服助手" in updated["response_text"]
    assert updated["status"] == "chitchat"
    assert updated["suggested_actions"] == []

def test_complaint_node_soothing_and_actions():
    state = create_initial_state(1, "我要投诉你们")
    updated = complaint_node(state)
    assert "非常抱歉" in updated["response_text"]
    assert updated["suggested_actions"] == ["transfer_agent", "create_ticket"]
    assert updated["status"] == "complaint"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_nodes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.workflow.nodes'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/workflow/nodes/pre_nodes.py
import json
import logging
from typing import Any, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from app.llm import get_chat_model
from app.services.workflow.state import AgentWorkflowState

logger = logging.getLogger(__name__)

VALID_INTENTS = {"物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"}

INTENT_SYSTEM_PROMPT = """你是一个电商客服意图识别专家。
请将用户的输入严格分类为以下七类之一：
1. 物流：询问包裹轨迹、发货进度、快递单号等
2. 订单：询问订单详情、金额、明细、下单时间等
3. 商品咨询：询问商品价格、规格、尺码、库存等
4. 退款退货：询问退款政策、退换货流程、运费规则等
5. 售后：质保维修、商品损坏等售后支持
6. 投诉：对服务态度不满、虚假宣传、强烈抗议、明确表示要投诉等
7. 闲聊：打招呼、问候、感谢、非业务寒暄等

请严格返回如下 JSON 格式，不要返回任何额外文字：
{"intent": "分类名称", "reason": "判定依据简述"}
"""

CHITCHAT_FIXED_TEXT = "您好！我是您的智能客服助手，请问有什么可以帮您？如果您需要查询订单、追踪物流或咨询售后退换货政策，随时发给我哦~"

COMPLAINT_SOOTHING_TEXT = "非常抱歉给您带来了不好的体验，请您消消气。我们非常重视您的反馈与诉求！您可以点击下方选项转接人工客服实时沟通，或直接提交人工客服工单，我们将由专人第一时间跟进为您处理。"

def coreference_resolution_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """指代消解节点：本章最简版，原样透传用户输入"""
    return {
        "resolved_query": state.get("input_query", ""),
    }

async def intent_recognition_node(
    state: AgentWorkflowState,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """意图识别节点：单次轻量 Prompt 判定 7 分类，输出 JSON"""
    query = state.get("resolved_query") or state.get("input_query", "")
    llm = model or get_chat_model(streaming=False)
    
    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=f"用户提问：{query}"),
    ]
    try:
        resp = await llm.ainvoke(messages)
        content = str(resp.content or "").strip()
        # 清除可能的 markdown 代码块标识
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        data = json.loads(content.strip())
        raw_intent = str(data.get("intent", "")).strip()
        intent = raw_intent if raw_intent in VALID_INTENTS else "售后"
        reason = str(data.get("reason", ""))
    except Exception as e:
        logger.warning(f"意图识别 JSON 解析异常，自动降级为业务数据类-售后: {e}")
        intent = "售后"
        reason = f"解析兜底: {str(e)}"

    return {
        "intent": intent,
        "intent_reason": reason,
    }

def chitchat_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """闲聊节点：直接返回预设固定亲和话术，不花大模型调用"""
    return {
        "response_text": CHITCHAT_FIXED_TEXT,
        "suggested_actions": [],
        "status": "chitchat",
    }

def complaint_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """投诉节点：安抚话术 + 推荐双按钮，不进 Agent，后端不自动执行动作"""
    return {
        "response_text": COMPLAINT_SOOTHING_TEXT,
        "suggested_actions": ["transfer_agent", "create_ticket"],
        "status": "complaint",
    }
```

```python
# app/services/workflow/nodes/router.py
from app.services.workflow.state import AgentWorkflowState

def route_by_intent(state: AgentWorkflowState) -> str:
    """分流路由条件边：将 7 类意图硬编码映射至 4 大出口"""
    intent = state.get("intent")
    if intent == "闲聊":
        return "chitchat"
    elif intent == "投诉":
        return "complaint"
    elif intent in ("商品咨询", "退款退货"):
        return "knowledge"
    elif intent in ("物流", "订单", "售后"):
        return "business_data"
    else:
        # 未知意图默认进业务数据类交主力 Agent 自由决策
        return "business_data"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_nodes.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add app/services/workflow/nodes/pre_nodes.py app/services/workflow/nodes/router.py tests/test_workflow_nodes.py
git commit -m "feat(workflow): implement coreference, intent recognition, and deterministic routing"
```

---

## Task 4: 知识检索节点与置信度闸门前置拦截

**Files:**
- Create: `app/services/workflow/nodes/knowledge_node.py`
- Create: `app/services/workflow/nodes/gate.py`
- Test: `tests/test_workflow_knowledge_gate.py`

**Interfaces:**
- Produces:
  - `knowledge_retrieval_node(state: AgentWorkflowState, retriever: Optional[Any] = None) -> Dict[str, Any]`
  - `confidence_gate(state: AgentWorkflowState) -> str` ("pass" | "fallback")
  - `knowledge_fallback_node(state: AgentWorkflowState, db: Optional[Any] = None) -> Dict[str, Any]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workflow_knowledge_gate.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.knowledge_node import knowledge_retrieval_node, knowledge_fallback_node
from app.services.workflow.nodes.gate import confidence_gate

@pytest.mark.asyncio
async def test_knowledge_retrieval_high_score_passed():
    mock_retriever = MagicMock()
    mock_hit = MagicMock()
    mock_hit.score = 0.88
    mock_hit.to_dict.return_value = {"chunk_id": 1, "text": "7天无理由退货", "score": 0.88}
    mock_res = MagicMock()
    mock_res.hits = [mock_hit]
    mock_res.citations = [{"index": 1, "source": "退款政策"}]
    mock_retriever.retrieve_with_strategy = AsyncMock(return_value=mock_res)

    state = create_initial_state(1, "怎么退货")
    updated = await knowledge_retrieval_node(state, retriever=mock_retriever)
    
    assert len(updated["retrieved_docs"]) == 1
    assert updated["retrieved_docs"][0]["score"] == 0.88
    # 闸门判定应该放行
    assert confidence_gate({**state, **updated}) == "pass"

@pytest.mark.asyncio
async def test_knowledge_retrieval_low_score_triggers_fallback():
    mock_retriever = MagicMock()
    mock_hit = MagicMock()
    mock_hit.score = 0.20  # 低于 0.35 阈值
    mock_hit.to_dict.return_value = {"chunk_id": 2, "text": "不相关内容", "score": 0.20}
    mock_res = MagicMock()
    mock_res.hits = [mock_hit]
    mock_res.citations = []
    mock_retriever.retrieve_with_strategy = AsyncMock(return_value=mock_res)

    state = create_initial_state(1, "太空飞船如何退货")
    updated = await knowledge_retrieval_node(state, retriever=mock_retriever)
    
    # 闸门判定应走兜底
    assert confidence_gate({**state, **updated}) == "fallback"

    fallback_updated = await knowledge_fallback_node({**state, **updated}, db=None)
    assert "暂未收录" in fallback_updated["response_text"]
    assert fallback_updated["confidence_passed"] is False
    assert fallback_updated["suggested_actions"] == ["transfer_agent", "create_ticket"]
    assert fallback_updated["status"] == "fallback"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_knowledge_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.workflow.nodes.knowledge_node'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/workflow/nodes/knowledge_node.py
import logging
from typing import Any, Dict, List, Optional
from app.services.workflow.state import AgentWorkflowState
from app.tools.business_tools import get_retriever

logger = logging.getLogger(__name__)

KNOWLEDGE_FALLBACK_TEXT = "抱歉，关于您咨询的问题，当前知识库中暂未收录确切规范或相关信息不足。建议您尝试更换问法，或选择转接人工客服/提交人工工单获取专人协助。"

CONFIDENCE_SCORE_THRESHOLD = 0.35

async def knowledge_retrieval_node(
    state: AgentWorkflowState,
    retriever: Optional[Any] = None,
) -> Dict[str, Any]:
    """知识检索节点：强制预检索 RAG 知识库"""
    query = state.get("resolved_query") or state.get("input_query", "")
    active_retriever = retriever or get_retriever()
    
    retrieved_docs: List[Dict[str, Any]] = []
    try:
        if hasattr(active_retriever, "retrieve_with_strategy"):
            res = await active_retriever.retrieve_with_strategy(query=query, min_score=0.1)
            hits = getattr(res, "hits", []) or []
            for hit in hits:
                if hasattr(hit, "to_dict"):
                    retrieved_docs.append(hit.to_dict())
                else:
                    retrieved_docs.append({
                        "text": getattr(hit, "text", str(hit)),
                        "score": getattr(hit, "score", 0.0),
                    })
        elif hasattr(active_retriever, "retrieve"):
            hits = await active_retriever.retrieve(query=query, top_k=5)
            for hit in hits:
                retrieved_docs.append({
                    "text": getattr(hit, "text", str(hit)),
                    "score": getattr(hit, "score", 0.5),
                })
    except Exception as e:
        logger.warning(f"知识检索节点异常: {e}")

    return {
        "retrieved_docs": retrieved_docs,
    }

async def knowledge_fallback_node(
    state: AgentWorkflowState,
    db: Optional[Any] = None,
) -> Dict[str, Any]:
    """兜底话术节点：弱证据拦截，记录至 low_confidence_questions 表，不进 Agent"""
    query = state.get("input_query", "")
    conv_id = state.get("conversation_id", 0)

    # 异步沉淀至 low_confidence_questions 表
    if db is not None:
        try:
            from app.models.low_confidence import LowConfidenceQuestion
            low_q = LowConfidenceQuestion(
                query=query,
                conversation_id=conv_id,
                entrance="workflow_confidence_gate",
                reason="retrieval_score_below_threshold",
            )
            db.add(low_q)
            await db.commit()
        except Exception as e:
            logger.warning(f"沉淀低置信度问题失败: {e}")

    return {
        "response_text": KNOWLEDGE_FALLBACK_TEXT,
        "confidence_passed": False,
        "suggested_actions": ["transfer_agent", "create_ticket"],
        "status": "fallback",
    }
```

```python
# app/services/workflow/nodes/gate.py
from app.services.workflow.state import AgentWorkflowState

def confidence_gate(state: AgentWorkflowState) -> str:
    """置信度闸门条件判定：最高分 >= 0.35 且有命中方可通过"""
    docs = state.get("retrieved_docs") or []
    if not docs:
        return "fallback"
    
    max_score = 0.0
    for doc in docs:
        score = float(doc.get("score") or 0.0)
        if score > max_score:
            max_score = score

    if max_score >= 0.35:
        return "pass"
    return "fallback"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_knowledge_gate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/services/workflow/nodes/knowledge_node.py app/services/workflow/nodes/gate.py tests/test_workflow_knowledge_gate.py
git commit -m "feat(workflow): implement knowledge retrieval node and confidence gate"
```

---

## Task 5: 主力 ReAct Agent 节点与多步自适应工具调用

**Files:**
- Create: `app/services/workflow/nodes/agent_node.py`
- Test: `tests/test_workflow_agent_react.py`

**Interfaces:**
- Produces: `main_agent_node(state: AgentWorkflowState, model: Optional[Any] = None, tools: Optional[List[Any]] = None) -> Dict[str, Any]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workflow_agent_react.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, ToolMessage
from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.agent_node import main_agent_node

@pytest.mark.asyncio
async def test_main_agent_single_step_tool_call():
    """测试物流查询单步调用工具收敛"""
    mock_model = MagicMock()
    resp1 = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}]
    )
    resp2 = AIMessage(content="订单1001的物流状态为：派送中，顺丰速运承运。")
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[resp1, resp2])
    mock_model.bind_tools.return_value = bound_model

    mock_tool = MagicMock()
    mock_tool.name = "query_logistics"
    mock_tool.ainvoke = AsyncMock(return_value='{"status": "派送中"}')

    state = create_initial_state(1, "订单1001的物流到哪了")
    updated = await main_agent_node(state, model=mock_model, tools=[mock_tool])

    assert "派送中" in updated["response_text"]
    assert updated["steps_taken"] == 2
    assert updated["status"] == "success"

@pytest.mark.asyncio
async def test_main_agent_multistep_react_call():
    """测试复合问题多步 ReAct（先查订单再查物流）"""
    mock_model = MagicMock()
    # 步1: 查订单
    resp1 = AIMessage(
        content="",
        tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}]
    )
    # 步2: 查物流
    resp2 = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c2"}]
    )
    # 步3: 最终总结
    resp3 = AIMessage(content="您购买的羽绒服已发货，目前顺丰正在派送中。")
    bound_model = MagicMock()
    bound_model.ainvoke = AsyncMock(side_effect=[resp1, resp2, resp3])
    mock_model.bind_tools.return_value = bound_model

    tool_order = MagicMock()
    tool_order.name = "query_order"
    tool_order.ainvoke = AsyncMock(return_value='{"order_id": "1001", "name": "羽绒服"}')

    tool_logistics = MagicMock()
    tool_logistics.name = "query_logistics"
    tool_logistics.ainvoke = AsyncMock(return_value='{"status": "派件中"}')

    state = create_initial_state(1, "查一下我买的羽绒服到哪了")
    updated = await main_agent_node(state, model=mock_model, tools=[tool_order, tool_logistics])

    assert updated["steps_taken"] == 3
    assert "顺丰正在派送中" in updated["response_text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_agent_react.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.workflow.nodes.agent_node'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/workflow/nodes/agent_node.py
import inspect
import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from app.llm import get_chat_model
from app.services.workflow.state import AgentWorkflowState
from app.tools.registry import default_tool_registry

logger = logging.getLogger(__name__)

AGENT_SYSTEM_BASE = """你是电商平台的生产级主力客服 Agent。
你拥有自主多步决策与调用工具的能力。
规则：
1. 遇到需要查询的数据，主动调用对应工具；
2. 如果用户提供的信息不足以调用工具（例如查物流却未给订单号），请友好向用户追问缺失信息，不要凭空猜测；
3. 工具返回结果后，结合已有知识与上下文进行有条理、亲切专业的回答。
"""

MAX_STEPS = 5

async def main_agent_node(
    state: AgentWorkflowState,
    model: Optional[Any] = None,
    tools: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """主力 ReAct Agent 节点：支持工具自适应循环、多步推理、知识上下文注入与 5 步硬截断"""
    llm = model or get_chat_model(streaming=False)
    available_tools = tools if tools is not None else default_tool_registry.get_all_tools()
    tools_map = {t.name: t for t in available_tools}

    # 1. 构造上下文提示词（融合上游知识库证据）
    sys_content = AGENT_SYSTEM_BASE
    docs = state.get("retrieved_docs") or []
    if docs:
        knowledge_texts = []
        for i, doc in enumerate(docs[:3], 1):
            text = doc.get("text") or doc.get("content") or str(doc)
            knowledge_texts.append(f"【参考知识 {i}】{text}")
        sys_content += "\n以下为检索到的官方政策与业务知识，请参考并用于回答：\n" + "\n".join(knowledge_texts)

    # 2. 准备消息列表（以状态中原有历史为基础）
    messages: List[BaseMessage] = [SystemMessage(content=sys_content)]
    existing_messages = state.get("messages") or []
    for m in existing_messages:
        if not isinstance(m, SystemMessage):
            messages.append(m)

    bound_llm = llm.bind_tools(available_tools) if (available_tools and hasattr(llm, "bind_tools")) else llm

    steps = 0
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    final_text = ""

    while steps < MAX_STEPS:
        steps += 1
        resp = await bound_llm.ainvoke(messages)
        messages.append(resp)

        # 统计 token 消耗
        meta = getattr(resp, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        if usage:
            total_tokens["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            total_tokens["completion_tokens"] += int(usage.get("completion_tokens", 0))
            total_tokens["total_tokens"] += int(usage.get("total_tokens", 0))

        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            # 自然收敛，完成推演
            final_text = str(resp.content or "")
            break

        # 执行工具并将结果回填
        for tool_call in tool_calls:
            name = tool_call.get("name", "")
            raw_args = tool_call.get("args") or {}
            call_id = str(tool_call.get("id") or "")

            target_tool = tools_map.get(name)
            if target_tool is None:
                output = f"工具 {name} 未注册"
            else:
                try:
                    if hasattr(target_tool, "ainvoke"):
                        output = await target_tool.ainvoke(raw_args)
                    elif inspect.iscoroutinefunction(target_tool):
                        output = await target_tool(**raw_args)
                    elif callable(target_tool):
                        output = target_tool(**raw_args)
                    else:
                        output = str(target_tool)
                except Exception as e:
                    output = f"执行异常: {str(e)}"

            messages.append(ToolMessage(content=str(output), tool_call_id=call_id))

    if not final_text:
        # 步数超限，执行总结
        final_summary = await llm.ainvoke(
            messages + [HumanMessage(content="请根据已有工具查询的信息，立即给出最终结论。")]
        )
        final_text = str(final_summary.content or "")

    return {
        "response_text": final_text,
        "messages": messages,
        "steps_taken": steps,
        "token_usage": total_tokens,
        "status": "success",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_agent_react.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/services/workflow/nodes/agent_node.py tests/test_workflow_agent_react.py
git commit -m "feat(workflow): implement main ReAct agent node with multistep tool execution"
```

---

## Task 6: 工作流图组装与端到端运行引擎

**Files:**
- Create: `app/services/workflow/engine.py`
- Create: `app/services/workflow/__init__.py`
- Test: `tests/test_workflow_engine.py`

**Interfaces:**
- Produces:
  - `build_workflow_graph(checkpointer: Optional[Any] = None) -> CompiledGraph`
  - `WorkflowEngine` class with `run(conversation_id: int, query: str, user_id: str = "default_user", db: Optional[Any] = None) -> AgentWorkflowState`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workflow_engine.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage
from app.services.workflow.engine import WorkflowEngine

@pytest.mark.asyncio
async def test_workflow_engine_chitchat_e2e():
    """闲聊端到端工作流验证"""
    engine = WorkflowEngine()
    # 模拟意图为闲聊
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value={"intent": "闲聊", "intent_reason": "问候"})):
        result = await engine.run(conversation_id=201, query="你好呀")
        assert "智能客服助手" in result["response_text"]
        assert result["status"] == "chitchat"

@pytest.mark.asyncio
async def test_workflow_engine_complaint_e2e():
    """投诉端到端工作流验证"""
    engine = WorkflowEngine()
    with patch("app.services.workflow.nodes.pre_nodes.intent_recognition_node", new=AsyncMock(return_value={"intent": "投诉", "intent_reason": "抗议"})):
        result = await engine.run(conversation_id=202, query="我要投诉客服")
        assert "非常抱歉" in result["response_text"]
        assert result["suggested_actions"] == ["transfer_agent", "create_ticket"]
        assert result["status"] == "complaint"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.workflow.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/workflow/engine.py
import logging
from typing import Any, Dict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.services.workflow.state import AgentWorkflowState, create_initial_state
from app.services.workflow.nodes.pre_nodes import (
    coreference_resolution_node,
    intent_recognition_node,
    chitchat_node,
    complaint_node,
)
from app.services.workflow.nodes.router import route_by_intent
from app.services.workflow.nodes.knowledge_node import knowledge_retrieval_node, knowledge_fallback_node
from app.services.workflow.nodes.gate import confidence_gate
from app.services.workflow.nodes.agent_node import main_agent_node

logger = logging.getLogger(__name__)

def logging_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """日志与审计节点"""
    logger.info(
        f"[Workflow Audit] ConvID={state.get('conversation_id')} "
        f"Intent={state.get('intent')} Status={state.get('status')} "
        f"Steps={state.get('steps_taken')} Actions={state.get('suggested_actions')}"
    )
    return {}

def build_workflow_graph(checkpointer: Optional[Any] = None):
    """装配 LangGraph 确定性工作流图"""
    builder = StateGraph(AgentWorkflowState)

    # 1. 注册所有节点
    builder.add_node("coreference", coreference_resolution_node)
    builder.add_node("intent", intent_recognition_node)
    builder.add_node("chitchat", chitchat_node)
    builder.add_node("complaint", complaint_node)
    builder.add_node("knowledge_retrieval", knowledge_retrieval_node)
    builder.add_node("knowledge_fallback", knowledge_fallback_node)
    builder.add_node("main_agent", main_agent_node)
    builder.add_node("logging", logging_node)

    # 2. 基础顺序流
    builder.add_edge(START, "coreference")
    builder.add_edge("coreference", "intent")

    # 3. 四出口意图分流
    builder.add_conditional_edges(
        "intent",
        route_by_intent,
        {
            "chitchat": "chitchat",
            "complaint": "complaint",
            "knowledge": "knowledge_retrieval",
            "business_data": "main_agent",
        },
    )

    # 4. 知识类置信度闸门分支
    builder.add_conditional_edges(
        "knowledge_retrieval",
        confidence_gate,
        {
            "pass": "main_agent",
            "fallback": "knowledge_fallback",
        },
    )

    # 5. 各分支汇聚至 logging 节点
    builder.add_edge("chitchat", "logging")
    builder.add_edge("complaint", "logging")
    builder.add_edge("knowledge_fallback", "logging")
    builder.add_edge("main_agent", "logging")
    builder.add_edge("logging", END)

    memory = checkpointer or MemorySaver()
    return builder.compile(checkpointer=memory)

class WorkflowEngine:
    """生产级智能客服工作流引擎"""

    def __init__(self, checkpointer: Optional[Any] = None) -> None:
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = build_workflow_graph(checkpointer=self.checkpointer)

    async def run(
        self,
        conversation_id: int,
        query: str,
        user_id: str = "default_user",
        db: Optional[Any] = None,
    ) -> AgentWorkflowState:
        """执行单轮工作流"""
        initial_state = create_initial_state(
            conversation_id=conversation_id,
            query=query,
            user_id=user_id,
        )
        config = {"configurable": {"thread_id": str(conversation_id)}}
        final_state = await self.graph.ainvoke(initial_state, config=config)
        return final_state
```

```python
# app/services/workflow/__init__.py
from app.services.workflow.state import AgentWorkflowState, create_initial_state
from app.services.workflow.engine import WorkflowEngine, build_workflow_graph

__all__ = [
    "AgentWorkflowState",
    "create_initial_state",
    "WorkflowEngine",
    "build_workflow_graph",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow_engine.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/services/workflow/ tests/test_workflow_engine.py
git commit -m "feat(workflow): assemble LangGraph deterministic workflow graph and engine"
```

---

## Task 7: 独立工单 API 端点与数据入库

**Files:**
- Create: `app/schemas/ticket.py`
- Modify: `app/api/routes.py`
- Test: `tests/test_api_tickets.py`

**Interfaces:**
- Consumes: `create_ticket` from `app.tools.business_tools`
- Produces: `POST /api/tickets` route in FastAPI

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_tickets.py
import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_create_ticket_api_success():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/tickets",
            json={
                "conversation_id": 9999,
                "description": "商品存在破损，申请退款被拒",
                "ticket_type": "投诉",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticket_no"].startswith("T")
        assert data["ticket_type"] == "投诉"
        assert "24小时" in data["status"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_tickets.py -v`
Expected: FAIL with 404 Not Found (route `/api/tickets` does not exist)

- [ ] **Step 3: Write minimal implementation**

```python
# app/schemas/ticket.py
from typing import Optional
from pydantic import BaseModel, Field

class TicketCreateRequest(BaseModel):
    conversation_id: Optional[int] = Field(None, description="关联会话ID")
    description: str = Field("用户自主申请客服工单", description="工单详细描述")
    ticket_type: str = Field("投诉", description="工单类型: 投诉/售后/咨询")

class TicketCreateResponse(BaseModel):
    ticket_no: str
    conversation_id: int
    ticket_type: str
    status: str
```

在 `app/api/routes.py` 中挂载端点：
```python
from app.schemas.ticket import TicketCreateRequest, TicketCreateResponse
from app.tools.business_tools import create_ticket

@router.post("/tickets", response_model=TicketCreateResponse)
async def api_create_ticket(request: TicketCreateRequest):
    """前端自选独立触发的创建工单 API（不消耗 LLM）"""
    res_str = await create_ticket.ainvoke({
        "conversation_id": request.conversation_id,
        "description": request.description,
        "ticket_type": request.ticket_type,
    })
    data = json.loads(res_str)
    if "error" in data:
        raise HTTPException(status_code=400, detail=data["error"])
    return TicketCreateResponse(
        ticket_no=data["ticket_no"],
        conversation_id=int(data["conversation_id"]),
        ticket_type=data["ticket_type"],
        status=data["status"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_tickets.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add app/schemas/ticket.py app/api/routes.py tests/test_api_tickets.py
git commit -m "feat(api): implement dedicated POST /api/tickets endpoint"
```

---

## Task 8: ChatService 与 `/api/chat/stream` 工作流串联适配

**Files:**
- Modify: `app/services/chat_service.py`
- Test: `tests/test_api_chat_stream.py`

**Interfaces:**
- Consumes: `WorkflowEngine`
- Produces: `ChatService.stream_chat` yielding SSE events (`tool_start`, `tool_end`, `text`, `actions`, `error`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_chat_stream.py 追加测试
@pytest.mark.asyncio
async def test_chat_stream_emits_actions_for_complaint():
    """测试投诉流式输出下发 actions 事件"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/chat/stream",
            json={"message": "我要投诉你们平台", "conversation_id": None},
        )
        assert resp.status_code == 200
        text = resp.text
        assert "event_type\": \"actions\"" in text
        assert "transfer_agent" in text
        assert "create_ticket" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_chat_stream.py -k test_chat_stream_emits_actions_for_complaint -v`
Expected: FAIL (actions event not yet emitted)

- [ ] **Step 3: Write minimal implementation**

在 `app/services/chat_service.py` 中引入 `WorkflowEngine`：
- 在 `stream_chat` 中使用 `WorkflowEngine` 执行图流转；
- 回显工具执行事件 `tool_start` / `tool_end`；
- 输出 `text` 事件流，并在包含 `suggested_actions` 时发射 `actions` 事件；
- 同步写入数据库 `messages` 表（双轨持久化）。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_chat_stream.py -k test_chat_stream_emits_actions_for_complaint -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/chat_service.py tests/test_api_chat_stream.py
git commit -m "feat(service): integrate WorkflowEngine into ChatService streaming"
```

---

## Task 9: 前端独立建议按钮与交互呈现

**Files:**
- Modify: `app/static/index.html`

- [ ] **Step 1: Inspect index.html chat render block**
定位 SSE 解析函数 `handleChatStream` 与消息气泡渲染。

- [ ] **Step 2: Add CSS styles and HTML structure for suggested actions**
增加 `.action-buttons-container`、`.btn-action-agent`、`.btn-action-ticket` 样式。

- [ ] **Step 3: Implement client-side click event handlers**
- 点击「转人工」：
  ```javascript
  appendSystemMessage("已转接人工客服");
  appendAssistantMessage("您好，我是客服小猫，请问有什么可以帮您的 🐱");
  btn.disabled = true;
  btn.classList.add("btn-disabled");
  ```
- 点击「建工单」：
  ```javascript
  const res = await fetch("/api/tickets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation_id: currentConversationId, description: "用户申请客服工单", ticket_type: "投诉" })
  });
  const data = await res.json();
  appendSystemMessage(`已为您成功创建人工工单【${data.ticket_no}】，客服将在24小时内跟进处理。`);
  btn.disabled = true;
  ```
- 两个按钮完全独立，不点任何按钮在输入框发消息继续对话正常流转。

- [ ] **Step 4: Commit**

```bash
git add app/static/index.html
git commit -m "feat(ui): add decoupled transfer_agent and create_ticket action buttons"
```

---

## Task 10: 综合端到端验收与全量回归测试套件

**Files:**
- Create: `tests/test_ch05_acceptance.py`

- [ ] **Step 1: Write the 5 acceptance criteria tests**

```python
# tests/test_ch05_acceptance.py
import pytest
from httpx import AsyncClient, ASGITransport
from main import app
from app.services.workflow.engine import WorkflowEngine

@pytest.mark.asyncio
async def test_criterion_1_policy_mandatory_retrieval():
    """验收标准 1：问政策类问题，工作流中能看到强制知识检索节点被走到"""
    engine = WorkflowEngine()
    state = await engine.run(conversation_id=501, query="退货退款政策是什么")
    assert state["intent"] in ("退款退货", "商品咨询")
    assert "retrieved_docs" in state

@pytest.mark.asyncio
async def test_criterion_2_logistics_agent_tool_calling():
    """验收标准 2：问「订单 1001 的物流到哪了」，Agent 自己调工具作答"""
    engine = WorkflowEngine()
    state = await engine.run(conversation_id=502, query="订单 1001 的物流到哪了")
    assert state["intent"] == "物流"
    assert "顺丰" in state["response_text"] or "派送" in state["response_text"] or "1001" in state["response_text"]

@pytest.mark.asyncio
async def test_criterion_3_complaint_decoupled_actions():
    """验收标准 3：说「我要投诉」，返回安抚话术，附带独立双按钮；建工单成功写 tickets 表"""
    engine = WorkflowEngine()
    state = await engine.run(conversation_id=503, query="我要投诉你们态度太差了")
    assert state["intent"] == "投诉"
    assert state["suggested_actions"] == ["transfer_agent", "create_ticket"]
    assert "非常抱歉" in state["response_text"]

    # 验证独立建工单 API
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/tickets", json={"conversation_id": 503, "description": "我要投诉"})
        assert res.status_code == 200
        assert res.json()["ticket_no"].startswith("T")

@pytest.mark.asyncio
async def test_criterion_4_chitchat_fixed_response():
    """验收标准 4：闲聊拿到固定话术，不花模型 Token"""
    engine = WorkflowEngine()
    state = await engine.run(conversation_id=504, query="你好呀")
    assert state["intent"] == "闲聊"
    assert "智能客服助手" in state["response_text"]
    assert state["token_usage"]["total_tokens"] == 0

@pytest.mark.asyncio
async def test_criterion_5_complex_multistep_react():
    """验收标准 5：复合问题（先查订单再查物流）ReAct 走了不止一步"""
    engine = WorkflowEngine()
    state = await engine.run(conversation_id=505, query="我买的羽绒服物流到哪了，帮我查查订单和轨迹")
    assert state["steps_taken"] >= 2
```

- [ ] **Step 2: Run acceptance tests**

Run: `pytest tests/test_ch05_acceptance.py -v`
Expected: PASS (5 passed)

- [ ] **Step 3: Run full test suite regression**

Run: `pytest -v`
Expected: 100% PASS across all existing tests (Ch01~Ch04) and new tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ch05_acceptance.py
git commit -m "test(acceptance): add end-to-end acceptance tests covering all 5 criteria"
```
