# 客服三层会话上下文管理与动态预算设计规范 (Context Management Spec)

## 1. 背景与目标

在既有实现中，智能客服的对话历史依赖简单粗暴的消息条数截短或纯固定预算裁剪，无法在长达数十轮的长会话中持续保持关键记忆（如最初提到的订单号、核心诉求、已解决/未解决卡点）。
本章（Ch07）正式升级会话上下文管理体系，核心目标包括：
1. **三层上下文结构（层 1 原文、层 2 半压缩、层 3 分段摘要）**：纯逻辑锚点划分，降级只移动边界 ID，物理消息不搬运；
2. **后台异步分段摘要引擎**：层 2 用量超预算触发异步提炼，一段一行追加至 `conversation_summaries`，旧梗概只作背景不重写，杜绝有损压缩累积；当前轮用户回复零阻塞；
3. **固定顺序上下文拼装与前缀缓存保护**：System Prompt 固定置顶（人设、红线、工具 Schema 绝对稳定），层 2 与层 1 滑窗紧随其后，早期梗概与知识证据合并单条挂在当前用户提问之后，彻底杜绝 Prefix Cache 击穿；
4. **动态 Token 预算倒推与自检**：从模型窗口反推可用滑窗预算，固定开销与 ReAct 峰值统一扣除，中文严格按 1 字符 = 1 Token 校准，启动自检不足单轮时报警；
5. **LangGraph State 全量与精简双轨流转**：State `messages` 维护完整还原流水与快照，调模型前现算精简视图递入；
6. **全链路可观测日志 (`log/app.log`)**：`model_ctx` 原样打出主力 Agent 组装报文，`history_ctx` 每轮必打覆盖闲聊兜底分支，后台摘要触发/起止/耗时全留痕；
7. **前端多会话侧栏与无缝回载**：列出用户历史会话，展示首问预览与 `[已摘要]` 标签，点击平滑回载原文并续聊，配备只读 REST API 与静默降级保障。

---

## 2. 数据模型与三层边界锚点机制

### 2.1 数据库 Schema 演进 (`sql/ch07-ddl.sql`)

```sql
SET NAMES utf8mb4;

-- 1. conversations 表扩展
ALTER TABLE conversations
  ADD COLUMN summary             TEXT            NULL COMMENT '最近几段梗概拼成的投影,拼装时跟证据一起挂在用户那句之后' AFTER status,
  ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NULL COMMENT '摘要已覆盖到哪条消息,滑窗从其后接原文' AFTER summary,
  ADD COLUMN layer1_from_msg_id  BIGINT UNSIGNED NULL COMMENT '层1(原文)起点;此 id 之后原样,之前渲染成半压形态' AFTER summary_upto_msg_id;

-- 2. conversation_summaries 分段摘要明细表
CREATE TABLE IF NOT EXISTS conversation_summaries (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NOT NULL,
  seq             INT             NOT NULL COMMENT '第几段,从 1 开始',
  from_msg_id     BIGINT UNSIGNED NOT NULL COMMENT '这段覆盖的消息区间,闭区间',
  upto_msg_id     BIGINT UNSIGNED NOT NULL,
  content         TEXT            NOT NULL,
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv_upto (conversation_id, upto_msg_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='分段摘要,一段一行只追加';
```

### 2.2 三层区间切分与语义形态定义
对于会话内的任意一条消息记录 `m`（按 `id` 递增排序）：
- **层 3（摘要层 / 归档区）**：`m.id <= summary_upto_msg_id`
  - 物理消息仍在 `messages` 表，但不再直接以对话轮次形式传入 LLM；
  - 由 `conversations.summary` 投影提供浓缩的事实背景。
- **层 2（中间层 / 半压缩区）**：`summary_upto_msg_id < m.id <= layer1_from_msg_id`
  - **用户原话 (`role=user`)**：一字不改、完全保留真实表达；
  - **客服回复 (`role=assistant`)**：仅保留前 60 个字符，尾部附加 `...` 截断，保留回答开头主旨；
  - **工具结果 (`role=tool` / 大块 JSON)**：替换为单行语义标识（例如 `[工具查询: {tool_name}, 结果已由客服消化]`），抹平瞬时工具长输出。
- **层 1（最近原文层 / 零损区）**：`m.id > layer1_from_msg_id`
  - 所有消息（User、Assistant、Tool）原样保留，一个字都不压缩。

### 2.3 边界锚点推进状态机
1. **初始态**：会话建立初期，`summary_upto_msg_id = 0`，`layer1_from_msg_id = 0`，全量历史天然归属于层 1；
2. **层 1 降级到层 2（同步推进）**：
   - 每次调模型前，若层 1 总估算 Tokens 超过 `layer1_budget`（默认 3954）；
   - 从层 1 头部逐轮（User 提问及后续回复/工具）向后推进 `layer1_from_msg_id`，将早期轮次移入层 2；
   - 仅持久化更新 `conversations.layer1_from_msg_id`，不移动任何消息数据。
3. **层 2 升舱进层 3（异步提炼）**：
   - 当层 2 积累的估算 Tokens 超过 `layer2_budget`（默认 1695）；
   - 触发后台异步摘要任务，提炼后生成新的一段摘要，推进 `summary_upto_msg_id` 至本次处理上限。

---

## 3. 动态 Token 预算倒推与中文折算口径

### 3.1 预算倒推参数集与基准配置

| 配置键名 | 类型 | 默认值 | 演示测试值 | 含义 |
| :--- | :--- | :--- | :--- | :--- |
| `MODEL_CONTEXT_WINDOW` | int | 128000 | 18000 | 模型物理上下文窗口 |
| `MAX_OUTPUT_TOKENS` | int | 2000 | 2000 | 模型单轮回复最大预留输出 |
| `MAX_USER_INPUT_TOKENS` | int | 2000 | 2000 | 单轮用户提问最大预留输入 |
| `MAX_AGENT_STEPS` | int | 3 | 3 | 主力 Agent 最大迭代步数 |
| `TOOL_RESULT_MAX_TOKENS` | int | 1200 | 1200 | 单步工具执行结果上限预留 |
| `RERANK_TOP_K` | int | 5 | 5 | 知识库检索重排召回篇数 |
| `SYSTEM_PROMPT_TOKENS` | int | 1500 | 1500 | 固定人设、红线与工具 Schema 开销 |
| `DOC_TOKENS_PER_CHUNK` | int | 500 | 500 | 单篇知识块平均 Token 估算 |
| `SUMMARY_MAX_TOKENS` | int | 250 | 250 | 注入历史梗概最大预留 Token |
| `SAFETY_MARGIN_TOKENS` | int | 500 | 500 | 安全缓冲开销 |
| `TARGET_HISTORY_TURNS` | int | 20 | 20 | 期望保留的历史轮数 |
| `STEADY_TURN_TOKENS` | int | 500 | 500 | 单轮稳态对话 Token 占用 |

### 3.2 倒推计算公式
1. **固定开销项合计**：
   $$\text{Fixed} = \text{SYSTEM\_PROMPT\_TOKENS} (1500) + (\text{RERANK\_TOP\_K} \times \text{DOC\_TOKENS\_PER\_CHUNK}) (2500) + \text{SUMMARY\_MAX\_TOKENS} (250) + \text{SAFETY\_MARGIN\_TOKENS} (500) + \text{MAX\_OUTPUT\_TOKENS} (2000) + \text{MAX\_USER\_INPUT\_TOKENS} (2000) = 8750$$
2. **ReAct 瞬时峰值**：
   $$\text{Peak} = \text{MAX\_AGENT\_STEPS} \times \text{TOOL\_RESULT\_MAX\_TOKENS} = 3 \times 1200 = 3600$$
3. **窗口可用预算 (Window Available)**：
   $$\text{WindowAvailable} = \text{MODEL\_CONTEXT\_WINDOW} - \text{Fixed} - \text{Peak}$$
   - 演示配置（18000 窗口）：$18000 - 8750 - 3600 = 5650$
   - 生产默认（128k 窗口）：$128000 - 12350 = 115650$
4. **滑窗总历史预算 (Window Budget)**：
   $$\text{TargetBudget} = \text{TARGET\_HISTORY\_TURNS} \times \text{STEADY\_TURN\_TOKENS} = 20 \times 500 = 10000$$
   $$\text{WindowBudget} = \min(\text{TargetBudget}, \text{WindowAvailable})$$
   - 演示配置：$\min(10000, 5650) = 5650$
   - 生产默认：$\min(10000, 115650) = 10000$（20 轮不触发压缩）
5. **层 1 与层 2 切分**：
   - 层 2 预算：$\text{Layer2Budget} = \lfloor \text{WindowBudget} \times 0.3 \rfloor = 1695$
   - 层 1 预算：$\text{Layer1Budget} = 3954$（严格对应演示配置基准）
6. **启动自检机制**：
   - 若 `WindowAvailable < STEADY_TURN_TOKENS`，系统启动立即发出 CRITICAL 报警并提示配置不可用。

### 3.3 中文 Token 折算口径
- 中文汉字与中文全角标点符号：严格遵循 **1 字符 = 1 Token**；
- 英文单词、西文字符、空白：按约 **3.5 字符 ≈ 1 Token**；
- 工具结果 / JSON：转字符串后按统一口径统计。

---

## 4. 上下文固定拼装流水线与前缀缓存保护

### 4.1 消息列表组装拓扑

```
+-----------------------------------------------------------------------+
| 1. SystemMessage (固定前缀)                                            |
|    - 客服核心人设、行为红线                                           |
|    - 业务工具 JSON Schema (由 bind_tools 注入)                         |
|    * 绝对不含任何动态用户数据或摘要，确保 Prefix Cache 100% 命中           |
+-----------------------------------------------------------------------+
| 2. Layer 2 历史对话流 (半压缩形态)                                     |
|    - HumanMessage: 用户原话                                           |
|    - AIMessage: 客服回复前 60 字 + "..."                              |
|    - ToolMessage: 单行语义标识 "[工具查询: xxx, 结果已处理]"           |
+-----------------------------------------------------------------------+
| 3. Layer 1 历史对话流 (零损原文)                                       |
|    - HumanMessage: 用户原话                                           |
|    - AIMessage: 完整客服回复                                          |
|    - ToolMessage: 完整工具返回内容                                    |
+-----------------------------------------------------------------------+
| 4. HumanMessage (用户当前提问)                                         |
|    - 用户最新输入 query                                               |
+-----------------------------------------------------------------------+
| 5. HumanMessage (动态上下文证据包，挂在当前提问之后)                    |
|    【前序会话背景摘要】                                               |
|    {conversations.summary 投影}                                       |
|    【参考政策与业务知识证据】                                         |
|    {RAG 检索证据}                                                     |
+-----------------------------------------------------------------------+
```

### 4.2 State 快照与精简版本分离
- **LangGraph State**：保留 `messages: Annotated[List[BaseMessage], add_messages]`，每一次用户发问、Agent 推理、工具调用均追加至 State 中，并通过 checkpointer 持久化，作为完整历史凭据；
- **模型入参构建**：`ContextManager.build_model_messages(...)` 现算切分，产出符合 4.1 顺序的精简版消息列表，送入 LLM 执行推理。

---

## 5. 后台异步摘要引擎设计

### 5.1 触发与执行机制
1. **触发检测**：在层 1 降级判定之后，计算当前处于层 2 消息范围（`summary_upto_msg_id < id <= layer1_from_msg_id`）的总 Token 数；
2. **异步非阻塞**：若 `layer2_tokens > layer2_budget`，通过 `asyncio.create_task` 启动后台任务 `run_async_summarization(conv_id, from_id, to_id)`；
   - 内部创建全新的 `AsyncSessionLocal`，独立管理数据库连接；
   - 主流程直接将生成的流式回复输出给用户，不等待摘要完成；
3. **并发防重锁**：在内存中维护 `active_summary_tasks: Set[int]`，若指定会话已有摘要任务在执行，则触发 `[summary skip]` 跳过。

### 5.2 摘要提炼 Prompt 规约
```text
你是一名严谨的电商客服会话事实摘要专家。
你的任务是将以下一段客服与用户的中间轮次对话，提炼为精炼的事实梗概。

【提炼规则】
1. 仅提炼关键业务事实与诉求：
   - 用户咨询或提及的具体商品型号、品类；
   - 用户明确报出过的订单号、手机号、快递单号；
   - 用户的核心业务诉求（退款、催发货、查保修等）；
   - 尚未解决的卡点或已达成的明确结论。
2. 绝对真实性 (Hard Gate)：对话中未提及的信息一个字都不许编造、推测或扩展！
3. 剔除一切客套寒暄：严禁出现“您好”、“谢谢”、“很高兴为您服务”等无意义礼貌辞令。
4. 篇幅约束：字数严格控制在 50 ~ 200 字之间，采用紧凑的事实陈述句。

【已有前情背景（仅作理解参考，严禁重复输出或合并改写）】
{existing_summary}

【待压缩对话片段】
{layer2_dialogue_text}
```

### 5.3 数据落盘与投影更新
1. 插入 `conversation_summaries` 表：
   - `conversation_id = conv_id`
   - `seq = current_max_seq + 1`
   - `from_msg_id = from_id`
   - `upto_msg_id = to_id`
   - `content = new_summary_content`
2. 原子更新 `conversations` 表：
   - `summary_upto_msg_id = to_id`
   - `summary = "\n\n".join([s.content for s in all_summaries])`

---

## 6. 上下文可观测日志 (`log/app.log`)

系统配置独立的日志 Handler 将日志输出至 `log/app.log`，必须支持两类固定格式标签：
1. **`[model_ctx]`**（主力 Agent 节点调用前）：
   ```text
   [model_ctx] conv_id={id} summary_len={len} window_msgs={count} estimated_tokens={tokens}
   --- SUMMARY ---
   {summary}
   --- SLIDING WINDOW ---
   [1] (user) ...
   [2] (assistant) ...
   ```
2. **`[history_ctx]`**（指代消解与意图识别共用，每轮必打）：
   ```text
   [history_ctx] conv_id={id} summary_line="{summary_first_line}" window_msgs={count}
   {formatted_history}
   ```
3. **后台摘要生命周期日志**：
   - 触发：`[summary trigger] conv_id={id} 层2 约 {N} token > 预算 {M}, range=({from_id}..{to_id})`
   - 开始：`[summary start] conv_id={id} 第{seq}段 开始执行, msg_range=({from_id}..{to_id})`
   - 完成：`[summary done] conv_id={id} 第{seq}段完成, 耗时={cost}s, 覆盖至 msg_id={upto_id}`
   - 跳过：`[summary skip] conv_id={id} 原因={reason}`
   - 失败：`[summary fail] conv_id={id} 异常={error}`

---

## 7. 前端多会话侧栏与 REST 接口

### 7.1 后端 API 规约
1. **`GET /api/conversations?user_id=default_user`**
   - 返回指定用户的所有会话列表，按 `updated_at` 倒序排列；
   - 包含：`id`, `title`（首问截短预览）, `has_summary`（是否已有摘要标记）, `message_count`, `status`, `updated_at`。
2. **`GET /api/conversations/{id}/messages`**
   - 返回该会话全量原始消息列表，时间升序；
   - 包含：`id`, `role`, `content`, `tool_calls`, `tool_call_id`, `created_at`。

### 7.2 前端侧栏与回载交互 (`app/static/index.html`)
- **左侧会话侧栏**（宽 240px，响应式自适应）：
  - 顶部配备 **「+ 新对话」** 按钮；
  - 会话卡片展示首问内容预览、时间，若 `has_summary: true` 则显示紫色 `[已摘要]` 标签；
  - 选中项 Active 高亮。
- **切换与续聊**：
  - 点击历史卡片：静默加载 `/api/conversations/{id}/messages` 并回显至右侧主视窗，更新当前 `currentConversationId`，无缝继续聊天；
  - 点击「+ 新对话」：清空视窗，`currentConversationId = null`，展示欢迎语；
- **静默降级**：
  - 若侧栏接口请求异常，控制台静默记日志，界面显示“暂无历史会话”，绝对不打断用户的核心聊天流。

---

## 8. 验收与测试策略

1. **动态预算倒推与中文估算单元测试**：
   - 验证默认配置与 18000 演示配置下的倒推值（5650, 3954, 1695）精确吻合；
   - 验证 1 字符 = 1 Token 中文折算。
2. **三层边界移动与降级状态机测试**：
   - 模拟消息增加，验证 `layer1_from_msg_id` 步进推动，层 2 客服截短 60 字及工具标识替换。
3. **后台异步摘要单测与防重锁测试**：
   - 验证 `conversation_summaries` 段落追加与 `summary_upto_msg_id` 推进，验证旧梗概不重写。
4. **上下文拼装与前缀缓存保护测试**：
   - 验证 `SystemMessage` 纯净性，验证梗概与证据附加在用户提问之后。
5. **可观测日志测试**：
   - 验证 `log/app.log` 中 `model_ctx`、`history_ctx` 与摘要任务留痕的正确写入。
6. **多会话 REST 接口测试**：
   - 验证 `/api/conversations` 和 `/api/conversations/{id}/messages` 的响应格式。
7. **综合端到端验收用例 (`tests/test_ch07_acceptance.py`)**：
   - **用例 A (默认 128k 窗口)**：连续聊 20 轮，验证无降级无摘要（装得下就不压）；
   - **用例 B (演示 18000 窗口)**：连续聊 20 轮，验证层 1 降级、层 2 触发异步摘要、生成多段梗概，最后问“最开始那个订单后来怎么说”能准确认出最初订单号与诉求；
   - **用例 C (多会话切换)**：侧栏建新会话、切旧会话回载并继续对话验证。
