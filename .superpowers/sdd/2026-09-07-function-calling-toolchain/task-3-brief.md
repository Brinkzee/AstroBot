# Task 3 Brief: LangChain 五大业务工具实现

## Files
- Create: `app/tools/__init__.py`
- Create: `app/tools/business_tools.py`
- Test: `tests/test_business_tools.py`

## Interfaces
- Consumes:
  - `app.models.FAQ`, `app.models.Ticket`
  - `app.db.session.AsyncSessionLocal`
- Produces:
  - Five `@tool` decorated functions in `app/tools/business_tools.py`:
    1. `query_order(order_id: str) -> str`
    2. `query_product(product_id_or_name: str) -> str`
    3. `query_logistics(order_id: str) -> str`
    4. `query_faq(keyword: str) -> str` (async function wrapped with `@tool`)
    5. `create_ticket(conversation_id: int, description: str, ticket_type: str = "售后") -> str` (async function wrapped with `@tool`)

## Exact Requirements
1. **Global Constraints**:
   - Use LangChain `@tool` decorator from `langchain_core.tools` (or `langchain.tools`)
   - Add detailed docstrings describing the tool purpose and arguments so the LLM understands when and how to call them.
   - For `query_order`, `query_product`, `query_logistics`: return simulated realistic mock data as structured JSON string or formatted text. Do not query external APIs, do not create new tables.
     - `query_order`: input `order_id`. Returns order status, payment amount, item details, order time.
     - `query_product`: input `product_id_or_name`. Returns product name, price, stock status, specs.
     - `query_logistics`: input `order_id`. Returns carrier (e.g. 顺丰速运), tracking number, status (运输中/派送中/已签收), and recent tracking checkpoints (e.g. 正在派件中，快递员已出发).
   - For `query_faq`:
     - Input `keyword: str`.
     - Uses `AsyncSessionLocal` to query `FAQ` table:
       `select(FAQ).where(or_(FAQ.question.like(f"%{keyword}%"), FAQ.answer.like(f"%{keyword}%"))).limit(3)`
     - If matched, returns formatted Q&A list.
     - If not matched, returns `"未找到与【{keyword}】相关的常见问题解答。"`
   - For `create_ticket`:
     - Input `conversation_id: int`, `description: str`, `ticket_type: str = "售后"`. Validate `ticket_type` in `["售后", "投诉", "咨询"]`, fallback to `"售后"`.
     - Generates ticket number: e.g. `f"T{datetime.now().strftime('%Y%m%d%H%M%S')}{random.randint(100, 999)}"`.
     - Inserts `Ticket` record into DB using `AsyncSessionLocal`, commits.
     - Returns JSON/string containing `ticket_no`, `conversation_id`, and status `"工单已创建，人工客服将在24小时内跟进处理"`.

2. **Step 1 (TDD RED)**:
   - Create `tests/test_business_tools.py`:
     - Test `query_order.invoke({"order_id": "1001"})`
     - Test `query_product.invoke({"product_id_or_name": "羽绒服"})`
     - Test `query_logistics.invoke({"order_id": "1001"})`
     - Test `await query_faq.ainvoke({"keyword": "退货"})` (with mock session or in-memory DB)
     - Test `await create_ticket.ainvoke({"conversation_id": 1, "description": "测试", "ticket_type": "售后"})`
   - Run `pytest tests/test_business_tools.py -v` and record RED failure.

3. **Step 3 (Implement)**:
   - Create `app/tools/__init__.py`.
   - Create `app/tools/business_tools.py` implementing all 5 tools with proper schemas and docstrings.

4. **Step 4 (TDD GREEN)**:
   - Run `pytest tests/test_business_tools.py -v` and full suite `pytest -v`.

5. **Step 5 (Commit)**:
   - Commit: `git add app/tools/ tests/test_business_tools.py`
   - Commit message: `feat(tools): implement five business tools using langchain @tool`
