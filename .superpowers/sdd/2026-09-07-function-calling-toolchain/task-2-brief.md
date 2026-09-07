# Task 2 Brief: SQLAlchemy ORM 数据模型与 FAQ 测试数据灌入

## Files
- Create: `app/models/__init__.py`
- Create: `app/models/conversation.py`
- Create: `app/models/message.py`
- Create: `app/models/faq.py`
- Create: `app/models/ticket.py`
- Create: `scripts/seed_data.py`
- Test: `tests/test_models.py`

## Interfaces
- Consumes:
  - `app.db.session.Base`
  - `app.db.session.AsyncSessionLocal`
- Produces:
  - `Conversation`: Table `conversations` (id: BIGINT UNSIGNED PK, user_id: VARCHAR(64), status: Enum('进行中', '已转人工', '已结束'), created_at, updated_at)
  - `Message`: Table `messages` (id: BIGINT UNSIGNED PK, conversation_id: BIGINT UNSIGNED FK, role: Enum('user', 'assistant', 'tool'), content: TEXT nullable, tool_calls: JSON nullable, tool_call_id: VARCHAR(64) nullable, created_at)
  - `FAQ`: Table `faq` (id: BIGINT UNSIGNED PK, question: VARCHAR(512), answer: TEXT, category: VARCHAR(64), created_at, updated_at)
  - `Ticket`: Table `tickets` (ticket_no: VARCHAR(32) PK, conversation_id: BIGINT UNSIGNED FK, description: TEXT, ticket_type: Enum('售后', '投诉', '咨询'), status: Enum('待处理', '已处理'), created_at)
  - `seed_all_data(session: AsyncSession)`: Injects seed data idempotently

## Exact Requirements
1. **DDL Alignment (`sql/ch02-ddl.sql`)**:
   - `conversations`: id (BigInteger/Integer, primary_key=True, autoincrement=True), user_id (String(64), nullable=False), status (Enum('进行中', '已转人工', '已结束', name='conv_status'), default='进行中'), created_at (DateTime, server_default=func.now()), updated_at (DateTime, server_default=func.now(), onupdate=func.now())
   - `messages`: id (BigInteger/Integer, primary_key=True, autoincrement=True), conversation_id (BigInteger/Integer, ForeignKey('conversations.id'), nullable=False), role (Enum('user', 'assistant', 'tool', name='message_role'), nullable=False), content (Text, nullable=True), tool_calls (JSON, nullable=True), tool_call_id (String(64), nullable=True), created_at (DateTime, server_default=func.now())
   - `faq`: id (BigInteger/Integer, primary_key=True, autoincrement=True), question (String(512), nullable=False), answer (Text, nullable=False), category (String(64), nullable=False), created_at (DateTime, server_default=func.now()), updated_at (DateTime, server_default=func.now(), onupdate=func.now())
   - `tickets`: ticket_no (String(32), primary_key=True), conversation_id (BigInteger/Integer, ForeignKey('conversations.id'), nullable=False), description (Text, nullable=False), ticket_type (Enum('售后', '投诉', '咨询', name='ticket_type'), nullable=False), status (Enum('待处理', '已处理', name='ticket_status'), default='待处理'), created_at (DateTime, server_default=func.now())

2. **Step 1 (TDD RED)**:
   - Create `tests/test_models.py` verifying model classes, columns, table names, and SQLite/in-memory or schema attribute assertions.
   - Run `pytest tests/test_models.py -v` and record RED failure.

3. **Step 3 (Implement)**:
   - Create `app/models/conversation.py`, `app/models/message.py`, `app/models/faq.py`, `app/models/ticket.py`, and `app/models/__init__.py` exporting all models.
   - Create `scripts/seed_data.py`:
     - Contains predefined FAQ data:
       - `{"question": "退货政策说明", "answer": "商城支持7天无理由退货，商品需保持吊牌完好、包装齐全，并在签收后7天内发起退货申请。", "category": "售后"}`
       - `{"question": "运费标准与包邮政策", "answer": "全场订单实付满99元免运费，不足99元收取10元基础运费。", "category": "物流"}`
       - `{"question": "发票开具说明", "answer": "确认收货后可在订单详情页申请开具电子发票，1-3个工作日内发送至绑定邮箱。", "category": "财务"}`
     - Function `async def seed_all_data(session: AsyncSession)` that queries existing FAQs and inserts if not already present.
     - `__main__` entry to run standalone: `python scripts/seed_data.py`.

4. **Step 4 (TDD GREEN)**:
   - Run `pytest tests/test_models.py -v` to ensure all tests pass.
   - Run `pytest -v` across whole suite.

5. **Step 5 (Commit)**:
   - Commit: `git add app/models/ scripts/seed_data.py tests/test_models.py`
   - Commit message: `feat(models): add orm entities and faq seed script aligned with ch02-ddl.sql`
