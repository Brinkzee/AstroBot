import asyncio
import sys
from pathlib import Path
from typing import List, Dict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.faq import FAQ
from app.db.session import AsyncSessionLocal

SEED_FAQS: List[Dict[str, str]] = [
    {
        "question": "退货政策说明",
        "answer": "商城支持7天无理由退货，商品需保持吊牌完好、包装齐全，并在签收后7天内发起退货申请。",
        "category": "售后",
    },
    {
        "question": "运费标准与包邮政策",
        "answer": "全场订单实付满99元免运费，不足99元收取10元基础运费。",
        "category": "物流",
    },
    {
        "question": "发票开具说明",
        "answer": "确认收货后可在订单详情页申请开具电子发票，1-3个工作日内发送至绑定邮箱。",
        "category": "财务",
    },
]


async def seed_all_data(session: AsyncSession) -> int:
    """
    Idempotently inject predefined FAQ seed data into the database.
    Returns the number of newly inserted FAQ records.
    """
    inserted_count = 0
    for item in SEED_FAQS:
        stmt = select(FAQ).where(FAQ.question == item["question"])
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is None:
            faq = FAQ(
                question=item["question"],
                answer=item["answer"],
                category=item["category"],
            )
            session.add(faq)
            inserted_count += 1

    if inserted_count > 0:
        await session.commit()

    return inserted_count


async def main():
    async with AsyncSessionLocal() as session:
        inserted = await seed_all_data(session)
        print(f"Seed finished: {inserted} FAQ item(s) inserted.")


if __name__ == "__main__":
    asyncio.run(main())
