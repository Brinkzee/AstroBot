import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import engine, Base, AsyncSessionLocal
from app.models import Conversation, Message, FAQ, Ticket
from scripts.seed_data import seed_all_data
from scripts.wsl_helper import ensure_mysql_ready

async def reinit():
    ensure_mysql_ready()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    print("All 4 tables (conversations, messages, faq, tickets) created successfully via SQLAlchemy ORM.")

    async with AsyncSessionLocal() as session:
        inserted = await seed_all_data(session)
        print(f"Seed data injected: {inserted} FAQ item(s).")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(reinit())
