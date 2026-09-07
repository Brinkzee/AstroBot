import asyncio
from sqlalchemy import text
from app.db.session import engine

async def check():
    async with engine.connect() as conn:
        res = await conn.execute(text("SHOW TABLES;"))
        tables = [r[0] for r in res.fetchall()]
        print("Existing tables in astro_bot:", tables)
        
        faqs = await conn.execute(text("SELECT id, question, category FROM faq;"))
        print("FAQ rows:")
        for r in faqs.fetchall():
            print(f"  [{r[0]}] {r[1]} ({r[2]})")
            
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(check())
