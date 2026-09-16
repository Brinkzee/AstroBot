import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from scripts.wsl_helper import ensure_mysql_ready
from app.db.session import engine

async def run_sql():
    ensure_mysql_ready()
    with open("sql/ch09-ddl.sql", "r", encoding="utf-8") as f:
        sql = f.read()
    
    lines = sql.split("\n")
    cleaned_lines = [line for line in lines if not line.strip().startswith("--")]
    cleaned_sql = "\n".join(cleaned_lines)
    statements = [s.strip() for s in cleaned_sql.split(';') if s.strip()]
    async with engine.begin() as conn:
        for stmt in statements:
            try:
                await conn.exec_driver_sql(stmt)
            except Exception as e:
                print(f"Skipping statement due to error: {e}")

if __name__ == "__main__":
    asyncio.run(run_sql())
