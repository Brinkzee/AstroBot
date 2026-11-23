from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

class Base(DeclarativeBase):
    pass

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
)

# SQLAlchemy 2.0.34 + aiomysql: AsyncAdapt_aiomysql_connection.ping(self, reconnect)
# requires an explicit boolean argument; enable _send_false_to_ping so pool_pre_ping
# passes ping(False) instead of crashing with TypeError.
if getattr(engine.sync_engine.dialect, "driver", None) == "aiomysql":
    engine.sync_engine.dialect._send_false_to_ping = True

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            try:
                await session.close()
            except Exception:
                pass
