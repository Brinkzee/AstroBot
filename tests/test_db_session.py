import pytest
from app.db.session import engine, AsyncSessionLocal, get_db, Base

@pytest.mark.asyncio
async def test_async_session_maker():
    assert AsyncSessionLocal is not None
    session = AsyncSessionLocal()
    assert session is not None
    await session.close()

@pytest.mark.asyncio
async def test_get_db_generator():
    gen = get_db()
    session = await anext(gen)
    assert session is not None
    await gen.aclose()

def test_base_declarative():
    assert Base is not None
