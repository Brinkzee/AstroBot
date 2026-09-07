# Task 1 Brief: 依赖更新与异步数据库基础设施

## Files
- Modify: `requirements.txt`
- Modify: `app/config.py`
- Create: `docker-compose.yml`
- Create: `scripts/start_mysql.ps1`
- Create: `app/db/__init__.py`
- Create: `app/db/session.py`
- Test: `tests/test_db_session.py`

## Interfaces
- Produces:
  - `app.config.settings.database_url: str`
  - `app.db.session.engine`: SQLAlchemy `AsyncEngine`
  - `app.db.session.AsyncSessionLocal`: `async_sessionmaker[AsyncSession]`
  - `app.db.session.get_db() -> AsyncGenerator[AsyncSession, None]`
  - `app.db.session.Base`: `DeclarativeBase`

## Exact Requirements
1. **Global Constraints**:
   - Python 3.12, Windows OS
   - SQLAlchemy 2.0 Async, aiomysql
   - Database URL default: `mysql+aiomysql://root:root123456@127.0.0.1:3306/astro_bot?charset=utf8mb4`
2. **Step 1 (TDD RED)**:
   - Create `tests/test_db_session.py`:
     ```python
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
     ```
3. **Step 2 (Verify Fail)**:
   - Run `pytest tests/test_db_session.py -v` and capture failure output.
4. **Step 3 (Implement)**:
   - Update `requirements.txt` with `aiomysql>=0.2.0`, `cryptography>=43.0.0`. Run `pip install aiomysql cryptography` if needed.
   - Update `app/config.py` to add `database_url: str = Field(default="mysql+aiomysql://root:root123456@127.0.0.1:3306/astro_bot?charset=utf8mb4", description="MySQL 异步连接串")`. Keep all existing fields in `app/config.py` intact!
   - Create `docker-compose.yml` with MySQL 8.0:
     image `mysql:8.0`, container_name `astro_bot_mysql`, environment: `MYSQL_ROOT_PASSWORD: root123456`, `MYSQL_DATABASE: astro_bot`, command `--default-authentication-plugin=mysql_native_password --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci`, ports `3306:3306`.
   - Create `scripts/start_mysql.ps1` that can run `wsl docker compose up -d` or `wsl docker run ...`.
   - Create `app/db/__init__.py`.
   - Create `app/db/session.py`:
     ```python
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
         pool_recycle=3600,
     )

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
                 await session.close()
     ```
5. **Step 4 (TDD GREEN)**:
   - Run `pytest tests/test_db_session.py -v` to ensure all tests pass.
   - Run existing suite `pytest -v` to ensure no regressions.
6. **Step 5 (Commit)**:
   - Commit changes: `git add requirements.txt app/config.py docker-compose.yml scripts/start_mysql.ps1 app/db/ tests/test_db_session.py`
   - Commit message: `feat(db): add async sqlalchemy session infrastructure and mysql docker config`
