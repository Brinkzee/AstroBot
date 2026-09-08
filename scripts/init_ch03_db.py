import asyncio
import re
import sys
from pathlib import Path
from typing import List, Optional
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import engine
from scripts.wsl_helper import ensure_mysql_ready


def parse_ddl_statements(ddl_path: Path | str) -> List[str]:
    """
    读取 DDL 文件，移除注释并拆分为独立的 SQL 语句。
    对于 CREATE TABLE 语句，自动注入 IF NOT EXISTS 确保幂等安全执行。
    """
    path = Path(ddl_path)
    content = path.read_text(encoding="utf-8")

    # 去除单行注释 (-- ...)
    cleaned_lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        cleaned_lines.append(line)
    cleaned_text = "\n".join(cleaned_lines)

    statements: List[str] = []
    for raw_stmt in cleaned_text.split(";"):
        stmt = raw_stmt.strip()
        if not stmt:
            continue
        # 确保 CREATE TABLE 为幂等的 CREATE TABLE IF NOT EXISTS
        stmt = re.sub(
            r"CREATE\s+TABLE\s+(?!IF\s+NOT\s+EXISTS)",
            "CREATE TABLE IF NOT EXISTS ",
            stmt,
            flags=re.IGNORECASE,
        )
        statements.append(stmt)

    return statements


async def init_ch03_db(
    engine_override=None,
    ddl_path: Optional[Path | str] = None,
) -> List[str]:
    """
    执行 sql/ch03-ddl.sql 脚本，幂等创建 knowledge_chunks 与 qa_extraction_staging 表。
    """
    if ddl_path is None:
        ddl_path = PROJECT_ROOT / "sql" / "ch03-ddl.sql"

    statements = parse_ddl_statements(ddl_path)

    if engine_override is None:
        ensure_mysql_ready()
        target_engine = engine
        should_dispose = True
    else:
        target_engine = engine_override
        should_dispose = False

    executed: List[str] = []
    async with target_engine.begin() as conn:
        for stmt in statements:
            await conn.execute(text(stmt))
            executed.append(stmt)

    if should_dispose:
        await target_engine.dispose()

    print(f"Successfully initialized ch03 database tables ({len(executed)} statements executed).")
    return executed


if __name__ == "__main__":
    asyncio.run(init_ch03_db())
