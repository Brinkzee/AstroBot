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


def split_sql_statements(sql_text: str) -> List[str]:
    """
    引号感知的 SQL 分句解析器。
    在字符级扫描时正确识别单引号 '...'、双引号 "..."、转义字符 \\ 及成对转义 ''/""，
    仅对不在引号内的半角分号 ';' 进行语句切分。
    """
    statements: List[str] = []
    current: List[str] = []
    in_single_quote = False
    in_double_quote = False
    i = 0
    n = len(sql_text)

    while i < n:
        char = sql_text[i]

        # 处理转义字符 \
        if char == "\\" and (in_single_quote or in_double_quote):
            current.append(char)
            if i + 1 < n:
                current.append(sql_text[i + 1])
                i += 2
                continue
            else:
                i += 1
                break

        if char == "'":
            if not in_double_quote:
                if in_single_quote:
                    # 判断 SQL 标准的连续两个单引号 '' 转义
                    if i + 1 < n and sql_text[i + 1] == "'":
                        current.append("''")
                        i += 2
                        continue
                    else:
                        in_single_quote = False
                else:
                    in_single_quote = True
            current.append(char)
        elif char == '"':
            if not in_single_quote:
                if in_double_quote:
                    # 判断 SQL 标准的连续两个双引号 "" 转义
                    if i + 1 < n and sql_text[i + 1] == '"':
                        current.append('""')
                        i += 2
                        continue
                    else:
                        in_double_quote = False
                else:
                    in_double_quote = True
            current.append(char)
        elif char == ";" and not in_single_quote and not in_double_quote:
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(char)

        i += 1

    remaining = "".join(current).strip()
    if remaining:
        statements.append(remaining)

    return statements


def parse_ddl_statements(ddl_path: Path | str) -> List[str]:
    """
    读取 DDL 文件，移除注释并使用引号感知的切分器拆分为独立的 SQL 语句。
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

    raw_statements = split_sql_statements(cleaned_text)

    statements: List[str] = []
    for raw_stmt in raw_statements:
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


async def init_ch04_db(
    engine_override=None,
    ddl_path: Optional[Path | str] = None,
) -> List[str]:
    """
    执行 sql/ch04_ddl.sql 脚本，幂等创建 low_confidence_questions 与 faith_cases 表。
    """
    if ddl_path is None:
        ddl_path = PROJECT_ROOT / "sql" / "ch04_ddl.sql"

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

    print(f"Successfully initialized ch04 database tables ({len(executed)} statements executed).")
    return executed


if __name__ == "__main__":
    asyncio.run(init_ch04_db())
