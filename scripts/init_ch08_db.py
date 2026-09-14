import asyncio
import inspect
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


async def _execute_stmt(conn, stmt_str: str):
    res = conn.execute(text(stmt_str))
    if inspect.isawaitable(res) or hasattr(res, "__await__"):
        return await res
    return res


async def init_ch08_db(
    engine_override=None,
    ddl_path: Optional[Path | str] = None,
) -> List[str]:
    """
    执行 sql/ch08-ddl.sql 脚本，幂等创建 tool_audit_logs 工具调用审计表。
    同时兼容 MySQL（带 utf8mb4 / SET NAMES）与 SQLite 内存测试模式（跳过 SET NAMES、适配类型与 JSON / AUTOINCREMENT）。
    """
    if ddl_path is None:
        ddl_path = PROJECT_ROOT / "sql" / "ch08-ddl.sql"

    statements = parse_ddl_statements(ddl_path)

    if engine_override is None:
        ensure_mysql_ready()
        target_engine = engine
        should_dispose = True
    else:
        target_engine = engine_override
        should_dispose = False

    # 检测底层数据库方言是否为 sqlite
    dialect_name = getattr(getattr(target_engine, "dialect", None), "name", None)
    is_sqlite = (isinstance(dialect_name, str) and dialect_name == "sqlite") or (
        "sqlite" in str(getattr(target_engine, "url", ""))
    )

    executed: List[str] = []
    begin_ctx = target_engine.begin()

    async def _run_statements(conn):
        if is_sqlite:
            # SQLite 特化幂等迁移逻辑
            for stmt in statements:
                upper_stmt = stmt.strip().upper()
                if upper_stmt.startswith("SET "):
                    # SQLite 不支持 SET NAMES 等变量配置语句，直接记录并跳过执行
                    executed.append(stmt)
                    continue

                if upper_stmt.startswith("CREATE TABLE"):
                    # SQLite 建表兼容处理：使用 INTEGER PRIMARY KEY AUTOINCREMENT 及兼容字段类型
                    await _execute_stmt(conn, """
                    CREATE TABLE IF NOT EXISTS tool_audit_logs (
                      id              INTEGER PRIMARY KEY AUTOINCREMENT,
                      conversation_id INTEGER NULL,
                      tool_call_id    TEXT NULL,
                      tool_name       TEXT NOT NULL,
                      tool_source     TEXT NOT NULL,
                      mcp_server      TEXT NULL,
                      arguments       JSON NULL,
                      result_summary  TEXT NULL,
                      status          TEXT NOT NULL,
                      error_message   TEXT NULL,
                      retry_count     INTEGER NOT NULL DEFAULT 0,
                      duration_ms     INTEGER NULL,
                      created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """)
                    await _execute_stmt(
                        conn,
                        "CREATE INDEX IF NOT EXISTS idx_conversation_id ON tool_audit_logs (conversation_id)"
                    )
                    await _execute_stmt(
                        conn,
                        "CREATE INDEX IF NOT EXISTS idx_tool_name ON tool_audit_logs (tool_name)"
                    )
                    await _execute_stmt(
                        conn,
                        "CREATE INDEX IF NOT EXISTS idx_status ON tool_audit_logs (status)"
                    )
                    executed.append(stmt)
                    continue

                # 其他语句直接执行
                await _execute_stmt(conn, stmt)
                executed.append(stmt)
        else:
            # MySQL 或 MockEngine 执行逻辑
            for stmt in statements:
                try:
                    await _execute_stmt(conn, stmt)
                    executed.append(stmt)
                except Exception as e:
                    err_msg = str(e).lower()
                    # 容错：表已存在报错视为幂等成功
                    if "already exists" in err_msg or "1050" in err_msg:
                        executed.append(stmt)
                    else:
                        raise

    if hasattr(begin_ctx, "__aenter__"):
        async with begin_ctx as conn:
            await _run_statements(conn)
    else:
        with begin_ctx as conn:
            await _run_statements(conn)

    if should_dispose:
        dispose_method = getattr(target_engine, "dispose", None)
        if callable(dispose_method):
            res = dispose_method()
            if inspect.isawaitable(res) or hasattr(res, "__await__"):
                await res

    print(f"Successfully initialized ch08 database tables ({len(executed)} statements executed).")
    return executed


if __name__ == "__main__":
    asyncio.run(init_ch08_db())
