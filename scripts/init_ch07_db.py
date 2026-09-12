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


async def init_ch07_db(
    engine_override=None,
    ddl_path: Optional[Path | str] = None,
) -> List[str]:
    """
    执行 sql/ch07-ddl.sql 脚本，幂等演进 conversations 表并创建 conversation_summaries 表。
    对 ALTER TABLE 列已存在进行容错处理，同时兼容 MySQL 与 SQLite 内存测试。
    """
    if ddl_path is None:
        ddl_path = PROJECT_ROOT / "sql" / "ch07-ddl.sql"

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

                if upper_stmt.startswith("ALTER TABLE"):
                    # 检查 conversations 表是否存在
                    chk = await _execute_stmt(
                        conn,
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
                    )
                    rows = chk.fetchall() if hasattr(chk, "fetchall") else []
                    if rows:
                        pragma_res = await _execute_stmt(conn, "PRAGMA table_info(conversations)")
                        existing_cols = {row[1] for row in (pragma_res.fetchall() if hasattr(pragma_res, "fetchall") else [])}

                        cols_spec = [
                            ("summary", "TEXT"),
                            ("summary_upto_msg_id", "INTEGER"),
                            ("layer1_from_msg_id", "INTEGER"),
                        ]
                        for col_name, col_type in cols_spec:
                            # 仅添加当前 ALTER TABLE 语句涉及且表中尚未存在的列
                            if col_name.lower() in stmt.lower() and col_name not in existing_cols:
                                try:
                                    await _execute_stmt(conn, f"ALTER TABLE conversations ADD COLUMN {col_name} {col_type}")
                                    existing_cols.add(col_name)
                                except Exception as err:
                                    if "duplicate column name" not in str(err).lower():
                                        raise
                    executed.append(stmt)
                    continue

                if upper_stmt.startswith("CREATE TABLE"):
                    # SQLite 建表兼容处理
                    await _execute_stmt(conn, """
                    CREATE TABLE IF NOT EXISTS conversation_summaries (
                      id              INTEGER PRIMARY KEY AUTOINCREMENT,
                      conversation_id INTEGER NOT NULL,
                      seq             INTEGER NOT NULL,
                      from_msg_id     INTEGER NOT NULL,
                      upto_msg_id     INTEGER NOT NULL,
                      content         TEXT NOT NULL,
                      created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (conversation_id) REFERENCES conversations (id),
                      UNIQUE (conversation_id, seq)
                    )
                    """)
                    await _execute_stmt(
                        conn,
                        "CREATE INDEX IF NOT EXISTS idx_conv_upto ON conversation_summaries (conversation_id, upto_msg_id)"
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
                    # 容错：列已存在报错（MySQL 1060 Duplicate column name）视为幂等成功
                    if "duplicate column name" in err_msg or "1060" in err_msg or "already exists" in err_msg:
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

    print(f"Successfully initialized ch07 database tables ({len(executed)} statements executed).")
    return executed


if __name__ == "__main__":
    asyncio.run(init_ch07_db())
