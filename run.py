#!/usr/bin/env python3
"""AstroBot 一键智能启动与环境自愈脚本。

执行流程：
1. 控制台编码配置与环境路径自适应；
2. 自动拉起与保活 WSL2 MySQL Docker 容器（探测 127.0.0.1:3306）；
3. 关系型存储（MySQL）表结构幂等自建与基础 FAQ 种子数据就绪；
4. 密集向量存储（Milvus-Lite）连接、未同步 pending 数据自愈补偿与空库自动构建；
5. 输出启动就绪仪表盘，启动 Uvicorn / FastAPI 主程序。

使用方法：
    python run.py                     # 默认开发模式（带热重载）
    python run.py --check-only        # 仅做环境与存储自检，不启动 Web 服务
    python run.py --port 8080         # 指定端口
    python run.py --no-reload         # 生产/稳定模式（禁用热重载）
    python run.py --clean-kb          # 重建知识库向量数据后启动
"""
import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# 确保 UTF-8 控制台输出与实时行缓冲刷新，避免 Windows cmd / PowerShell 乱码与日志堆积
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_environment() -> bool:
    """检查基础运行环境与配置。"""
    print("\n" + "=" * 65)
    print("📋 [第 1 步] 检查基础运行环境与依赖配置")
    print("=" * 65)

    # 1. 检查 Python 版本
    py_version = sys.version_info
    print(f"  • Python 版本: {py_version.major}.{py_version.minor}.{py_version.micro}", end="")
    if py_version < (3, 10):
        print("  ❌ (需要 Python 3.10+)")
        return False
    print("  ✅ [OK]")

    # 2. 检查 .env 配置文件
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        print("  • 环境变量配置: 发现 .env 配置文件  ✅ [OK]")
    else:
        print("  • 环境变量配置: 未发现 .env 文件，将使用系统环境变量或默认配置  ⚠️ [提示]")

    # 3. 检查关键依赖库导入
    core_packages = [
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("SQLAlchemy", "sqlalchemy"),
        ("PyMilvus", "pymilvus"),
    ]
    missing = []
    for name, mod in core_packages:
        try:
            __import__(mod)
            print(f"  • 依赖组件 {name:<12}: 已安装  ✅ [OK]")
        except ImportError:
            print(f"  • 依赖组件 {name:<12}: 缺失  ❌")
            missing.append(name)

    if missing:
        print(f"\n[错误] 缺少必要依赖: {', '.join(missing)}，请先执行 pip install -r requirements.txt")
        return False

    # 4. 检查会话上下文管理与 Token 预算
    try:
        from app.services.context.budget import check_budget_on_startup, calculate_context_budget
        from app.config import Settings, settings

        active_settings = Settings() if "MODEL_CONTEXT_WINDOW" in os.environ else settings
        if not check_budget_on_startup(active_settings):
            print("  ❌ 上下文预算自检未通过：模型窗口无法容纳单轮稳态交互")
            return False

        budget = calculate_context_budget(active_settings)
        print(
            f"  • 上下文管理窗口: {active_settings.model_context_window} Tokens "
            f"(总滑窗预算: {budget.window_budget} Tokens, 层1: {budget.layer1_budget}, 层2: {budget.layer2_budget})  ✅ [OK]"
        )
    except Exception as e:
        print(f"  ❌ 上下文预算自检异常: {e}")
        return False

    return True


async def check_storage_readiness(clean_kb: bool = False) -> bool:
    """检查并确保存储基础设施就绪（MySQL 容器、表结构、FAQ、Milvus 向量库）。"""
    print("\n" + "=" * 65)
    print("🐬 [第 2 步] 关系型存储与 Docker 容器就绪检查")
    print("=" * 65)

    # 1. 自动拉起与保活 WSL2 MySQL 容器
    from scripts.wsl_helper import ensure_mysql_ready
    ensure_mysql_ready(verbose=True)

    # 2. 数据库连接与 ORM 表结构检查
    from app.db.session import engine, Base, AsyncSessionLocal
    # 导入全部 ORM 模型以注册 metadata
    from app.models import Conversation, Message, FAQ, Ticket, KnowledgeChunk, QAExtractionStaging
    from scripts.seed_data import seed_all_data

    print("\n  • 正在验证 MySQL 数据库连接与表结构完整性...")
    try:
        async with engine.begin() as conn:
            # 幂等建表：已存在的表不会受到影响，缺失的表将自动创建
            await conn.run_sync(Base.metadata.create_all)
        print("    ↳ 数据库全部 6 张核心业务表已验证就绪 ✅")
    except Exception as e:
        print(f"    ❌ 数据库连接或建表失败: {e}")
        return False

    # 2.1 第七章三层会话上下文数据表与字段迁移
    from scripts.init_ch07_db import init_ch07_db
    try:
        await init_ch07_db()
        print("    ↳ 第七章三层会话上下文数据表与字段迁移已就绪 ✅")
    except Exception as e:
        print(f"    ❌ 第七章数据表与字段迁移失败: {e}")
        return False

    # 2.2 第八章工具调用审计数据表 (tool_audit_logs) 迁移
    from scripts.init_ch08_db import init_ch08_db
    try:
        await init_ch08_db()
        print("    ↳ 第八章工具调用审计数据表 (tool_audit_logs) 迁移已就绪 ✅")
    except Exception as e:
        print(f"    ❌ 第八章数据表与迁移失败: {e}")
        return False

    # 2.3 第九章飞轮待审队列与评估记录数据表迁移
    from scripts.init_ch09_db import init_ch09_db
    try:
        await init_ch09_db()
        print("    ↳ 第九章飞轮待审队列与评估记录数据表迁移已就绪 ✅")
    except Exception as e:
        print(f"    ❌ 第九章数据表与迁移失败: {e}")
        return False

    # 3. 基础 FAQ 种子数据填充
    async with AsyncSessionLocal() as session:
        try:
            inserted = await seed_all_data(session)
            if inserted > 0:
                print(f"    ↳ 自动注入初始基础 FAQ 种子数据: 新增 {inserted} 条记录 ✅")
            else:
                print("    ↳ 基础 FAQ 知识已存在，跳过初始种子填充 ✅")
        except Exception as e:
            print(f"    ⚠️ 检查 FAQ 种子数据时提示: {e}")

    # 4. 密集向量知识库（Milvus-Lite）与自愈补齐检查
    print("\n" + "=" * 65)
    print("🧠 [第 3 步] 密集向量存储 (Milvus-Lite) 与知识库自愈检查")
    print("=" * 65)

    from sqlalchemy import select, func
    from app.services.rag.milvus_client import MilvusKnowledgeStore
    from app.services.rag.dual_writer import KnowledgeDualWriter
    from scripts.build_knowledge_base import build_knowledge_base

    # 4.1 检查是否存在待同步的 pending 孤立块（断点续跑自愈机制）
    async with AsyncSessionLocal() as session:
        pending_res = await session.execute(
            select(func.count(KnowledgeChunk.id)).where(KnowledgeChunk.vectorize_status == "pending")
        )
        pending_count = pending_res.scalar() or 0

        if pending_count > 0:
            print(f"  • 发现 {pending_count} 条 pending 孤立知识块，正在启动断点续跑自动补偿自愈...")
            writer = KnowledgeDualWriter()
            try:
                repaired = await writer.repair_pending_chunks(session)
                print(f"    ↳ 自愈完成: 成功补齐并索引了 {repaired} 个知识块 ✅")
            finally:
                writer.close()
        else:
            print("  • 未发现 pending 遗留块，MySQL 与 Milvus 状态健康 ✅")

    # 4.2 检查知识库有效数据量
    async with AsyncSessionLocal() as session:
        done_res = await session.execute(
            select(func.count(KnowledgeChunk.id)).where(KnowledgeChunk.vectorize_status == "done")
        )
        done_count = done_res.scalar() or 0

    store = MilvusKnowledgeStore()
    try:
        milvus_count = store.count("knowledge")
    finally:
        store.close(release_server=True)

    print(f"  • MySQL 权威源有效知识块数: {done_count}")
    print(f"  • Milvus 向量库已索引记录数: {milvus_count}")

    # 4.3 若知识库为空或显式指定了 --clean-kb，触发知识库全量构建
    if done_count == 0 or milvus_count == 0 or clean_kb:
        reason = "检测到知识库为空" if (done_count == 0 or milvus_count == 0) else "用户指定了 --clean-kb"
        print(f"\n  ▶ [{reason}] 正在自动执行全量知识库构建与双写落库...")
        try:
            stats = await build_knowledge_base(kb_dir="data/kb", clean=True)
            print(f"    ↳ 自动建库完成: 成功写入 {stats['total_chunks']} 个知识块 ✅")
        except Exception as e:
            print(f"    ❌ 自动构建知识库异常: {e}")
            return False
    else:
        print("  • 知识库向量索引就绪，无须重新建库 ✅")

    # 5. MCP 独立协议服务与动态工具注册中心就绪检查
    print("\n" + "=" * 65)
    print("🔌 [第 4 步] MCP 独立协议服务与动态工具注册中心就绪检查")
    print("=" * 65)

    from urllib.parse import urlparse
    from app.config import settings
    from scripts.wsl_helper import is_port_open
    from app.tools.registry import default_tool_registry

    # 5.1 MCP Server 连通性探测 (8001 物流 / 8002 售后)
    logistics_url = urlparse(settings.MCP_LOGISTICS_SERVER_URL)
    aftersale_url = urlparse(settings.MCP_AFTERSALE_SERVER_URL)
    logistics_host = logistics_url.hostname or "127.0.0.1"
    logistics_port = logistics_url.port or 8001
    aftersale_host = aftersale_url.hostname or "127.0.0.1"
    aftersale_port = aftersale_url.port or 8002

    logistics_online = is_port_open(logistics_host, logistics_port, timeout=0.2)
    aftersale_online = is_port_open(aftersale_host, aftersale_port, timeout=0.2)

    if logistics_online:
        print(f"  • 物流 MCP 服务 ({logistics_host}:{logistics_port}): 在线  ✅ [OK]")
    else:
        print(f"  • 物流 MCP 服务 ({logistics_host}:{logistics_port}): 未启动 (已降级为内置/模拟处理)  ⚠️ [提示]")

    if aftersale_online:
        print(f"  • 售后 MCP 服务 ({aftersale_host}:{aftersale_port}): 在线  ✅ [OK]")
    else:
        print(f"  • 售后 MCP 服务 ({aftersale_host}:{aftersale_port}): 未启动 (已降级为内置/模拟处理)  ⚠️ [提示]")

    # 5.2 工具注册中心动态加载与拓扑展示
    client = getattr(default_tool_registry, "mcp_client", None)
    orig_connections = getattr(client, "connections", None) if client else None

    try:
        if not logistics_online and not aftersale_online:
            # 双 Server 均离线时快速降级，避免 10s 无谓超时等待
            default_tool_registry.mcp_client = None
            all_tools = await default_tool_registry.get_all_tools()
        elif isinstance(orig_connections, dict):
            # 仅保留已在线的服务连接进行工具动态发现
            filtered = {}
            if logistics_online and "logistics" in orig_connections:
                filtered["logistics"] = orig_connections["logistics"]
            if aftersale_online and "aftersale" in orig_connections:
                filtered["aftersale"] = orig_connections["aftersale"]
            client.connections = filtered
            all_tools = await default_tool_registry.get_all_tools()
        else:
            all_tools = await default_tool_registry.get_all_tools()
    finally:
        if not logistics_online and not aftersale_online:
            default_tool_registry.mcp_client = client
        elif client and orig_connections is not None:
            client.connections = orig_connections

    builtin_tools = [t for t in all_tools if getattr(t, "tool_source", "builtin") == "builtin"]
    mcp_tools = [t for t in all_tools if getattr(t, "tool_source", "") == "mcp"]

    print(f"  • 工具注册中心: 动态就绪 {len(all_tools)} 个工具 (内置: {len(builtin_tools)}, MCP: {len(mcp_tools)})  ✅ [OK]")
    if builtin_tools:
        print(f"    ↳ 内置工具: {', '.join(t.name for t in builtin_tools)}")
    if mcp_tools:
        mcp_names = [f"{t.name} ({getattr(t, 'mcp_server', 'mcp')})" for t in mcp_tools]
        print(f"    ↳ MCP 动态工具: {', '.join(mcp_names)}")
    else:
        print("    ↳ MCP 动态工具: 无 (服务离线或未注册动态工具，系统降级运行)")

    # 释放连接池，避免与后续 Uvicorn 子进程发生连接冲突
    await engine.dispose()
    return True


def check_port_availability(host: str, port: int) -> bool:
    """检查主服务端口是否空闲可用。若已被占用则返回 False 并输出友好诊断提示。"""
    from scripts.wsl_helper import is_port_open

    if is_port_open(host, port, timeout=0.5):
        print("\n" + "!" * 65)
        print(f"❌ [端口冲突] 检测到端口 {host}:{port} 已被其它程序占用！")
        print("   💡 建议解决方案:")
        print("      1. 关闭正在占用该端口的旧进程/服务；")
        print("      2. 或启动时指定其它可用端口: python run.py --port 8080")
        print("!" * 65 + "\n")
        return False
    return True


def open_browser_async(url: str, delay: float = 0.8):
    """异步延迟在系统默认浏览器中打开指定 URL，避免阻塞主服务就绪循环。"""
    import threading
    import webbrowser

    def _worker():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def print_banner(host: str, port: int):
    """打印漂亮的启动仪表盘面板。"""
    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print("\n" + "=" * 65)
    print("       🚀  AstroBot 智能电商客服系统已完全就绪  🚀")
    print("=" * 65)
    print(f"  💬 客户服务对话界面:  http://{display_host}:{port}/chat")
    print(f"  📚 知识库管理工作台:  http://{display_host}:{port}/kb")
    print(f"  📖 交互式 API 文档:   http://{display_host}:{port}/docs")
    print(f"  🩺 系统健康检查接口:  http://{display_host}:{port}/health")
    print("  " + "-" * 61)
    print(f"  🐬 数据库服务 (MySQL): 127.0.0.1:3306 (Docker / 自动保活)")
    print(f"  🧠 向量存储 (Milvus):  ./data/milvus/astro_bot.db (BGE-M3 Dense)")
    try:
        from app.services.context.budget import calculate_context_budget
        from app.config import Settings, settings
        active_settings = Settings() if "MODEL_CONTEXT_WINDOW" in os.environ else settings
        budget = calculate_context_budget(active_settings)
        print(f"  🪟 上下文管理 (Tokens): 窗口 {active_settings.model_context_window} | 滑窗 {budget.window_budget} (L1:{budget.layer1_budget}/L2:{budget.layer2_budget})")
    except Exception:
        pass
    try:
        from app.tools.registry import default_tool_registry
        builtin_count = len(default_tool_registry.get_all_builtin_tools())
        print(f"  🔌 协议拓展 (MCP Server): 物流(:8001) | 售后(:8002) (Streamable HTTP)")
        print(f"  🛠️ 工具注册中心 (Tools): 内置 {builtin_count} 项 | 审计表 (tool_audit_logs 就绪)")
    except Exception:
        pass
    print("=" * 65)
    print("  [提示] 按 Ctrl+C 即可安全停止服务。\n")


def main():
    parser = argparse.ArgumentParser(description="AstroBot 智能启动与环境自检服务")
    parser.add_argument("--host", default="127.0.0.1", help="绑定监听地址 (默认: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="绑定端口号 (默认: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="禁用代码热重载 (默认开启便于调试)")
    parser.add_argument("--check-only", action="store_true", help="仅执行环境与存储状态自检，不启动 Web 服务")
    parser.add_argument("--clean-kb", action="store_true", help="启动前强制重建知识库向量数据")
    parser.add_argument("--no-mcp", action="store_true", help="禁用自动拉起外部 MCP 独立服务子进程")
    parser.add_argument("--no-browser", action="store_true", help="禁用主服务就绪后自动在浏览器打开聊天界面")
    args = parser.parse_args()

    # 1. 环境基础检查
    if not check_environment():
        print("\n❌ 环境检查未通过，启动终止。")
        sys.exit(1)

    # 2. 联动拉起 MCP 独立子进程（若未显式禁用）
    mcp_manager = None
    if not args.no_mcp:
        print("\n" + "=" * 65)
        print("🚀 [联动服务] 检查并拉起外部 MCP 独立进程服务")
        print("=" * 65)
        try:
            from scripts.mcp_process_manager import MCPServerManager
            mcp_manager = MCPServerManager()
            mcp_manager.start_servers(verbose=True)
        except Exception as e:
            print(f"  ⚠️ MCP 子进程管理器启动异常: {e}，将降级运行")

    # 3. 存储与容器就绪检查
    storage_ready = asyncio.run(check_storage_readiness(clean_kb=args.clean_kb))
    if not storage_ready:
        print("\n❌ 存储就绪检查未通过，启动终止。")
        if mcp_manager:
            mcp_manager.stop_all()
        sys.exit(1)

    if args.check_only:
        print("\n🎉 [检查完成] 所有环境、容器与存储状态均已就绪！(--check-only 模式，退出启动)")
        if mcp_manager:
            mcp_manager.stop_all()
        sys.exit(0)

    # 4. 主服务端口冲突检查
    if not check_port_availability(args.host, args.port):
        if mcp_manager:
            mcp_manager.stop_all()
        sys.exit(1)

    # 5. 打印仪表盘
    print_banner(args.host, args.port)

    # 6. 自动异步打开系统浏览器
    if not args.no_browser:
        display_host = "127.0.0.1" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
        chat_url = f"http://{display_host}:{args.port}/chat"
        open_browser_async(chat_url, delay=0.8)

    # 7. 启动 Uvicorn 主服务
    import uvicorn
    reload_enabled = not args.no_reload
    try:
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=reload_enabled,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n👋 AstroBot 服务已安全停止。")
    finally:
        if mcp_manager:
            mcp_manager.stop_all()


if __name__ == "__main__":
    main()
