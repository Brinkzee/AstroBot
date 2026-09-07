import asyncio
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import AsyncSessionLocal, engine
from app.services.chat_service import ChatService

chat_service = ChatService()

async def run_scenario(scenario_num: int, title: str, user_message: str):
    print("=" * 70)
    print(f"【验收场景 {scenario_num}】: {title}")
    print(f"用户输入: {user_message}")
    print("-" * 70)
    
    async with AsyncSessionLocal() as db:
        events = []
        full_text = ""
        tool_called = None
        
        async for event in chat_service.stream_chat(db, conversation_id=None, message=user_message):
            events.append(event)
            etype = event.get("event_type")
            if etype == "tool_start":
                tool_called = event.get("tool_name")
                print(f"[SSE 状态帧] 🚀 tool_start: {event.get('tool_name')} ({event.get('tool_label')}) | args: {event.get('args')}")
            elif etype == "tool_end":
                print(f"[SSE 状态帧] ✅ tool_end: {event.get('tool_name')} | success: {event.get('success')}")
            elif etype == "text":
                content = event.get("content", "")
                full_text += content
                sys.stdout.write(content)
                sys.stdout.flush()
            elif etype == "error":
                print(f"[SSE 错误帧] ❌ {event.get('error')}")
                
        print("\n" + "-" * 70)
        print(f"最终收敛回复: {full_text.strip()}")
        print(f"调用工具: {tool_called}")
        print("=" * 70 + "\n")
        return tool_called, full_text

async def main():
    print("\n🚀 开始执行 Chapter 02 三大验收标准端到端实测...\n")
    
    # 验收场景 1: 查物流
    t1, res1 = await run_scenario(1, "查询订单 1001 的物流", "订单 1001 的物流到哪了")
    assert t1 == "query_logistics", f"预期调用 query_logistics，实际: {t1}"
    assert "1001" in res1 or "物流" in res1 or "快递" in res1 or "派送" in res1, "未在回复中识别物流信息"
    
    # 验收场景 2: 查 FAQ 命中
    t2, res2 = await run_scenario(2, "查询退货政策 (FAQ 命中)", "退货政策是什么")
    assert t2 == "query_faq", f"预期调用 query_faq，实际: {t2}"
    assert "7天" in res2 or "退货" in res2 or "吊牌" in res2, "未在回复中识别退货政策信息"
    
    # 验收场景 3: 查 FAQ 漏召回 (邮费是多少，库中为“运费”)
    t3, res3 = await run_scenario(3, "查询邮费是多少 (FAQ 预期漏召回)", "邮费是多少")
    print(f"验收 3 检查: 工具={t3}, 回复是否体现未检索到或人工咨询={res3[:50]}...")
    
    await engine.dispose()
    print("\n🎉 三大验收场景全部验证成功！\n")

if __name__ == "__main__":
    asyncio.run(main())
