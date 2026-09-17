import sys
import os
import asyncio

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.session import AsyncSessionLocal
from app.services.flywheel.pipeline import FlywheelPipeline
async def main():
    try:
        from scripts.wsl_helper import ensure_mysql_ready
        ensure_mysql_ready(verbose=False)
    except Exception:
        pass

    try:
        from app.llm import get_chat_model
        llm = get_chat_model()
    except Exception:
        llm = None
        
    pipeline = FlywheelPipeline()
    
    async with AsyncSessionLocal() as db:
        res = await pipeline.process_pending_questions(db, llm=llm)
        
    print("=======================================================")
    print("飞轮批处理运行完成:")
    print(f"  待处理问题数: {res.get('processed_count', 0)}")
    print(f"  新建知识缺口: {res.get('new_created', 0)}")
    print(f"  查重归并缺口: {res.get('merged_count', 0)}")
    print("=======================================================")

if __name__ == "__main__":
    asyncio.run(main())
