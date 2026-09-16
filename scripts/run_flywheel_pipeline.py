import sys
import os
import asyncio

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.session import AsyncSessionLocal
from app.services.flywheel.pipeline import FlywheelPipeline

async def main():
    # Try to import get_llm
    try:
        from app.services.chat_service import get_llm  # guessing where it might be
        llm = get_llm()
    except Exception:
        # Fallback if get_llm is not available or mock is needed
        try:
            from langchain_openai import ChatOpenAI
            from app.config import settings
            llm = ChatOpenAI(model=settings.openai_api_model, api_key=settings.openai_api_key)
        except:
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
