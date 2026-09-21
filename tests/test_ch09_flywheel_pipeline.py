import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import select

from app.db.session import Base
from app.services.flywheel.pipeline import FlywheelPipeline
from app.models.review_queue import ReviewQueue, ReviewStatus
from app.models.low_confidence import LowConfidenceQuestion

@pytest_asyncio.fixture
async def memory_db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestingSessionLocal = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with TestingSessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

class MockLLM:

    def invoke(self, prompt):
        # We can implement simple logic for testing
        prompt_str = str(prompt)
        if "标准FAQ" in prompt_str:
            return type('Obj', (object,), {'content': 'Normalized: Why is it not working?\n\nAnswer: Please check connection.'})()
        if "是否表达了相同的客户意图" in prompt_str:
            if "same" in prompt_str:
                return type('Obj', (object,), {'content': '2'})()
            return type('Obj', (object,), {'content': 'NONE'})()
        return type('Obj', (object,), {'content': ''})()

@pytest.mark.asyncio
async def test_normalize_question():
    pipeline = FlywheelPipeline()
    llm = MockLLM()
    q, a = pipeline.normalize_question("为啥我的东西不动了啊？", llm=llm)
    assert q == "Normalized: Why is it not working?"
    assert a == "Answer: Please check connection."

@pytest.mark.asyncio
async def test_find_duplicate_question():
    pipeline = FlywheelPipeline()
    llm = MockLLM()
    c1 = ReviewQueue(id=1, normalized_question="Why is it not working?")
    c2 = ReviewQueue(id=2, normalized_question="same meaning?")
    
    # Test identical text
    assert pipeline.find_duplicate_question("Why is it not working?", [c1, c2], llm=llm) == 1
    
    # Test LLM semantic match
    # Since candidates are formatted in prompt, let's mock carefully or just rely on MockLLM simplistic check
    llm2 = MockLLM()
    def mock_invoke(prompt):
        if '新问题："same intent"' in str(prompt):
            return type('Obj', (object,), {'content': '2'})()
        return type('Obj', (object,), {'content': 'NONE'})()
    llm2.invoke = mock_invoke
    
    assert pipeline.find_duplicate_question("same intent", [c1, c2], llm=llm2) == 2
    assert pipeline.find_duplicate_question("different", [c1, c2], llm=llm2) is None


@pytest.mark.asyncio
async def test_flywheel_pipeline_creates_and_merges(memory_db):
    pipeline = FlywheelPipeline()
    
    # Custom Mock LLM for this test to control merge behavior
    class TestMockLLM:
        def __init__(self):
            self.normalize_calls = 0
            
        def invoke(self, prompt):
            prompt_str = str(prompt)
            if "FAQ" in prompt_str:
                self.normalize_calls += 1
                if self.normalize_calls == 1:
                    return type('Obj', (object,), {'content': 'Q1\n\nA1'})()
                elif self.normalize_calls == 2:
                    return type('Obj', (object,), {'content': 'Q1_same\n\nA1_alt'})()
                else:
                    return type('Obj', (object,), {'content': 'Q3_diff\n\nA3'})()
            if "NONE" in prompt_str:
                print("MOCK RECEIVED:", prompt_str)
                if "Q1_same" in prompt_str:
                    return type('Obj', (object,), {'content': 'YES'})()
                return type('Obj', (object,), {'content': 'NONE'})()
            return type('Obj', (object,), {'content': ''})()

    test_llm = TestMockLLM()
    
    # 1. Insert first raw question
    q1 = LowConfidenceQuestion(raw_question="q1", source="self_check")
    memory_db.add(q1)
    await memory_db.commit()
    
    res = await pipeline.process_pending_questions(memory_db, batch_size=50, llm=test_llm)
    assert res["processed_count"] == 1
    assert res["new_created"] == 1
    assert res["merged_count"] == 0
    
    # Check db
    await memory_db.refresh(q1)
    assert q1.matched_review_id is not None
    r1 = await memory_db.get(ReviewQueue, q1.matched_review_id)
    assert r1.occurrence_count == 1
    assert r1.normalized_question == "Q1"
    
    # 2. Insert second raw question (similar intent)
    q2 = LowConfidenceQuestion(raw_question="q2", source="self_check")
    memory_db.add(q2)
    await memory_db.commit()
    
    res2 = await pipeline.process_pending_questions(memory_db, batch_size=50, llm=test_llm)
    assert res2["processed_count"] == 1
    assert res2["new_created"] == 0
    assert res2["merged_count"] == 1
    
    await memory_db.refresh(q2)
    assert q2.matched_review_id == r1.id
    
    await memory_db.refresh(r1)
    assert r1.occurrence_count == 2
    
    # 3. Insert third raw question (different intent)
    q3 = LowConfidenceQuestion(raw_question="q3", source="self_check")
    memory_db.add(q3)
    await memory_db.commit()
    
    res3 = await pipeline.process_pending_questions(memory_db, batch_size=50, llm=test_llm)
    assert res3["processed_count"] == 1
    assert res3["new_created"] == 1
    assert res3["merged_count"] == 0
    
    await memory_db.refresh(q3)
    assert q3.matched_review_id is not None
    assert q3.matched_review_id != r1.id
    
    r3 = await memory_db.get(ReviewQueue, q3.matched_review_id)
    assert r3.occurrence_count == 1
    assert r3.normalized_question == "Q3_diff"
