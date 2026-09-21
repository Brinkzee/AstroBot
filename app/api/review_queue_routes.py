from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from app.db.session import get_db
from app.models.review_queue import ReviewQueue, ReviewStatus
from app.models.low_confidence import LowConfidenceQuestion
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.services.rag.splitter import DocChunk

review_queue_router = APIRouter(prefix="/api/review-queue", tags=["review-queue"])

class ApproveRequest(BaseModel):
    approved_answer: str = Field(..., description="核准答案")

@review_queue_router.get("")
async def get_review_queue(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    stmt = select(ReviewQueue)
    if status:
        stmt = stmt.where(ReviewQueue.review_status == status)
    
    stmt = stmt.order_by(desc(ReviewQueue.occurrence_count), desc(ReviewQueue.id))
    stmt = stmt.offset(offset).limit(limit)
    
    result = await db.execute(stmt)
    items_db = result.scalars().all()
    
    # Get total count
    count_stmt = select(func.count(ReviewQueue.id))
    if status:
        count_stmt = count_stmt.where(ReviewQueue.review_status == status)
    total_result = await db.execute(count_stmt)
    total = total_result.scalar_one()

    # For raw_questions_count, since lazy="select", we need a separate count or query. 
    # Better to just count in python or do a subquery, but simplest is a separate count query per item or joinedload
    # Given the requirements, we'll manually fetch count for simplicity or assume it's fine
    items = []
    for item in items_db:
        # fetch raw_questions_count
        rq_stmt = select(func.count(LowConfidenceQuestion.id)).where(LowConfidenceQuestion.matched_review_id == item.id)
        rq_result = await db.execute(rq_stmt)
        rq_count = rq_result.scalar_one()
        
        items.append({
            "id": item.id,
            "normalized_question": item.normalized_question,
            "ai_suggested_answer": item.ai_suggested_answer,
            "occurrence_count": item.occurrence_count,
            "review_status": item.review_status,
            "approved_answer": item.approved_answer,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "raw_questions_count": rq_count
        })

    return {"total": total, "items": items}

@review_queue_router.get("/{id}")
async def get_review_queue_detail(id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(ReviewQueue).where(ReviewQueue.id == id)
    result = await db.execute(stmt)
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(status_code=404, detail="ReviewQueue not found")
        
    lcq_stmt = select(LowConfidenceQuestion).where(LowConfidenceQuestion.matched_review_id == id)
    lcq_result = await db.execute(lcq_stmt)
    lcqs = lcq_result.scalars().all()
    
    raw_queries = []
    for lcq in lcqs:
        raw_queries.append({
            "id": lcq.id,
            "raw_question": lcq.raw_question,
            "source": lcq.source,
            "reason": lcq.reason,
            "retrieved_chunks": lcq.retrieved_chunks,
            "created_at": lcq.created_at
        })
        
    return {
        "id": item.id,
        "normalized_question": item.normalized_question,
        "ai_suggested_answer": item.ai_suggested_answer,
        "occurrence_count": item.occurrence_count,
        "review_status": item.review_status,
        "approved_answer": item.approved_answer,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "raw_queries": raw_queries
    }

@review_queue_router.post("/{id}/reject")
async def reject_review_queue(id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(ReviewQueue).where(ReviewQueue.id == id)
    result = await db.execute(stmt)
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(status_code=404, detail="ReviewQueue not found")
        
    item.review_status = ReviewStatus.REJECTED.value
    await db.commit()
    return {"status": "rejected", "id": id}

@review_queue_router.post("/{id}/approve")
async def approve_review_queue(id: int, request: ApproveRequest, db: AsyncSession = Depends(get_db)):
    if not request.approved_answer.strip():
        raise HTTPException(status_code=400, detail="Approved answer cannot be empty")
        
    stmt = select(ReviewQueue).where(ReviewQueue.id == id)
    result = await db.execute(stmt)
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(status_code=404, detail="ReviewQueue not found")
        
    item.review_status = ReviewStatus.APPROVED.value
    item.approved_answer = request.approved_answer
    
    chunk = DocChunk(
        category="常见问题",
        questions=item.normalized_question,
        answer=request.approved_answer,
        section_path="数据飞轮/FAQ",
        content_type="faq",
        is_key_clause=True
    )
    
    writer = KnowledgeDualWriter()
    await writer.write_chunks(db, [chunk])
    
    await db.commit()
    return {"status": "approved", "id": id, "written_chunk_questions": item.normalized_question}
