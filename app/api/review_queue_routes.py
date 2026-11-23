from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from app.core.auth import verify_admin_token
from app.db.session import get_db
from app.models.review_queue import ReviewQueue, ReviewStatus
from app.models.low_confidence import LowConfidenceQuestion
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.services.rag.splitter import DocChunk

review_queue_router = APIRouter(prefix="/api/review-queue", tags=["review-queue"])


class ApproveRequest(BaseModel):
    approved_answer: str = Field(..., description="核准答案")


class RejectRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=500, description="可选驳回原因")


@review_queue_router.get("")
async def get_review_queue(
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200, description="每页返回条数"),
    offset: int = Query(0, ge=0, description="偏移量"),
    db: AsyncSession = Depends(get_db),
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

    # Batch-count raw questions with a single GROUP BY query (eliminates N+1)
    counts_map: Dict[int, int] = {}
    if items_db:
        item_ids = [item.id for item in items_db]
        rq_stmt = (
            select(
                LowConfidenceQuestion.matched_review_id,
                func.count(LowConfidenceQuestion.id),
            )
            .where(LowConfidenceQuestion.matched_review_id.in_(item_ids))
            .group_by(LowConfidenceQuestion.matched_review_id)
        )
        rq_result = await db.execute(rq_stmt)
        counts_map = {row[0]: int(row[1]) for row in rq_result.all() if row[0] is not None}

    items = []
    for item in items_db:
        items.append({
            "id": item.id,
            "normalized_question": item.normalized_question,
            "ai_suggested_answer": item.ai_suggested_answer,
            "occurrence_count": item.occurrence_count,
            "review_status": item.review_status,
            "approved_answer": item.approved_answer,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "raw_questions_count": counts_map.get(item.id, 0),
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
            "created_at": lcq.created_at,
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
        "raw_queries": raw_queries,
    }


@review_queue_router.post("/{id}/reject", dependencies=[Depends(verify_admin_token)])
async def reject_review_queue(
    id: int,
    request: Optional[RejectRequest] = Body(default=None),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ReviewQueue).where(ReviewQueue.id == id)
    result = await db.execute(stmt)
    item = result.scalar_one_or_none()

    if not item:
        raise HTTPException(status_code=404, detail="ReviewQueue not found")

    item.review_status = ReviewStatus.REJECTED.value
    await db.commit()
    resp: Dict[str, Any] = {"status": "rejected", "id": id}
    if request is not None and request.reason is not None and request.reason.strip():
        resp["reason"] = request.reason.strip()
    return resp


@review_queue_router.post("/{id}/approve", dependencies=[Depends(verify_admin_token)])
async def approve_review_queue(
    id: int,
    request: ApproveRequest,
    db: AsyncSession = Depends(get_db),
):
    if not request.approved_answer.strip():
        raise HTTPException(status_code=400, detail="Approved answer cannot be empty")

    stmt = select(ReviewQueue).where(ReviewQueue.id == id)
    result = await db.execute(stmt)
    item = result.scalar_one_or_none()

    if not item:
        raise HTTPException(status_code=404, detail="ReviewQueue not found")

    if item.review_status == ReviewStatus.APPROVED.value:
        raise HTTPException(status_code=409, detail="ReviewQueue item is already approved")

    chunk = DocChunk(
        category="常见问题",
        questions=item.normalized_question,
        answer=request.approved_answer,
        section_path="数据飞轮/FAQ",
        content_type="faq",
        is_key_clause=True,
    )

    writer = KnowledgeDualWriter()
    try:
        await writer.write_chunks(db, [chunk])
    finally:
        writer.close()

    item.review_status = ReviewStatus.APPROVED.value
    item.approved_answer = request.approved_answer
    await db.commit()
    return {"status": "approved", "id": id, "written_chunk_questions": item.normalized_question}
