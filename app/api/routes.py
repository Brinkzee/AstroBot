import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func, or_, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.chat import ChatStreamRequest, ChatResumeRequest
from app.schemas.after_sale import AfterSaleExtractRequest, AfterSaleTicket
from app.schemas.ticket import TicketCreateRequest, TicketCreateResponse
from app.schemas.conversation import ConversationItem, ConversationMessageItem
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.ticket import Ticket
from app.models.summary import ConversationSummary
from app.models.tool_audit_log import ToolAuditLog
from app.models.low_confidence import LowConfidenceQuestion
from app.services.after_sale_service import extract_after_sale_ticket
from app.services.chat_service import ChatService
from app.tools.business_tools import create_ticket
from app.db.session import get_db
from app.api.rag_eval_routes import router as rag_eval_router

router = APIRouter(prefix="/api")
router.include_router(rag_eval_router)

chat_service = ChatService(use_workflow=True)
default_chat_service = chat_service


@router.post("/chat/stream")
async def chat_stream(
    request: ChatStreamRequest,
    db: AsyncSession = Depends(get_db),
):
    async def event_generator():
        try:
            async for event in chat_service.stream_chat(
                db=db,
                conversation_id=request.effective_conversation_id,
                message=request.message,
            ):
                payload = json.dumps(event, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_payload = json.dumps(
                {
                    "event_type": "error",
                    "conversation_id": request.effective_conversation_id,
                    "error": str(e),
                },
                ensure_ascii=False,
            )
            yield f"data: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/resume")
async def chat_resume(
    request: ChatResumeRequest,
    db: AsyncSession = Depends(get_db),
):
    async def event_generator():
        try:
            async for event in chat_service.resume_chat(
                db=db,
                conversation_id=request.conversation_id,
                action=request.action,
            ):
                payload = json.dumps(event, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_payload = json.dumps(
                {
                    "event_type": "error",
                    "conversation_id": request.conversation_id,
                    "error": str(e),
                },
                ensure_ascii=False,
            )
            yield f"data: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



@router.post("/after-sale/extract", response_model=AfterSaleTicket)
async def extract_ticket(request: AfterSaleExtractRequest):
    try:
        ticket = extract_after_sale_ticket(request.description)
        return ticket
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提取失败: {str(e)}")


@router.post("/tickets", response_model=TicketCreateResponse)
async def api_create_ticket(request: TicketCreateRequest):
    """前端自选独立触发的创建工单 API（不消耗 LLM）"""
    res_str = await create_ticket.ainvoke({
        "conversation_id": request.conversation_id,
        "description": request.description,
        "ticket_type": request.ticket_type,
    })
    data = json.loads(res_str)
    if "error" in data:
        raise HTTPException(status_code=400, detail=data["error"])
    return TicketCreateResponse(
        ticket_no=data["ticket_no"],
        conversation_id=int(data["conversation_id"]),
        ticket_type=data["ticket_type"],
        status=data["status"],
    )


# ==============================================================================
# 多会话管理与历史回溯 API 接口 (/api/conversations/...)
# ==============================================================================

def _format_datetime(dt) -> str:
    if dt is None:
        return ""
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt)


@router.get("/conversations", response_model=List[ConversationItem])
async def list_conversations(
    user_id: str = "default_user",
    db: AsyncSession = Depends(get_db),
):
    """查询指定用户的会话列表，按更新时间倒序排列，并附带首问预览与摘要标记"""
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
    )
    res = await db.execute(stmt)
    conversations = res.scalars().all()

    items: List[ConversationItem] = []
    for conv in conversations:
        # 统计该会话消息总条数
        count_stmt = select(func.count(Message.id)).where(Message.conversation_id == conv.id)
        count_res = await db.execute(count_stmt)
        msg_count = count_res.scalar() or 0

        # 提取首条 user 提问作为 title 预览
        first_msg_stmt = (
            select(Message.content)
            .where(Message.conversation_id == conv.id, Message.role == "user")
            .order_by(Message.created_at.asc(), Message.id.asc())
            .limit(1)
        )
        first_msg_res = await db.execute(first_msg_stmt)
        first_content = first_msg_res.scalar()

        if first_content and first_content.strip():
            title = first_content.strip()[:25]
        else:
            title = "新会话"

        # has_summary: 当 conv.summary 存在且非空或 conv.summary_upto_msg_id > 0 时为 True，否则为 False
        has_summary = bool(
            (conv.summary and conv.summary.strip())
            or (conv.summary_upto_msg_id is not None and conv.summary_upto_msg_id > 0)
        )

        status_str = conv.status.value if hasattr(conv.status, "value") else str(conv.status or "进行中")

        items.append(
            ConversationItem(
                id=conv.id,
                title=title,
                has_summary=has_summary,
                status=status_str,
                message_count=msg_count,
                created_at=_format_datetime(conv.created_at),
                updated_at=_format_datetime(conv.updated_at),
            )
        )

    return items


@router.get("/conversations/{id}/messages", response_model=List[ConversationMessageItem])
async def get_conversation_messages(
    id: int,
    db: AsyncSession = Depends(get_db),
):
    """获取指定会话 ID 的全部历史消息，按时间升序排列"""
    conv_stmt = select(Conversation.id).where(Conversation.id == id)
    conv_res = await db.execute(conv_stmt)
    if not conv_res.scalar():
        raise HTTPException(status_code=404, detail=f"会话 {id} 不存在")

    msg_stmt = (
        select(Message)
        .where(Message.conversation_id == id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    res = await db.execute(msg_stmt)
    messages = res.scalars().all()

    items: List[ConversationMessageItem] = []
    for m in messages:
        tool_calls = m.tool_calls
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except Exception:
                pass

        role_str = m.role.value if hasattr(m.role, "value") else str(m.role)

        items.append(
            ConversationMessageItem(
                id=m.id,
                role=role_str,
                content=m.content,
                tool_calls=tool_calls,
                tool_call_id=m.tool_call_id,
                created_at=_format_datetime(m.created_at),
            )
        )

    return items


@router.delete("/conversations/{id}")
async def delete_conversation(
    id: int,
    db: AsyncSession = Depends(get_db),
):
    """删除指定会话并原子级联清理或解绑关联数据"""
    conv = await db.get(Conversation, id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"会话 {id} 不存在")

    # 1. LowConfidenceQuestion: 来源会话解绑（设 conversation_id = None）
    await db.execute(
        update(LowConfidenceQuestion)
        .where(LowConfidenceQuestion.conversation_id == id)
        .values(conversation_id=None)
    )

    # 2. ToolAuditLog: 删除该会话的调用审计记录
    await db.execute(delete(ToolAuditLog).where(ToolAuditLog.conversation_id == id))

    # 3. Ticket: 删除该会话创建的人工工单
    await db.execute(delete(Ticket).where(Ticket.conversation_id == id))

    # 4. ConversationSummary: 删除该会话的分段摘要
    await db.execute(delete(ConversationSummary).where(ConversationSummary.conversation_id == id))

    # 5. Message: 删除该会话的消息流水
    await db.execute(delete(Message).where(Message.conversation_id == id))

    # 6. Conversation: 删除会话实体
    await db.delete(conv)

    # 7. 提交事务
    await db.commit()

    return {
        "success": True,
        "message": f"会话 #{id} 已成功删除",
        "conversation_id": id,
    }


# ==============================================================================
# 知识库可视化管理与自测工作台 API 接口 (/api/kb/...)
# ==============================================================================
import time
from datetime import datetime
from pathlib import Path
from sqlalchemy import select, func, or_
from app.models.knowledge import KnowledgeChunk
from app.schemas.kb import (
    ManualChunkCreateRequest,
    KBSearchRequest,
    KBStatsResponse,
    MaterialItem,
    ChunkItem,
    ChunkListResponse,
    KBSearchResponse,
    KBSearchHit,
    DocumentPreviewRequest,
    DocChunkPreviewItem,
    DocumentPreviewResponse,
    ManualDocumentCreateRequest,
    ManualDocumentCreateResponse,
)
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.services.rag.milvus_client import MilvusKnowledgeStore
from app.services.rag.retriever import KnowledgeRetriever
from app.services.rag.splitter import DocChunk, MarkdownSectionSplitter


@router.get("/kb/stats", response_model=KBStatsResponse)
async def get_kb_stats(db: AsyncSession = Depends(get_db)):
    """获取知识库整体统计指标 (MySQL 原文 vs Milvus 向量索引)"""
    total_res = await db.execute(select(func.count(KnowledgeChunk.id)))
    total_chunks = total_res.scalar() or 0

    done_res = await db.execute(
        select(func.count(KnowledgeChunk.id)).where(KnowledgeChunk.vectorize_status == "done")
    )
    done_chunks = done_res.scalar() or 0

    pending_res = await db.execute(
        select(func.count(KnowledgeChunk.id)).where(KnowledgeChunk.vectorize_status == "pending")
    )
    pending_chunks = pending_res.scalar() or 0

    failed_res = await db.execute(
        select(func.count(KnowledgeChunk.id)).where(KnowledgeChunk.vectorize_status == "failed")
    )
    failed_chunks = failed_res.scalar() or 0

    store = MilvusKnowledgeStore()
    try:
        milvus_count = store.count("knowledge")
    finally:
        store.close()

    return KBStatsResponse(
        total_chunks=total_chunks,
        done_chunks=done_chunks,
        pending_chunks=pending_chunks,
        failed_chunks=failed_chunks,
        milvus_count=milvus_count,
        is_aligned=(done_chunks == milvus_count),
    )


@router.get("/kb/materials", response_model=List[MaterialItem])
async def get_kb_materials():
    """扫描 data/kb/ 目录下的原材料文档清单及切块估算"""
    kb_dir = Path("data/kb")
    materials: List[MaterialItem] = []
    if kb_dir.exists():
        splitter = MarkdownSectionSplitter()
        for file_path in sorted(kb_dir.glob("*.md")):
            stat = file_path.stat()
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            try:
                content = file_path.read_text(encoding="utf-8")
                sections = splitter.split_markdown(content)
                est_chunks = len(sections)
            except Exception:
                est_chunks = 0

            materials.append(
                MaterialItem(
                    filename=file_path.name,
                    file_path=str(file_path).replace("\\", "/"),
                    file_size_bytes=stat.st_size,
                    modified_time=mod_time,
                    estimated_chunks=est_chunks,
                )
            )
    return materials


@router.get("/kb/chunks", response_model=ChunkListResponse)
async def get_kb_chunks(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """分页与条件筛选知识切块列表"""
    stmt = select(KnowledgeChunk)
    count_stmt = select(func.count(KnowledgeChunk.id))

    filters = []
    if status:
        filters.append(KnowledgeChunk.vectorize_status == status)
    if category:
        filters.append(KnowledgeChunk.category == category)
    if search:
        filters.append(
            or_(
                KnowledgeChunk.questions.like(f"%{search}%"),
                KnowledgeChunk.answer.like(f"%{search}%"),
                KnowledgeChunk.section_path.like(f"%{search}%"),
            )
        )

    if filters:
        stmt = stmt.where(*filters)
        count_stmt = count_stmt.where(*filters)

    total_res = await db.execute(count_stmt)
    total = total_res.scalar() or 0

    stmt = stmt.order_by(KnowledgeChunk.id.desc()).offset((page - 1) * page_size).limit(page_size)
    res = await db.execute(stmt)
    records = res.scalars().all()

    items = []
    for r in records:
        created_str = (
            r.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(r.created_at, "strftime")
            else str(r.created_at)
        )
        items.append(
            ChunkItem(
                id=r.id,
                category=r.category,
                questions=r.questions,
                answer=r.answer,
                section_path=r.section_path,
                content_type=r.content_type,
                is_key_clause=r.is_key_clause,
                vectorize_status=r.vectorize_status,
                vector_id=r.vector_id,
                created_at=created_str,
            )
        )

    return ChunkListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=items,
    )


@router.post("/kb/chunks/manual", response_model=ChunkItem)
async def create_manual_chunk(
    request: ManualChunkCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """手工录入单条知识块，支持立即双写落库与向量化"""
    try:
        if request.sync_vector:
            doc_chunk = DocChunk(
                category=request.category,
                questions=request.questions,
                answer=request.answer,
                section_path=request.section_path,
                content_type=request.content_type,
                is_key_clause=request.is_key_clause,
            )
            writer = KnowledgeDualWriter()
            try:
                saved = await writer.write_chunks(db, [doc_chunk])
                if not saved:
                    raise HTTPException(status_code=500, detail="保存知识块失败")
                chunk = saved[0]
            finally:
                writer.close()
        else:
            chunk = KnowledgeChunk(
                category=request.category,
                questions=request.questions,
                answer=request.answer,
                section_path=request.section_path,
                content_type=request.content_type,
                is_key_clause=request.is_key_clause,
                vectorize_status="pending",
                vector_id=None,
            )
            db.add(chunk)
            await db.commit()
            await db.refresh(chunk)

        created_str = (
            chunk.created_at.strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(chunk.created_at, "strftime")
            else str(chunk.created_at)
        )
        return ChunkItem(
            id=chunk.id,
            category=chunk.category,
            questions=chunk.questions,
            answer=chunk.answer,
            section_path=chunk.section_path,
            content_type=chunk.content_type,
            is_key_clause=chunk.is_key_clause,
            vectorize_status=chunk.vectorize_status,
            vector_id=chunk.vector_id,
            created_at=created_str,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"知识录入异常: {str(e)}")


@router.post("/kb/repair-pending")
async def repair_pending_chunks(db: AsyncSession = Depends(get_db)):
    """一键触发断点续跑补齐所有未向量化的 pending 知识块"""
    try:
        writer = KnowledgeDualWriter()
        try:
            repaired = await writer.repair_pending_chunks(db)
        finally:
            writer.close()
        return {"success": True, "repaired_count": repaired}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"修复补偿失败: {str(e)}")


@router.post("/kb/search", response_model=KBSearchResponse)
async def search_knowledge(request: KBSearchRequest):
    """密集语义检索自测沙盒"""
    start_time = time.perf_counter()
    try:
        retriever = KnowledgeRetriever()
        hits = await retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            category=request.category,
        )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        preview = retriever.format_faq_hits(hits, keyword=request.query)

        hit_items = []
        for h in hits:
            hit_items.append(
                KBSearchHit(
                    id=h["id"],
                    distance=round(float(h.get("distance", 0.0)), 4),
                    category=h.get("category", ""),
                    questions=h.get("questions", ""),
                    answer=h.get("answer", ""),
                    section_path=h.get("section_path"),
                    content_type=h.get("content_type", "faq"),
                    is_key_clause=bool(h.get("is_key_clause", False)),
                )
            )

        return KBSearchResponse(
            query=request.query,
            top_k=request.top_k,
            latency_ms=elapsed_ms,
            total_hits=len(hit_items),
            hits=hit_items,
            formatted_preview=preview,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索自测异常: {str(e)}")


@router.post("/kb/documents/preview", response_model=DocumentPreviewResponse)
async def preview_document_chunks(request: DocumentPreviewRequest):
    """完整 Markdown 文档智能切块预览（不落库）"""
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="文档正文内容不能为空")

    try:
        splitter = MarkdownSectionSplitter()
        doc_chunks = splitter.split_markdown(request.content)

        default_cat = (request.category or "").strip()
        preview_items: List[DocChunkPreviewItem] = []
        for idx, c in enumerate(doc_chunks, start=1):
            cat = c.category
            if default_cat and (not cat or cat == "未分类"):
                cat = default_cat
            preview_items.append(
                DocChunkPreviewItem(
                    id=idx,
                    category=cat,
                    questions=c.questions,
                    answer=c.answer,
                    section_path=c.section_path,
                    content_type=c.content_type or "policy",
                    is_key_clause=bool(c.is_key_clause),
                )
            )

        return DocumentPreviewResponse(
            filename=request.filename,
            total_chunks=len(preview_items),
            chunks=preview_items,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档切块预览异常: {str(e)}")


@router.post("/kb/documents/manual", response_model=ManualDocumentCreateResponse)
async def create_manual_document(
    request: ManualDocumentCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """完整 Markdown 文档录入，支持保存原材料文件与智能切分向量化双写"""
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="文档正文内容不能为空")

    clean_name = Path(request.filename).name.strip()
    if not clean_name:
        clean_name = f"doc_{int(time.time())}.md"
    elif not clean_name.endswith(".md"):
        clean_name = f"{clean_name}.md"

    # 可选保存原材料文件至 data/kb/
    file_saved = False
    if request.save_file:
        try:
            kb_dir = Path("data/kb")
            kb_dir.mkdir(parents=True, exist_ok=True)
            target_path = kb_dir / clean_name
            target_path.write_text(request.content, encoding="utf-8")
            file_saved = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"保存原材料文件失败: {str(e)}")

    try:
        splitter = MarkdownSectionSplitter()
        doc_chunks = splitter.split_markdown(request.content)
        if not doc_chunks:
            raise HTTPException(status_code=400, detail="文档未能切分出有效知识块，请检查内容格式")

        default_cat = (request.category or "").strip()
        for c in doc_chunks:
            if default_cat and (not c.category or c.category == "未分类"):
                c.category = default_cat

        if request.sync_vector:
            writer = KnowledgeDualWriter()
            try:
                saved = await writer.write_chunks(db, doc_chunks)
            finally:
                writer.close()
            return ManualDocumentCreateResponse(
                success=True,
                total_chunks=len(doc_chunks),
                saved_chunks=len(saved),
                vectorize_status="done",
                file_saved=file_saved,
                filename=clean_name,
            )
        else:
            db_chunks = []
            for c in doc_chunks:
                db_chunks.append(
                    KnowledgeChunk(
                        category=c.category,
                        questions=c.questions,
                        answer=c.answer,
                        section_path=c.section_path,
                        content_type=c.content_type or "policy",
                        is_key_clause=c.is_key_clause,
                        vectorize_status="pending",
                        vector_id=None,
                    )
                )
            db.add_all(db_chunks)
            await db.commit()
            return ManualDocumentCreateResponse(
                success=True,
                total_chunks=len(doc_chunks),
                saved_chunks=len(db_chunks),
                vectorize_status="pending",
                file_saved=file_saved,
                filename=clean_name,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"完整文档录入异常: {str(e)}")


