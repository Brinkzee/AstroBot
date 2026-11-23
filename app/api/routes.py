import asyncio
import inspect
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func, or_, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import verify_admin_token
from app.schemas.chat import ChatStreamRequest, ChatResumeRequest
from app.schemas.after_sale import AfterSaleExtractRequest, AfterSaleTicket
from app.schemas.ticket import TicketCreateRequest, TicketCreateResponse
from app.schemas.conversation import ConversationItem, ConversationMessageItem
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
from app.models.conversation import Conversation
from app.models.knowledge import KnowledgeChunk
from app.models.message import Message
from app.models.ticket import Ticket
from app.models.summary import ConversationSummary
from app.models.tool_audit_log import ToolAuditLog
from app.models.low_confidence import LowConfidenceQuestion, record_low_confidence
from app.services.after_sale_service import (
    aextract_after_sale_ticket,
    extract_after_sale_ticket,
    extract_after_sale_ticket as _orig_extract_after_sale_ticket,
)
from app.services.chat_service import ChatService
from app.services.rag.dual_writer import KnowledgeDualWriter
from app.services.rag.milvus_client import MilvusKnowledgeStore
from app.services.rag.retriever import KnowledgeRetriever
from app.services.rag.splitter import DocChunk, MarkdownSectionSplitter
from app.tools.business_tools import create_ticket
from app.db.session import get_db
from app.api.rag_eval_routes import router as rag_eval_router

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = ROOT_DIR / "data" / "kb"

_WINDOWS_RESERVED_STEMS = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_SAFE_MD_FILENAME_RE = re.compile(r"^[^./\\\x00-\x1f][^/\\\x00-\x1f]*$")

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
        if extract_after_sale_ticket is _orig_extract_after_sale_ticket:
            ticket = await aextract_after_sale_ticket(request.description)
        else:
            res = await asyncio.to_thread(extract_after_sale_ticket, request.description)
            ticket = await res if inspect.isawaitable(res) else res
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

class FeedbackRequest(BaseModel):
    conversation_id: int
    message_id: Optional[int] = None
    feedback_type: Literal["up", "down"] = Field(..., description="赞同/点踩: 'up' 或 'down'")
    reason: Optional[str] = None
    query: Optional[str] = None


@router.post("/chat/feedback")
async def chat_feedback(request: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    conv = await db.get(Conversation, request.conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"会话 {request.conversation_id} 不存在")

    if request.feedback_type == "down":
        raw_question: Optional[str] = request.query
        target_ast_msg: Optional[Message] = None

        if request.message_id is not None:
            ast_stmt = select(Message).where(
                Message.conversation_id == request.conversation_id,
                Message.id == request.message_id,
            )
            ast_res = await db.execute(ast_stmt)
            candidate = ast_res.scalar_one_or_none()
            if candidate is not None and candidate.role == "assistant":
                target_ast_msg = candidate
            if not raw_question:
                user_stmt = (
                    select(Message)
                    .where(
                        Message.conversation_id == request.conversation_id,
                        Message.role == "user",
                        Message.id <= request.message_id,
                    )
                    .order_by(Message.id.desc())
                    .limit(1)
                )
                user_res = await db.execute(user_stmt)
                msg = user_res.scalar()
                if msg and msg.content:
                    raw_question = msg.content

        if not raw_question:
            stmt = (
                select(Message)
                .where(
                    Message.conversation_id == request.conversation_id,
                    Message.role == "user",
                )
                .order_by(Message.id.desc())
                .limit(1)
            )
            res = await db.execute(stmt)
            msg = res.scalar()
            raw_question = (msg.content if msg and msg.content else None) or "未知用户问题"

        safe_raw_question: str = raw_question.strip() or "未知用户问题"

        retrieved_chunks = None
        ast_msg = target_ast_msg
        if ast_msg is None:
            stmt_tool = (
                select(Message)
                .where(
                    Message.conversation_id == request.conversation_id,
                    Message.role == "assistant",
                )
                .order_by(Message.id.desc())
                .limit(1)
            )
            res_tool = await db.execute(stmt_tool)
            ast_msg = res_tool.scalar()

        if ast_msg and ast_msg.tool_calls:
            tc = ast_msg.tool_calls
            if isinstance(tc, str):
                try:
                    tc = json.loads(tc)
                except Exception as e:
                    logger.warning(f"Failed to parse tool_calls: {e}")
                    tc = []
            has_kb_tool = False
            if isinstance(tc, list):
                for call in tc:
                    if isinstance(call, dict) and call.get("name") in ("query_faq", "search_knowledge"):
                        has_kb_tool = True
                        break
            if has_kb_tool:
                stmt_audit = (
                    select(ToolAuditLog)
                    .where(
                        ToolAuditLog.conversation_id == request.conversation_id,
                        ToolAuditLog.tool_name.in_(["query_faq", "search_knowledge"]),
                    )
                    .order_by(ToolAuditLog.id.desc())
                    .limit(1)
                )
                res_audit = await db.execute(stmt_audit)
                audit_log = res_audit.scalar()

                if audit_log and audit_log.result_summary:
                    try:
                        res_data = (
                            json.loads(audit_log.result_summary)
                            if isinstance(audit_log.result_summary, str)
                            else audit_log.result_summary
                        )
                        if isinstance(res_data, dict) and "hits" in res_data:
                            hits = res_data["hits"]
                        elif isinstance(res_data, list):
                            hits = res_data
                        else:
                            hits = []

                        parsed_chunks = []
                        for hit in hits:
                            if not isinstance(hit, dict):
                                continue
                            text = hit.get("text") or hit.get("content") or hit.get("answer") or ""
                            raw_score = hit.get("score")
                            if raw_score is None:
                                raw_score = hit.get("distance")
                            score = float(raw_score) if raw_score is not None else 0.0
                            section = hit.get("section") or hit.get("section_path") or ""
                            parsed_chunks.append({
                                "text": str(text)[:300],
                                "score": score,
                                "section": str(section),
                            })
                        if parsed_chunks:
                            retrieved_chunks = parsed_chunks
                    except Exception as e:
                        logger.warning(f"Failed to parse tool result for chunks: {e}")

        entry = await record_low_confidence(
            db=db,
            raw_question=safe_raw_question,
            source="user_feedback",
            conversation_id=request.conversation_id,
            reason=request.reason or "用户点踩未解决",
            retrieved_chunks=retrieved_chunks,
        )
        return {"status": "success", "message": "Feedback recorded", "entry_id": entry.id}
    return {"status": "success", "message": "Thanks for your feedback"}

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
    limit: int = Query(50, ge=1, le=200, description="每页返回会话数量"),
    offset: int = Query(0, ge=0, description="分页偏移量"),
    db: AsyncSession = Depends(get_db),
):
    """查询指定用户的会话列表，按更新时间倒序排列，并附带首问预览与摘要标记（批量查询避免 N+1）"""
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .offset(offset)
        .limit(limit)
    )
    res = await db.execute(stmt)
    conversations = list(res.scalars().all())
    if not conversations:
        return []

    conv_ids = [c.id for c in conversations]

    # 批量统计每个会话的消息总条数
    count_stmt = (
        select(Message.conversation_id, func.count(Message.id))
        .where(Message.conversation_id.in_(conv_ids))
        .group_by(Message.conversation_id)
    )
    count_res = await db.execute(count_stmt)
    msg_count_map = {row[0]: int(row[1]) for row in count_res.all()}

    # 批量提取每个会话的首条 user 提问
    rn = func.row_number().over(
        partition_by=Message.conversation_id,
        order_by=(Message.created_at.asc(), Message.id.asc()),
    ).label("rn")
    ranked_subq = (
        select(
            Message.conversation_id.label("conversation_id"),
            Message.content.label("content"),
            rn,
        )
        .where(Message.conversation_id.in_(conv_ids), Message.role == "user")
        .subquery()
    )
    first_msg_stmt = select(ranked_subq.c.conversation_id, ranked_subq.c.content).where(
        ranked_subq.c.rn == 1
    )
    first_msg_res = await db.execute(first_msg_stmt)
    first_content_map = {row[0]: row[1] for row in first_msg_res.all()}

    items: List[ConversationItem] = []
    for conv in conversations:
        msg_count = msg_count_map.get(conv.id, 0)
        first_content = first_content_map.get(conv.id)

        if first_content and str(first_content).strip():
            title = str(first_content).strip()[:25]
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
    user_id: Optional[str] = Query(default=None, description="可选用户标识，传入时校验归属权"),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
):
    """获取指定会话 ID 的全部历史消息，按时间升序排列"""
    conv = await db.get(Conversation, id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"会话 {id} 不存在")

    effective_user_id = (user_id or x_user_id or "").strip() or None
    if effective_user_id is not None and conv.user_id != effective_user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")

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
    user_id: Optional[str] = Query(default=None, description="可选用户标识，传入时校验归属权"),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
):
    """删除指定会话并原子级联清理或解绑关联数据"""
    conv = await db.get(Conversation, id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"会话 {id} 不存在")

    effective_user_id = (user_id or x_user_id or "").strip() or None
    if effective_user_id is not None and conv.user_id != effective_user_id:
        raise HTTPException(status_code=403, detail="无权删除该会话")

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

def _count_milvus_knowledge() -> int:
    store = MilvusKnowledgeStore()
    try:
        return store.count("knowledge")
    finally:
        store.close()


def _scan_kb_materials(kb_dir: Path) -> List[MaterialItem]:
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


def _validate_kb_filename(raw_filename: str) -> str:
    """严格校验知识库文档文件名，杜绝路径穿越、点文件及 Windows 保留设备名。"""
    raw = (raw_filename or "").strip()
    if not raw or "/" in raw or "\\" in raw or ".." in raw:
        raise HTTPException(status_code=400, detail="非法文件名：不允许包含路径分隔符或 '..'")
    if raw.startswith(".") or not _SAFE_MD_FILENAME_RE.match(raw):
        raise HTTPException(status_code=400, detail="非法文件名：不允许以 '.' 开头或包含控制字符")
    clean_name = raw if raw.endswith(".md") else f"{raw}.md"
    stem = Path(clean_name).stem.strip()
    if not stem or stem.upper() in _WINDOWS_RESERVED_STEMS:
        raise HTTPException(status_code=400, detail=f"非法文件名：'{clean_name}' 为保留名称或空名称")
    return clean_name


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

    milvus_count = await asyncio.to_thread(_count_milvus_knowledge)

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
    return await asyncio.to_thread(_scan_kb_materials, KB_DIR)


@router.get("/kb/chunks", response_model=ChunkListResponse)
async def get_kb_chunks(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
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


@router.post("/kb/chunks/manual", response_model=ChunkItem, dependencies=[Depends(verify_admin_token)])
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
    except HTTPException:
        raise
    except Exception as e:
        try:
            await db.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"知识录入异常: {str(e)}")


@router.post("/kb/repair-pending", dependencies=[Depends(verify_admin_token)])
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
    retriever: Optional[KnowledgeRetriever] = None
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
    finally:
        if retriever is not None:
            try:
                retriever.close()
            except Exception:
                pass


@router.post("/kb/documents/preview", response_model=DocumentPreviewResponse)
async def preview_document_chunks(request: DocumentPreviewRequest):
    """完整 Markdown 文档智能切块预览（不落库）"""
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="文档正文内容不能为空")

    try:
        splitter = MarkdownSectionSplitter()
        doc_chunks = await asyncio.to_thread(splitter.split_markdown, request.content)

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


@router.post(
    "/kb/documents/manual",
    response_model=ManualDocumentCreateResponse,
    dependencies=[Depends(verify_admin_token)],
)
async def create_manual_document(
    request: ManualDocumentCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """完整 Markdown 文档录入，支持保存原材料文件与智能切分向量化双写"""
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="文档正文内容不能为空")

    clean_name = _validate_kb_filename(request.filename)
    target_path = KB_DIR / clean_name

    # 若需落盘原材料文件，先校验覆盖冲突（不提前写盘，避免失败留下孤立文件）
    if request.save_file and not request.overwrite and target_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"原材料文件 '{clean_name}' 已存在，如需覆盖请显式设置 overwrite=true",
        )

    try:
        splitter = MarkdownSectionSplitter()
        doc_chunks = await asyncio.to_thread(splitter.split_markdown, request.content)
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
            saved_count = len(saved)
            vec_status = "done"
        else:
            db_chunks = [
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
                for c in doc_chunks
            ]
            db.add_all(db_chunks)
            await db.commit()
            saved_count = len(db_chunks)
            vec_status = "pending"

        file_saved = False
        if request.save_file:
            def _write_doc_file() -> None:
                KB_DIR.mkdir(parents=True, exist_ok=True)
                target_path.write_text(request.content, encoding="utf-8")

            await asyncio.to_thread(_write_doc_file)
            file_saved = True

        return ManualDocumentCreateResponse(
            success=True,
            total_chunks=len(doc_chunks),
            saved_chunks=saved_count,
            vectorize_status=vec_status,
            file_saved=file_saved,
            filename=clean_name,
        )
    except HTTPException:
        raise
    except Exception as e:
        try:
            await db.rollback()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"完整文档录入异常: {str(e)}")


