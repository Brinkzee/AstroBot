import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.chat import ChatStreamRequest
from app.schemas.after_sale import AfterSaleExtractRequest, AfterSaleTicket
from app.services.after_sale_service import extract_after_sale_ticket
from app.services.chat_service import ChatService
from app.db.session import get_db

router = APIRouter(prefix="/api")

chat_service = ChatService()
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


@router.post("/after-sale/extract", response_model=AfterSaleTicket)
async def extract_ticket(request: AfterSaleExtractRequest):
    try:
        ticket = extract_after_sale_ticket(request.description)
        return ticket
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提取失败: {str(e)}")
