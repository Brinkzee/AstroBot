import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatStreamRequest
from app.schemas.after_sale import AfterSaleExtractRequest, AfterSaleTicket
from app.services.session_manager import session_manager
from app.services.after_sale_service import extract_after_sale_ticket
from app.prompts.customer_service import customer_service_prompt
from app.llm import get_chat_model
from app.config import settings

router = APIRouter(prefix="/api")

@router.post("/chat/stream")
async def chat_stream(request: ChatStreamRequest):
    session_id = session_manager.get_or_create_session(request.session_id)
    history = session_manager.get_trimmed_history(session_id, max_tokens=settings.max_context_tokens)

    formatted_messages = customer_service_prompt.format_messages(
        history=history,
        input=request.message
    )

    # 记录当前用户消息到会话
    session_manager.add_user_message(session_id, request.message)

    llm = get_chat_model(streaming=True)

    async def event_generator():
        collected_chunks = []
        try:
            async for chunk in llm.astream(formatted_messages):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    collected_chunks.append(content)
                    payload = json.dumps({"session_id": session_id, "content": content}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

            # 流式传输完成后，将 AI 完整回复写入会话历史
            full_response = "".join(collected_chunks)
            if full_response:
                session_manager.add_ai_message(session_id, full_response)

            yield "data: [DONE]\n\n"
        except Exception as e:
            err_payload = json.dumps({"session_id": session_id, "error": str(e)}, ensure_ascii=False)
            yield f"data: {err_payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/after-sale/extract", response_model=AfterSaleTicket)
async def extract_ticket(request: AfterSaleExtractRequest):
    try:
        ticket = extract_after_sale_ticket(request.description)
        return ticket
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提取失败: {str(e)}")
