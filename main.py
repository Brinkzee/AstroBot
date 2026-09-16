import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import router

app = FastAPI(
    title="AstroBot - 电商智能客服系统",
    description="支持多轮 SSE 流式对话、客服角色约束与售后工单结构化提取",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.review_queue_routes import review_queue_router
from app.api.observability import router as observability_router

app.include_router(router)
app.include_router(review_queue_router)
app.include_router(observability_router)

static_dir = os.path.join(os.path.dirname(__file__), "app", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
@app.get("/chat")
def get_chat_page():
    index_file = os.path.join(static_dir, "index.html")
    return FileResponse(index_file)

@app.get("/kb")
def get_kb_page():
    kb_file = os.path.join(static_dir, "kb.html")
    return FileResponse(kb_file)

@app.get("/rag-eval")
@app.get("/rag_eval")
def get_rag_eval_page():
    eval_file = os.path.join(static_dir, "rag_eval.html")
    return FileResponse(eval_file)

@app.get("/observability")
def get_observability_page():
    file_path = os.path.join(static_dir, "observability.html")
    return FileResponse(file_path)

@app.get("/review-queue")
def get_review_queue_page():
    file_path = os.path.join(static_dir, "review_queue.html")
    return FileResponse(file_path)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
