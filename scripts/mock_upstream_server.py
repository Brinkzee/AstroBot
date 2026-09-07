"""本地 OpenAI 兼容协议上游模拟服务 (Mock Upstream Server)
用于在无外部 API Key 或离线开发环境下，端到端验证 FastAPI 与 LangChain 体系的流式与结构化输出。
"""
import json
import time
import re
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

mock_app = FastAPI(title="Mock OpenAI Upstream Server")

@mock_app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    streaming = body.get("stream", False)
    tools = body.get("tools", [])
    response_format = body.get("response_format", {})

    last_user_msg = ""
    full_context = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        full_context += f"[{role}]: {content}\n"
        if role == "user":
            last_user_msg = content

    # 1. 判断是否为结构化提取请求（with_structured_output 通常使用 tool_choice 或 response_format 或 prompt 指导）
    is_structured = False
    if tools or (response_format and response_format.get("type") in ["json_object", "json_schema"]):
        is_structured = True

    if is_structured:
        # 提取订单号
        order_match = re.search(r"([A-Za-z0-9]{8,18})", last_user_msg)
        order_id = order_match.group(1) if order_match else None

        # 归纳诉求与方案
        issue_type = "商品质量问题/破损" if any(w in last_user_msg for w in ["卡住", "坏", "破", "开胶", "漏", "起球"]) else "七天无理由退货"
        expected_solution = "换货" if "换" in last_user_msg else ("退款" if "退" in last_user_msg else "客服核实处理")

        structured_data = {
            "order_id": order_id,
            "issue_type": issue_type,
            "expected_solution": expected_solution,
            "raw_description": last_user_msg
        }

        # 如果是通过 tool_calls 实现
        if tools:
            tool_name = tools[0].get("function", {}).get("name", "AfterSaleTicket")
            response_obj = {
                "id": f"chatcmpl-mock-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "mock-model"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_mock_123",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(structured_data, ensure_ascii=False)
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }
            return JSONResponse(response_obj)
        else:
            # response_format 模式
            response_obj = {
                "id": f"chatcmpl-mock-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "mock-model"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(structured_data, ensure_ascii=False)
                    },
                    "finish_reason": "stop"
                }]
            }
            return JSONResponse(response_obj)

    # 2. 常规对话：根据多轮上下文生成贴合电商客服角色的回复内容
    reply_tokens = []
    if "羽绒服" in last_user_msg and ("保修" in last_user_msg or "多久" in last_user_msg):
        reply_tokens = ["您", "好", "！", "星", "光", "优", "选", "羽", "绒", "服", "自", "收", "货", "起", "享", "有", "全", "年", "365", "天", "官", "方", "质", "保", "，", "请", "您", "放", "心", "选", "购", "。"]
    elif "内胆" in last_user_msg:
        # 上下文关联识别
        reply_tokens = ["您", "好", "！", "这", "款", "黑", "色", "羽", "绒", "服", "是", "含", "有", "可", "拆", "卸", "白", "鸭", "绒", "内", "胆", "的", "，", "保", "暖", "性", "能", "非", "常", "优", "秀", "。"]
    else:
        reply_tokens = ["您", "好", "！", "我", "是", "星", "光", "优", "选", "客", "服", "小", "星", "，", "请", "问", "有", "什", "么", "可", "以", "帮", "您", "？"]

    if streaming:
        async def stream_generator():
            for t in reply_tokens:
                chunk = {
                    "id": f"chatcmpl-mock-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.get("model", "mock-model"),
                    "choices": [{
                        "index": 0,
                        "delta": {"content": t},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            # 发送结束
            stop_chunk = {
                "id": f"chatcmpl-mock-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.get("model", "mock-model"),
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        full_content = "".join(reply_tokens)
        return JSONResponse({
            "id": f"chatcmpl-mock-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock-model"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_content
                },
                "finish_reason": "stop"
            }]
        })

if __name__ == "__main__":
    uvicorn.run(mock_app, host="127.0.0.1", port=8001)
