"""客服历史对话通用问答抽取 Prompt 与 Schema 定义。

负责引导大模型从客服历史对话流水分离口语闲聊、脱敏个人隐私（单号、姓名、地址等），
提炼出高质量、标准通用的电商知识库问答对，并强制以结构化 JSON 输出。
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class ExtractedQAPair(BaseModel):
    """抽取的单条标准问答对"""
    question: str = Field(description="抽取的标准通用问题")
    answer: str = Field(description="提炼的标准客服回答")


class ExtractedQAList(BaseModel):
    """抽取的问答对列表包装"""
    items: List[ExtractedQAPair] = Field(
        default_factory=list,
        description="抽取的问答对列表",
    )


QA_EXTRACTION_SYSTEM_PROMPT = """你是由“星光优选”电商平台研发的知识工程专家。
你的核心任务是分析历史客服对话流水，从中提取出具有沉淀价值的标准通用问答对（FAQ），供企业知识库使用。

【处理与抽取规范】
1. 彻底剔除寒暄与客套：
   - 过滤掉日常问候（如“你好”、“在吗”、“早上好”、“收到”、“谢谢”、“不客气”等）以及无业务含义的日常唠嗑。
2. 严格脱敏隐私信息：
   - 将所有具体的个人隐私与交易数据（包括具体订单号、用户姓名、手机号、具体收货地址、快递单号等）进行抽象化或脱敏处理，不得泄露用户和订单隐私。
3. 提炼标准通用问答：
   - 问法（question）应提炼为规范、清晰、符合广大顾客普遍搜索习惯的标准通用问题；
   - 回答（answer）应总结客服回复中的权威业务规则、操作指引或政策说明，语言精炼准确；
   - 若对话无实质业务问答（例如纯打招呼、纯重复催促或未能解决的无效聊天），输出空列表。
4. 强制输出标准 JSON 格式：
   - 输出必须为合法有效的 JSON 数组，严禁附带任何额外的开场白、解释或总结；
   - 格式规范示例：
[
  {
    "question": "退货运费由谁承担？",
    "answer": "商品质量问题由商家承担往返运费；个人原因退换货由买家自行承担寄回运费。"
  }
]
"""

qa_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_EXTRACTION_SYSTEM_PROMPT),
    ("human", "请从以下客服对话中抽取提炼通用问答对：\n\n{dialogue_text}"),
])


def parse_qa_extraction_output(text: str) -> List[ExtractedQAPair]:
    """解析大模型输出文本为 ExtractedQAPair 列表。
    
    支持标准 JSON 数组、包含 ```json 代码块的文本，以及包装在 {"items": [...]} 的格式。
    遇到解析失败或空输入时优雅容错返回空列表。
    """
    if not text or not text.strip():
        return []

    clean_text = text.strip()

    # 剥离 Markdown 代码块标记（如 ```json ... ``` 或 ``` ... ```）
    if "```" in clean_text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text, re.IGNORECASE)
        if match:
            clean_text = match.group(1).strip()

    # 尝试提取最外层的 JSON 结构（[...] 或 {...}）
    start_bracket = clean_text.find("[")
    start_brace = clean_text.find("{")

    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError:
        # 若直接反序列化失败，尝试子串截取
        if start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
            end_bracket = clean_text.rfind("]")
            if end_bracket != -1 and end_bracket > start_bracket:
                try:
                    data = json.loads(clean_text[start_bracket : end_bracket + 1])
                except json.JSONDecodeError:
                    logger.warning(f"无法从大模型输出中解析 JSON 数组: {text[:200]}")
                    return []
            else:
                return []
        elif start_brace != -1:
            end_brace = clean_text.rfind("}")
            if end_brace != -1 and end_brace > start_brace:
                try:
                    data = json.loads(clean_text[start_brace : end_brace + 1])
                except json.JSONDecodeError:
                    logger.warning(f"无法从大模型输出中解析 JSON 对象: {text[:200]}")
                    return []
            else:
                return []
        else:
            return []

    # 将解析出的结构归一化为 List[ExtractedQAPair]
    results: List[ExtractedQAPair] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                q = str(item["question"]).strip()
                a = str(item["answer"]).strip()
                if q and a:
                    results.append(ExtractedQAPair(question=q, answer=a))
    elif isinstance(data, dict):
        items = data.get("items") or data.get("qa_list") or data.get("data")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    q = str(item["question"]).strip()
                    a = str(item["answer"]).strip()
                    if q and a:
                        results.append(ExtractedQAPair(question=q, answer=a))

    return results
