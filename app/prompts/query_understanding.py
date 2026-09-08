"""用户查询理解与口语改写归一 Prompt 与解析工具。

职责：
1. 口语改写归一 (standard_query)：
   - 将用户的口语化、情绪化长句（如“我买了那个pro x99的手表，要是用着不顺心能退吗”）改写为专业、规范的电商标准问法（如“星光PRO-X99智能手表退换货政策与流程”）；
   - 提取并规范化专有名词与核心型号（如“pro x99” -> “PRO-X99”）；
   - 彻底剔除语气助词（如“啊”、“呢”、“吧”、“嘛”）与冗余修饰。
2. 检索同义词与关键词扩展 (expanded_keywords)：
   - 抽取 1 个核心型号/专有名词，以及 2~4 个强相关的精准同义词、行业近义词（如 ["PRO-X99", "退货", "换货", "退换货"]）；
   - 严禁宽泛无据发散。
3. 强制输出结构化 JSON 规范。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


QUERY_UNDERSTANDING_SYSTEM_PROMPT = """你是由“星光优选”电商平台研发的智能搜索与知识问答意图理解专家。
你的核心任务是对用户的口语化长句提问进行【口语改写归一】与【检索侧同义词扩展】，以便精准检索知识库。

【处理规则与规范】
1. 口语改写归一（standard_query）：
   - 消除用户的口语化用词、情绪化表达（如“用着不顺心”、“我买了那个”、“能退吗”、“坏了咋办”）；
   - 规范化专有名词与产品型号（例如：“pro x99” 归一为 “PRO-X99” 或标准型号，“手表” 归一为 “智能手表”）；
   - 剔除语气助词（如“啊”、“呢”、“吧”、“嘛”）与无业务含义的日常唠嗑；
   - 归纳为符合企业知识库索引规范的标准提问（例如：“星光PRO-X99智能手表退换货政策与流程”）。

2. 检索同义词与关键词扩展（expanded_keywords）：
   - 提炼 1 个核心专有名词/产品型号（例如：“PRO-X99”）；
   - 提取与用户搜索意图强相关的 2~4 个精准同义词、近义词或行业标准词（例如：["PRO-X99", "退货", "换货", "退换货"]）；
   - 严禁随意发散或扩展无关词汇，严格控制在 2~4 个精准词。

3. 强制输出标准 JSON 格式：
   - 输出必须为合法有效的 JSON 对象，严禁附带任何开场白、代码解释或多余文字；
   - 格式规范示例：
{
  "standard_query": "星光PRO-X99智能手表退换货政策与流程",
  "expanded_keywords": ["PRO-X99", "退货", "换货", "退换货"]
}
"""

query_understanding_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=QUERY_UNDERSTANDING_SYSTEM_PROMPT),
    ("human", "请对以下用户查询进行口语改写归一与同义词扩展：\n\n{query}"),
])


def parse_query_understanding_output(text: str) -> Dict[str, Any]:
    """解析大模型输出文本为查询理解字典。
    
    返回字典结构：
    {
        "standard_query": str,
        "expanded_keywords": List[str]
    }
    
    支持标准 JSON、Markdown ```json ... ``` 代码块包裹以及异常容错解析。
    若解析失败或输入为空，返回空字典 {}。
    """
    if not text or not str(text).strip():
        return {}

    clean_text = str(text).strip()

    # 1. 剥离 Markdown 代码块标记（```json ... ``` 或 ``` ... ```）
    if "```" in clean_text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text, re.IGNORECASE)
        if match:
            clean_text = match.group(1).strip()

    # 2. 截取最外层 JSON 对象花括号
    start_brace = clean_text.find("{")
    end_brace = clean_text.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        clean_text = clean_text[start_brace : end_brace + 1]

    data = None
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError:
        # 针对大模型末尾多逗号或带单引号的容错正则提取
        standard_match = re.search(r'["\']standard_query["\']\s*:\s*["\']([^"\']+)["\']', clean_text)
        keywords_match = re.search(r'["\']expanded_keywords["\']\s*:\s*\[([^\]]*)\]', clean_text)
        if standard_match:
            sq = standard_match.group(1).strip()
            kws: List[str] = []
            if keywords_match:
                kws = [
                    k.strip()
                    for k in re.findall(r'["\']([^"\']+)["\']', keywords_match.group(1))
                    if k.strip()
                ]
            return {
                "standard_query": sq,
                "expanded_keywords": kws,
            }
        logger.warning(f"无法从大模型输出中解析有效 JSON: {text[:200]}")
        return {}

    # 若为数组包装，取首个对象
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        data = data[0]

    if not isinstance(data, dict):
        return {}

    standard_query = str(data.get("standard_query", "")).strip()
    raw_keywords = data.get("expanded_keywords", [])

    expanded_keywords: List[str] = []
    if isinstance(raw_keywords, list):
        expanded_keywords = [
            str(k).strip() for k in raw_keywords if k is not None and str(k).strip()
        ]
    elif isinstance(raw_keywords, str) and raw_keywords.strip():
        expanded_keywords = [
            k.strip() for k in re.split(r"[,，\s]+", raw_keywords.strip()) if k.strip()
        ]

    return {
        "standard_query": standard_query,
        "expanded_keywords": expanded_keywords,
    }
