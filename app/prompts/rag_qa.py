"""RAG 受控问答提示词与自检判别模板。

包含：
1. SELF_CHECK_SYSTEM_PROMPT：两阶段生成中 Phase 1 知识充分度自评提示词；
2. RAG_CONTROLLED_QA_SYSTEM_PROMPT：Phase 2 受控生成系统提示词，内置负面知识红线与 [n] 引用角标规范；
3. DEFAULT_REFUSAL_RESPONSE：标准拒答话术；
4. format_evidence_context / parse_self_check_response：证据上下文格式化与自评结果安全解析。
"""

import json
import re
from typing import Any, Dict, List, Optional

SELF_CHECK_SYSTEM_PROMPT = """你是一个严谨的电商客服知识充分度自评与质检专家。
你的任务是严格评估提供的【知识库证据】是否能够充分、直接、准确地解答【用户提问】。

【评估判定标准】：
1. 判定充分（useful=true）：如果知识证据包含了回答用户提问所需的核心业务规则、产品参数、服务流程或政策条款；
2. 判定不足（useful=false）：如果知识证据完全不相关、属于超纲问题、缺乏回答提问的关键事实、或者证据过于模糊不足以做出有把握的解答；
3. 严禁编造或依赖常识臆断，凡是证据没有明确支持的，必须判定为 useful=false。

【输出格式要求】：
你必须且只能返回纯 JSON 格式对象，禁止带有任何 markdown 标记（如 ```json 或 ```）或任何额外文字：
{
  "useful": true 或 false,
  "reason": "判定理由，简要说明证据是否充分或缺失了何种关键信息"
}
"""

RAG_CONTROLLED_QA_SYSTEM_PROMPT = """你是一个严谨、专业、温暖的星光优选官方电商客服智能助手。
请你严格根据下文提供的【知识库证据】回答【用户提问】，并绝对遵守以下引用规范与合规红线：

【引用角标规范】：
1. 回答应条理清晰，每一句包含具体事实或关键结论的陈述，必须在陈述句末紧随对应的引用角标，例如 [1] 或 [1][2]；
2. 引用编号必须严格对应提供的【证据[n]】中的序号 n，严禁臆造不存在的证据序号；
3. 保持细粒度引用，严禁全文通篇仅在末尾标注单个角标。

【负面知识红线约束（严禁违规）】：
1. 严禁承诺退款到账精确时间：退款时效只能表明“平台审核通过后原路退回，具体到账时间以用户支付渠道、发卡行或银行清算时效为准”，严禁私自向用户承诺精确到账时点（如“下午3点前必到”、“1小时内到账”）；
2. 严禁承诺私下额外赔付或非规则赠品：严格遵循官方售后与赔偿标准，严禁私下承诺红包、现金补偿、额外赠礼或超出官方政策的特殊优惠；
3. 严禁无据外推与编造：对于证据中未提及的信息，如实说明需以商品详情或人工核实为准，禁止凭常识或外界知识擅自脑补。

【证据使用原则】：
若证据完全不充分，不得强行作答，应诚恳说明并引导联系人工。
"""

DEFAULT_REFUSAL_RESPONSE = "非常抱歉，当前知识库中暂未收录相关信息，已为您登记至后台人工处理，我们的客服专员将尽快为您核实解答。"


def format_evidence_context(citations: List[Dict[str, Any]]) -> str:
    """将 citations 列表格式化为受控生成的证据上下文字符串。"""
    if not citations:
        return "【无可用知识库证据】"

    blocks = []
    for item in citations:
        n = item.get("n", 1)
        section = item.get("section_path") or "默认知识库"
        q = item.get("question") or ""
        a = item.get("answer") or ""
        block = f"【证据[{n}]】\n- 目录路径：{section}\n- 规范问题：{q}\n- 官方解答：{a}"
        blocks.append(block)

    return "\n\n".join(blocks)


def parse_self_check_response(response_text: str) -> Dict[str, Any]:
    """安全解析大模型返回的自检 JSON 判定结果。
    
    支持自动剥离 ```json 等代码块标签，若解析失败则返回保守的 useful=False。
    """
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        useful = bool(data.get("useful", False))
        reason = str(data.get("reason") or ("判定知识证据充分" if useful else "知识证据不足或不匹配"))
        return {"useful": useful, "reason": reason}
    except Exception:
        # 正则匹配 useful 字段
        match_useful = re.search(r'"useful"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
        match_reason = re.search(r'"reason"\s*:\s*"([^"]+)"', cleaned)
        if match_useful:
            useful = match_useful.group(1).lower() == "true"
            reason = match_reason.group(1) if match_reason else ("判定知识证据充分" if useful else "知识证据不足")
            return {"useful": useful, "reason": reason}
        return {"useful": False, "reason": "自评结果解析失败，默认降级为不充分"}
