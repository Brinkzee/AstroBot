"""会话事实摘要提炼提示词与模板定义。

规约约束：
1. 仅提炼关键业务事实与诉求（商品型号、明确报出过的订单号/手机号、核心诉求、未解决问题）；
2. 绝对真实性 (Hard Gate)：对话中未提及的信息严禁编造、推测或扩展；
3. 剔除一切客套寒暄与礼貌套话（如“您好”、“谢谢”、“很高兴为您服务”等）；
4. 长度约束：字数严格控制在 50 ~ 200 字之间，采用紧凑的事实陈述句；
5. 旧梗概隔离：已有前情背景仅作理解参考，严禁重复输出或合并改写已有事实，仅提炼当前片段的新事实。
"""

from typing import Optional

SUMMARY_SYSTEM_PROMPT = """你是一名严谨的电商客服会话事实摘要专家。
你的任务是将以下一段客服与用户的中间轮次对话，提炼为精炼的事实梗概。

【提炼规则】
1. 仅提炼关键业务事实与诉求：
   - 用户咨询或提及的具体商品型号、品类；
   - 用户明确报出过的订单号、手机号、快递单号；
   - 用户的核心业务诉求（退款、催发货、查保修、换货等）；
   - 尚未解决的卡点或已达成的明确结论。
2. 绝对真实性 (Hard Gate)：严禁编造任何未出现的事实，对话中未提及的信息一个字都不许编造、推测或扩展！
3. 剔除一切客套寒暄：剔除所有寒暄与礼貌套话，严禁出现“您好”、“谢谢”、“很高兴为您服务”等无意义客套礼貌辞令。
4. 篇幅约束：字数严格控制在 50 ~ 200 字之间，采用紧凑的事实陈述句。
5. 前情背景隔离：旧梗概（已有前情背景）仅作为背景材料参考，严禁重复输出已有事实，严禁合并改写已有事实，仅聚焦提炼当前待压缩片段中的新事实。
"""

SUMMARY_USER_PROMPT_TEMPLATE = """【已有前情背景（仅作理解参考，严禁重复输出或合并改写）】
{existing_summary}

【待压缩对话片段】
{dialogue_text}
"""


def format_summary_prompt(
    dialogue_text: str,
    existing_summary: Optional[str] = None,
) -> str:
    """格式化摘要提炼的用户输入 Prompt。"""
    summary_bg = (
        existing_summary.strip()
        if existing_summary and existing_summary.strip()
        else "暂无"
    )
    return SUMMARY_USER_PROMPT_TEMPLATE.format(
        existing_summary=summary_bg,
        dialogue_text=dialogue_text.strip(),
    )


format_summary_user_prompt = format_summary_prompt

__all__ = [
    "SUMMARY_SYSTEM_PROMPT",
    "SUMMARY_USER_PROMPT_TEMPLATE",
    "format_summary_prompt",
    "format_summary_user_prompt",
]
