import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def test_system_prompt_core_constraints():
    from app.prompts.customer_service import CUSTOMER_SERVICE_SYSTEM_PROMPT
    # 角色人设验证
    assert "星光优选" in CUSTOMER_SERVICE_SYSTEM_PROMPT
    assert "客服" in CUSTOMER_SERVICE_SYSTEM_PROMPT
    # 礼貌服务规范
    assert "礼貌" in CUSTOMER_SERVICE_SYSTEM_PROMPT or "热情" in CUSTOMER_SERVICE_SYSTEM_PROMPT
    # 业务边界约束
    assert "电商" in CUSTOMER_SERVICE_SYSTEM_PROMPT
    # 售后权责红线约束：索要凭证、不私自承诺违规现金赔偿、引导正规流程
    assert "凭证" in CUSTOMER_SERVICE_SYSTEM_PROMPT or "订单号" in CUSTOMER_SERVICE_SYSTEM_PROMPT
    assert "承诺" in CUSTOMER_SERVICE_SYSTEM_PROMPT
    assert "流程" in CUSTOMER_SERVICE_SYSTEM_PROMPT or "申请" in CUSTOMER_SERVICE_SYSTEM_PROMPT

def test_prompt_template_structure_and_formatting():
    from app.prompts.customer_service import customer_service_prompt
    history = [
        HumanMessage(content="你好，我在你们这买了件衣服"),
        AIMessage(content="您好！我是星光优选智能客服小星，请问有什么可以帮您？")
    ]
    user_input = "请问尺码偏大还是偏小？"
    
    messages = customer_service_prompt.format_messages(history=history, input=user_input)
    
    assert len(messages) == 4
    # 首条必须是 SystemMessage
    assert isinstance(messages[0], SystemMessage)
    assert "小星" in messages[0].content
    # 接着是历史消息
    assert messages[1] == history[0]
    assert messages[2] == history[1]
    # 最后一条是当前提问
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == user_input

def test_eval_sample_scenarios():
    """评估集验证：拿3组标注场景测试 Prompt 是否包含对不同场景的应对指令要求"""
    from app.prompts.customer_service import CUSTOMER_SERVICE_SYSTEM_PROMPT
    eval_cases = [
        {"scenario": "破损质量问题", "expected_keywords": ["质量问题", "情绪", "凭证"]},
        {"scenario": "越权索赔", "expected_keywords": ["承诺", "赔付"]},
        {"scenario": "业务边界", "expected_keywords": ["电商", "拒绝"]}
    ]
    for case in eval_cases:
        for kw in case["expected_keywords"]:
            assert kw in CUSTOMER_SERVICE_SYSTEM_PROMPT, f"场景 '{case['scenario']}' 缺少关键指令要求: '{kw}'"
