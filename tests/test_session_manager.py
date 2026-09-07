import pytest
from langchain_core.messages import HumanMessage, AIMessage

def test_session_lifecycle():
    from app.services.session_manager import SessionManager
    sm = SessionManager()
    
    # 自动生成 session_id
    sid = sm.get_or_create_session()
    assert sid is not None
    assert len(sid) > 0
    assert sm.get_history(sid) == []

    # 指定 session_id
    sid2 = sm.get_or_create_session("custom-session-1")
    assert sid2 == "custom-session-1"

    # 追加消息
    sm.add_user_message(sid2, "你好")
    sm.add_ai_message(sid2, "您好，我是客服小星！")
    history = sm.get_history(sid2)
    assert len(history) == 2
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "你好"
    assert isinstance(history[1], AIMessage)
    assert history[1].content == "您好，我是客服小星！"

def test_session_trimming_under_token_budget():
    from app.services.session_manager import SessionManager, count_tokens_fallback
    sm = SessionManager()
    sid = sm.get_or_create_session("trim-test-session")

    # 注入 10 轮对话（每条约 15 tokens，10轮共20条，总约 300+ tokens）
    for i in range(10):
        sm.add_user_message(sid, f"用户提问第{i}轮：羽绒服多少钱？")
        sm.add_ai_message(sid, f"客服回复第{i}轮：现价399元，支持保价。")

    raw_history = sm.get_history(sid)
    assert len(raw_history) == 20
    total_tokens = count_tokens_fallback(raw_history)
    assert total_tokens > 200

    # 设定 token 预算为 120 tokens，进行滑动窗口裁剪
    trimmed = sm.get_trimmed_history(sid, max_tokens=120)

    # 验证裁剪生效
    assert len(trimmed) < 20
    assert len(trimmed) >= 2
    # 裁剪后总 token 不应超过预算
    assert count_tokens_fallback(trimmed) <= 120
    # 按照 trim_messages(strategy='last', start_on='human')，第一条必须是 HumanMessage
    assert isinstance(trimmed[0], HumanMessage)
    # 保留的是最近的对话（如第8、9轮）
    last_msg = trimmed[-1]
    assert isinstance(last_msg, AIMessage)
    assert "第9轮" in last_msg.content

def test_session_clear():
    from app.services.session_manager import SessionManager
    sm = SessionManager()
    sid = sm.get_or_create_session("to-clear")
    sm.add_user_message(sid, "test")
    assert len(sm.get_history(sid)) == 1
    sm.clear_session(sid)
    assert sm.get_history(sid) == []
