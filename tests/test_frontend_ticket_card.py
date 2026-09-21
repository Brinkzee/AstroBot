import pytest
import os
from bs4 import BeautifulSoup


def test_frontend_ticket_preview_card_elements():
    """测试前端 index.html 是否完整包含工单预览卡片与 resume 流式交互契约"""
    file_path = os.path.join("app", "static", "index.html")
    assert os.path.exists(file_path), "app/static/index.html 不存在"
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. 契约关键字断言
    assert "ticket_preview" in html, "必须包含 ticket_preview 事件监听"
    assert "/api/chat/resume" in html, "必须包含 /api/chat/resume 恢复调用接口"
    assert "renderTicketPreviewCard" in html, "必须实现 renderTicketPreviewCard 渲染函数"
    assert "handleTicketDecision" in html, "必须实现 handleTicketDecision 决策函数"
    assert "resumeChatWorkflow" in html, "必须实现 resumeChatWorkflow 流式恢复处理函数"

    # 2. 按钮与样式类断言
    assert "btn-ticket-confirm" in html, "必须包含确认提交按钮类 btn-ticket-confirm"
    assert "btn-ticket-cancel" in html, "必须包含取消按钮类 btn-ticket-cancel"
    assert "ticket-preview-container" in html, "必须包含容器类 ticket-preview-container"
    assert "ticket-preview-card" in html, "必须包含卡片类 ticket-preview-card"
    assert "ticket-type-badge" in html, "必须包含工单类型标签"
    assert "ticket-preview-status" in html, "必须包含状态反馈类 ticket-preview-status"


def test_frontend_html_syntax_validity():
    """确保 index.html 能够被 BeautifulSoup 正常解析，DOM 结构无致命语法损坏"""
    file_path = os.path.join("app", "static", "index.html")
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    assert soup.find("title") is not None
    assert "星光优选" in soup.find("title").text
