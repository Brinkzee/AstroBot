import pytest
import os
from bs4 import BeautifulSoup


def test_frontend_conversation_delete_elements():
    """测试前端 index.html 是否包含会话删除按钮、样式及交互逻辑契约"""
    file_path = os.path.join("app", "static", "index.html")
    assert os.path.exists(file_path), "app/static/index.html 不存在"
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. 样式规则契约断言
    assert ".session-delete-btn" in html, "必须包含 .session-delete-btn 样式规则"
    assert ".session-item:hover .session-delete-btn" in html, "必须包含悬浮显示删除按钮的样式规则"

    # 2. 模板渲染契约断言
    assert 'class="session-delete-btn"' in html, "模板中必须渲染 session-delete-btn 元素"
    assert "deleteSession(event," in html, "模板中必须绑定 deleteSession(event, ...) 点击事件"
    assert 'title="删除此会话"' in html, "删除按钮必须包含删除提示 title"

    # 3. JS 逻辑函数契约断言
    assert "deleteSession" in html, "必须实现 deleteSession 函数"
    assert "stopPropagation" in html, "deleteSession 必须调用 event.stopPropagation() 阻止事件冒泡"
    assert "confirm(" in html, "deleteSession 必须包含 confirm 二次确认机制"
    assert "'DELETE'" in html or '"DELETE"' in html, "deleteSession 必须使用 DELETE 请求方法"
    assert "/api/conversations/" in html, "deleteSession 必须请求 /api/conversations/ 接口"
    assert "loadSessions" in html, "deleteSession 完成后必须刷新会话列表"


def test_frontend_conversation_delete_html_syntax():
    """确保 index.html 能够被 BeautifulSoup 正常解析，DOM 结构无致命语法损坏"""
    file_path = os.path.join("app", "static", "index.html")
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    assert soup.find("title") is not None
    assert "星光优选" in soup.find("title").text
