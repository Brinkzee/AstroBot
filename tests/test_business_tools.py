import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.models import FAQ, Ticket
from app.tools.business_tools import (
    query_order,
    query_product,
    query_logistics,
    query_faq,
    create_ticket,
)


def test_query_order():
    res = query_order.invoke({"order_id": "1001"})
    assert "1001" in res
    assert "订单状态" in res
    assert "支付金额" in res
    assert "商品明细" in res
    data = json.loads(res)
    assert data["order_id"] == "1001"
    assert data["订单状态"] == "已发货"


def test_query_product():
    res = query_product.invoke({"product_id_or_name": "羽绒服"})
    assert "羽绒服" in res
    assert "库存" in res
    assert "价格" in res
    data = json.loads(res)
    assert "商品名称" in data
    assert "极简保暖羽绒服" in data["商品名称"]


def test_query_logistics():
    res = query_logistics.invoke({"order_id": "1001"})
    assert "1001" in res
    assert "物流" in res or "快递" in res
    assert "顺丰速运" in res
    data = json.loads(res)
    assert data["order_id"] == "1001"
    assert data["快递公司"] == "顺丰速运"
    assert "物流状态" in data


@pytest.mark.asyncio
async def test_query_faq_matched():
    mock_faq = FAQ(
        id=1,
        question="退货政策说明",
        answer="支持7天无理由退货，商品需保持原状及吊牌完好。",
        category="售后",
    )
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_faq]
    mock_session.execute.return_value = mock_result

    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None
    mock_session_local = MagicMock(return_value=mock_ctx)

    with patch("app.tools.business_tools.AsyncSessionLocal", mock_session_local):
        res = await query_faq.ainvoke({"keyword": "退货"})
        assert "退货政策说明" in res
        assert "支持7天无理由退货" in res
        mock_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_query_faq_not_found():
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_session.execute.return_value = mock_result

    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None
    mock_session_local = MagicMock(return_value=mock_ctx)

    with patch("app.tools.business_tools.AsyncSessionLocal", mock_session_local):
        res = await query_faq.ainvoke({"keyword": "宇宙飞船"})
        assert "未找到与【宇宙飞船】相关的常见问题解答。" in res


@pytest.mark.asyncio
async def test_query_faq_dense_vector_retrieval():
    """测试 query_faq 命中 Dense 向量检索时的正常响应流程"""
    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(return_value=[
        {
            "id": 1,
            "distance": 0.95,
            "questions": "如何申请退货退款？",
            "answer": "联系在线客服并提交申请，我们将在24小时内审核。",
        }
    ])
    mock_retriever.format_faq_hits.return_value = (
        "1. 问：如何申请退货退款？\n   答：联系在线客服并提交申请，我们将在24小时内审核。"
    )

    with patch("app.tools.business_tools.get_retriever", return_value=mock_retriever):
        res = await query_faq.ainvoke({"keyword": "退款"})
        assert "如何申请退货退款？" in res
        assert "联系在线客服并提交申请" in res
        mock_retriever.retrieve.assert_awaited_once_with("退款", top_k=3)


@pytest.mark.asyncio
async def test_create_ticket_success():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None
    mock_session_local = MagicMock(return_value=mock_ctx)

    with patch("app.tools.business_tools.AsyncSessionLocal", mock_session_local):
        res = await create_ticket.ainvoke({
            "conversation_id": 1,
            "description": "收到商品尺码不合适，申请换货",
            "ticket_type": "售后",
        })
        assert "工单已创建，人工客服将在24小时内跟进处理" in res
        data = json.loads(res)
        assert data["conversation_id"] == 1
        assert data["ticket_type"] == "售后"
        assert data["ticket_no"].startswith("T")
        mock_session.add.assert_called_once()
        added_ticket = mock_session.add.call_args[0][0]
        assert isinstance(added_ticket, Ticket)
        assert added_ticket.conversation_id == 1
        assert added_ticket.description == "收到商品尺码不合适，申请换货"
        assert added_ticket.ticket_type == "售后"
        mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_ticket_invalid_type_fallback():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None
    mock_session_local = MagicMock(return_value=mock_ctx)

    with patch("app.tools.business_tools.AsyncSessionLocal", mock_session_local):
        res = await create_ticket.ainvoke({
            "conversation_id": 2,
            "description": "未知分类的问题反馈",
            "ticket_type": "非法类型",
        })
        data = json.loads(res)
        assert data["ticket_type"] == "售后"
        added_ticket = mock_session.add.call_args[0][0]
        assert added_ticket.ticket_type == "售后"


def test_tool_metadata_and_docstrings():
    tools = [query_order, query_product, query_logistics, query_faq, create_ticket]
    names = [t.name for t in tools]
    assert names == ["query_order", "query_product", "query_logistics", "query_faq", "create_ticket"]
    for t in tools:
        assert t.description is not None and len(t.description.strip()) > 0
