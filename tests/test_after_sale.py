from unittest.mock import MagicMock
import pytest

def test_after_sale_ticket_schema():
    from app.schemas.after_sale import AfterSaleTicket
    ticket = AfterSaleTicket(
        order_id="TB123456",
        issue_type="商品质量问题/破损",
        expected_solution="退货退款",
        raw_description="衣服破了个洞想退货"
    )
    assert ticket.order_id == "TB123456"
    assert ticket.issue_type == "商品质量问题/破损"
    assert ticket.expected_solution == "退货退款"
    assert ticket.raw_description == "衣服破了个洞想退货"

def test_after_sale_ticket_optional_order_id():
    from app.schemas.after_sale import AfterSaleTicket
    ticket = AfterSaleTicket(
        order_id=None,
        issue_type="物流延迟",
        expected_solution="催促物流",
        raw_description="物流三天没更新了"
    )
    assert ticket.order_id is None

def test_extract_after_sale_ticket_with_mock_model():
    from app.schemas.after_sale import AfterSaleTicket
    from app.services.after_sale_service import extract_after_sale_ticket

    mock_model = MagicMock()
    mock_structured = MagicMock()
    expected_ticket = AfterSaleTicket(
        order_id="ORD999",
        issue_type="商品质量问题/破损",
        expected_solution="换货",
        raw_description="鞋子开胶了"
    )
    mock_structured.invoke.return_value = expected_ticket
    mock_model.with_structured_output.return_value = mock_structured

    input_text = "订单ORD999鞋子开胶了要换货"
    result = extract_after_sale_ticket(input_text, model=mock_model)
    
    mock_model.with_structured_output.assert_called_once_with(AfterSaleTicket)
    assert result.order_id == "ORD999"
    assert result.issue_type == "商品质量问题/破损"
    assert result.expected_solution == "换货"
    assert result.raw_description == input_text

def test_eval_sample_dataset_definition():
    """评估集定义测试：提供5组典型电商售后真实描述样本，供真实模型联调与评测时验证"""
    from app.services.after_sale_service import EVALUATION_SAMPLES
    assert len(EVALUATION_SAMPLES) >= 5
    for sample in EVALUATION_SAMPLES:
        assert "description" in sample
        assert "expected_order_id" in sample
        assert "expected_issue_type" in sample
        assert "expected_solution" in sample
