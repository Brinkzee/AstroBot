import json
from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient, ASGITransport
from main import app
from scripts.wsl_helper import ensure_mysql_ready


@pytest.fixture(scope="module", autouse=True)
def setup_mysql_ready():
    """保证 WSL2 MySQL 容器已拉起且保活进程处于激活状态"""
    ensure_mysql_ready(verbose=False)


@pytest.mark.asyncio
async def test_create_ticket_api_success():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/tickets",
            json={
                "conversation_id": 9999,
                "description": "商品存在破损，申请退款被拒",
                "ticket_type": "投诉",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticket_no"].startswith("T")
        assert data["ticket_type"] == "投诉"
        assert "24小时" in data["status"]


@pytest.mark.asyncio
async def test_create_ticket_api_defaults():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/tickets",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticket_no"].startswith("T")
        assert data["ticket_type"] == "投诉"
        assert "24小时" in data["status"]


@pytest.mark.asyncio
async def test_create_ticket_api_error_returns_400():
    transport = ASGITransport(app=app)
    mock_tool = AsyncMock()
    mock_tool.ainvoke = AsyncMock(return_value=json.dumps({"error": "模拟工单数据库入库失败"}))
    with patch("app.api.routes.create_ticket", mock_tool):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/tickets",
                json={"conversation_id": 1234, "description": "测试异常"},
            )
            assert resp.status_code == 400
            assert resp.json()["detail"] == "模拟工单数据库入库失败"

