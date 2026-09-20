"""Tests for Chapter 10 Acceptance API system and read-only endpoints.

Tests:
- GET /api/acceptance/overview (9 gates with pass/fail/missing, passed_count, service status)
- GET /api/acceptance/eval (micro vs macro, 17 classes, candidate scores, confusion matrices)
- GET /api/acceptance/data (dataset inventory, zero-leakage hard gate, training & ONNX status)
- GET /api/acceptance/errors (3-way error accounting, error samples, boundary notes)
- GET /api/acceptance/service (live healthz probe)
- POST /api/acceptance/classify (proxy :8110/classify single query)
- Graceful degradation on missing artifacts (present=false + make_target, never 500)
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import ASGITransport, AsyncClient
from main import app


@pytest.mark.asyncio
async def test_acceptance_overview_structure_and_gates():
    """Verify GET /api/acceptance/overview returns 9 gates with 3-state status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/acceptance/overview")
        assert res.status_code == 200
        data = res.json()

        assert "gates" in data
        assert "passed_count" in data
        assert "total_count" in data
        assert data["total_count"] == 9
        assert len(data["gates"]) == 9
        assert "service_status" in data

        expected_gate_ids = [
            "data_leakage",
            "f1_redlines",
            "error_accounting",
            "threshold_replay",
            "onnx_consistency",
            "service_live",
            "batch_classification",
            "job_security",
            "e2e_regression",
        ]

        actual_gate_ids = [g["id"] for g in data["gates"]]
        assert actual_gate_ids == expected_gate_ids

        # Verify each gate has required fields and valid 3-state status
        for gate in data["gates"]:
            assert "id" in gate
            assert "name" in gate
            assert "status" in gate
            assert gate["status"] in ("pass", "fail", "missing")
            assert "make_target" in gate
            assert "job_name" in gate
            assert "description" in gate
            assert "detail" in gate

        # Check passed_count arithmetic
        computed_passed = sum(1 for g in data["gates"] if g["status"] == "pass")
        assert data["passed_count"] == computed_passed


@pytest.mark.asyncio
async def test_acceptance_overview_missing_artifacts_gives_missing_not_fail():
    """Verify that when artifacts are missing, gate status is 'missing', NEVER 'fail'."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Patch Path.exists to return False for all ch10 artifact files
        real_exists = Path.exists

        def fake_exists(path_obj):
            p_str = str(path_obj).replace("\\", "/")
            if "reports/ch10_evaluation_report.json" in p_str or "mewhelp-ch10-dataset" in p_str or "data/ch10" in p_str:
                return False
            return real_exists(path_obj)

        with patch.object(Path, "exists", fake_exists):
            res = await client.get("/api/acceptance/overview")
            assert res.status_code == 200
            data = res.json()

            gate_map = {g["id"]: g for g in data["gates"]}
            # When files are missing, these must be 'missing', NOT 'fail'
            assert gate_map["data_leakage"]["status"] == "missing"
            assert gate_map["f1_redlines"]["status"] == "missing"
            assert gate_map["error_accounting"]["status"] == "missing"
            assert gate_map["threshold_replay"]["status"] == "missing"
            assert gate_map["onnx_consistency"]["status"] == "missing"


@pytest.mark.asyncio
async def test_acceptance_eval_endpoint_present():
    """Verify GET /api/acceptance/eval returns evaluation data when report exists."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/acceptance/eval")
        assert res.status_code == 200
        data = res.json()

        assert "present" in data
        if data["present"]:
            assert "summary" in data
            assert "macro_f1" in data["summary"]
            assert "micro_f1" in data["summary"]
            assert "per_class" in data
            assert len(data["per_class"]) == 17
            assert "threshold" in data
            assert "candidate_scores" in data
            assert "confusion_matrices" in data
            assert "error_accounting" in data


@pytest.mark.asyncio
async def test_acceptance_eval_endpoint_missing_graceful():
    """Verify GET /api/acceptance/eval returns present=false + make_target when report is missing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        real_exists = Path.exists

        def fake_exists(path_obj):
            p_str = str(path_obj).replace("\\", "/")
            if "ch10_evaluation_report.json" in p_str:
                return False
            return real_exists(path_obj)

        with patch.object(Path, "exists", fake_exists):
            res = await client.get("/api/acceptance/eval")
            assert res.status_code == 200
            data = res.json()
            assert data["present"] is False
            assert "make_target" in data
            assert data["make_target"] == "make eval-ch10"


@pytest.mark.asyncio
async def test_acceptance_data_endpoint_present():
    """Verify GET /api/acceptance/data returns dataset and artifacts inventory."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/acceptance/data")
        assert res.status_code == 200
        data = res.json()

        assert "present" in data
        if data["present"]:
            assert "dataset" in data
            assert "train_count" in data["dataset"]
            assert "val_count" in data["dataset"]
            assert "test_count" in data["dataset"]
            assert "leakage_check" in data
            assert "passed" in data["leakage_check"]
            assert data["leakage_check"]["passed"] is True
            assert "training_artifacts" in data
            assert "onnx_artifacts" in data


@pytest.mark.asyncio
async def test_acceptance_data_endpoint_missing_graceful():
    """Verify GET /api/acceptance/data returns present=false + make_target when dataset is missing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        real_exists = Path.exists

        def fake_exists(path_obj):
            p_str = str(path_obj).replace("\\", "/")
            if "mewhelp-ch10-dataset" in p_str:
                return False
            return real_exists(path_obj)

        with patch.object(Path, "exists", fake_exists):
            res = await client.get("/api/acceptance/data")
            assert res.status_code == 200
            data = res.json()
            assert data["present"] is False
            assert "make_target" in data
            assert data["make_target"] == "make data-prep-ch10"


@pytest.mark.asyncio
async def test_acceptance_errors_endpoint_present():
    """Verify GET /api/acceptance/errors returns error accounting and sample details."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/acceptance/errors")
        assert res.status_code == 200
        data = res.json()

        assert "present" in data
        if data["present"]:
            assert "error_accounting" in data
            acct = data["error_accounting"]
            assert "under_tag_count" in acct
            assert "over_tag_count" in acct
            assert "displaced_count" in acct
            assert "total_penalty" in acct
            assert "confusion_matrix_sum" in acct
            assert "balanced" in acct
            assert acct["balanced"] is True
            assert "error_samples" in data
            assert isinstance(data["error_samples"], list)


@pytest.mark.asyncio
async def test_acceptance_errors_endpoint_missing_graceful():
    """Verify GET /api/acceptance/errors returns present=false + make_target when missing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        real_exists = Path.exists

        def fake_exists(path_obj):
            p_str = str(path_obj).replace("\\", "/")
            if "ch10_evaluation_report.json" in p_str or "error_samples.md" in p_str:
                return False
            return real_exists(path_obj)

        with patch.object(Path, "exists", fake_exists):
            res = await client.get("/api/acceptance/errors")
            assert res.status_code == 200
            data = res.json()
            assert data["present"] is False
            assert "make_target" in data
            assert data["make_target"] == "make eval-ch10"


@pytest.mark.asyncio
async def test_acceptance_service_probe_online_and_offline():
    """Verify GET /api/acceptance/service returns probe details when online and offline."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Case 1: Online mock
        with patch("app.api.acceptance.query_classifier_healthz", new_callable=AsyncMock) as mock_probe:
            mock_probe.return_value = {
                "online": True,
                "status": "ok",
                "details": {"model_loaded": True, "categories_count": 17, "threshold": 0.45},
            }
            res = await client.get("/api/acceptance/service")
            assert res.status_code == 200
            data = res.json()
            assert data["online"] is True
            assert data["status"] == "ok"
            assert data["details"]["model_loaded"] is True

        # Case 2: Offline mock
        with patch("app.api.acceptance.query_classifier_healthz", new_callable=AsyncMock) as mock_probe:
            mock_probe.return_value = {
                "online": False,
                "status": "offline",
                "error": "Connection refused",
                "make_target": "make classifier-up",
            }
            res = await client.get("/api/acceptance/service")
            assert res.status_code == 200
            data = res.json()
            assert data["online"] is False
            assert data["status"] == "offline"
            assert data["make_target"] == "make classifier-up"


@pytest.mark.asyncio
async def test_acceptance_classify_proxy_success():
    """Verify POST /api/acceptance/classify proxies single query to :8110/classify."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        mock_classify_response = {
            "text": "买大了想退",
            "labels": ["尺码", "退换货"],
            "scores": {"尺码": 0.88, "退换货": 0.92, "物流": 0.05},
        }

        with patch("app.api.acceptance.proxy_classify_request", new_callable=AsyncMock) as mock_proxy:
            mock_proxy.return_value = mock_classify_response
            res = await client.post("/api/acceptance/classify", json={"text": "买大了想退"})
            assert res.status_code == 200
            data = res.json()
            assert data["text"] == "买大了想退"
            assert "labels" in data
            assert "尺码" in data["labels"]
            assert "退换货" in data["labels"]
            assert "scores" in data


@pytest.mark.asyncio
async def test_acceptance_classify_proxy_service_offline_graceful():
    """Verify POST /api/acceptance/classify returns 503 with make_target when service is offline."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("app.api.acceptance.proxy_classify_request", new_callable=AsyncMock) as mock_proxy:
            from fastapi import HTTPException
            mock_proxy.side_effect = HTTPException(
                status_code=503,
                detail="推理服务未启动，请先运行 make classifier-up",
            )
            res = await client.post("/api/acceptance/classify", json={"text": "买大了想退"})
            assert res.status_code == 503
            data = res.json()
            assert "make classifier-up" in data["detail"]


@pytest.mark.asyncio
async def test_topics_distribution_endpoint_structure():
    """Verify GET /api/topics/distribution returns complete 17-class statistics."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/topics/distribution")
        assert res.status_code == 200
        data = res.json()

        assert "total_classified" in data
        assert "total_unclassified" in data
        assert "total_pool" in data
        assert "top_3_topics" in data
        assert "top_1_topic" in data
        assert "distribution" in data
        assert len(data["distribution"]) == 17

        for item in data["distribution"]:
            assert "category" in item
            assert "count" in item
            assert "percentage" in item
            assert "tier" in item
            assert "is_top3" in item
            assert "priority_hint" in item

        # Also verify alias endpoint
        alias_res = await client.get("/api/acceptance/topics/distribution")
        assert alias_res.status_code == 200
        alias_data = alias_res.json()
        assert len(alias_data["distribution"]) == 17


@pytest.mark.asyncio
async def test_topics_distribution_with_mocked_data():
    """Verify topic distribution calculation with mock DB session."""
    from app.db.session import get_db

    mock_session = AsyncMock()
    # Mock class_res.scalars().all() -> sample labels
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [
        ["尺码", "退换货"],
        ["退换货"],
        ["物流"],
        ["尺码", "物流"],
        ["退换货", "发票"],
    ]
    mock_class_res = MagicMock()
    mock_class_res.scalars.return_value = mock_scalars

    # Mock unclass_res.scalar_one() -> 10 unclassified
    mock_unclass_res = MagicMock()
    mock_unclass_res.scalar_one.return_value = 10

    mock_session.execute.side_effect = [mock_class_res, mock_unclass_res]

    app.dependency_overrides[get_db] = lambda: mock_session
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get("/api/topics/distribution")
            assert res.status_code == 200
            data = res.json()

            assert data["total_classified"] == 5
            assert data["total_unclassified"] == 10
            assert data["total_pool"] == 15
            # "退换货": 3, "尺码": 2, "物流": 2, "发票": 1
            assert "退换货" in data["top_3_topics"]
            assert data["top_1_topic"] == "退换货"

            dist_map = {item["category"]: item for item in data["distribution"]}
            assert dist_map["退换货"]["count"] == 3
            assert dist_map["退换货"]["is_top3"] is True
            assert dist_map["退换货"]["priority_hint"] == "知识库优先补充重点"
            assert dist_map["退换货"]["tier"] == "strict"

            assert dist_map["尺码"]["count"] == 2
            assert dist_map["物流"]["count"] == 2
            assert dist_map["发票"]["count"] == 1
            assert dist_map["账号"]["count"] == 0
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_proxy_classify_request_trust_env_and_error_handling():
    """Verify proxy_classify_request handles offline/502 without exposing raw proxy error."""
    from app.api.acceptance import proxy_classify_request
    from fastapi import HTTPException

    # 1. 模拟 502 Bad Gateway (代理拦截或未就绪)，应转译为 503 并提示 make classifier-up
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.text = ""
        mock_post.return_value = mock_resp

        with pytest.raises(HTTPException) as exc_info:
            await proxy_classify_request("买大了想退")
        assert exc_info.value.status_code == 503
        assert "make classifier-up" in exc_info.value.detail

    # 2. 真实离线端口请求，应捕获 ConnectError/NetworkError 并抛出友好 503
    with pytest.raises(HTTPException) as exc_info:
        await proxy_classify_request("买大了想退", port=59999)
    assert exc_info.value.status_code == 503
    assert "make classifier-up" in exc_info.value.detail
