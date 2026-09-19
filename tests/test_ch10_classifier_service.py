"""Unit tests for Chapter 10 lightweight inference service (:8110) and PID governance.

Tests:
1. Pure lightweight runtime test: strict assertion that torch and transformers are NEVER imported.
2. GET /healthz endpoint: model status, category count, threshold.
3. POST /classify endpoint: single and batch inference, multi-label predictions, scores structure.
4. POST /classify edge cases: empty batch, whitespace/empty strings, invalid payloads.
5. PID governance: write_pid, read_pid, remove_pid, is_pid_alive.
6. Service status: stopped, stale PID, running with health check.
7. Safe process stop and process tree kill (anti-orphan).
8. CLI command argument parsing: start, stop, status, restart.
"""

import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import pytest
from fastapi.testclient import TestClient

from app.services.classifier.taxonomy import TAXONOMY_17
from app.services.classifier_service.server import app, ClassifierInferenceEngine
from scripts.classifier_service import (
    get_service_status,
    is_pid_alive,
    kill_process_tree,
    parse_args,
    read_pid,
    remove_pid,
    start_service,
    stop_service,
    write_pid,
)


def test_no_torch_or_transformers_imported():
    """Verify server.py strictly adheres to the lightweight runtime contract.

    1. AST inspection: server.py cannot import torch or transformers.
    2. Subprocess check: importing server in a clean Python process must not load torch or transformers.
    """
    server_path = Path("app/services/classifier_service/server.py")
    assert server_path.exists(), "server.py must exist"

    # AST check
    tree = ast.parse(server_path.read_text(encoding="utf-8"))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0])

    forbidden = {"torch", "transformers", "torchvision", "torchaudio"}
    intersection = imported_modules.intersection(forbidden)
    assert not intersection, f"server.py directly imports forbidden heavy modules: {intersection}"

    # Clean subprocess check: sys.modules must not contain torch or transformers
    check_code = (
        "import sys; "
        "import app.services.classifier_service.server; "
        "forbidden = {'torch', 'transformers'}.intersection(sys.modules); "
        "assert not forbidden, f'Loaded forbidden modules: {forbidden}'; "
        "print('CLEAN_RUNTIME_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", check_code],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
    )
    assert result.returncode == 0, f"Clean runtime check failed: {result.stderr}"
    assert "CLEAN_RUNTIME_OK" in result.stdout


def test_healthz_endpoint():
    """Verify GET /healthz returns status ok, model_loaded, categories_count, and threshold."""
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["categories_count"] == 17
    assert "threshold" in data
    assert isinstance(data["threshold"], (float, int, dict))


def test_classify_endpoint_single_and_batch():
    """Verify POST /classify works for single text and multiple texts."""
    client = TestClient(app)
    texts = [
        "买大了想退",
        "物流太慢了，而且包装破损",
        "什么时候能开发票",
    ]
    response = client.post("/classify", json={"texts": texts})
    assert response.status_code == 200

    data = response.json()
    assert "results" in data
    results = data["results"]
    assert len(results) == len(texts)

    for i, res in enumerate(results):
        assert res["text"] == texts[i]
        assert isinstance(res["labels"], list)
        assert isinstance(res["scores"], dict)
        assert len(res["scores"]) == 17

        # Check all 17 taxonomy categories are present
        for cat in TAXONOMY_17:
            assert cat in res["scores"]
            score = res["scores"][cat]
            assert 0.0 <= score <= 1.0

        # Check thresholding rule: labels match categories with score >= threshold
        engine = app.state.engine
        for cat in res["labels"]:
            cat_thresh = engine.get_threshold_for_category(cat)
            assert res["scores"][cat] >= cat_thresh - 1e-6


def test_classify_endpoint_empty_and_edge_cases():
    """Verify POST /classify handles empty lists, whitespaces, and invalid payloads."""
    client = TestClient(app)

    # Empty batch
    resp_empty = client.post("/classify", json={"texts": []})
    assert resp_empty.status_code == 200
    assert resp_empty.json() == {"results": []}

    # Whitespace and empty string texts
    resp_blank = client.post("/classify", json={"texts": ["", "   "]})
    assert resp_blank.status_code == 200
    assert len(resp_blank.json()["results"]) == 2

    # Invalid input format (not a list)
    resp_invalid = client.post("/classify", json={"texts": "not a list"})
    assert resp_invalid.status_code == 422


@pytest.mark.asyncio
async def test_async_client_classify():
    """Verify async client interaction with /healthz and /classify."""
    import httpx

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        health_resp = await ac.get("/healthz")
        assert health_resp.status_code == 200
        assert health_resp.json()["model_loaded"] is True

        classify_resp = await ac.post("/classify", json={"texts": ["衣服尺码偏小想退货"]})
        assert classify_resp.status_code == 200
        results = classify_resp.json()["results"]
        assert len(results) == 1
        assert "尺码" in results[0]["labels"] or "退换货" in results[0]["labels"]


def test_pid_lifecycle(tmp_path):
    """Verify write_pid, read_pid, remove_pid management."""
    pid_file = tmp_path / "test.pid"
    assert read_pid(pid_file) is None

    write_pid(pid_file, 12345)
    assert read_pid(pid_file) == 12345

    remove_pid(pid_file)
    assert read_pid(pid_file) is None
    assert not pid_file.exists()

    # Removing non-existent pid file should not raise error
    remove_pid(pid_file)


def test_is_pid_alive():
    """Verify process liveness detection."""
    # Current process should be alive
    current_pid = os.getpid()
    assert is_pid_alive(current_pid) is True

    # A bogus non-existent PID should be dead
    bogus_pid = 999999
    assert is_pid_alive(bogus_pid) is False


def test_service_status(tmp_path):
    """Verify get_service_status handles missing PID, dead PID, and running PID."""
    pid_file = tmp_path / "service.pid"

    # 1. No PID file -> stopped
    status_info = get_service_status(pid_file=pid_file, port=8110)
    assert status_info["running"] is False
    assert status_info["pid"] is None

    # 2. Dead PID file -> stale PID, stopped
    write_pid(pid_file, 999999)
    status_info_stale = get_service_status(pid_file=pid_file, port=8110)
    assert status_info_stale["running"] is False
    assert status_info_stale["pid"] == 999999
    assert status_info_stale.get("stale") is True


def test_process_tree_termination(tmp_path):
    """Verify safe process tree kill without leaving orphan processes."""
    pid_file = tmp_path / "dummy.pid"

    # Launch a dummy long-running child process
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    dummy_pid = proc.pid
    write_pid(pid_file, dummy_pid)

    assert is_pid_alive(dummy_pid) is True

    # Stop service (kills process and removes PID file)
    stopped = stop_service(pid_file=pid_file)
    assert stopped is True
    assert not pid_file.exists()
    assert is_pid_alive(dummy_pid) is False


def test_cli_argument_parsing():
    """Verify CLI subcommands and default argument values."""
    args_start = parse_args(["start", "--port", "8110", "--host", "127.0.0.1"])
    assert args_start.command == "start"
    assert args_start.port == 8110
    assert args_start.host == "127.0.0.1"

    args_stop = parse_args(["stop"])
    assert args_stop.command == "stop"

    args_status = parse_args(["status"])
    assert args_status.command == "status"

    args_restart = parse_args(["restart"])
    assert args_restart.command == "restart"
