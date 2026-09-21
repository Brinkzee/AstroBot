import json
from unittest.mock import patch, MagicMock
from app.services.workflow.state import AgentWorkflowState
from app.services.workflow.nodes.gate import compute_evidence_confidence, confidence_gate
from app.config import settings
import subprocess
import os

def test_compute_evidence_confidence_empty():
    assert compute_evidence_confidence([]) == 0.0

def test_compute_evidence_confidence_multi_signals():
    # test multiple signals calculation
    # score1 = 0.8, score2 = 0.5, score3 = 0.2
    # top1_score = 0.8
    # valid_count = 2 (0.8, 0.5 >= 0.25)
    # k = 3 -> valid_count/k = 2/3 = 0.666...
    # score_margin = 0.8 - 0.5 = 0.3
    # formula: 0.60 * 0.8 + 0.25 * (2/3) + 0.15 * 0.3
    # = 0.48 + 0.1666666... + 0.045 = 0.691666...
    docs = [{"score": 0.8}, {"score": 0.5}, {"score": 0.2}]
    score = compute_evidence_confidence(docs, k=3)
    assert abs(score - 0.6916666666666667) < 1e-6

def test_compute_evidence_confidence_single_doc():
    docs = [{"score": 0.6}]
    # top1 = 0.6
    # valid_count = 1
    # margin = 0.6 * 0.5 = 0.3
    # formula: 0.6 * 0.6 + 0.25 * (1/3) + 0.15 * 0.3
    # = 0.36 + 0.08333333... + 0.045 = 0.48833333...
    score = compute_evidence_confidence(docs, k=3)
    assert abs(score - 0.4883333333333333) < 1e-6

def test_confidence_gate_fallback_branch():
    state = AgentWorkflowState(retrieved_docs=[{"score": 0.1}])
    with patch("app.services.workflow.nodes.gate.settings") as mock_settings:
        mock_settings.evidence_confidence_threshold = 0.40
        res = confidence_gate(state)
        assert res == "fallback"
        assert state["evidence_confidence"] < 0.40

def test_confidence_gate_pass_branch():
    state = AgentWorkflowState(retrieved_docs=[{"score": 0.9}])
    with patch("app.services.workflow.nodes.gate.settings") as mock_settings:
        mock_settings.evidence_confidence_threshold = 0.40
        res = confidence_gate(state)
        assert res == "pass"
        assert state["evidence_confidence"] >= 0.40

def test_calibrate_confidence_gate_script():
    # Write a dummy eval_ch04.jsonl if it doesn't exist
    os.makedirs("tests/data", exist_ok=True)
    if not os.path.exists("tests/data/eval_ch04.jsonl"):
        with open("tests/data/eval_ch04.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"question": "Q1", "answer": "A1"}) + "\n")
            f.write(json.dumps({"question": "Q2", "answer": "A2"}) + "\n")

    # Run the script
    result = subprocess.run(["python", "scripts/calibrate_confidence_gate.py"], capture_output=True, text=True)
    assert result.returncode == 0
    
    # Check output report
    report_path = "reports/confidence_calibration.json"
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "calibrated_at" in data
        assert "recommended_threshold" in data
        assert "best_youden_j" in data
        assert "best_f1" in data
