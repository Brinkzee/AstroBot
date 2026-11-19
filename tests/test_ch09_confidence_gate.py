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


def test_compute_evidence_confidence_supports_rerank_score_and_distance():
    # rerank_score 应与 score 等效计算
    docs_rerank = [{"rerank_score": 0.8}, {"rerank_score": 0.5}, {"rerank_score": 0.2}]
    score_rerank = compute_evidence_confidence(docs_rerank, k=3)
    assert abs(score_rerank - 0.6916666666666667) < 1e-6

    # 仅有 distance（如 vector_only / bm25_only）时也应正确读取
    docs_distance = [{"distance": 0.6}]
    score_distance = compute_evidence_confidence(docs_distance, k=3)
    assert abs(score_distance - 0.4883333333333333) < 1e-6

    # 显式 score=0.0 不应被 rerank_score 覆盖
    docs_explicit_zero = [{"score": 0.0, "rerank_score": 0.9}]
    assert compute_evidence_confidence(docs_explicit_zero, k=3) == 0.0


def test_calibrate_confidence_gate_uses_real_retriever_and_should_refuse(tmp_path):
    from scripts.calibrate_confidence_gate import run_calibration
    from app.services.rag.advanced_retriever import AdvancedRetrievalResult

    dataset_file = tmp_path / "eval_mini.jsonl"
    report_file = tmp_path / "calib_report.json"

    # 特意按 [正, 正, 负, 负] 顺序排列，验证脚本真实读取 should_refuse 而非按 i % 2 奇偶构造标签
    rows = [
        {"id": "P1", "query": "退货政策是什么", "should_refuse": False},
        {"id": "P2", "query": "多久发货", "should_refuse": False},
        {"id": "N1", "query": "你们卖活体猫吗", "should_refuse": True},
        {"id": "N2", "query": "实体门店在哪", "should_refuse": True},
    ]
    dataset_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    class StubRetriever:
        def __init__(self):
            self.called_queries = []

        async def retrieve_with_strategy(self, query: str, strategy: str = "hybrid_rerank", min_score: float = 0.1, **kwargs):
            self.called_queries.append(query)
            if query in ("退货政策是什么", "多久发货"):
                return AdvancedRetrievalResult(
                    docs=[{"rerank_score": 0.78, "chunk_text": "相关规则"}, {"rerank_score": 0.42, "chunk_text": "补充"}],
                    strategy=strategy,
                )
            return AdvancedRetrievalResult(
                docs=[{"rerank_score": 0.14, "chunk_text": "弱相关"}],
                strategy=strategy,
            )

    stub = StubRetriever()
    report = run_calibration(
        dataset_path=str(dataset_file),
        output_path=str(report_file),
        retriever=stub,
    )

    assert stub.called_queries == ["退货政策是什么", "多久发货", "你们卖活体猫吗", "实体门店在哪"]
    assert report["sample_count"] == 4
    assert report["positive_count"] == 2
    assert report["negative_count"] == 2
    assert report["best_youden_j"] == 1.0
    assert 0.12 <= report["recommended_threshold"] <= 0.65
    assert report_file.exists()
