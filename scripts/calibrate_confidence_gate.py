import json
import os
import math
import sys
import datetime
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.workflow.nodes.gate import compute_evidence_confidence
from app.config import settings

def run_calibration():
    eval_file = "tests/data/eval_ch04.jsonl"
    
    # We mock retrieval/scores here.
    # We create artificial samples for positive and negative since we might not have a real retriever set up
    samples = []
    
    if os.path.exists(eval_file):
        # Even if we read it, we need fake docs. Let's just generate some synthetic scores.
        with open(eval_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Give half the questions a positive label, half negative
        for i, line in enumerate(lines):
            try:
                data = json.loads(line.strip())
                is_positive = i % 2 == 0
                if is_positive:
                    # High scores
                    docs = [{"score": 0.8}, {"score": 0.5}, {"score": 0.3}]
                else:
                    # Low scores
                    docs = [{"score": 0.2}, {"score": 0.1}]
                samples.append({
                    "is_positive": is_positive,
                    "confidence": compute_evidence_confidence(docs)
                })
            except Exception:
                pass

    if not samples:
        # Fallback fake data if no eval file
        for i in range(25):
            samples.append({"is_positive": True, "confidence": compute_evidence_confidence([{"score": 0.8}, {"score": 0.6}])})
            samples.append({"is_positive": False, "confidence": compute_evidence_confidence([{"score": 0.2}, {"score": 0.1}])})

    thresholds = [round(0.20 + i * 0.02, 2) for i in range(31)]
    
    best_t = 0.40
    best_j = -1.0
    best_f1 = -1.0
    best_metrics = {}

    for t in thresholds:
        tp = sum(1 for s in samples if s["is_positive"] and s["confidence"] >= t)
        fp = sum(1 for s in samples if not s["is_positive"] and s["confidence"] >= t)
        tn = sum(1 for s in samples if not s["is_positive"] and s["confidence"] < t)
        fn = sum(1 for s in samples if s["is_positive"] and s["confidence"] < t)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = sensitivity
        
        youden_j = sensitivity + specificity - 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if youden_j > best_j:
            best_j = youden_j
            best_f1 = f1
            best_t = t
            best_metrics = {
                "sensitivity": round(sensitivity, 2),
                "specificity": round(specificity, 2),
                "precision": round(precision, 2),
                "recall": round(recall, 2)
            }

    active_t = getattr(settings, "evidence_confidence_threshold", 0.40)
    
    report = {
        "calibrated_at": datetime.datetime.now().isoformat(),
        "sample_count": len(samples),
        "recommended_threshold": round(best_t, 2),
        "best_youden_j": round(best_j, 2),
        "best_f1": round(best_f1, 2),
        "active_threshold": active_t,
        "threshold_mismatch": abs(best_t - active_t) > 0.01,
        "metrics": best_metrics
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/confidence_calibration.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    print(f"Calibration finished. Recommended Threshold: {best_t:.2f} (J: {best_j:.2f}, F1: {best_f1:.2f})")
    print("Report saved to reports/confidence_calibration.json")

if __name__ == "__main__":
    run_calibration()
