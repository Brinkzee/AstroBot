"""Lightweight inference service for Chapter 10 Multi-Label Topic Classifier.

Runs on :8110.
Strict lightweight runtime: ZERO dependencies on PyTorch or HuggingFace Transformers.
Uses onnxruntime, tokenizers, numpy, fastapi, and uvicorn.
"""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from app.services.classifier.taxonomy import TAXONOMY_17

# Default paths
DEFAULT_MODEL_PATH = "data/ch10/onnx/model.onnx"
DEFAULT_TOKENIZER_PATH = "data/ch10/onnx/tokenizer.json"
DEFAULT_THRESHOLD_PATH = "data/ch10/onnx/threshold.json"
DEFAULT_THRESHOLD_VALUE = 0.45


class ClassifierInferenceEngine:
    """Pure ONNX + Tokenizers lightweight inference engine."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        threshold_path: Optional[Union[str, Path]] = None,
        taxonomy: Optional[List[str]] = None,
    ):
        self.model_path = Path(model_path or os.getenv("ONNX_MODEL_PATH", DEFAULT_MODEL_PATH))
        self.tokenizer_path = Path(
            tokenizer_path or os.getenv("TOKENIZER_PATH", DEFAULT_TOKENIZER_PATH)
        )
        self.threshold_path = Path(
            threshold_path or os.getenv("THRESHOLD_PATH", DEFAULT_THRESHOLD_PATH)
        )
        self.taxonomy: List[str] = taxonomy or list(TAXONOMY_17)

        self.tokenizer: Optional[Tokenizer] = None
        self.session: Optional[ort.InferenceSession] = None
        self.threshold_info: Any = DEFAULT_THRESHOLD_VALUE
        self.per_class_thresholds: Dict[str, float] = {}
        self.model_loaded: bool = False

    def load(self) -> None:
        """Load tokenizer, ONNX runtime session, and threshold configuration."""
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        # Load Tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_padding()

        # Load ONNX Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Load Threshold
        self._load_threshold()
        self.model_loaded = True

    def _load_threshold(self) -> None:
        """Parse threshold from threshold.json, float, or dict."""
        if not self.threshold_path.exists():
            self.threshold_info = DEFAULT_THRESHOLD_VALUE
            self.per_class_thresholds = {cat: DEFAULT_THRESHOLD_VALUE for cat in self.taxonomy}
            return

        try:
            content = self.threshold_path.read_text(encoding="utf-8")
            data = json.loads(content)
            self.threshold_info = data
            if isinstance(data, dict):
                if "threshold" in data and isinstance(data["threshold"], (int, float)):
                    t_val = float(data["threshold"])
                    self.per_class_thresholds = {cat: t_val for cat in self.taxonomy}
                else:
                    # Possibly per-class dict
                    self.per_class_thresholds = {
                        cat: float(data.get(cat, DEFAULT_THRESHOLD_VALUE))
                        for cat in self.taxonomy
                    }
            elif isinstance(data, (int, float)):
                t_val = float(data)
                self.threshold_info = t_val
                self.per_class_thresholds = {cat: t_val for cat in self.taxonomy}
            else:
                self.threshold_info = DEFAULT_THRESHOLD_VALUE
                self.per_class_thresholds = {cat: DEFAULT_THRESHOLD_VALUE for cat in self.taxonomy}
        except Exception:
            self.threshold_info = DEFAULT_THRESHOLD_VALUE
            self.per_class_thresholds = {cat: DEFAULT_THRESHOLD_VALUE for cat in self.taxonomy}

    def get_threshold_for_category(self, category: str) -> float:
        """Get decision threshold for a specific category."""
        return self.per_class_thresholds.get(category, DEFAULT_THRESHOLD_VALUE)

    def classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run batch inference on given list of texts and return labels and scores."""
        if not self.model_loaded:
            self.load()

        if not texts:
            return []

        # Tokenize batch
        assert self.tokenizer is not None
        assert self.session is not None

        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        # Run ONNX inference
        outputs = self.session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        logits = outputs[0]  # shape: (batch_size, 17)

        # Compute Sigmoid probabilities
        probs = 1.0 / (1.0 + np.exp(-logits))

        results = []
        for i, text in enumerate(texts):
            scores: Dict[str, float] = {}
            matched: List[tuple] = []

            for idx, cat in enumerate(self.taxonomy):
                prob = float(probs[i, idx])
                rounded_prob = round(prob, 4)
                scores[cat] = rounded_prob

                thresh = self.get_threshold_for_category(cat)
                if prob >= thresh:
                    matched.append((cat, rounded_prob))

            # Sort matched labels by confidence score descending
            matched.sort(key=lambda x: x[1], reverse=True)
            labels = [cat for cat, _ in matched]

            results.append({
                "text": text,
                "labels": labels,
                "scores": scores,
            })

        return results


# Global inference engine instance
engine = ClassifierInferenceEngine()
try:
    engine.load()
except Exception:
    # Allow deferring load if model files are not yet present at import
    pass


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Lifespan context manager to ensure engine is loaded on service start."""
    if not engine.model_loaded:
        try:
            engine.load()
        except Exception as e:
            # Service can still boot and report error in /healthz
            pass
    yield


app = FastAPI(
    title="Ch10 Multi-Label Topic Classifier Service",
    description="Lightweight ONNX inference service running on :8110",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.engine = engine


def create_classifier_app() -> FastAPI:
    """Return the classifier service FastAPI app instance."""
    return app


class ClassifyRequest(BaseModel):
    texts: List[str] = Field(..., description="List of user queries to classify")


class ItemResult(BaseModel):
    text: str
    labels: List[str]
    scores: Dict[str, float]


class ClassifyResponse(BaseModel):
    results: List[ItemResult]


@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    if not engine.model_loaded:
        try:
            engine.load()
        except Exception as e:
            return {
                "status": "error",
                "model_loaded": False,
                "error": str(e),
            }

    # Extract clean threshold for response
    thresh_val = engine.threshold_info
    if isinstance(thresh_val, dict) and "threshold" in thresh_val:
        thresh_val = thresh_val["threshold"]

    return {
        "status": "ok",
        "model_loaded": engine.model_loaded,
        "categories_count": len(engine.taxonomy),
        "threshold": thresh_val,
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """Classify user queries into multi-label topic categories."""
    try:
        results = engine.classify(request.texts)
        return ClassifyResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """CLI entrypoint to run uvicorn server directly."""
    import uvicorn

    port = int(os.getenv("PORT", "8110"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("app.services.classifier_service.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
