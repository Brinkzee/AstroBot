"""Chapter 10 Lightweight Classifier Service package."""

from app.services.classifier_service.server import app, ClassifierInferenceEngine

__all__ = ["app", "ClassifierInferenceEngine"]
