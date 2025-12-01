"""
FastAPI application that exposes inference endpoints for the
GoEmotions multi-label classifier and the SuicideWatch risk detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_GO_DIR = BASE_DIR / "model_go"
MODEL_SW_DIR = BASE_DIR / "model_sw"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_label_map(raw_map: Dict) -> Dict[int, str]:
    """Ensure id2label keys are integers (HF may store them as strings)."""
    if not raw_map:
        return {}
    first_key = next(iter(raw_map.keys()))
    if isinstance(first_key, str):
        return {int(k): v for k, v in raw_map.items()}
    return raw_map


tokenizer_go = AutoTokenizer.from_pretrained(MODEL_GO_DIR)
model_go = AutoModelForSequenceClassification.from_pretrained(MODEL_GO_DIR)
model_go.to(DEVICE)
model_go.eval()
id2label_go_map = _load_label_map(getattr(model_go.config, "id2label", {}))
labels_go = [
    id2label_go_map.get(i, f"label_{i}") for i in range(model_go.config.num_labels)
]

tokenizer_sw = AutoTokenizer.from_pretrained(MODEL_SW_DIR)
model_sw = AutoModelForSequenceClassification.from_pretrained(MODEL_SW_DIR)
model_sw.to(DEVICE)
model_sw.eval()
id2label_sw_map = _load_label_map(getattr(model_sw.config, "id2label", {}))
labels_sw = [
    id2label_sw_map.get(i, f"class_{i}") for i in range(model_sw.config.num_labels)
]


class EmotionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to analyze.")
    threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Probability threshold for keeping an emotion label.",
    )


class SuicideRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input post or message.")


app = FastAPI(
    title="Emotion Profiling & Suicide Risk API",
    description=(
        "Serves the fine-tuned DistilBERT models for GoEmotions (multi-label) "
        "and SuicideWatch (binary) classification."
    ),
    version="0.1.0",
)


def _tokenize(tokenizer, text: str) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=tokenizer.model_max_length,
    )
    return {k: v.to(DEVICE) for k, v in encoded.items()}


@app.get("/healthz")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict/emotions")
def predict_emotions(payload: EmotionRequest) -> Dict:
    inputs = _tokenize(tokenizer_go, payload.text)
    with torch.no_grad():
        logits = model_go(**inputs).logits
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    distribution = [
        {"label": label, "probability": prob}
        for label, prob in zip(labels_go, probabilities)
    ]

    predictions = [
        entry for entry in distribution if entry["probability"] >= payload.threshold
    ]

    if not predictions:
        top_entry = max(distribution, key=lambda item: item["probability"])
        predictions = [top_entry]

    return {
        "threshold": payload.threshold,
        "predictions": predictions,
        "full_distribution": distribution,
    }


@app.post("/predict/suicide")
def predict_suicide(payload: SuicideRequest) -> Dict:
    inputs = _tokenize(tokenizer_sw, payload.text)
    with torch.no_grad():
        logits = model_sw(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    distribution = {
        label: prob for label, prob in zip(labels_sw, probabilities)
    }
    predicted_label = max(distribution.items(), key=lambda item: item[1])

    return {
        "label": predicted_label[0],
        "confidence": predicted_label[1],
        "probabilities": distribution,
    }
