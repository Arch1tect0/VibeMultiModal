"""ViT binary fire/no-fire detector.

Responsible output columns:
- ViT_Fire_Status
- ViT_Fire_Confidence
- ViT_Model_Label
"""

from pathlib import Path
from typing import Dict

from transformers import pipeline

from src import config

_fire_classifier = None


def get_fire_classifier():
    global _fire_classifier
    if _fire_classifier is None:
        _fire_classifier = pipeline(
            "image-classification",
            model=config.FIRE_CLASSIFIER_MODEL,
            device=config.DEVICE,
        )
    return _fire_classifier


def normalize_fire_label(label: str) -> str:
    text = label.lower().replace("_", " ").replace("-", " ")
    if "non" in text or "no fire" in text or "normal" in text or text.strip() in {"0", "negative"}:
        return "No Fire"
    if "fire" in text or "flame" in text or text.strip() in {"1", "positive"}:
        return "Fire"
    return label


def detect(image_path: Path) -> Dict:
    classifier = get_fire_classifier()
    results = classifier(str(image_path))
    if not results:
        return {"status": "Unknown", "confidence": 0.0, "model_label": ""}

    fire_best = None
    non_best = None
    for item in results:
        label = str(item.get("label", ""))
        score = float(item.get("score", 0.0))
        norm = normalize_fire_label(label)
        if norm == "Fire" and (fire_best is None or score > fire_best["score"]):
            fire_best = {"label": label, "score": score}
        if norm == "No Fire" and (non_best is None or score > non_best["score"]):
            non_best = {"label": label, "score": score}

    if fire_best is not None:
        if fire_best["score"] >= config.FIRE_THRESHOLD:
            return {"status": "Fire", "confidence": fire_best["score"], "model_label": fire_best["label"]}
        if non_best is not None and non_best["score"] > fire_best["score"]:
            return {"status": "No Fire", "confidence": non_best["score"], "model_label": non_best["label"]}
        return {"status": "Uncertain", "confidence": fire_best["score"], "model_label": fire_best["label"]}

    top = max(results, key=lambda x: float(x.get("score", 0.0)))
    return {
        "status": normalize_fire_label(str(top.get("label", ""))),
        "confidence": float(top.get("score", 0.0)),
        "model_label": str(top.get("label", "")),
    }
