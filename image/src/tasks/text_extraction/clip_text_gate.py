"""Optional CLIP gate for deciding whether an OCR crop likely contains readable text.

This module is intentionally separate from OCR. It does not read text; it only
answers whether a proposed crop looks like a street sign, license plate, or
other readable text target instead of background clutter such as fire, smoke,
trees, cars, sky, or pavement.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src import config

_clip_model = None
_clip_processor = None

POSITIVE_LABELS = ("readable_text", "street_sign", "license_plate")
NEGATIVE_LABELS = ("no_readable_text", "natural_scene", "vehicle_body")

PROMPTS = {
    "readable_text": "a close-up photo of readable words, letters, or numbers",
    "street_sign": "a close-up photo of a road sign or street sign with readable text",
    "license_plate": "a close-up photo of a vehicle license plate with readable numbers and letters",
    "no_readable_text": "a photo crop with no readable text or writing",
    "natural_scene": "a crop of fire, smoke, trees, sky, bushes, grass, or natural background with no sign text",
    "vehicle_body": "a crop of a car body, bumper, tail light, wheel, reflection, or road with no readable plate text",
}


def _cfg(name: str, default):
    return getattr(config, name, default)


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        _clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(config.TORCH_DEVICE)
        _clip_model.eval()
    return _clip_model, _clip_processor


def _crop_to_pil(crop) -> Image.Image:
    if crop is None or crop.size == 0:
        return Image.new("RGB", (1, 1), color=(0, 0, 0))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def score_crop(crop) -> Dict:
    """Return CLIP scores for a candidate OCR crop."""
    model, processor = _get_clip()
    image = _crop_to_pil(crop)
    labels = list(PROMPTS.keys())
    text_prompts = [PROMPTS[label] for label in labels]
    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(config.TORCH_DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    scores = {labels[i]: float(probs[i].item()) for i in range(len(labels))}
    best_label = max(scores, key=scores.get)
    positive_score = sum(scores[label] for label in POSITIVE_LABELS)
    negative_score = sum(scores[label] for label in NEGATIVE_LABELS)
    return {
        "best_label": best_label,
        "best_score": scores[best_label],
        "positive_score": positive_score,
        "negative_score": negative_score,
        "scores": scores,
    }


def should_run_ocr(crop, sign_reason: str = "") -> Tuple[bool, Dict]:
    """Decide whether OCR should run on this crop.

    The gate is intentionally conservative for natural-scene false positives,
    but allows strong license-plate candidates with a slightly lower positive
    threshold because plate crops are often small.
    """
    if not _cfg("CLIP_TEXT_GATE_ENABLED", False):
        return True, {"enabled": False}

    try:
        result = score_crop(crop)
    except Exception as exc:  # Keep OCR usable if CLIP weights are unavailable.
        if _cfg("CLIP_TEXT_GATE_FAIL_OPEN", True):
            return True, {"enabled": True, "error": str(exc), "fail_open": True}
        return False, {"enabled": True, "error": str(exc), "fail_open": False}

    reason = (sign_reason or "").lower()
    min_positive = float(_cfg("CLIP_TEXT_GATE_MIN_POSITIVE", 0.42))
    min_margin = float(_cfg("CLIP_TEXT_GATE_MIN_MARGIN", 0.04))
    if "license_plate" in reason:
        min_positive = float(_cfg("CLIP_TEXT_GATE_PLATE_MIN_POSITIVE", 0.34))
    elif "sign_color" in reason or "sign" in reason:
        min_positive = float(_cfg("CLIP_TEXT_GATE_SIGN_MIN_POSITIVE", 0.38))

    positive = float(result["positive_score"])
    negative = float(result["negative_score"])
    margin = positive - negative
    best_label = result["best_label"]

    allow = positive >= min_positive and margin >= min_margin and best_label not in NEGATIVE_LABELS
    result.update({
        "enabled": True,
        "allow": allow,
        "min_positive": min_positive,
        "min_margin": min_margin,
        "margin": margin,
    })
    return allow, result
