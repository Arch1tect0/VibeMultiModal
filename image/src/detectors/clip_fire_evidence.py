"""CLIP visible-fire evidence detector.

Responsible output columns:
- CLIP_Fire_Evidence
- CLIP_Fire_Evidence_Confidence

This detector no longer emits smoke labels. Smoke is handled separately by
src/detectors/clip_smoke.py and src/core/smoke_decision_engine.py.
"""

from pathlib import Path
from typing import Dict

from src import config
from src.detectors.clip_common import zero_shot


def detect(image_path: Path) -> Dict:
    label, confidence, scores = zero_shot(image_path, config.FIRE_EVIDENCE_PROMPTS)
    status = "Visible Fire" if label == "Visible Fire" else "No Visible Fire"
    return {"label": status, "confidence": confidence, "scores": scores}
