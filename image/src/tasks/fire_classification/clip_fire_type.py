"""CLIP fire-type context classifier.

Responsible output columns:
- Fire_Type_CLIP
- Fire_Type_Confidence

The final fire type is decided in core/decision_engine.py so object context can
correct weak CLIP classifications.
"""

from pathlib import Path
from typing import Dict

from src import config
from src.tasks.shared.clip_common import zero_shot


def detect(image_path: Path, final_fire_status: str) -> Dict:
    if final_fire_status != "Fire":
        return {"label": "None", "confidence": 0.0, "scores": {}}
    label, confidence, scores = zero_shot(image_path, config.FIRE_TYPE_PROMPTS)
    return {"label": label, "confidence": confidence, "scores": scores}
