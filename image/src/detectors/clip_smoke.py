"""Independent CLIP smoke detector.

Responsible output columns:
- CLIP_Smoke_Status
- CLIP_Smoke_Confidence
- CLIP_Smoke_Label
"""

from pathlib import Path
from typing import Dict

from src import config
from src.detectors.clip_common import zero_shot


def detect(image_path: Path) -> Dict:
    label, confidence, scores = zero_shot(image_path, config.SMOKE_PROMPTS)
    status = "Smoke" if label == "Smoke" and confidence >= config.CLIP_SMOKE_THRESHOLD else "No Smoke"
    return {
        "status": status,
        "label": label,
        "confidence": confidence,
        "scores": scores,
        "method": "clip_smoke_prompt",
        "reason": f"best_label={label}, confidence={confidence:.3f}, threshold={config.CLIP_SMOKE_THRESHOLD}",
    }
