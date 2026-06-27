"""CLIP smoke-plume classifier with explicit cloud/fog negatives.

This detector is separate from the generic CLIP smoke detector. It is intended to
help with large plume false negatives by asking CLIP to choose among specific
large-smoke and common-lookalike scene descriptions.

Responsible engine report method:
- clip_smoke_plume
"""

from pathlib import Path
from typing import Dict

from src import config
from src.tasks.shared.clip_common import zero_shot


PLUME_PROMPTS = {
    "Large Smoke Plume": "a photo of a large rising smoke plume from a fire, wildfire, industrial fire, or burning vehicle",
    "Thin Smoke Or Haze": "a photo with thin smoke, drifting haze, fumes, or ash in the air",
    "Clouds Or Fog": "a photo of natural clouds, fog, mist, steam, or overcast sky with no fire smoke",
    "Clear No Smoke": "a clear photo with no visible smoke plume, haze, fumes, or ash cloud",
}


def detect(image_path: Path) -> Dict:
    label, confidence, scores = zero_shot(image_path, PLUME_PROMPTS)
    smoke_labels = {"Large Smoke Plume", "Thin Smoke Or Haze"}
    threshold = config.CLIP_SMOKE_PLUME_THRESHOLD
    status = "Smoke" if label in smoke_labels and confidence >= threshold else "No Smoke"
    return {
        "status": status,
        "label": label,
        "confidence": confidence,
        "scores": scores,
        "method": "clip_smoke_plume_prompt",
        "reason": f"best_label={label}, confidence={confidence:.3f}, threshold={threshold}",
    }
