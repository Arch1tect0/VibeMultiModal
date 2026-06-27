"""Smoke detector module.

Round 1 preserves the original pipeline behavior by using the existing CLIP
fire-evidence prompt result. This creates a clean smoke module without changing
model performance.

Responsible output columns:
- Smoke_Status
- Smoke_Confidence
- Smoke_Method
- Smoke_Reason
"""

from typing import Dict

from src import config


def detect_from_clip_evidence(clip_evidence: Dict, flame_region: Dict) -> Dict:
    label = clip_evidence.get("label", "")
    confidence = float(clip_evidence.get("confidence", 0.0))
    flame_pct = float(flame_region.get("image_percent", 0.0))

    if label == "Smoke Only" and confidence >= config.CLIP_SMOKE_THRESHOLD:
        if flame_pct < config.COLOR_FIRE_OVERRIDE_PERCENT:
            return {
                "status": "Smoke Only",
                "confidence": confidence,
                "method": "clip_fire_evidence_prompt",
                "reason": f"clip_smoke_only {confidence:.3f} >= {config.CLIP_SMOKE_THRESHOLD} and flame_pct {flame_pct:.4f} below override",
            }
        return {
            "status": "Smoke With Flame Color",
            "confidence": confidence,
            "method": "clip_fire_evidence_prompt",
            "reason": f"smoke evidence present but flame_pct {flame_pct:.4f} meets/approaches fire evidence",
        }

    return {
        "status": "No Smoke Evidence",
        "confidence": confidence if label == "Smoke Only" else 0.0,
        "method": "clip_fire_evidence_prompt",
        "reason": f"best_clip_label={label}, confidence={confidence:.3f}",
    }
