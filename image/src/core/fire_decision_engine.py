"""Fire-only decision rules.

This module decides only whether visible fire is present. It does not produce
Smoke Only. Smoke is decided independently in smoke_decision_engine.py.

OpenCV flame-color detection is intentionally treated as localization/support
evidence. Colored traffic signs, license plates, taillights, sunsets, safety
vests, and vehicle paint can all match flame HSV ranges, so color alone should
not override image-level classifiers.
"""

from typing import Dict, Tuple

from src import config


def _cfg(name: str, default):
    return getattr(config, name, default)


def _clip_no_fire(clip_label: str, clip_conf: float) -> bool:
    return clip_label in {"No Visible Fire", "No Fire"} and clip_conf >= _cfg("CLIP_NO_FIRE_VETO_THRESHOLD", 0.65)


def _vit_no_fire(vit_status: str, vit_conf: float) -> bool:
    return vit_status in {"No Fire", "No_Fire", "no_fire"} and vit_conf >= _cfg("VIT_NO_FIRE_VETO_THRESHOLD", 0.65)


def decide_fire_status(detector_results: Dict[str, Dict]) -> Tuple[str, str, str]:
    vit_result = detector_results.get("vit_fire", {}) or {}
    flame_region = detector_results.get("opencv_flame", {}) or {}
    clip_evidence = detector_results.get("clip_fire_evidence", {}) or {}

    flame_pct = float(flame_region.get("image_percent", 0.0))
    vit_status = vit_result.get("status", "Unknown")
    vit_conf = float(vit_result.get("confidence", 0.0))
    clip_label = clip_evidence.get("label", "")
    clip_conf = float(clip_evidence.get("confidence", 0.0))

    vit_fire = vit_status == "Fire" and vit_conf >= config.FIRE_THRESHOLD
    clip_fire = clip_label == "Visible Fire" and clip_conf >= config.CLIP_VISIBLE_FIRE_THRESHOLD
    color_fire = flame_pct >= config.COLOR_FIRE_OVERRIDE_PERCENT
    no_fire_veto = _clip_no_fire(clip_label, clip_conf) or _vit_no_fire(vit_status, vit_conf)

    # Classifier-confirmed fire. Color strengthens/localizes but is not required
    # for ViT, because ViT is the dedicated fire classifier.
    if vit_fire:
        triggered = ["vit_fire"]
        if color_fire:
            triggered.append("opencv_flame")
        if clip_fire:
            triggered.append("clip_fire_evidence")
        return "Fire", f"vit_fire {vit_conf:.3f} >= {config.FIRE_THRESHOLD}; flame_color={flame_pct:.4f}", ", ".join(triggered)

    # Strong No-Fire classifiers veto color-supported CLIP/OpenCV detections.
    # This prevents traffic signs/license plates/taillights/red vehicles from
    # becoming Fire when the dedicated ViT fire classifier is very confident
    # the scene is normal. This must run before the CLIP+color fire rule below.
    if no_fire_veto and (color_fire or clip_fire):
        veto_parts = []
        if _clip_no_fire(clip_label, clip_conf):
            veto_parts.append(f"clip_no_fire {clip_conf:.3f}")
        if _vit_no_fire(vit_status, vit_conf):
            veto_parts.append(f"vit_no_fire {vit_conf:.3f}")
        evidence_parts = []
        if clip_fire:
            evidence_parts.append(f"clip_visible_fire {clip_conf:.3f}")
        if color_fire:
            evidence_parts.append(f"flame_color={flame_pct:.4f}")
        return (
            "No Fire",
            f"strong no-fire classifier vetoed color/CLIP fire evidence; veto={' and '.join(veto_parts)}; evidence={'; '.join(evidence_parts)}",
            ", ".join(veto_parts) if veto_parts else "None",
        )

    # CLIP visible fire should be supported by at least some flame-colored pixels,
    # and it is not allowed to override a strong No-Fire classifier vote.
    if clip_fire and flame_pct > 0:
        triggered = ["clip_fire_evidence", "opencv_flame"]
        return "Fire", f"clip_visible_fire {clip_conf:.3f} with flame_color {flame_pct:.4f}", ", ".join(triggered)

    # Optional legacy behavior: allow color-only to declare Fire. Default false.
    if color_fire and _cfg("FIRE_ALLOW_COLOR_ONLY_OVERRIDE", False):
        return "Fire", f"color_flame_region {flame_pct:.4f} >= {config.COLOR_FIRE_OVERRIDE_PERCENT}", "opencv_flame"

    if vit_status == "Uncertain" or flame_pct > 0 or clip_fire:
        weak = []
        if vit_status == "Uncertain":
            weak.append("vit_fire")
        if flame_pct > 0:
            weak.append("opencv_flame")
        if clip_fire:
            weak.append("clip_fire_evidence")
        return "Uncertain", "weak or color-only fire evidence; no classifier-confirmed fire", ", ".join(weak)

    return "No Fire", "no strong fire evidence", "None"
