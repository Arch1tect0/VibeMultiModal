"""Final scene fusion.

Combines independent fire and smoke decisions into a scene-level decision.
"""

from typing import List, Tuple

from src import config


def decide_scene_status(fire_status: str, smoke_status: str) -> Tuple[str, str]:
    fire_positive = fire_status == "Fire"
    smoke_positive = smoke_status == "Smoke"
    fire_uncertain = fire_status == "Uncertain"
    smoke_uncertain = smoke_status == "Uncertain Smoke"

    if fire_positive and smoke_positive:
        return "Fire + Smoke", "independent fire and smoke decisions are positive"
    if fire_positive:
        return "Fire", "fire decision is positive"
    if smoke_positive:
        return "Smoke Only", "smoke decision is positive and fire decision is not positive"
    if fire_uncertain or smoke_uncertain:
        return "Uncertain", "at least one independent decision is uncertain"
    return "No Fire / No Smoke", "no independent fire or smoke evidence met thresholds"


def classify_scene(final_decision: str, fire_status: str, smoke_status: str, final_fire_type: str, object_names: List[str]) -> str:
    """Return the executive scene label.

    If a fire is present and a fire type was determined, the Scene_Type is the
    fire type itself (for example, Vehicle Fire or Structure Fire). Smoke remains
    visible through Smoke_Detection_Confidence and the scene report instead of
    replacing the fire type in the main CSV.
    """
    if fire_status == "Fire" and final_fire_type not in {"", "None", "Unknown"}:
        return final_fire_type
    if final_decision == "Fire + Smoke":
        return "Fire + Smoke"
    if final_decision == "Fire":
        return "Fire"
    if final_decision == "Smoke Only":
        return "Smoke Only"
    if any(name in config.VEHICLE_CLASSES for name in object_names):
        return "Vehicle / Accident Scene"
    if final_decision == "Uncertain":
        return "Uncertain Fire/Smoke Scene"
    return "No Fire / Unknown Scene"
