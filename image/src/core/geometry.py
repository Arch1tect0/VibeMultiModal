"""Geometry helpers shared by detection and decision modules."""

from typing import Dict, List, Optional, Tuple

from src.config import REFERENCE_CLASSES

Box = Tuple[float, float, float, float]


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_intersection(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def center_distance(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def choose_reference_object(fire_box: Optional[Box], detections: List[Dict]) -> Optional[Dict]:
    if fire_box is None:
        return None
    candidates = [d for d in detections if d.get("is_reference") or d.get("class_name") in REFERENCE_CLASSES]
    if not candidates:
        return None

    scored = []
    for det in candidates:
        ref_box = det["box"]
        ref_area = box_area(ref_box)
        if ref_area <= 0:
            continue
        inter = box_intersection(fire_box, ref_box)
        overlap_ratio = inter / ref_area
        distance = center_distance(fire_box, ref_box)
        score = overlap_ratio * 100000.0 - distance
        scored.append((score, overlap_ratio, distance, det))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    _, overlap_ratio, distance, best = scored[0]
    return {**best, "overlap_ratio": overlap_ratio, "center_distance": distance}
