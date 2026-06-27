"""Generic annotated-image writer.

Any detector that returns boxes through DetectorResult can be drawn here. Image-level
classifiers such as ViT and CLIP are still recorded in CSV, but they do not draw
boxes because they do not localize the finding.
"""

from typing import Dict, Iterable, List, Optional

import cv2

try:
    from src.core.detector_result import DetectorResult, from_legacy_region, from_yolo_detections
except Exception:  # pragma: no cover - keeps old imports safe in ad-hoc execution
    DetectorResult = None


FAMILY_COLORS = {
    "fire": (0, 0, 255),       # red in BGR
    "smoke": (220, 220, 220),  # light gray/white
    "context": (255, 0, 0),    # blue
    "text": (0, 255, 255),
}


def _draw_box(annotated, box, label: str, family: str):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = FAMILY_COLORS.get(family, (0, 255, 0))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        annotated,
        label[:60],
        (x1, max(y1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def draw_detector_annotations(image, detector_results: Iterable) -> Optional[object]:
    if image is None:
        return image
    annotated = image.copy()
    for result in detector_results:
        boxes = getattr(result, "boxes", []) or []
        family = getattr(result, "family", "other")
        name = getattr(result, "name", "detector")
        decision = getattr(result, "decision", "")
        confidence = getattr(result, "confidence", 0.0)
        label = f"{name}: {decision}"
        if confidence:
            label += f" {confidence:.2f}"
        for box in boxes:
            _draw_box(annotated, box, label, family)
    return annotated


def build_annotation_results(detections: List[Dict], flame_region: Dict, smoke_region: Optional[object] = None) -> List:
    """Adapter from the existing pipeline dictionaries to DetectorResult objects."""
    results = []
    if DetectorResult is None:
        return results
    # Context boxes from YOLO.
    results.extend(from_yolo_detections([d for d in detections if d.get("is_reference")]))
    # Fire localization from OpenCV flame contours.
    if flame_region:
        results.append(from_legacy_region("opencv_flame", "fire", flame_region))
    # Smoke localization from any OpenCV smoke localizers.
    if smoke_region:
        smoke_regions = smoke_region if isinstance(smoke_region, list) else [smoke_region]
        for region in smoke_regions:
            if region:
                results.append(from_legacy_region(region.get("method", "opencv_smoke"), "smoke", region))
    return results


def annotation_sources(detector_results: Iterable) -> str:
    names = []
    for result in detector_results:
        if getattr(result, "localized", False):
            names.append(getattr(result, "name", "unknown"))
    return ", ".join(sorted(set(names))) if names else "None"


# Backward-compatible wrapper used by earlier pipeline code.
def draw_annotations(image, detections: List[Dict], flame_region: Dict, final_type: str, size_info: Dict, smoke_region: Optional[Dict] = None):
    results = build_annotation_results(detections, flame_region, smoke_region)
    return draw_detector_annotations(image, results)
