"""Standard detector result shape for modular classification and localization.

A detector can classify, localize, or both. Classification-only models such as
ViT and CLIP return no boxes/masks. Localization models such as OpenCV contour
modules or YOLO return boxes and can be annotated generically.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Box = Tuple[float, float, float, float]


@dataclass
class DetectorResult:
    name: str
    family: str  # fire, smoke, context, text, etc.
    kind: str  # classifier, localizer, detector, ocr
    decision: str
    confidence: float = 0.0
    reason: str = ""
    boxes: List[Box] = field(default_factory=list)
    mask: Optional[Any] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def localized(self) -> bool:
        return bool(self.boxes) or self.mask is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "kind": self.kind,
            "decision": self.decision,
            "confidence": self.confidence,
            "reason": self.reason,
            "boxes": self.boxes,
            "mask": self.mask,
            "localized": self.localized,
            "details": self.details,
        }


def from_legacy_region(name: str, family: str, region: Dict[str, Any], decision_key: str = "status") -> DetectorResult:
    boxes = []
    for item in region.get("regions", []) or []:
        box = item.get("box")
        if box is not None:
            boxes.append(tuple(float(v) for v in box))
    if not boxes and region.get("union_box") is not None:
        boxes.append(tuple(float(v) for v in region["union_box"]))
    return DetectorResult(
        name=name,
        family=family,
        kind="localizer",
        decision=region.get(decision_key, "Not Run"),
        confidence=float(region.get("image_percent", 0.0)),
        reason=region.get("reason", ""),
        boxes=boxes,
        details={k: v for k, v in region.items() if k not in {"regions", "union_box"}},
    )


def from_yolo_detections(detections: List[Dict[str, Any]]) -> List[DetectorResult]:
    results: List[DetectorResult] = []
    for det in detections:
        box = det.get("box")
        if box is None:
            continue
        results.append(DetectorResult(
            name="yolo_objects",
            family="context",
            kind="detector",
            decision=det.get("class_name", "object"),
            confidence=float(det.get("confidence", 0.0)),
            reason="YOLO object detection",
            boxes=[tuple(float(v) for v in box)],
            details=det,
        ))
    return results
