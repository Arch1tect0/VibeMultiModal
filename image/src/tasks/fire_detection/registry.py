"""Fire detector registry."""

from src.core.detector_registry import DetectorRegistry
from src.tasks.fire_detection import clip_fire_evidence, opencv_flame, vit_fire


def empty_vit(name: str):
    return {"status": "Not Run", "confidence": 0.0, "model_label": f"{name} disabled", "method": name}


def empty_region(name: str):
    return {
        "status": "Not Run",
        "regions": [],
        "union_box": None,
        "area": 0.0,
        "image_percent": 0.0,
        "method": name,
        "reason": f"{name} disabled",
    }


def empty_clip_fire(name: str):
    return {"label": "Not Run", "status": "Not Run", "confidence": 0.0, "scores": {}, "method": name, "reason": f"{name} disabled"}


REGISTRY = DetectorRegistry("fire")
REGISTRY.register("vit_fire", lambda image_path, image: vit_fire.detect(image_path), empty_vit)
REGISTRY.register("opencv_flame", lambda image_path, image: opencv_flame.detect(image), empty_region)
REGISTRY.register("clip_fire_evidence", lambda image_path, image: clip_fire_evidence.detect(image_path), empty_clip_fire)


def run_enabled(enabled, image_path, image):
    return REGISTRY.run_enabled(enabled, image_path, image)
