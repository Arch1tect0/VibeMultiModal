"""Smoke detector registry.

Add new smoke detectors here, then include the detector name in
ENABLED_SMOKE_DETECTORS or pass --smoke-detectors on the command line.
"""

from src.core.detector_registry import DetectorRegistry
from src.tasks.smoke_detection import clip_smoke, clip_smoke_plume, opencv_bright_plume, opencv_dark_smoke, opencv_smoke


def empty_clip(name: str):
    return {
        "status": "Not Run",
        "label": "Not Run",
        "confidence": 0.0,
        "scores": {},
        "method": name,
        "reason": f"{name} disabled",
    }


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


REGISTRY = DetectorRegistry("smoke")
REGISTRY.register("clip_smoke", lambda image_path, image: clip_smoke.detect(image_path), empty_clip)
REGISTRY.register("clip_smoke_plume", lambda image_path, image: clip_smoke_plume.detect(image_path), empty_clip)
REGISTRY.register("opencv_smoke", lambda image_path, image: opencv_smoke.detect(image), empty_region)
REGISTRY.register("opencv_dark_smoke", lambda image_path, image: opencv_dark_smoke.detect(image), empty_region)
REGISTRY.register("opencv_bright_plume", lambda image_path, image: opencv_bright_plume.detect(image), empty_region)


def run_enabled(enabled, image_path, image):
    return REGISTRY.run_enabled(enabled, image_path, image)
