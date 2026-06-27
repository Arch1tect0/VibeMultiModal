"""OpenCV dark-smoke detector.

Targets black/dark-gray smoke that the lighter gray/white smoke detector can miss.
This detector localizes smoke-colored connected regions and can annotate boxes.
"""

from typing import Dict

import cv2
import numpy as np

from src import config


def _build_regions(mask, image_area: float):
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(150.0, config.OPENCV_DARK_SMOKE_MIN_REGION_PERCENT * image_area)
    regions = []
    total_area = 0.0
    xs1, ys1, xs2, ys2 = [], [], [], []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        regions.append({"box": (float(x), float(y), float(x + w), float(y + h)), "area": area})
        total_area += area
        xs1.append(x); ys1.append(y); xs2.append(x + w); ys2.append(y + h)
    union_box = None
    if regions:
        union_box = (float(min(xs1)), float(min(ys1)), float(max(xs2)), float(max(ys2)))
    return regions, union_box, total_area, min_area


def detect(image) -> Dict:
    if image is None:
        return {"status": "No Image", "regions": [], "union_box": None, "area": 0.0, "image_percent": 0.0, "method": "no_image", "reason": "image could not be read"}

    h, w = image.shape[:2]
    image_area = float(h * w)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Dark smoke: low/medium saturation and low/medium value. Avoid fully black borders.
    lower = np.array([0, 0, 35], dtype=np.uint8)
    upper = np.array([179, 110, 170], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Reduce very colorful/brown objects and tiny high-contrast texture.
    regions, union_box, total_area, min_area = _build_regions(mask, image_area)
    image_percent = total_area / image_area if image_area > 0 else 0.0
    threshold = config.OPENCV_DARK_SMOKE_THRESHOLD
    status = "Dark Smoke Detected" if image_percent >= threshold else "No Dark Smoke"
    reason = f"{len(regions)} region(s), image_percent={image_percent:.4f}, threshold={threshold}, min_area={min_area:.1f}"
    return {
        "status": status,
        "regions": regions,
        "union_box": union_box,
        "area": total_area,
        "image_percent": image_percent,
        "method": "opencv_dark_smoke",
        "reason": reason,
    }
