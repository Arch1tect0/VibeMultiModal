"""Simple OpenCV smoke/haze color-region detector.

This is intentionally conservative and transparent. It looks for low-saturation,
mid/high-value gray-white regions and reports them as smoke-color evidence. It is
not expected to be perfect; it exists as a separate detector that can be replaced
or supplemented later.

Responsible output columns:
- OpenCV_Smoke_Status
- OpenCV_Smoke_Image_Percent
- OpenCV_Smoke_Region_Count
- OpenCV_Smoke_Reason
"""

from typing import Dict

import cv2
import numpy as np

from src import config


def detect(image) -> Dict:
    if image is None:
        return {
            "status": "No Image",
            "regions": [],
            "union_box": None,
            "area": 0.0,
            "image_percent": 0.0,
            "method": "no_image",
            "reason": "image could not be read",
        }

    h, w = image.shape[:2]
    image_area = float(h * w)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Smoke often appears as gray/white haze: low saturation, medium to high value.
    lower = np.array([0, 0, 80], dtype=np.uint8)
    upper = np.array([179, 80, 245], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Reduce common false positives from nearly pure white sky/highlights.
    very_bright = cv2.inRange(hsv, np.array([0, 0, 246], dtype=np.uint8), np.array([179, 50, 255], dtype=np.uint8))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(very_bright))

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(100.0, 0.003 * image_area)
    regions = []
    total_area = 0.0
    xs1, ys1, xs2, ys2 = [], [], [], []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        regions.append({"box": (float(x), float(y), float(x + bw), float(y + bh)), "area": area})
        total_area += area
        xs1.append(x); ys1.append(y); xs2.append(x + bw); ys2.append(y + bh)

    union_box = None
    if regions:
        union_box = (float(min(xs1)), float(min(ys1)), float(max(xs2)), float(max(ys2)))

    image_percent = total_area / image_area if image_area > 0 else 0.0
    status = "Smoke Color Detected" if image_percent >= config.OPENCV_SMOKE_THRESHOLD else "No Smoke Color"
    reason = f"{len(regions)} region(s), image_percent={image_percent:.4f}, threshold={config.OPENCV_SMOKE_THRESHOLD}, min_area={min_area:.1f}"

    return {
        "status": status,
        "regions": regions,
        "union_box": union_box,
        "area": total_area,
        "image_percent": image_percent,
        "method": "opencv_smoke_color",
        "reason": reason,
    }
