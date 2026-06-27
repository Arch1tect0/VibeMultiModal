"""OpenCV flame-color detector.

Responsible output columns:
- OpenCV_Flame_Status
- OpenCV_Flame_Image_Percent
- OpenCV_Flame_Region_Count
- OpenCV_Flame_Reason
- Fire_Image_Percent, via size estimator/final output compatibility
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

    ranges = [
        (np.array([0, 80, 120], dtype=np.uint8), np.array([5, 255, 255], dtype=np.uint8)),
        (np.array([5, 80, 120], dtype=np.uint8), np.array([45, 255, 255], dtype=np.uint8)),
        (np.array([170, 80, 120], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
    ]
    mask = np.zeros((h, w), dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(25.0, config.MIN_FIRE_PIXEL_PERCENT * image_area)
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
        xs1.append(x)
        ys1.append(y)
        xs2.append(x + bw)
        ys2.append(y + bh)

    union_box = None
    if regions:
        union_box = (float(min(xs1)), float(min(ys1)), float(max(xs2)), float(max(ys2)))

    image_percent = total_area / image_area if image_area > 0 else 0.0
    status = "Flame Color Detected" if image_percent > 0 else "No Flame Color"
    reason = f"{len(regions)} region(s), image_percent={image_percent:.4f}, min_area={min_area:.1f}"

    return {
        "status": status,
        "regions": regions,
        "union_box": union_box,
        "area": total_area,
        "image_percent": image_percent,
        "method": "opencv_flame_color",
        "reason": reason,
    }
