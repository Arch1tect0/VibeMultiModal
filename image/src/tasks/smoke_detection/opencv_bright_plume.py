"""OpenCV bright/white smoke plume detector.

Targets large white or light-gray smoke plumes. The original smoke detector
intentionally excluded very bright pixels to avoid sky/highlight false positives;
that can miss large white smoke plumes. This module allows bright smoke but uses
connected-region size and edge/texture checks to reduce obvious uniform sky.
"""

from typing import Dict

import cv2
import numpy as np

from src import config


def detect(image) -> Dict:
    if image is None:
        return {"status": "No Image", "regions": [], "union_box": None, "area": 0.0, "image_percent": 0.0, "texture_score": 0.0, "method": "no_image", "reason": "image could not be read"}

    h, w = image.shape[:2]
    image_area = float(h * w)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bright plume candidate: low saturation and bright value, including very bright whites.
    lower = np.array([0, 0, 135], dtype=np.uint8)
    upper = np.array([179, 95, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Exclude broad, perfectly uniform white/blue sky by requiring local intensity variation.
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    texture = cv2.absdiff(gray, blur)
    texture_mask = cv2.inRange(texture, config.OPENCV_BRIGHT_PLUME_MIN_TEXTURE, 255)
    mask = cv2.bitwise_and(mask, texture_mask)

    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(250.0, config.OPENCV_BRIGHT_PLUME_MIN_REGION_PERCENT * image_area)
    regions = []
    total_area = 0.0
    texture_weighted = 0.0
    xs1, ys1, xs2, ys2 = [], [], [], []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        region_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=-1)
        mean_texture = float(cv2.mean(texture, mask=region_mask)[0])
        if mean_texture < config.OPENCV_BRIGHT_PLUME_REGION_MIN_TEXTURE:
            continue
        regions.append({"box": (float(x), float(y), float(x + bw), float(y + bh)), "area": area, "mean_texture": mean_texture})
        total_area += area
        texture_weighted += mean_texture * area
        xs1.append(x); ys1.append(y); xs2.append(x + bw); ys2.append(y + bh)

    union_box = None
    if regions:
        union_box = (float(min(xs1)), float(min(ys1)), float(max(xs2)), float(max(ys2)))

    image_percent = total_area / image_area if image_area > 0 else 0.0
    texture_score = texture_weighted / total_area if total_area > 0 else 0.0
    threshold = config.OPENCV_BRIGHT_PLUME_THRESHOLD
    status = "Bright Smoke Plume Detected" if image_percent >= threshold else "No Bright Smoke Plume"
    reason = f"{len(regions)} region(s), image_percent={image_percent:.4f}, threshold={threshold}, texture_score={texture_score:.2f}, min_area={min_area:.1f}"
    return {
        "status": status,
        "regions": regions,
        "union_box": union_box,
        "area": total_area,
        "image_percent": image_percent,
        "texture_score": texture_score,
        "method": "opencv_bright_plume",
        "reason": reason,
    }
