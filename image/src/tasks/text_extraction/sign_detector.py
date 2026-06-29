"""Lightweight sign-region detector for OCR gating.

This module does not try to read text. It finds likely sign/placard/panel
regions and returns crop boxes. OCR should run only on these crops so
watermarks and image-wide overlays are ignored by default.

This version adds a traffic/street-sign color candidate path for blue, green,
and red signs. It can also use an optional learned detector (Ultralytics YOLO or
YOLO-World) before falling back to OpenCV heuristics.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src import config

Box = Tuple[int, int, int, int]


def _cfg(name: str, default):
    """Read optional config values without requiring a config.py patch."""
    return getattr(config, name, default)


def _clip_box(box: Box, width: int, height: int) -> Box:
    x, y, w, h = box
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def _edge_touch_fraction(box: Box, width: int, height: int) -> float:
    x, y, w, h = box
    margin_x = max(2, int(width * config.SIGN_EDGE_MARGIN_PERCENT))
    margin_y = max(2, int(height * config.SIGN_EDGE_MARGIN_PERCENT))
    touches = 0
    touches += int(x <= margin_x)
    touches += int(y <= margin_y)
    touches += int(x + w >= width - margin_x)
    touches += int(y + h >= height - margin_y)
    return touches / 4.0


def _fire_like_fraction(crop) -> float:
    """Return fraction of pixels that look flame-colored.

    This helps reject false sign candidates created by fire, glowing windows,
    and flame edges. Legitimate signs may contain some red/orange, but a crop
    dominated by flame colors should not be treated as a sign region.
    """
    if crop is None or crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    orange_yellow = cv2.inRange(hsv, np.array([8, 70, 90]), np.array([45, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), orange_yellow)
    return float(cv2.countNonZero(mask)) / float(crop.shape[0] * crop.shape[1])


def _candidate_reject_reason(image, box: Box, contour_area: float) -> str:
    height, width = image.shape[:2]
    x, y, w, h = _clip_box(box, width, height)
    area = float(w * h)
    image_area = float(width * height)
    if area <= 0 or image_area <= 0:
        return "empty candidate"
    area_pct = area / image_area
    aspect = w / max(float(h), 1.0)
    extent = float(contour_area) / area if contour_area > 0 else 0.0
    crop = image[y:y + h, x:x + w]
    fire_fraction = _fire_like_fraction(crop)

    if area_pct < _cfg("SIGN_MIN_AREA_PERCENT", 0.001):
        return f"area too small {area_pct:.4f} < {_cfg("SIGN_MIN_AREA_PERCENT", 0.001):.4f}"
    if area_pct > _cfg("SIGN_MAX_AREA_PERCENT", 0.25):
        return f"area too large {area_pct:.4f} > {_cfg("SIGN_MAX_AREA_PERCENT", 0.25):.4f}"
    if w < _cfg("SIGN_MIN_WIDTH", 24) or h < _cfg("SIGN_MIN_HEIGHT", 12):
        return f"dimensions too small {w}x{h}"
    if not (_cfg("SIGN_MIN_ASPECT_RATIO", 0.4) <= aspect <= _cfg("SIGN_MAX_ASPECT_RATIO", 10.0)):
        return f"aspect ratio {aspect:.2f} outside sign range"
    if extent < _cfg("SIGN_MIN_RECTANGULARITY", 0.12):
        return f"rectangularity {extent:.3f} below {_cfg("SIGN_MIN_RECTANGULARITY", 0.12):.3f}"
    if fire_fraction > _cfg("SIGN_MAX_FIRE_COLOR_FRACTION", 0.75):
        return f"fire-colored crop fraction {fire_fraction:.3f} > {_cfg("SIGN_MAX_FIRE_COLOR_FRACTION", 0.75):.3f}"
    if _edge_touch_fraction((x, y, w, h), width, height) >= _cfg("SIGN_MAX_EDGE_TOUCH_FRACTION", 0.75):
        return "candidate touches image edge like overlay/watermark"
    return ""


def _box_score(image, box: Box, contour_area: float) -> float:
    height, width = image.shape[:2]
    x, y, w, h = _clip_box(box, width, height)
    area = float(w * h)
    if area <= 0:
        return 0.0

    crop = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    aspect = w / max(float(h), 1.0)
    extent = min(1.0, float(contour_area) / area) if contour_area > 0 else 0.0
    contrast = max(0.0, min(1.0, float(gray.std()) / 80.0))
    mean_s = float(hsv[:, :, 1].mean())
    mean_v = float(hsv[:, :, 2].mean())
    light_panel = 1.0 if mean_s < 90 and mean_v > 130 else 0.0
    colored_panel = 1.0 if mean_s > 70 and mean_v > 80 else 0.0
    panel_score = max(light_panel, colored_panel)

    aspect_score = 1.0 if 0.6 <= aspect <= 8.0 else 0.4
    edge_penalty = _edge_touch_fraction((x, y, w, h), width, height)
    huge_penalty = 0.5 if area / float(width * height) > _cfg("SIGN_MAX_AREA_PERCENT", 0.25) else 0.0

    score = (0.30 * extent) + (0.25 * contrast) + (0.25 * panel_score) + (0.20 * aspect_score)
    score -= 0.35 * edge_penalty
    score -= huge_penalty
    return max(0.0, min(1.0, score))



@lru_cache(maxsize=1)
def _load_yolo_model():
    """Load an optional learned sign detector.

    Works with either a custom traffic-sign YOLO checkpoint or an Ultralytics
    YOLO-World checkpoint. If the dependency/model is unavailable, return None
    and let the OpenCV paths continue.
    """
    if not _cfg("SIGN_YOLO_ENABLED", True):
        return None
    model_name = str(_cfg("SIGN_YOLO_MODEL", "yolov8s-worldv2.pt")).strip()
    if not model_name:
        return None
    try:
        from ultralytics import YOLO, YOLOWorld
    except Exception:
        return None

    try:
        looks_like_world = "world" in model_name.lower()
        model_cls = YOLOWorld if looks_like_world else YOLO
        model = model_cls(model_name)
        prompts = [
            item.strip()
            for item in str(_cfg(
                "SIGN_YOLO_PROMPTS",
                "street sign,road sign,highway sign,traffic sign,stop sign,route sign,exit sign",
            )).split(",")
            if item.strip()
        ]
        if looks_like_world and prompts and hasattr(model, "set_classes"):
            model.set_classes(prompts)
        return model
    except Exception:
        return None


def _model_names(model) -> Dict[int, str]:
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {idx: str(name) for idx, name in enumerate(names)}


def _is_sign_class(label: str) -> bool:
    label_l = (label or "").lower().replace("_", " ").replace("-", " ")
    keywords = [
        item.strip().lower()
        for item in str(_cfg(
            "SIGN_YOLO_CLASS_KEYWORDS",
            "sign,street sign,road sign,highway sign,traffic sign,stop sign,route sign,exit sign",
        )).split(",")
        if item.strip()
    ]
    return any(keyword in label_l for keyword in keywords)


def _learned_sign_candidates(image) -> List[Dict]:
    """Return sign boxes from an optional YOLO/YOLO-World detector."""
    model = _load_yolo_model()
    if model is None or image is None:
        return []

    height, width = image.shape[:2]
    image_area = float(width * height)
    if image_area <= 0:
        return []

    try:
        results = model.predict(
            image,
            conf=float(_cfg("SIGN_YOLO_CONFIDENCE", 0.05)),
            iou=float(_cfg("SIGN_YOLO_IOU", 0.50)),
            verbose=False,
        )
    except Exception:
        return []

    names = _model_names(model)
    regions: List[Dict] = []
    max_area = float(_cfg("SIGN_YOLO_MAX_AREA_PERCENT", 0.40))
    min_area = float(_cfg("SIGN_YOLO_MIN_AREA_PERCENT", 0.0002))
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            try:
                xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                cls_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1
                conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            except Exception:
                continue
            label = names.get(cls_id, str(cls_id))
            if not _is_sign_class(label):
                continue
            x1, y1, x2, y2 = xyxy
            x = int(round(max(0, min(x1, width - 1))))
            y = int(round(max(0, min(y1, height - 1))))
            w = int(round(max(1, min(x2, width) - x)))
            h = int(round(max(1, min(y2, height) - y)))
            area_pct = (w * h) / image_area
            if area_pct < min_area or area_pct > max_area:
                continue
            if w < _cfg("SIGN_YOLO_MIN_WIDTH", 12) or h < _cfg("SIGN_YOLO_MIN_HEIGHT", 8):
                continue
            regions.append({
                "box": (x, y, w, h),
                "confidence": round(conf, 3),
                "reason": f"learned_sign_detector label={label}, area={area_pct:.4f}, score={conf:.3f}",
            })
    return regions

def _color_sign_masks(image) -> List[Tuple[str, np.ndarray]]:
    """Return HSV masks for common traffic/street-sign colors.

    This path is intentionally separate from the generic contour path. It is
    designed to catch blue/green/red street signs that are not clean white
    panels and may be angled, partially occluded, or non-rectangular.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Blue street signs: fairly saturated, medium-bright to bright.
    blue = cv2.inRange(hsv, np.array([90, 55, 55]), np.array([135, 255, 255]))
    # Green street signs.
    green = cv2.inRange(hsv, np.array([35, 45, 45]), np.array([90, 255, 255]))
    # Red signs, including STOP signs. Red wraps around the HSV hue boundary.
    red1 = cv2.inRange(hsv, np.array([0, 55, 55]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 55, 55]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(red1, red2)
    return [("blue_sign_color", blue), ("green_sign_color", green), ("red_sign_color", red)]


def _white_text_fraction(crop) -> float:
    """Estimate whether a colored sign crop has bright text/border pixels."""
    if crop is None or crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # White paint/letters: bright and low saturation.
    white = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 85, 255]))
    return float(cv2.countNonZero(white)) / float(crop.shape[0] * crop.shape[1])


def _color_candidate_reject_reason(image, box: Box, contour_area: float, color_name: str) -> str:
    height, width = image.shape[:2]
    x, y, w, h = _clip_box(box, width, height)
    area = float(w * h)
    image_area = float(width * height)
    if area <= 0 or image_area <= 0:
        return "empty color candidate"
    area_pct = area / image_area
    aspect = w / max(float(h), 1.0)
    extent = float(contour_area) / area if contour_area > 0 else 0.0
    crop = image[y:y + h, x:x + w]
    white_fraction = _white_text_fraction(crop)
    fire_fraction = _fire_like_fraction(crop)

    min_area = _cfg("SIGN_COLOR_MIN_AREA_PERCENT", 0.0025)
    max_area = _cfg("SIGN_COLOR_MAX_AREA_PERCENT", 0.20)
    min_w = _cfg("SIGN_COLOR_MIN_WIDTH", 24)
    min_h = _cfg("SIGN_COLOR_MIN_HEIGHT", 12)
    max_edge_touch = _cfg("SIGN_COLOR_MAX_EDGE_TOUCH_FRACTION", 0.75)
    min_white = _cfg("SIGN_COLOR_MIN_WHITE_TEXT_FRACTION", 0.006)
    min_extent = _cfg("SIGN_COLOR_MIN_EXTENT", 0.18)

    if area_pct < min_area:
        return f"color sign area too small {area_pct:.4f} < {min_area:.4f}"
    if area_pct > max_area:
        return f"color sign area too large {area_pct:.4f} > {max_area:.4f}"
    if w < min_w or h < min_h:
        return f"color sign dimensions too small {w}x{h}"
    if _edge_touch_fraction((x, y, w, h), width, height) >= max_edge_touch:
        return "color candidate touches image edge like overlay/watermark"
    # Red STOP signs are not rectangular, so allow a wider aspect range. Blue
    # and green street signs are usually wide rectangles.
    if color_name == "red_sign_color":
        if not (0.55 <= aspect <= 2.2):
            return f"red sign aspect {aspect:.2f} outside stop/sign range"
        # Allow octagons and partial contours.
        if extent < 0.28:
            return f"red sign extent {extent:.3f} below 0.280"
        # Avoid flame-dominated red/orange crops.
        if fire_fraction > _cfg("SIGN_RED_MAX_FIRE_COLOR_FRACTION", 0.45):
            return f"red candidate too flame-like {fire_fraction:.3f}"
    else:
        if not (1.2 <= aspect <= 12.0):
            return f"street sign aspect {aspect:.2f} outside wide sign range"
        if extent < min_extent:
            return f"street sign extent {extent:.3f} below {min_extent:.3f}"
    if white_fraction < min_white:
        return f"not enough white lettering/border {white_fraction:.4f} < {min_white:.4f}"
    return ""


def _color_box_score(image, box: Box, contour_area: float, color_name: str) -> float:
    height, width = image.shape[:2]
    x, y, w, h = _clip_box(box, width, height)
    area = float(w * h)
    if area <= 0:
        return 0.0
    crop = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    aspect = w / max(float(h), 1.0)
    extent = min(1.0, float(contour_area) / area) if contour_area > 0 else 0.0
    saturation = max(0.0, min(1.0, float(hsv[:, :, 1].mean()) / 180.0))
    contrast = max(0.0, min(1.0, float(gray.std()) / 75.0))
    white = max(0.0, min(1.0, _white_text_fraction(crop) / 0.06))
    if color_name == "red_sign_color":
        aspect_score = 1.0 if 0.7 <= aspect <= 1.6 else 0.65
    else:
        aspect_score = 1.0 if 1.8 <= aspect <= 9.0 else 0.65
    score = (0.25 * extent) + (0.25 * saturation) + (0.20 * contrast) + (0.20 * white) + (0.10 * aspect_score)
    score -= 0.25 * _edge_touch_fraction((x, y, w, h), width, height)
    return max(0.0, min(1.0, score))


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _rotated_crop_from_contour(image, contour):
    """Return a perspective-normalized crop for angled sign contours."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype("float32")
    ordered = _order_points(box)
    (tl, tr, br, bl) = ordered
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))
    if max_width < 10 or max_height < 10:
        return None
    # Keep street signs landscape when appropriate.
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    if warped.shape[0] > warped.shape[1] and warped.shape[0] / max(warped.shape[1], 1) > 1.4:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def _color_sign_candidates(image) -> List[Dict]:
    height, width = image.shape[:2]
    image_area = float(width * height)
    if image_area <= 0:
        return []

    regions: List[Dict] = []
    # Wider kernel connects street sign panels that are split by glare, text,
    # tree branches, or reflective highlights.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(7, int(width * 0.025)), max(5, int(height * 0.012))),
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for color_name, mask in _color_sign_masks(image):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            if contour_area <= 0:
                continue
            x, y, w, h = _clip_box(cv2.boundingRect(contour), width, height)
            reject_reason = _color_candidate_reject_reason(image, (x, y, w, h), contour_area, color_name)
            if reject_reason:
                continue
            score = _color_box_score(image, (x, y, w, h), contour_area, color_name)
            if score < _cfg("SIGN_COLOR_MIN_CONFIDENCE", 0.45):
                continue
            area_pct = (w * h) / image_area
            reason = f"{color_name} region area={area_pct:.4f}, score={score:.3f}"
            aspect = w / max(float(h), 1.0)
            # A colorful license plate in the lower half of a vehicle image can
            # otherwise look like a small blue/green road sign. Reclassify these
            # compact lower-image color panels so OCR uses plate validation and
            # does not emit a false Street Sign result.
            if (
                _cfg("SIGN_COLOR_LOWER_IMAGE_PLATE_RECLASSIFY", True)
                and color_name in {"blue_sign_color", "green_sign_color"}
                and (y / float(height)) >= _cfg("SIGN_COLOR_PLATE_RECLASSIFY_MIN_Y_PERCENT", 0.45)
                and area_pct <= _cfg("SIGN_COLOR_PLATE_RECLASSIFY_MAX_AREA_PERCENT", 0.030)
                and aspect <= _cfg("SIGN_COLOR_PLATE_RECLASSIFY_MAX_ASPECT", 3.8)
            ):
                reason = f"license_plate_candidate from_{color_name} area={area_pct:.4f}, aspect={aspect:.2f}, score={score:.3f}"
            region = {
                "box": (x, y, w, h),
                "confidence": round(score, 3),
                "reason": reason,
            }
            rotated_crop = _rotated_crop_from_contour(image, contour)
            if rotated_crop is not None and rotated_crop.size > 0:
                region["ocr_crop"] = rotated_crop
                region["ocr_crop_source"] = "rotated_color_contour"
            regions.append(region)
    return regions


def _license_plate_candidates(image) -> List[Dict]:
    """Detect likely license-plate regions as OCR candidates.

    This is intentionally conservative: plates are small, bright rectangular
    panels with internal dark text/edges. It should not turn every white object
    into OCR text, but it gives the OCR engine a crop for vehicle plates that
    are not blue/green/red signs.
    """
    if not _cfg("LICENSE_PLATE_DETECTION_ENABLED", True):
        return []

    height, width = image.shape[:2]
    image_area = float(width * height)
    if image_area <= 0:
        return []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Near-grayscale road scenes are a common source of false plate reads.
    # Handling them well needs slower OCR/detection retries, so skip this
    # optional path by default.
    if _cfg("LICENSE_PLATE_SKIP_LOW_SATURATION", True):
        mean_saturation = float(hsv[:, :, 1].mean())
        if mean_saturation < _cfg("LICENSE_PLATE_MIN_IMAGE_SATURATION", 12.0):
            return []

    # Bright, low-to-moderate saturation rectangular panels. Include slightly
    # colored plates by allowing saturation up to ~120.
    plate_mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 125, 255]))
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), iterations=2)

    contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: List[Dict] = []

    min_area_pct = _cfg("LICENSE_PLATE_MIN_AREA_PERCENT", 0.00035)
    max_area_pct = _cfg("LICENSE_PLATE_MAX_AREA_PERCENT", 0.030)
    min_w = _cfg("LICENSE_PLATE_MIN_WIDTH", 26)
    min_h = _cfg("LICENSE_PLATE_MIN_HEIGHT", 10)
    min_conf = _cfg("LICENSE_PLATE_MIN_CONFIDENCE", 0.42)

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0:
            continue
        x, y, w, h = _clip_box(cv2.boundingRect(contour), width, height)
        area_pct = (w * h) / image_area
        aspect = w / max(float(h), 1.0)
        # Avoid tree/sky/overhead-sign false positives being treated as license plates.
        # Vehicle plates are normally in the lower portion of incident photos.
        min_y_pct = _cfg("LICENSE_PLATE_MIN_Y_PERCENT", 0.30)
        if (y / float(height)) < min_y_pct:
            continue
        if area_pct < min_area_pct or area_pct > max_area_pct:
            continue
        if w < min_w or h < min_h:
            continue
        if not (1.4 <= aspect <= 6.8):
            continue
        if _edge_touch_fraction((x, y, w, h), width, height) >= 0.75:
            continue

        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        crop_gray = gray[y:y + h, x:x + w]
        crop_hsv = hsv[y:y + h, x:x + w]

        # Plates have high contrast characters inside a bright panel.
        contrast = max(0.0, min(1.0, float(crop_gray.std()) / 65.0))
        bright_fraction = float(cv2.countNonZero(cv2.inRange(crop_hsv, np.array([0, 0, 145]), np.array([180, 140, 255])))) / float(w * h)
        edge_density = float(cv2.countNonZero(cv2.Canny(crop_gray, 50, 150))) / float(w * h)
        extent = min(1.0, contour_area / float(w * h))
        aspect_score = 1.0 if 2.0 <= aspect <= 5.2 else 0.65

        score = (0.25 * contrast) + (0.25 * bright_fraction) + (0.20 * min(1.0, edge_density / 0.18)) + (0.15 * extent) + (0.15 * aspect_score)
        if score < min_conf:
            continue

        regions.append({
            "box": (x, y, w, h),
            "confidence": round(max(0.0, min(1.0, score)), 3),
            "reason": f"license_plate_candidate area={area_pct:.4f}, aspect={aspect:.2f}, score={score:.3f}",
        })

    return regions


def _nms(regions: List[Dict], iou_threshold: float = 0.35) -> List[Dict]:
    def iou(a: Box, b: Box) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    kept: List[Dict] = []
    for region in sorted(regions, key=lambda r: r["confidence"], reverse=True):
        if all(iou(region["box"], kept_region["box"]) < iou_threshold for kept_region in kept):
            kept.append(region)
    return kept


def _generic_candidates(image) -> List[Dict]:
    height, width = image.shape[:2]
    image_area = float(width * height)
    if image_area <= 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    # Combine edge rectangles with adaptive-threshold text/panel blobs.
    edges = cv2.Canny(gray, _cfg("SIGN_CANNY_LOW", 50), _cfg("SIGN_CANNY_HIGH", 150))
    kernel_w = max(9, int(width * 0.025))
    kernel_h = max(5, int(height * 0.015))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    combined = cv2.bitwise_or(closed_edges, closed_thresh)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Dict] = []
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0:
            continue
        x, y, w, h = _clip_box(cv2.boundingRect(contour), width, height)
        box_area_pct = (w * h) / image_area
        reject_reason = _candidate_reject_reason(image, (x, y, w, h), contour_area)
        if reject_reason:
            continue
        score = _box_score(image, (x, y, w, h), contour_area)
        if score < _cfg("SIGN_MIN_CONFIDENCE", 0.45):
            continue
        candidates.append({
            "box": (x, y, w, h),
            "confidence": round(score, 3),
            "reason": f"sign-like region area={box_area_pct:.4f}, score={score:.3f}",
        })
    return candidates


def detect(image) -> List[Dict]:
    """Return likely sign regions as dictionaries with box/confidence/reason."""
    if image is None:
        return []

    candidates: List[Dict] = []
    # Learned detector first: it catches street/highway signs that color/contour
    # heuristics often miss. The heuristic paths remain as a no-install fallback.
    candidates.extend(_learned_sign_candidates(image))
    if _cfg("SIGN_COLOR_DETECTION_ENABLED", True):
        candidates.extend(_color_sign_candidates(image))
    candidates.extend(_license_plate_candidates(image))
    candidates.extend(_generic_candidates(image))

    # Keep enough candidates for multiple street signs plus a stop sign.
    max_regions = max(_cfg("SIGN_MAX_REGIONS", 8), _cfg("SIGN_COLOR_MAX_REGIONS", 8))
    return _nms(candidates, iou_threshold=_cfg("SIGN_NMS_IOU", 0.35))[:max_regions]
