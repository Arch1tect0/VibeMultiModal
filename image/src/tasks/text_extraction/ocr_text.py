"""Sign-gated Tesseract OCR detector.

Responsible final output column:
- Text_Extracted

The detector first finds likely sign/placard regions and only runs OCR inside
those crops. This prevents most watermarks, timestamps, and image-wide overlay
text from being written to the final output.

This version is more tolerant of street signs. Street signs often have white
text on blue/green/red reflective backgrounds, angled crops, and partial tree
occlusion. The OCR step now tries multiple sign-specific preprocessing modes
and only accepts text that survives conservative content checks.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytesseract

from src import config
from src.tasks.text_extraction import sign_detector, watermark_filter

Box = Tuple[int, int, int, int]

ROAD_SUFFIXES = {
    "st", "street", "rd", "road", "ln", "lane", "ave", "avenue", "blvd",
    "drive", "dr", "ct", "court", "way", "hwy", "highway", "pkwy", "parkway",
    "pl", "place", "ter", "terrace", "circle", "cir", "trl", "trail",
}


def _cfg(name: str, default):
    return getattr(config, name, default)


def _clean_text(text: str) -> str:
    text = watermark_filter.normalize_text(text)
    # Keep common sign characters, but drop OCR noise runs.
    text = re.sub(r"[^A-Za-z0-9#&./\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -./|")
    return text


def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text or "")


def _has_road_suffix(text: str) -> bool:
    toks = [t.lower().rstrip(".") for t in _tokens(text)]
    return any(t in ROAD_SUFFIXES for t in toks)


def _looks_like_stop_sign(text: str) -> bool:
    return "stop" in (text or "").lower()


def _looks_like_license_plate(text: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
    if not (4 <= len(cleaned) <= 10):
        return False
    # Require a mix of letters and numbers for ordinary plates. This rejects
    # most random words and OCR fragments while accepting values like XD017,
    # 29A33185, ABC123, etc.
    has_alpha = any(ch.isalpha() for ch in cleaned)
    has_digit = any(ch.isdigit() for ch in cleaned)
    return has_alpha and has_digit


def _is_meaningful_sign_text(text: str, confidence: float, sign_reason: str = "") -> bool:
    text = _clean_text(text)
    if not text:
        return False
    toks = _tokens(text)
    alnum_chars = sum(ch.isalnum() for ch in text)
    punctuation_chars = sum((not ch.isalnum() and not ch.isspace()) for ch in text)
    punct_fraction = punctuation_chars / max(len(text), 1)

    if alnum_chars < _cfg("OCR_MIN_ALNUM_CHARS", 3):
        return False
    if punct_fraction > _cfg("OCR_MAX_PUNCTUATION_FRACTION", 0.35):
        return False

    long_tokens = [t for t in toks if len(t) >= _cfg("OCR_MIN_TOKEN_LENGTH", 2)]
    has_sign_keyword = _has_road_suffix(text) or _looks_like_stop_sign(text)
    is_color_sign = "sign_color" in (sign_reason or "")
    is_license_plate = "license_plate" in (sign_reason or "")

    # License plates are short alphanumeric strings; they should not be rejected
    # just because they lack dictionary words or road suffixes.
    if is_license_plate and _looks_like_license_plate(text) and confidence >= _cfg("OCR_LICENSE_PLATE_MIN_CONFIDENCE", 0.18):
        return True

    # Street signs are often difficult for Tesseract. Allow lower OCR confidence
    # when the text contains a road suffix or STOP and came from a color-sign crop.
    if has_sign_keyword and is_color_sign and confidence >= _cfg("OCR_STREET_SIGN_MIN_CONFIDENCE", 0.20):
        lower_tokens = [t.lower().rstrip(".") for t in toks]
        # A road suffix should usually appear at the end of a street-sign read,
        # e.g. "Glencrest Ln". If it appears in the middle of a long noisy
        # string, treat it as OCR clutter instead of accepting the whole line.
        suffix_positions = [i for i, t in enumerate(lower_tokens) if t in ROAD_SUFFIXES]
        suffix_near_end = any(i >= max(0, len(lower_tokens) - 3) for i in suffix_positions)
        has_reasonable_length = len(lower_tokens) <= 6
        if _looks_like_stop_sign(text) or (suffix_near_end and has_reasonable_length):
            return True

    if confidence < _cfg("OCR_MIN_CONFIDENCE", 0.60):
        return False
    return len(long_tokens) >= _cfg("OCR_MIN_MEANINGFUL_TOKENS", 1)


def _resize_for_ocr(crop, scale: float = None):
    scale = scale or max(2.0, _cfg("OCR_SCALE_FACTOR", 2.0))
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _preprocess_variants(crop, sign_reason: str = ""):
    """Yield (name, processed_image) OCR variants for a sign crop."""
    variants = []
    scaled = _resize_for_ocr(crop, scale=_cfg("OCR_STREET_SIGN_SCALE_FACTOR", 3.0))
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)

    # License plates usually have dark text on a bright panel.
    if "license_plate" in (sign_reason or ""):
        variants.append(("gray", gray))
        variants.append(("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))
        variants.append(("otsu_inv", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]))
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
        variants.append(("adaptive", adaptive))
    # Color street signs usually have white text on saturated backgrounds. A
    # white-pixel mask often reads street names better than ordinary thresholding.
    elif "sign_color" in (sign_reason or ""):
        hsv = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 95]), np.array([180, 105, 255]))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
        variants.append(("white_text_mask", white_mask))
        variants.append(("gray", gray))
        variants.append(("otsu_inv", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]))
    else:
        variants.append(("gray", gray))
        variants.append(("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))

    return variants


def _ocr_image(processed, psm: int) -> Dict:
    cfg = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT, config=cfg)
    words: List[str] = []
    confidences: List[float] = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        text = _clean_text(text)
        if not text:
            continue
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence <= 0:
            continue
        words.append(text)
        confidences.append(confidence / 100.0 if confidence > 1 else confidence)

    text = _clean_text(" ".join(words))
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Fallback to image_to_string because Tesseract data sometimes splits sign
    # lettering poorly and returns no useful word-level confidence.
    if not text:
        raw = _clean_text(pytesseract.image_to_string(processed, config=cfg))
        if raw:
            text = raw
            avg_confidence = 0.0
    return {"text": text, "confidence": round(avg_confidence, 3)}


def _ocr_string_candidate(processed, psm: int) -> Dict:
    cfg = f"--oem 3 --psm {psm}"
    text = _clean_text(pytesseract.image_to_string(processed, config=cfg))
    # image_to_string has no word-level confidence. Use a modest synthetic
    # confidence so it can rescue clear street-sign text with a road suffix,
    # but cannot beat high-confidence OCR data by itself.
    return {"text": text, "confidence": 0.25 if text else 0.0}


def _candidate_score(text: str, confidence: float, variant_name: str) -> float:
    toks = _tokens(text)
    score = confidence
    if _has_road_suffix(text):
        score += 0.35
    if _looks_like_stop_sign(text):
        score += 0.35
    if _looks_like_license_plate(text):
        score += 0.32
    if any(len(t) >= 4 for t in toks):
        score += 0.12
    score += min(0.15, 0.03 * len(toks))
    if "white_text_mask" in variant_name:
        score += 0.06
    return score


def _ocr_crop(crop, crop_box: Box, full_image_shape, sign_reason: str = "") -> Dict:
    candidates: List[Dict] = []
    psm_values = _cfg("OCR_PSM_VALUES", "6,7,11")
    if isinstance(psm_values, str):
        psm_values = [int(v.strip()) for v in psm_values.split(",") if v.strip()]

    for variant_name, processed in _preprocess_variants(crop, sign_reason):
        for psm in psm_values:
            result_sources = [("data", _ocr_image(processed, psm))]
            if psm in (6, 7):
                result_sources.append(("string", _ocr_string_candidate(processed, psm)))
            for source, result in result_sources:
                text = _clean_text(result["text"])
                confidence = float(result.get("confidence", 0.0))
                if not text:
                    continue
                # Keep rejected candidates for diagnostics, but only accept meaningful
                # sign text. This blocks tiny OCR hallucinations such as "A | 2".
                accepted_by_content = _is_meaningful_sign_text(text, confidence, sign_reason)
                rejected_as_watermark = watermark_filter.is_watermark_like(text, crop_box, full_image_shape, confidence)
                if accepted_by_content and "sign_color" in (sign_reason or "") and (_has_road_suffix(text) or _looks_like_stop_sign(text)):
                    # A road suffix/STOP from a sign-colored crop is not a watermark.
                    rejected_as_watermark = False
                if accepted_by_content and "license_plate" in (sign_reason or "") and _looks_like_license_plate(text):
                    # A plausible plate from a plate crop is target text, not an overlay.
                    rejected_as_watermark = False
                candidates.append({
                    "text": "" if (not accepted_by_content or rejected_as_watermark) else text,
                    "raw_text": text,
                    "confidence": round(confidence, 3),
                    "variant": f"{variant_name}:{source}",
                    "psm": psm,
                    "accepted_by_content": accepted_by_content,
                    "rejected_as_watermark": rejected_as_watermark,
                    "score": _candidate_score(text, confidence, variant_name),
                })

    accepted = [c for c in candidates if c.get("text")]
    if accepted:
        best = max(accepted, key=lambda c: c["score"])
        return {
            "text": best["text"],
            "raw_text": best["raw_text"],
            "confidence": round(max(best["confidence"], min(1.0, best["score"])), 3),
            "box": crop_box,
            "variant": best["variant"],
            "psm": best["psm"],
            "rejected_as_watermark": False,
            "all_candidates": candidates[:20],
        }

    # Report the best raw candidate for diagnostics even when nothing was accepted.
    best_raw = max(candidates, key=lambda c: c["score"], default={"raw_text": "", "confidence": 0.0, "rejected_as_watermark": True})
    return {
        "text": "",
        "raw_text": best_raw.get("raw_text", ""),
        "confidence": round(float(best_raw.get("confidence", 0.0)), 3),
        "box": crop_box,
        "variant": best_raw.get("variant", ""),
        "psm": best_raw.get("psm", ""),
        "rejected_as_watermark": True,
        "all_candidates": candidates[:20],
    }


def detect_details(image) -> Dict:
    """Return OCR details including sign boxes and rejected watermark text."""
    if image is None or config.OCR_ENGINE == "none":
        return {"text": "", "sign_regions": [], "ocr_results": [], "watermark_filtered": []}

    height, width = image.shape[:2]
    sign_regions = sign_detector.detect(image) if config.SIGN_DETECTION_ENABLED else []

    # By default, do not OCR the whole image. Whole-image OCR is what causes
    # watermark text to dominate Text_Extracted.
    if not sign_regions and not config.OCR_ALLOW_FULL_IMAGE_FALLBACK:
        return {"text": "", "sign_regions": [], "ocr_results": [], "watermark_filtered": []}

    if not sign_regions:
        sign_regions = [{
            "box": (0, 0, width, height),
            "confidence": 0.0,
            "reason": "full-image fallback enabled",
        }]

    accepted_text: List[str] = []
    rejected_text: List[str] = []
    ocr_results: List[Dict] = []
    base_pad = config.SIGN_CROP_PADDING

    for region in sign_regions:
        x, y, w, h = region["box"]
        sign_reason = region.get("reason", "")
        # Street-sign OCR benefits from a little more padding, but too much
        # padding admits trees/poles. Keep it modest.
        pad = max(base_pad, 4) if "sign_color" in sign_reason else base_pad
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)
        crop = region.get("ocr_crop")
        crop_box = (x1, y1, x2 - x1, y2 - y1)
        if crop is None:
            crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        result = _ocr_crop(crop, crop_box, image.shape, sign_reason=sign_reason)
        if region.get("ocr_crop_source"):
            result["ocr_crop_source"] = region.get("ocr_crop_source")
        result["sign_confidence"] = region.get("confidence", 0.0)
        result["sign_reason"] = sign_reason
        ocr_results.append(result)
        if result["text"]:
            accepted_text.append(result["text"])
        elif result["raw_text"]:
            rejected_text.append(result["raw_text"])

    return {
        "text": watermark_filter.join_filtered_text(accepted_text),
        "sign_regions": sign_regions,
        "ocr_results": ocr_results,
        "watermark_filtered": rejected_text,
    }


def detect(image) -> str:
    """Backward-compatible API used by analyze_images.py."""
    return detect_details(image).get("text", "")
