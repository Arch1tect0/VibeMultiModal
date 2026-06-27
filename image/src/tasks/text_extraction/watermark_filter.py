"""Heuristics for rejecting OCR text that is likely a watermark/overlay or OCR noise."""

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

from src import config

Box = Tuple[int, int, int, int]


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def has_watermark_keyword(text: str) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in config.WATERMARK_KEYWORDS)


def is_edge_overlay(box: Optional[Box], image_shape) -> bool:
    if box is None or image_shape is None:
        return False
    height, width = image_shape[:2]
    x, y, w, h = box
    if width <= 0 or height <= 0:
        return False
    margin_x = width * config.WATERMARK_EDGE_MARGIN_PERCENT
    margin_y = height * config.WATERMARK_EDGE_MARGIN_PERCENT
    near_edge = x <= margin_x or y <= margin_y or (x + w) >= width - margin_x or (y + h) >= height - margin_y
    spans_width = w / float(width) >= config.WATERMARK_MAX_WIDTH_FRACTION
    spans_height = h / float(height) >= config.WATERMARK_MAX_HEIGHT_FRACTION
    return bool(near_edge and (spans_width or spans_height))


def _meaningful_word_count(text: str) -> int:
    # Count tokens that look like real sign text, not single OCR fragments.
    tokens = re.findall(r"[A-Za-z0-9]+", text or "")
    return sum(1 for token in tokens if len(token) >= config.OCR_MIN_TOKEN_LENGTH)


def rejection_reason(text: str, box: Optional[Box] = None, image_shape=None, avg_confidence: float = 1.0) -> str:
    """Return empty string if accepted, otherwise the reason text was rejected."""
    text = normalize_text(text)
    if not text:
        return "empty OCR text"
    if avg_confidence < config.OCR_MIN_CONFIDENCE:
        return f"OCR confidence {avg_confidence:.3f} below {config.OCR_MIN_CONFIDENCE:.3f}"
    if has_watermark_keyword(text):
        return "watermark keyword detected"
    if is_edge_overlay(box, image_shape):
        return "edge-spanning overlay text"

    lowered = text.lower()
    if "http" in lowered or ".com" in lowered or ".net" in lowered or ".org" in lowered:
        return "URL-like watermark text"

    alnum_chars = sum(ch.isalnum() for ch in text)
    alpha_chars = sum(ch.isalpha() for ch in text)
    if alnum_chars < config.OCR_MIN_ALNUM_CHARS:
        return f"too few alphanumeric characters ({alnum_chars} < {config.OCR_MIN_ALNUM_CHARS})"
    if _meaningful_word_count(text) < config.OCR_MIN_MEANINGFUL_TOKENS:
        return "too few meaningful OCR tokens"

    # Strings such as 'A | 2' often come from flames/window edges. They have
    # very few alphanumerics and a high punctuation ratio.
    punctuation_chars = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
    if punctuation_chars / max(len(text), 1) > config.OCR_MAX_PUNCTUATION_FRACTION:
        return "OCR text is mostly punctuation/fragments"

    # Watermarks often read as long non-word fragments, URLs, or file stamps.
    if len(text) >= 8 and alpha_chars / max(len(text), 1) < 0.35:
        return "long low-alpha OCR fragment"
    return ""


def is_watermark_like(text: str, box: Optional[Box] = None, image_shape=None, avg_confidence: float = 1.0) -> bool:
    return bool(rejection_reason(text, box, image_shape, avg_confidence))


def join_filtered_text(chunks: Iterable[str]) -> str:
    seen = set()
    output = []
    for chunk in chunks:
        chunk = normalize_text(chunk)
        key = chunk.lower()
        if chunk and key not in seen:
            seen.add(key)
            output.append(chunk)
    return " | ".join(output)
