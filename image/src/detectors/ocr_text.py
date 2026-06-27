"""Tesseract OCR detector.

Responsible output column:
- Text_Extracted
"""

import cv2
import pytesseract

from src import config


def detect(image) -> str:
    if image is None or config.OCR_ENGINE == "none":
        return ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    text = pytesseract.image_to_string(gray)
    return " ".join(text.split())
