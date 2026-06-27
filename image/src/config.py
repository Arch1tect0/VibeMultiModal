"""Shared configuration for the modular incident-image analyzer.

This version separates fire detection from smoke detection. Fire and smoke have
independent detector lists, independent decision engines, and a final scene
fusion step.
"""

import os
from pathlib import Path

import cv2
import pytesseract
import torch
from dotenv import load_dotenv

APP_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(APP_ROOT / ".env")

DATASET_ROOT = Path(os.getenv("DATASET_ROOT", APP_ROOT / "dataset"))
if not DATASET_ROOT.is_absolute():
    DATASET_ROOT = APP_ROOT / DATASET_ROOT

DEFAULT_SPLITS = ["train", "valid", "test"]
OUTPUT_DIR = APP_ROOT / "outputs"
ANNOTATED_DIR = OUTPUT_DIR / "annotated_images"
CSV_PATH = OUTPUT_DIR / "results.csv"
REPORTS_DIR = OUTPUT_DIR / "engine_reports"

OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract").strip().lower()
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
CSV_UPDATE_INTERVAL = int(os.getenv("CSV_UPDATE_INTERVAL", "10"))

YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov8m.pt").strip()
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
USE_YOLO = os.getenv("USE_YOLO", "true").strip().lower() == "true"

FIRE_CLASSIFIER_MODEL = os.getenv("FIRE_CLASSIFIER_MODEL", "EdBianchi/vit-fire-detection")
FIRE_THRESHOLD = float(os.getenv("FIRE_THRESHOLD", "0.60"))
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
CLIP_CONTEXT_THRESHOLD = float(os.getenv("CLIP_CONTEXT_THRESHOLD", "0.24"))
CLIP_VISIBLE_FIRE_THRESHOLD = float(os.getenv("CLIP_VISIBLE_FIRE_THRESHOLD", "0.42"))

# Smoke is now independent of the fire decision.
CLIP_SMOKE_THRESHOLD = float(os.getenv("CLIP_SMOKE_THRESHOLD", "0.42"))
CLIP_SMOKE_PLUME_THRESHOLD = float(os.getenv("CLIP_SMOKE_PLUME_THRESHOLD", "0.34"))
OPENCV_SMOKE_THRESHOLD = float(os.getenv("OPENCV_SMOKE_THRESHOLD", "0.015"))
OPENCV_DARK_SMOKE_THRESHOLD = float(os.getenv("OPENCV_DARK_SMOKE_THRESHOLD", "0.010"))
OPENCV_DARK_SMOKE_MIN_REGION_PERCENT = float(os.getenv("OPENCV_DARK_SMOKE_MIN_REGION_PERCENT", "0.002"))
OPENCV_BRIGHT_PLUME_THRESHOLD = float(os.getenv("OPENCV_BRIGHT_PLUME_THRESHOLD", "0.020"))
OPENCV_BRIGHT_PLUME_MIN_REGION_PERCENT = float(os.getenv("OPENCV_BRIGHT_PLUME_MIN_REGION_PERCENT", "0.004"))
OPENCV_BRIGHT_PLUME_MIN_TEXTURE = int(os.getenv("OPENCV_BRIGHT_PLUME_MIN_TEXTURE", "3"))
OPENCV_BRIGHT_PLUME_REGION_MIN_TEXTURE = float(os.getenv("OPENCV_BRIGHT_PLUME_REGION_MIN_TEXTURE", "4.0"))
SMOKE_REQUIRE_AGREEMENT = os.getenv("SMOKE_REQUIRE_AGREEMENT", "true").strip().lower() == "true"
SMOKE_WEAK_EVIDENCE_SCORE = float(os.getenv("SMOKE_WEAK_EVIDENCE_SCORE", "0.55"))
SMOKE_WEAK_EVIDENCE_MIN_METHODS = int(os.getenv("SMOKE_WEAK_EVIDENCE_MIN_METHODS", "2"))
SMOKE_UNCERTAIN_SCORE = float(os.getenv("SMOKE_UNCERTAIN_SCORE", "0.40"))
SMOKE_LOCALIZERS_SUPPORT_ONLY = os.getenv("SMOKE_LOCALIZERS_SUPPORT_ONLY", "true").strip().lower() == "true"
SMOKE_CLASSIFIER_WEAK_SUPPORT_SCORE = float(os.getenv("SMOKE_CLASSIFIER_WEAK_SUPPORT_SCORE", "0.30"))
SMOKE_NO_SMOKE_VETO_SCORE = float(os.getenv("SMOKE_NO_SMOKE_VETO_SCORE", "0.70"))

# Smoke fusion tuning. These prevent weak plume/color hits from overriding a
# strong explicit No Smoke classifier result. OpenCV/localizer detectors only
# add support/confidence; they do not independently declare Smoke.
SMOKE_STRONG_NO_SMOKE_VETO_SCORE = float(os.getenv("SMOKE_STRONG_NO_SMOKE_VETO_SCORE", "0.80"))
SMOKE_PRIMARY_NO_SMOKE_VETO_SCORE = float(os.getenv("SMOKE_PRIMARY_NO_SMOKE_VETO_SCORE", "0.85"))
SMOKE_POSITIVE_CLASSIFIER_OVERRIDE_SCORE = float(os.getenv("SMOKE_POSITIVE_CLASSIFIER_OVERRIDE_SCORE", "0.65"))
SMOKE_LOCALIZER_CONFIDENCE_BONUS = float(os.getenv("SMOKE_LOCALIZER_CONFIDENCE_BONUS", "0.15"))

MIN_FIRE_PIXEL_PERCENT = float(os.getenv("MIN_FIRE_PIXEL_PERCENT", "0.002"))
COLOR_FIRE_OVERRIDE_PERCENT = float(os.getenv("COLOR_FIRE_OVERRIDE_PERCENT", "0.006"))

DEVICE = 0 if torch.cuda.is_available() else -1
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

REFERENCE_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train", "boat",
}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train", "boat"}

# Detector-level modularity. Add/remove names here or pass --fire-detectors / --smoke-detectors.
ENABLED_FIRE_DETECTORS = [
    name.strip() for name in os.getenv(
        "ENABLED_FIRE_DETECTORS",
        "vit_fire,opencv_flame,clip_fire_evidence",
    ).split(",") if name.strip()
]
ENABLED_SMOKE_DETECTORS = [
    name.strip() for name in os.getenv(
        "ENABLED_SMOKE_DETECTORS",
        "clip_smoke,clip_smoke_plume,opencv_smoke,opencv_dark_smoke,opencv_bright_plume",
    ).split(",") if name.strip()
]
ENABLED_CONTEXT_DETECTORS = [
    name.strip() for name in os.getenv(
        "ENABLED_CONTEXT_DETECTORS",
        "yolo_objects,ocr_text",
    ).split(",") if name.strip()
]

FIRE_TYPE_PROMPTS = {
    "Structure Fire": "a photo of a building, apartment, house, warehouse, room, balcony, or structure with visible flames",
    "Vehicle Fire": "a photo of a car, truck, bus, van, motorcycle, or other vehicle with visible flames",
    "Forest Fire": "a photo of a forest wildfire with trees and flames",
    "Grass Fire": "a photo of grass, brush, field, roadside vegetation, or open land burning",
    "HazMat Fire": "a photo of an industrial, chemical, fuel, tanker, hazardous materials, refinery, or factory fire",
    "Container / Trash Fire": "a photo of a small fire in a dumpster, trash bin, barrel, wheelbarrow, grill, or metal container",
    "Other Fire": "a photo of visible flames or active burning that does not fit another category",
}

FIRE_EVIDENCE_PROMPTS = {
    "Visible Fire": "a photo with visible orange flames or active burning",
    "No Visible Fire": "a photo with no visible flames or active burning",
}

SMOKE_PROMPTS = {
    "Smoke": "a photo with visible smoke, smoke plume, haze, fumes, or ash cloud",
    "No Smoke": "a clear photo with no visible smoke, haze, fumes, or ash cloud",
}

RESULT_COLUMNS = ["Image_ID", "Scene_Type", "Objects_Detected", "Text_Extracted", "Fire_Detection_Confidence", "Smoke_Detection_Confidence", "Fire_Classification_Confidence", "Scene_Decision_Confidence"]

# Sign-first OCR settings. OCR now runs only inside sign-like regions by default
# so watermark/overlay text is not written to Text_Extracted.
SIGN_DETECTION_ENABLED = os.getenv("SIGN_DETECTION_ENABLED", "true").strip().lower() == "true"
OCR_ALLOW_FULL_IMAGE_FALLBACK = os.getenv("OCR_ALLOW_FULL_IMAGE_FALLBACK", "false").strip().lower() == "true"
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.45"))
OCR_SCALE_FACTOR = float(os.getenv("OCR_SCALE_FACTOR", "2.0"))

# OCR filtering/tuning. These are referenced by the text extraction code and
# must exist even when not set in .env. They keep OCR conservative while still
# allowing short license plates and difficult street-sign reads.
OCR_MIN_ALNUM_CHARS = int(os.getenv("OCR_MIN_ALNUM_CHARS", "3"))
OCR_MAX_PUNCTUATION_FRACTION = float(os.getenv("OCR_MAX_PUNCTUATION_FRACTION", "0.35"))
OCR_MIN_TOKEN_LENGTH = int(os.getenv("OCR_MIN_TOKEN_LENGTH", "2"))
OCR_MIN_MEANINGFUL_TOKENS = int(os.getenv("OCR_MIN_MEANINGFUL_TOKENS", "1"))
OCR_STREET_SIGN_MIN_CONFIDENCE = float(os.getenv("OCR_STREET_SIGN_MIN_CONFIDENCE", "0.20"))
OCR_STREET_SIGN_SCALE_FACTOR = float(os.getenv("OCR_STREET_SIGN_SCALE_FACTOR", "3.0"))
OCR_LICENSE_PLATE_MIN_CONFIDENCE = float(os.getenv("OCR_LICENSE_PLATE_MIN_CONFIDENCE", "0.18"))
OCR_PSM_VALUES = os.getenv("OCR_PSM_VALUES", "6,7,11")

TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG", "--oem 3 --psm 6")

SIGN_MIN_AREA_PERCENT = float(os.getenv("SIGN_MIN_AREA_PERCENT", "0.002"))
SIGN_MAX_AREA_PERCENT = float(os.getenv("SIGN_MAX_AREA_PERCENT", "0.35"))
SIGN_MIN_WIDTH = int(os.getenv("SIGN_MIN_WIDTH", "28"))
SIGN_MIN_HEIGHT = int(os.getenv("SIGN_MIN_HEIGHT", "14"))
SIGN_MAX_REGIONS = int(os.getenv("SIGN_MAX_REGIONS", "5"))
SIGN_MIN_CONFIDENCE = float(os.getenv("SIGN_MIN_CONFIDENCE", "0.38"))
SIGN_CROP_PADDING = int(os.getenv("SIGN_CROP_PADDING", "8"))
SIGN_EDGE_MARGIN_PERCENT = float(os.getenv("SIGN_EDGE_MARGIN_PERCENT", "0.035"))
SIGN_MAX_EDGE_TOUCH_FRACTION = float(os.getenv("SIGN_MAX_EDGE_TOUCH_FRACTION", "0.5"))
SIGN_CANNY_LOW = int(os.getenv("SIGN_CANNY_LOW", "60"))
SIGN_CANNY_HIGH = int(os.getenv("SIGN_CANNY_HIGH", "160"))

WATERMARK_EDGE_MARGIN_PERCENT = float(os.getenv("WATERMARK_EDGE_MARGIN_PERCENT", "0.08"))
WATERMARK_MAX_WIDTH_FRACTION = float(os.getenv("WATERMARK_MAX_WIDTH_FRACTION", "0.55"))
WATERMARK_MAX_HEIGHT_FRACTION = float(os.getenv("WATERMARK_MAX_HEIGHT_FRACTION", "0.20"))
WATERMARK_KEYWORDS = [
    term.strip().lower()
    for term in os.getenv(
        "WATERMARK_KEYWORDS",
        "roboflow,shutterstock,alamy,getty,istock,watermark,preview,sample,dreamstime,depositphotos,rf.,jpg.rf",
    ).split(",")
    if term.strip()
]
