# Task folders

Each task owns the detector modules for one area of the pipeline. Add or remove detectors by editing `src/config.py` or by passing CLI arguments.

## fire_detection
Fire/no-fire evidence modules.

- `vit_fire.py`: ViT image-level fire classifier. No boxes.
- `opencv_flame.py`: HSV flame-region detector. Produces regions/boxes.
- `clip_fire_evidence.py`: CLIP visible-fire evidence classifier. No boxes.

## smoke_detection
Smoke evidence modules.

- `clip_smoke.py`: CLIP image-level smoke classifier. No boxes.
- `opencv_smoke.py`: OpenCV gray/low-saturation smoke-region detector. Produces regions/boxes.
- `smoke_detection.py`: compatibility helpers for smoke detection.

## fire_classification
Fire type/category classification after the fire decision.

- `clip_fire_type.py`: CLIP fire type classifier, such as vehicle/structure/grass/fire.

## object_detection
Scene/object context detectors.

- `yolo_objects.py`: YOLO object detector for vehicles, people, and reference objects.

## text_extraction
OCR and text extraction.

- `ocr_text.py`: Tesseract OCR wrapper.

## shared
Code reused by multiple task folders.

- `clip_common.py`: shared CLIP zero-shot helper.

## Decision engines
Task detectors do not make the final scene decision directly. The outputs are fused by:

- `src/core/fire_decision_engine.py`
- `src/core/smoke_decision_engine.py`
- `src/core/scene_decision_engine.py`
