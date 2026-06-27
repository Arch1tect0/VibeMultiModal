"""YOLOv8 object detector.

Responsible output columns:
- Objects_Detected
- Vehicle_Objects
- YOLO_Model
- Object_Detection_Confidence
"""

from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO

from src import config

_yolo_model = None


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(config.YOLO_MODEL_NAME)
    return _yolo_model


def detect(image_path: Path) -> Dict:
    if not config.USE_YOLO:
        return {"detections": [], "model_results": None, "model_name": "disabled"}

    model = get_yolo_model()
    results = model(str(image_path), conf=config.YOLO_CONFIDENCE, verbose=False)
    detections: List[Dict] = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = str(model.names[class_id])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "is_reference": class_name in config.REFERENCE_CLASSES,
            })
    return {"detections": detections, "model_results": results, "model_name": config.YOLO_MODEL_NAME}
