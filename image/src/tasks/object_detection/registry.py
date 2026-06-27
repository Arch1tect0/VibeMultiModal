"""Context detector registry."""

from src.core.detector_registry import DetectorRegistry
from src.tasks.object_detection import yolo_objects
from src.tasks.text_extraction import ocr_text


REGISTRY = DetectorRegistry("context")
REGISTRY.register("yolo_objects", lambda image_path, image: yolo_objects.detect(image_path))


def run_ocr_text(image_path, image):
    details = ocr_text.detect_details(image)
    text = details.get("text", "")
    sign_regions = details.get("sign_regions", [])
    ocr_results = details.get("ocr_results", [])
    accepted_results = [r for r in ocr_results if r.get("text")]
    rejected_results = [r for r in ocr_results if r.get("raw_text") and not r.get("text")]

    if text:
        status = "Text Extracted"
        reason = f"accepted OCR from {len(accepted_results)} sign crop(s)"
    elif sign_regions:
        status = "Sign Detected - No Text"
        reason = f"{len(sign_regions)} sign-like region(s), but OCR produced no accepted text"
    else:
        status = "No Sign Detected"
        reason = "sign-gated OCR skipped because no sign-like region was found"

    confidences = [float(r.get("confidence", 0.0)) for r in accepted_results]
    if not confidences:
        confidences = [float(r.get("sign_confidence", 0.0)) for r in ocr_results]
    confidence = max(confidences) if confidences else 0.0

    details.update({
        "status": status,
        "confidence": confidence,
        "method": "ocr_text",
        "reason": reason,
        "accepted_count": len(accepted_results),
        "rejected_count": len(rejected_results),
    })
    return details


REGISTRY.register("ocr_text", run_ocr_text)


def run_enabled(enabled, image_path, image):
    return REGISTRY.run_enabled(enabled, image_path, image)
