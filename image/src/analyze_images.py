"""Task-folder incident scene analyzer with slim output and engine reports.

Main CSV intentionally contains only executive output:
Image_ID, Scene_Type, Objects_Detected, Text_Extracted,
Fire_Detection_Confidence, Smoke_Detection_Confidence,
Fire_Classification_Confidence, Scene_Decision_Confidence

Detailed detector/engine audit output is written separately to:
outputs/engine_reports/
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import pandas as pd
from tqdm import tqdm

from src import config
from src.core import decision_engine, fire_decision_engine, scene_decision_engine, smoke_decision_engine
from src.core.io import get_image_paths, get_split_name
from src.output.annotator import annotation_sources, build_annotation_results, draw_detector_annotations
from src.output.report_writer import MAIN_COLUMNS, add_method_report_row, save_all_outputs
from src.tasks.fire_classification import clip_fire_type
from src.tasks.fire_detection import registry as fire_registry
from src.tasks.object_detection import registry as context_registry
from src.tasks.smoke_detection import registry as smoke_registry


def run_fire_detectors(image_path: Path, image) -> Dict[str, Dict]:
    """Run the configured fire detectors through the fire registry."""
    return fire_registry.run_enabled(config.ENABLED_FIRE_DETECTORS, image_path, image)


def run_smoke_detectors(image_path: Path, image) -> Dict[str, Dict]:
    """Run the configured smoke detectors through the smoke registry.

    Only configured detectors appear in the report. If a configured detector is
    misspelled or fails, the registry returns an explicit Unknown Detector or
    Execution Error row.
    """
    return smoke_registry.run_enabled(config.ENABLED_SMOKE_DETECTORS, image_path, image)


def run_context_detectors(image_path: Path, image) -> Dict[str, Dict]:
    """Run configured context detectors, such as YOLO objects and sign-gated OCR."""
    return context_registry.run_enabled(config.ENABLED_CONTEXT_DETECTORS, image_path, image)

def object_summary(reference_detections: List[Dict]) -> Tuple[str, float]:
    if not reference_detections:
        return "None", 0.0
    object_names = sorted({d["class_name"] for d in reference_detections})
    avg_confidence = sum(float(d.get("confidence", 0.0)) for d in reference_detections) / len(reference_detections)
    return ", ".join(object_names), avg_confidence


def fire_engine_confidence(fire_status: str, fire_results: Dict[str, Dict]) -> float:
    flame_pct = float(fire_results.get("opencv_flame", {}).get("image_percent", 0.0))
    vit_conf = float(fire_results.get("vit_fire", {}).get("confidence", 0.0))
    clip_conf = float(fire_results.get("clip_fire_evidence", {}).get("confidence", 0.0))
    opencv_conf = min(1.0, flame_pct / max(config.COLOR_FIRE_OVERRIDE_PERCENT, 1e-9)) if flame_pct > 0 else 0.0
    if fire_status == "Fire":
        return max(vit_conf, clip_conf, opencv_conf)
    if fire_status == "Uncertain":
        return max(vit_conf, clip_conf, opencv_conf) * 0.5
    return max(0.0, 1.0 - max(vit_conf, clip_conf, opencv_conf))


def _smoke_method_confidence(method_name: str, result: Dict) -> float:
    if method_name.startswith("clip_"):
        return float(result.get("confidence", 0.0))
    pct = float(result.get("image_percent", 0.0))
    threshold_map = {
        "opencv_smoke": config.OPENCV_SMOKE_THRESHOLD,
        "opencv_dark_smoke": config.OPENCV_DARK_SMOKE_THRESHOLD,
        "opencv_bright_plume": config.OPENCV_BRIGHT_PLUME_THRESHOLD,
    }
    threshold = threshold_map.get(method_name, config.OPENCV_SMOKE_THRESHOLD)
    return min(1.0, pct / max(threshold, 1e-9)) if pct > 0 else 0.0


def smoke_engine_confidence(smoke_status: str, smoke_results: Dict[str, Dict]) -> float:
    """Return engine confidence without letting color masks dominate.

    CLIP/classifier confidence is the base. OpenCV/localizer detectors only add
    a small support bonus when the engine already decided Smoke. This keeps
    smoke-color detections from turning a strong CLIP No Smoke result into a
    low-confidence or positive smoke result.
    """
    classifier_positive = []
    classifier_negative = []
    localizer_confs = []

    for name, result in smoke_results.items():
        result = result or {}
        if name.startswith("clip_"):
            conf = float(result.get("confidence", 0.0))
            if result.get("status") == "Smoke":
                classifier_positive.append(conf)
            elif result.get("status") == "No Smoke":
                classifier_negative.append(conf)
        elif name.startswith("opencv_"):
            localizer_confs.append(_smoke_method_confidence(name, result))

    best_positive = max(classifier_positive) if classifier_positive else 0.0
    best_negative = max(classifier_negative) if classifier_negative else 0.0
    best_localizer = max(localizer_confs) if localizer_confs else 0.0
    localizer_bonus = getattr(config, "SMOKE_LOCALIZER_CONFIDENCE_BONUS", 0.15) * best_localizer

    if smoke_status == "Smoke":
        return min(1.0, best_positive + localizer_bonus)
    if smoke_status == "Uncertain Smoke":
        return min(1.0, max(best_positive, best_localizer * 0.5))
    if best_negative > 0:
        return best_negative
    return max(0.0, 1.0 - max(best_positive, best_localizer * 0.25))


def fire_type_engine_confidence(final_fire_type: str, clip_type_result: Dict) -> float:
    if final_fire_type in {"None", "Unknown"}:
        return 0.0
    return float(clip_type_result.get("confidence", 0.0))


def scene_confidence(final_decision: str, fire_conf: float, smoke_conf: float) -> float:
    if final_decision == "Fire + Smoke":
        return (fire_conf + smoke_conf) / 2.0
    if final_decision == "Fire":
        return fire_conf
    if final_decision == "Smoke Only":
        return smoke_conf
    if final_decision == "Uncertain":
        return max(fire_conf, smoke_conf) * 0.5
    return min(fire_conf, smoke_conf)


def _fire_method_confidence(method_name: str, result: Dict) -> float:
    if method_name == "opencv_flame":
        flame_pct = float(result.get("image_percent", 0.0))
        return min(1.0, flame_pct / max(config.COLOR_FIRE_OVERRIDE_PERCENT, 1e-9)) if flame_pct > 0 else 0.0
    return float(result.get("confidence", 0.0))


def add_fire_report(image_id: str, rows: List[Dict], fire_status: str, fire_reason: str, fire_conf: float, fire_results: Dict[str, Dict]) -> None:
    for method_name, result in fire_results.items():
        method_conf = _fire_method_confidence(method_name, result)
        decision = result.get("status") or result.get("label", "")
        reason = result.get("reason") or result.get("model_label") or fire_reason
        localized = "Yes" if result.get("union_box") else "No"
        add_method_report_row(
            rows,
            image_id,
            "Fire Detection",
            fire_status,
            fire_conf,
            method_name,
            decision,
            method_conf,
            reason,
            localized,
            localized,
        )


def add_smoke_report(image_id: str, rows: List[Dict], smoke_status: str, smoke_reason: str, smoke_conf: float, smoke_results: Dict[str, Dict]) -> None:
    for method_name, result in smoke_results.items():
        method_conf = _smoke_method_confidence(method_name, result)
        decision = result.get("status") or result.get("label", "")
        localized = "Yes" if result.get("union_box") else "No"
        add_method_report_row(
            rows,
            image_id,
            "Smoke Detection",
            smoke_status,
            smoke_conf,
            method_name,
            decision,
            method_conf,
            result.get("reason", smoke_reason),
            localized,
            localized,
        )


def add_fire_type_report(image_id: str, rows: List[Dict], final_fire_type: str, fire_type_reason: str, fire_type_conf: float, clip_type_result: Dict) -> None:
    add_method_report_row(rows, image_id, "Fire Classification", final_fire_type, fire_type_conf, "clip_fire_type", clip_type_result.get("label", ""), clip_type_result.get("confidence", 0.0), fire_type_reason)


def add_scene_report(image_id: str, rows: List[Dict], final_decision: str, final_reason: str, final_conf: float, fire_status: str, fire_conf: float, smoke_status: str, smoke_conf: float, scene_type: str, object_conf: float) -> None:
    add_method_report_row(rows, image_id, "Scene Decision", final_decision, final_conf, "fire_decision_engine", fire_status, fire_conf, final_reason)
    add_method_report_row(rows, image_id, "Scene Decision", final_decision, final_conf, "smoke_decision_engine", smoke_status, smoke_conf, final_reason)
    add_method_report_row(rows, image_id, "Scene Decision", final_decision, final_conf, "scene_classifier", scene_type, final_conf, "scene type derived from final decision and object context")
    add_method_report_row(rows, image_id, "Scene Decision", final_decision, final_conf, "object_context", "Objects Detected" if object_conf > 0 else "No Reference Objects", object_conf, "YOLO reference-object context")


def text_engine_confidence(ocr_result: Dict) -> float:
    if not ocr_result:
        return 0.0
    return float(ocr_result.get("confidence", 0.0))


def add_text_report(image_id: str, rows: List[Dict], ocr_result: Dict) -> None:
    ocr_result = ocr_result or {}
    engine_decision = ocr_result.get("status", "Not Run")
    engine_conf = text_engine_confidence(ocr_result)
    sign_regions = ocr_result.get("sign_regions", []) or []
    ocr_results = ocr_result.get("ocr_results", []) or []
    watermark_filtered = ocr_result.get("watermark_filtered", []) or []

    if sign_regions:
        best_sign = max(sign_regions, key=lambda r: float(r.get("confidence", 0.0)))
        sign_decision = f"{len(sign_regions)} Sign Region(s)"
        sign_conf = float(best_sign.get("confidence", 0.0))
        sign_reason = best_sign.get("reason", "sign-like region detected")
        localized = "Yes"
    else:
        sign_decision = "No Sign Detected"
        sign_conf = 0.0
        sign_reason = "no sign-like crop region found"
        localized = "No"

    add_method_report_row(
        rows, image_id, "Text Extraction", engine_decision, engine_conf,
        "sign_detector", sign_decision, sign_conf, sign_reason, localized, localized,
    )

    accepted = [r for r in ocr_results if r.get("text")]
    rejected = [r for r in ocr_results if r.get("raw_text") and not r.get("text")]
    if accepted:
        ocr_text = " | ".join(r.get("text", "") for r in accepted if r.get("text"))
        ocr_conf = max(float(r.get("confidence", 0.0)) for r in accepted)
        ocr_decision = "Text Extracted"
        ocr_reason = ocr_text
    elif ocr_results:
        ocr_conf = max(float(r.get("confidence", 0.0)) for r in ocr_results)
        ocr_decision = "No Accepted Text"
        reasons = [r.get("reject_reason", "") for r in ocr_results if r.get("reject_reason")]
        ocr_reason = "; ".join(reasons[:3]) if reasons else "OCR ran on sign crop(s), but no text survived confidence/watermark filters"
    else:
        ocr_conf = 0.0
        ocr_decision = "OCR Not Run"
        ocr_reason = ocr_result.get("reason", "OCR skipped")

    add_method_report_row(
        rows, image_id, "Text Extraction", engine_decision, engine_conf,
        "ocr_text", ocr_decision, ocr_conf, ocr_reason, "No", "No",
    )

    if rejected or watermark_filtered:
        wm_decision = "Watermark/Overlay Filtered"
        if watermark_filtered:
            wm_reason = " | ".join(watermark_filtered[:5])
        else:
            wm_reason = "; ".join(r.get("reject_reason", "OCR crop rejected") for r in rejected[:5])
        wm_conf = max((float(r.get("confidence", 0.0)) for r in rejected), default=0.0)
    else:
        wm_decision = "No Watermark Text Filtered"
        wm_reason = "no OCR text was rejected as watermark/overlay"
        wm_conf = 0.0

    add_method_report_row(
        rows, image_id, "Text Extraction", engine_decision, engine_conf,
        "watermark_filter", wm_decision, wm_conf, wm_reason, "No", "No",
    )


def build_main_row(
    image_path: Path,
    scene_type: str,
    objects_detected: str,
    text: str,
    fire_confidence: float,
    smoke_confidence: float,
    fire_classification_confidence: float,
    scene_confidence_score: float,
) -> Dict:
    return {
        "Image_ID": image_path.name,
        "Scene_Type": scene_type,
        "Objects_Detected": objects_detected,
        "Text_Extracted": text,
        "Fire_Detection_Confidence": round(float(fire_confidence), 3),
        "Smoke_Detection_Confidence": round(float(smoke_confidence), 3),
        "Fire_Classification_Confidence": round(float(fire_classification_confidence), 3),
        "Scene_Decision_Confidence": round(float(scene_confidence_score), 3),
    }


def analyze_images(
    image_dir: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    splits: Optional[Iterable[str]] = None,
    csv_update_interval: int = config.CSV_UPDATE_INTERVAL,
) -> pd.DataFrame:
    image_dir = Path(image_dir) if image_dir else None
    csv_path = Path(csv_path) if csv_path else config.CSV_PATH
    reports_dir = config.REPORTS_DIR
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    config.ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(image_dir=image_dir, splits=splits)
    main_rows: List[Dict] = []
    report_rows = {
        "fire_detection_report": [],
        "smoke_detection_report": [],
        "fire_classification_report": [],
        "scene_decision_report": [],
        "text_extraction_report": [],
    }

    if not image_paths:
        save_all_outputs(main_rows, report_rows, csv_path, reports_dir)
        print("No images found.")
        return pd.DataFrame(columns=MAIN_COLUMNS)

    for idx, image_path in enumerate(tqdm(image_paths, desc="Analyzing images"), start=1):
        image = cv2.imread(str(image_path))
        split_name = get_split_name(image_path, custom_image_dir=image_dir)

        fire_results = run_fire_detectors(image_path, image)
        smoke_results = run_smoke_detectors(image_path, image)

        context_results = run_context_detectors(image_path, image)
        yolo_result = context_results.get("yolo_objects", {"detections": [], "model_results": None, "model_name": "disabled"})
        reference_detections = [d for d in yolo_result.get("detections", []) if d.get("is_reference")]
        objects_detected, object_conf = object_summary(reference_detections)

        fire_status, fire_reason, _fire_triggered = fire_decision_engine.decide_fire_status(fire_results)
        smoke_status, smoke_reason, _smoke_triggered = smoke_decision_engine.decide_smoke_status(smoke_results)
        final_decision, final_reason = scene_decision_engine.decide_scene_status(fire_status, smoke_status)

        flame_region = fire_results.get("opencv_flame", {})
        clip_type_result = clip_fire_type.detect(image_path, fire_status)
        final_fire_type, fire_type_reason = decision_engine.infer_fire_type(fire_status, clip_type_result, reference_detections, flame_region)
        object_names = [d["class_name"] for d in reference_detections]
        scene_type = scene_decision_engine.classify_scene(final_decision, fire_status, smoke_status, final_fire_type, object_names)
        text = context_results.get("ocr_text", {}).get("text", "")

        fire_conf = fire_engine_confidence(fire_status, fire_results)
        smoke_conf = smoke_engine_confidence(smoke_status, smoke_results)
        fire_type_conf = fire_type_engine_confidence(final_fire_type, clip_type_result)
        final_conf = scene_confidence(final_decision, fire_conf, smoke_conf)

        main_rows.append(build_main_row(image_path, scene_type, objects_detected, text, fire_conf, smoke_conf, fire_type_conf, final_conf))

        image_id = image_path.name
        add_fire_report(image_id, report_rows["fire_detection_report"], fire_status, fire_reason, fire_conf, fire_results)
        add_smoke_report(image_id, report_rows["smoke_detection_report"], smoke_status, smoke_reason, smoke_conf, smoke_results)
        add_fire_type_report(image_id, report_rows["fire_classification_report"], final_fire_type, fire_type_reason, fire_type_conf, clip_type_result)
        add_scene_report(image_id, report_rows["scene_decision_report"], final_decision, final_reason, final_conf, fire_status, fire_conf, smoke_status, smoke_conf, scene_type, object_conf)
        add_text_report(image_id, report_rows["text_extraction_report"], context_results.get("ocr_text", {}))

        annotation_results = build_annotation_results(reference_detections, flame_region, [smoke_results.get("opencv_smoke"), smoke_results.get("opencv_dark_smoke"), smoke_results.get("opencv_bright_plume")])
        annotated = draw_detector_annotations(image, annotation_results)
        if annotated is not None:
            cv2.imwrite(str(config.ANNOTATED_DIR / f"{split_name}_{image_path.name}"), annotated)

        if csv_update_interval > 0 and (idx % csv_update_interval == 0 or idx == len(image_paths)):
            save_all_outputs(main_rows, report_rows, csv_path, reports_dir)
            print(f"Outputs updated: {idx}/{len(image_paths)} images -> {csv_path} and {reports_dir}")

    save_all_outputs(main_rows, report_rows, csv_path, reports_dir)
    print(f"Done. Main CSV saved to: {csv_path}")
    print(f"Engine reports saved to: {reports_dir}")
    return pd.DataFrame(main_rows, columns=MAIN_COLUMNS)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze images with task-based modular fire and smoke detectors.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Custom image folder. Overrides dataset splits.")
    parser.add_argument("--splits", nargs="+", default=config.DEFAULT_SPLITS, help="Dataset splits to analyze, such as --splits test")
    parser.add_argument("--csv-path", type=Path, default=config.CSV_PATH, help="Output CSV path.")
    parser.add_argument("--csv-update-interval", type=int, default=config.CSV_UPDATE_INTERVAL, help="Write CSV and reports every N images.")
    parser.add_argument("--yolo-model", default=config.YOLO_MODEL_NAME, help="YOLOv8 model, e.g. yolov8m.pt, yolov8l.pt, yolov8x.pt")
    parser.add_argument("--yolo-confidence", type=float, default=config.YOLO_CONFIDENCE, help="YOLO detection confidence threshold.")
    parser.add_argument("--fire-threshold", type=float, default=config.FIRE_THRESHOLD, help="ViT fire classifier threshold.")
    parser.add_argument("--fire-detectors", nargs="*", default=None, help="Fire detectors to run: vit_fire opencv_flame clip_fire_evidence")
    parser.add_argument("--smoke-detectors", nargs="*", default=None, help="Smoke detectors to run: clip_smoke clip_smoke_plume opencv_smoke opencv_dark_smoke opencv_bright_plume. Pass no values to disable smoke.")
    parser.add_argument("--context-detectors", nargs="*", default=None, help="Context detectors to run: yolo_objects ocr_text. Pass no values to disable context detectors.")
    parser.add_argument("--clip-smoke-threshold", type=float, default=config.CLIP_SMOKE_THRESHOLD, help="CLIP smoke threshold.")
    parser.add_argument("--opencv-smoke-threshold", type=float, default=config.OPENCV_SMOKE_THRESHOLD, help="OpenCV light smoke image-percent threshold.")
    parser.add_argument("--clip-smoke-plume-threshold", type=float, default=config.CLIP_SMOKE_PLUME_THRESHOLD, help="CLIP smoke-plume threshold.")
    parser.add_argument("--opencv-dark-smoke-threshold", type=float, default=config.OPENCV_DARK_SMOKE_THRESHOLD, help="OpenCV dark-smoke image-percent threshold.")
    parser.add_argument("--opencv-bright-plume-threshold", type=float, default=config.OPENCV_BRIGHT_PLUME_THRESHOLD, help="OpenCV bright-plume image-percent threshold.")
    parser.add_argument("--smoke-require-agreement", action="store_true", help="Require both CLIP smoke and OpenCV smoke to classify Smoke. Default behavior already prevents OpenCV-only Smoke decisions.")
    parser.add_argument("--allow-opencv-smoke-alone", action="store_true", help="Allow OpenCV/localizer-only smoke hits to classify Smoke. Not recommended.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config.YOLO_MODEL_NAME = args.yolo_model
    config.YOLO_CONFIDENCE = args.yolo_confidence
    config.FIRE_THRESHOLD = args.fire_threshold
    config.CLIP_SMOKE_THRESHOLD = args.clip_smoke_threshold
    config.OPENCV_SMOKE_THRESHOLD = args.opencv_smoke_threshold
    config.CLIP_SMOKE_PLUME_THRESHOLD = args.clip_smoke_plume_threshold
    config.OPENCV_DARK_SMOKE_THRESHOLD = args.opencv_dark_smoke_threshold
    config.OPENCV_BRIGHT_PLUME_THRESHOLD = args.opencv_bright_plume_threshold
    if args.fire_detectors is not None:
        config.ENABLED_FIRE_DETECTORS = args.fire_detectors
    if args.smoke_detectors is not None:
        config.ENABLED_SMOKE_DETECTORS = args.smoke_detectors
    if args.context_detectors is not None:
        config.ENABLED_CONTEXT_DETECTORS = args.context_detectors
    if args.smoke_require_agreement:
        config.SMOKE_REQUIRE_AGREEMENT = True
    if args.allow_opencv_smoke_alone:
        config.SMOKE_LOCALIZERS_SUPPORT_ONLY = False
        config.SMOKE_REQUIRE_AGREEMENT = False
    analyze_images(
        image_dir=args.image_dir,
        csv_path=args.csv_path,
        splits=args.splits,
        csv_update_interval=args.csv_update_interval,
    )
