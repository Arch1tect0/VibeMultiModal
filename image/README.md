# FireDetection - independent modular fire and smoke pipeline

This version separates **fire detection** from **smoke detection** so smoke is no longer an offshoot or fallback label inside fire logic.

The pipeline now works as:

```text
fire detectors  -> Fire_Status
smoke detectors -> Smoke_Status
scene fusion    -> Final_Decision
```

## Run

From the project folder:

```bash
python -m src.analyze_images --image-dir path/to/images
```

Or run selected dataset splits if you have `dataset/train/images`, `dataset/valid/images`, and `dataset/test/images`:

```bash
python -m src.analyze_images --splits test
```

## Add or remove detectors

Use command-line detector lists:

```bash
python -m src.analyze_images \
  --image-dir path/to/images \
  --fire-detectors vit_fire opencv_flame clip_fire_evidence \
  --smoke-detectors clip_smoke opencv_smoke
```

Disable smoke completely:

```bash
python -m src.analyze_images --image-dir path/to/images --smoke-detectors
```

Run only CLIP smoke:

```bash
python -m src.analyze_images --image-dir path/to/images --smoke-detectors clip_smoke
```

Run only OpenCV smoke:

```bash
python -m src.analyze_images --image-dir path/to/images --smoke-detectors opencv_smoke
```

You can also edit `.env`:

```text
ENABLED_FIRE_DETECTORS=vit_fire,opencv_flame,clip_fire_evidence
ENABLED_SMOKE_DETECTORS=clip_smoke,opencv_smoke
ENABLED_CONTEXT_DETECTORS=yolo_objects,ocr_text
```

## New decision columns

The most important output columns are:

| Column | Meaning |
|---|---|
| `Fire_Status` | Fire-only result: `Fire`, `No Fire`, or `Uncertain` |
| `Fire_Status_Reason` | Why the fire decision was made |
| `Fire_Triggered_Detectors` | Fire detectors that drove the fire decision |
| `Smoke_Status` | Smoke-only result: `Smoke`, `No Smoke`, or `Uncertain Smoke` |
| `Smoke_Reason` | Why the smoke decision was made |
| `Smoke_Triggered_Detectors` | Smoke detectors that drove the smoke decision |
| `Final_Decision` | Combined scene result: `Fire + Smoke`, `Fire`, `Smoke Only`, `Uncertain`, or `No Fire / No Smoke` |
| `Final_Reason` | Why the final scene decision was made |

Detector-specific columns remain separate:

```text
ViT_Fire_Status
OpenCV_Flame_Status
CLIP_Fire_Evidence
CLIP_Smoke_Status
OpenCV_Smoke_Status
```

## Files that control the modular logic

```text
src/core/fire_decision_engine.py     # combines only fire detectors
src/core/smoke_decision_engine.py    # combines only smoke detectors
src/core/scene_decision_engine.py    # combines fire + smoke into final scene decision
src/detectors/clip_smoke.py          # independent CLIP smoke detector
src/detectors/opencv_smoke.py        # independent OpenCV smoke/haze detector
src/analyze_images.py                # detector orchestration and CSV output
src/config.py                        # detector lists, prompts, thresholds, output columns
```

## Current smoke modules

### `clip_smoke`

Uses separate CLIP prompts:

```text
Smoke
No Smoke
```

This is separate from fire prompts and no longer depends on `CLIP_Fire_Evidence` returning `Smoke Only`.

### `opencv_smoke`

A simple HSV-based smoke/haze color detector. It is included as a transparent starting point and can be replaced later with a stronger smoke model.

## Smoke tuning

Lower thresholds = more sensitive.

```bash
python -m src.analyze_images \
  --image-dir path/to/images \
  --clip-smoke-threshold 0.35 \
  --opencv-smoke-threshold 0.010
```

Require both smoke detectors to agree:

```bash
python -m src.analyze_images \
  --image-dir path/to/images \
  --smoke-require-agreement
```

## Annotation behavior

Boxes only appear for detectors that localize image regions:

| Detector | Annotation |
|---|---|
| OpenCV flame | Red fire-region box |
| OpenCV smoke | Light smoke-region box |
| YOLO objects | Blue object boxes |
| ViT fire | No box; image-level classifier |
| CLIP fire/smoke | No box; image-level classifier |

## Notes

This first implementation keeps the existing fire detector behavior largely intact while making smoke independent. The smoke logic is intentionally modular so you can add a future YOLO smoke model, segmentation model, or custom smoke classifier without changing fire logic.


## Task-based source layout

Detector code is now organized by task rather than in one flat detector folder:

```text
src/tasks/
  fire_detection/       # ViT fire, OpenCV flame, CLIP visible-fire evidence
  smoke_detection/      # CLIP smoke and OpenCV smoke localization
  fire_classification/  # fire type classification
  object_detection/     # YOLO scene/reference objects
  text_extraction/      # OCR
  shared/               # shared CLIP helpers
```

The decision engines remain in `src/core/` so detection logic and final decision logic stay separate.

To add or remove whole detectors, edit `src/config.py` or use CLI options such as:

```bash
python -m src.analyze_images --fire-detectors vit_fire opencv_flame --smoke-detectors opencv_smoke
```

## Output structure

The main CSV is intentionally slim and is written to:

```text
outputs/results.csv
```

It contains only executive fields:

```text
Image_ID, Scene_Type, Objects_Detected, Text_Extracted, Fire_Detection_Confidence, Smoke_Detection_Confidence, Fire_Classification_Confidence, Scene_Decision_Confidence
```

`Scene_Type` now returns the determined fire type when a fire exists, such as `Vehicle Fire`, `Structure Fire`, `Grass Fire`, `Forest Fire`, `Container / Trash Fire`, or `Other Fire`. If no fire is detected, it falls back to labels such as `Smoke Only`, `Vehicle / Accident Scene`, `Uncertain Fire/Smoke Scene`, or `No Fire / Unknown Scene`.

The confidence columns are broken out by engine instead of using one generic score:

| Column | Meaning |
|---|---|
| `Fire_Detection_Confidence` | Confidence from the fire detection engine |
| `Smoke_Detection_Confidence` | Confidence from the smoke detection engine |
| `Fire_Classification_Confidence` | Confidence from the fire-type classification engine |
| `Scene_Decision_Confidence` | Confidence from the final scene fusion engine |

Detailed method-level audit reports are written separately to:

```text
outputs/engine_reports/
  fire_detection_report.csv
  smoke_detection_report.csv
  fire_classification_report.csv
  scene_decision_report.csv
```

Each engine report includes the engine final decision, the engine confidence, each method's decision, each method's confidence, and any available reason/localization/annotation information.

## Smoke detection iteration: additional plume detectors

This version expands smoke detection so it no longer depends on a single CLIP prompt plus one gray-color OpenCV mask.

Default smoke detectors now run as separate modules:

- `clip_smoke`: generic image-level smoke classifier.
- `clip_smoke_plume`: CLIP classifier with explicit large-plume and cloud/fog negatives.
- `opencv_smoke`: original light gray/white smoke-color localizer.
- `opencv_dark_smoke`: dark/black smoke localizer.
- `opencv_bright_plume`: bright/white plume localizer that allows very bright smoke and filters uniform sky using texture.

Smoke localizers can annotate images. CLIP smoke modules classify the image but do not draw boxes.

Useful tuning options:

```bash
python -m src.analyze_images \
  --image-dir path/to/images \
  --clip-smoke-threshold 0.42 \
  --clip-smoke-plume-threshold 0.34 \
  --opencv-smoke-threshold 0.015 \
  --opencv-dark-smoke-threshold 0.010 \
  --opencv-bright-plume-threshold 0.020
```

To test detectors independently:

```bash
python -m src.analyze_images --image-dir path/to/images --smoke-detectors clip_smoke_plume
python -m src.analyze_images --image-dir path/to/images --smoke-detectors opencv_bright_plume
python -m src.analyze_images --image-dir path/to/images --smoke-detectors opencv_dark_smoke
```

The smoke report at `outputs/engine_reports/smoke_detection_report.csv` shows each method's decision and confidence so false positives and false negatives can be traced to a specific detector.

## Registry-based detector execution

This version runs detectors through explicit registries instead of hardcoded `if` blocks in `src/analyze_images.py`.

Registry files:

```text
src/core/detector_registry.py
src/tasks/fire_detection/registry.py
src/tasks/smoke_detection/registry.py
src/tasks/object_detection/registry.py
```

To add a smoke detector:

1. Add the detector implementation under `src/tasks/smoke_detection/`.
2. Register it in `src/tasks/smoke_detection/registry.py`.
3. Add its name to `ENABLED_SMOKE_DETECTORS` in `src/config.py` or pass it with `--smoke-detectors`.

Only configured detectors are shown in the engine reports. If a configured detector is misspelled or fails at runtime, the report shows `Unknown Detector` or `Execution Error` instead of silently displaying stale `Not Run` rows.
