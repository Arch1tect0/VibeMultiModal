"""Report writers for concise CSV output and per-engine audit reports."""

from pathlib import Path
from typing import Dict, List

import pandas as pd


MAIN_COLUMNS = [
    "Image_ID",
    "Scene_Type",
    "Objects_Detected",
    "Text_Extracted",
    "Fire_Detection_Confidence",
    "Smoke_Detection_Confidence",
    "Fire_Classification_Confidence",
    "Scene_Decision_Confidence",
]


def save_main_csv(rows: List[Dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=MAIN_COLUMNS).to_csv(csv_path, index=False)


def save_engine_reports(report_rows: Dict[str, List[Dict]], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for report_name, rows in report_rows.items():
        pd.DataFrame(rows).to_csv(reports_dir / f"{report_name}.csv", index=False)


def save_all_outputs(main_rows: List[Dict], report_rows: Dict[str, List[Dict]], csv_path: Path, reports_dir: Path) -> None:
    save_main_csv(main_rows, csv_path)
    save_engine_reports(report_rows, reports_dir)


def add_method_report_row(
    rows: List[Dict],
    image_id: str,
    engine_name: str,
    engine_decision: str,
    engine_confidence: float,
    method_name: str,
    method_decision: str,
    method_confidence: float,
    reason: str = "",
    localized: str = "No",
    annotation_source: str = "No",
) -> None:
    rows.append({
        "Image_ID": image_id,
        "Engine": engine_name,
        "Engine_Final_Decision": engine_decision,
        "Engine_Confidence": round(float(engine_confidence), 3),
        "Method": method_name,
        "Method_Decision": method_decision,
        "Method_Confidence": round(float(method_confidence), 3),
        "Method_Reason": reason,
        "Localized": localized,
        "Annotation_Source": annotation_source,
    })
