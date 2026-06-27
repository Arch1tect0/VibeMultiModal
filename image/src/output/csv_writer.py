"""CSV output helpers."""

from pathlib import Path
from typing import Dict, List

import pandas as pd


def save_checkpoint(rows: List[Dict], csv_path: Path, processed: int, total: int) -> None:
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"CSV updated: {processed}/{total} images -> {csv_path}")
