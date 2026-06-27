"""Image discovery and split-name helpers."""

from pathlib import Path
from typing import Iterable, List, Optional

from src import config


def get_image_paths(image_dir: Optional[Path] = None, splits: Optional[Iterable[str]] = None) -> List[Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if image_dir is not None:
        image_dir = Path(image_dir)
        return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in valid_exts)

    paths: List[Path] = []
    for split in list(splits or config.DEFAULT_SPLITS):
        split_dir = config.DATASET_ROOT / split / "images"
        if split_dir.exists():
            paths.extend(p for p in split_dir.rglob("*") if p.suffix.lower() in valid_exts)
    return sorted(paths)


def get_split_name(image_path: Path, custom_image_dir: Optional[Path] = None) -> str:
    if custom_image_dir is not None:
        return "custom"
    try:
        return image_path.relative_to(config.DATASET_ROOT).parts[0]
    except ValueError:
        return "unknown"
