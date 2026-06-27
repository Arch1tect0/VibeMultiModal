"""Small registry/dispatcher used by engine task folders.

The registry is intentionally explicit: a detector only runs if its name is in
configuration and the name exists in the registry. If a configured detector is
missing or raises an exception, that fact is returned as a detector result so the
engine report shows the problem instead of silently producing stale Not Run rows.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional


DetectorCallable = Callable[[Path, object], Dict]


@dataclass(frozen=True)
class RegisteredDetector:
    name: str
    run: DetectorCallable
    family: str
    default_result: Optional[Callable[[str], Dict]] = None


class DetectorRegistry:
    def __init__(self, family: str):
        self.family = family
        self._detectors: Dict[str, RegisteredDetector] = {}

    def register(
        self,
        name: str,
        run: DetectorCallable,
        default_result: Optional[Callable[[str], Dict]] = None,
    ) -> None:
        if not name:
            raise ValueError("Detector name cannot be empty")
        if name in self._detectors:
            raise ValueError(f"Duplicate detector registration: {name}")
        self._detectors[name] = RegisteredDetector(name=name, run=run, family=self.family, default_result=default_result)

    @property
    def names(self):
        return tuple(self._detectors.keys())

    def registered(self) -> Mapping[str, RegisteredDetector]:
        return dict(self._detectors)

    def run_enabled(self, enabled: Iterable[str], image_path: Path, image) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {}
        for name in [item for item in enabled if item]:
            detector = self._detectors.get(name)
            if detector is None:
                results[name] = unknown_detector_result(name, self.family, self.names)
                continue
            try:
                result = detector.run(image_path, image)
            except Exception as exc:  # keep batch processing alive and make the failure auditable
                result = execution_error_result(name, self.family, exc)
            results[name] = normalize_result(name, self.family, result)
        return results


def normalize_result(name: str, family: str, result: Optional[Dict]) -> Dict:
    if not isinstance(result, dict):
        result = {}
    normalized = dict(result)
    normalized.setdefault("method", name)
    normalized.setdefault("detector", name)
    normalized.setdefault("family", family)
    normalized.setdefault("status", normalized.get("label", "No Result"))
    normalized.setdefault("confidence", 0.0)
    normalized.setdefault("reason", "")
    return normalized


def unknown_detector_result(name: str, family: str, known_names: Iterable[str]) -> Dict:
    return {
        "status": "Unknown Detector",
        "label": "Unknown Detector",
        "confidence": 0.0,
        "scores": {},
        "regions": [],
        "union_box": None,
        "area": 0.0,
        "image_percent": 0.0,
        "method": name,
        "detector": name,
        "family": family,
        "reason": f"{name} is enabled but not registered. Registered {family} detectors: {', '.join(known_names) or 'None'}",
    }


def execution_error_result(name: str, family: str, exc: Exception) -> Dict:
    return {
        "status": "Execution Error",
        "label": "Execution Error",
        "confidence": 0.0,
        "scores": {},
        "regions": [],
        "union_box": None,
        "area": 0.0,
        "image_percent": 0.0,
        "method": name,
        "detector": name,
        "family": family,
        "reason": f"{type(exc).__name__}: {exc}",
    }
