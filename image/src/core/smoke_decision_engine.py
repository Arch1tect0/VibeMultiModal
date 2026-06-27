"""Smoke-only decision rules.

Smoke is decided independently from fire, but OpenCV/color/texture smoke
localizers are now treated as supporting evidence only. That is intentional:
color-based smoke detection is useful for annotation and for strengthening a
classifier decision, but it is too noisy to declare Smoke by itself.

Decision policy:
- The primary smoke classifier (clip_smoke) is authoritative for strong No Smoke votes.
- Plume/color/texture methods can support smoke confidence, but cannot override a strong primary No Smoke vote.
- A weak smoke classifier plus OpenCV/localizer support can declare Smoke.
- OpenCV/localizer-only hits produce Uncertain Smoke, not Smoke, unless vetoed.
- No meaningful evidence produces No Smoke.
"""

from typing import Dict, Tuple

from src import config


CLASSIFIER_PREFIXES = ("clip_",)
LOCALIZER_PREFIXES = ("opencv_",)
PRIMARY_SMOKE_CLASSIFIER = "clip_smoke"
PLUME_SUPPORT_CLASSIFIERS = {"clip_smoke_plume"}


def _is_primary_smoke_classifier(name: str) -> bool:
    return name == PRIMARY_SMOKE_CLASSIFIER


def _is_support_classifier(name: str) -> bool:
    return name in PLUME_SUPPORT_CLASSIFIERS


def _localizer_thresholds() -> Dict[str, float]:
    return {
        "opencv_smoke": config.OPENCV_SMOKE_THRESHOLD,
        "opencv_dark_smoke": config.OPENCV_DARK_SMOKE_THRESHOLD,
        "opencv_bright_plume": config.OPENCV_BRIGHT_PLUME_THRESHOLD,
    }


def _is_classifier(name: str) -> bool:
    return name.startswith(CLASSIFIER_PREFIXES)


def _is_localizer(name: str) -> bool:
    return name.startswith(LOCALIZER_PREFIXES)


def _method_score(name: str, result: Dict) -> float:
    """Return normalized *positive smoke evidence* score from 0 to 1.

    Important: labels like "No Smoke" contain the word "Smoke", so do not
    infer positive smoke evidence by substring matching on the label. Use the
    detector status as the source of truth.
    """
    result = result or {}
    if _is_classifier(name):
        status = result.get("status")
        conf = float(result.get("confidence", 0.0))
        return conf if status == "Smoke" else 0.0

    if _is_localizer(name):
        pct = float(result.get("image_percent", 0.0))
        threshold = _localizer_thresholds().get(name, config.OPENCV_SMOKE_THRESHOLD)
        return min(1.0, pct / max(threshold, 1e-9)) if pct > 0 else 0.0

    return 0.0


def _method_hit(name: str, result: Dict) -> bool:
    result = result or {}
    if _is_classifier(name):
        return result.get("status") == "Smoke"
    if _is_localizer(name):
        threshold = _localizer_thresholds().get(name, config.OPENCV_SMOKE_THRESHOLD)
        return float(result.get("image_percent", 0.0)) >= threshold
    return False


def _negative_classifier_votes(detector_results: Dict[str, Dict]) -> Dict[str, float]:
    """Return classifier methods that explicitly voted No Smoke with confidence."""
    votes = {}
    for name, result in detector_results.items():
        result = result or {}
        if _is_classifier(name) and result.get("status") == "No Smoke":
            votes[name] = float(result.get("confidence", 0.0))
    return votes


def decide_smoke_status(detector_results: Dict[str, Dict]) -> Tuple[str, str, str]:
    scores = {name: _method_score(name, result) for name, result in detector_results.items()}
    hits = [name for name, result in detector_results.items() if _method_hit(name, result)]

    classifier_hits = [name for name in hits if _is_classifier(name)]
    localizer_hits = [name for name in hits if _is_localizer(name)]

    classifier_scores = {name: score for name, score in scores.items() if _is_classifier(name)}
    localizer_scores = {name: score for name, score in scores.items() if _is_localizer(name)}
    best_classifier = max(classifier_scores, key=classifier_scores.get) if classifier_scores else "None"
    best_localizer = max(localizer_scores, key=localizer_scores.get) if localizer_scores else "None"
    best_classifier_score = classifier_scores.get(best_classifier, 0.0)
    best_localizer_score = localizer_scores.get(best_localizer, 0.0)

    negative_classifier_votes = _negative_classifier_votes(detector_results)
    best_negative_classifier = max(negative_classifier_votes, key=negative_classifier_votes.get) if negative_classifier_votes else "None"
    best_negative_score = negative_classifier_votes.get(best_negative_classifier, 0.0)

    # Strong primary No Smoke should win over plume/color/texture evidence.
    # clip_smoke_plume is intentionally treated as support-only because it is
    # prone to mistaking bright sky, reflections, or haze-like background for
    # smoke. It can help when the primary smoke classifier is weak, but it must
    # not override a strong primary No Smoke result.
    primary_result = detector_results.get(PRIMARY_SMOKE_CLASSIFIER) or {}
    primary_status = primary_result.get("status")
    primary_confidence = float(primary_result.get("confidence", 0.0))
    primary_no_smoke_veto_score = getattr(
        config,
        "SMOKE_PRIMARY_NO_SMOKE_VETO_SCORE",
        getattr(config, "SMOKE_STRONG_NO_SMOKE_VETO_SCORE", config.SMOKE_NO_SMOKE_VETO_SCORE),
    )

    positive_classifier_scores = {
        name: score
        for name, score in classifier_scores.items()
        if (detector_results.get(name) or {}).get("status") == "Smoke"
    }
    best_positive_classifier = max(positive_classifier_scores, key=positive_classifier_scores.get) if positive_classifier_scores else "None"
    best_positive_score = positive_classifier_scores.get(best_positive_classifier, 0.0)

    if primary_status == "No Smoke" and primary_confidence >= primary_no_smoke_veto_score:
        support_hits = [name for name in classifier_hits if _is_support_classifier(name)] + localizer_hits
        return (
            "No Smoke",
            f"strong primary no-smoke classifier vetoed plume/color support; {PRIMARY_SMOKE_CLASSIFIER}:{primary_confidence:.3f}; support={', '.join(support_hits) if support_hits else 'none'}; best_positive={best_positive_classifier}:{best_positive_score:.3f}",
            PRIMARY_SMOKE_CLASSIFIER,
        )

    # Strong No Smoke from any other classifier still wins over color-only hits.
    # It does not veto a primary clip_smoke Smoke decision.
    strong_no_smoke_veto = getattr(config, "SMOKE_STRONG_NO_SMOKE_VETO_SCORE", config.SMOKE_NO_SMOKE_VETO_SCORE)
    primary_positive = primary_status == "Smoke"
    if not primary_positive and best_negative_score >= strong_no_smoke_veto and not classifier_hits:
        evidence = []
        if localizer_hits:
            evidence.append(f"localizers={', '.join(localizer_hits)}")
        evidence_text = "; ".join(evidence) if evidence else "no positive smoke evidence"
        return (
            "No Smoke",
            f"strong no-smoke classifier vetoed color-only evidence; no_smoke={best_negative_classifier}:{best_negative_score:.3f}; {evidence_text}",
            best_negative_classifier,
        )

    # A strong explicit No Smoke classifier vote also vetoes localizer-only color hits.
    # This prevents gray/white backgrounds, sky, pavement, clouds, or water vapor
    # from becoming Smoke just because the OpenCV color mask fired.
    if not classifier_hits and localizer_hits and best_negative_score >= config.SMOKE_NO_SMOKE_VETO_SCORE:
        return (
            "No Smoke",
            f"strong no-smoke classifier vetoed localizer-only evidence; no_smoke={best_negative_classifier}:{best_negative_score:.3f}; localizers={', '.join(localizer_hits)}",
            best_negative_classifier,
        )

    if classifier_hits and localizer_hits:
        triggered = classifier_hits + localizer_hits
        return (
            "Smoke",
            f"smoke classifier confirmed with localizer support; classifiers={', '.join(classifier_hits)}; localizers={', '.join(localizer_hits)}",
            ", ".join(triggered),
        )

    if classifier_hits:
        primary_hits = [name for name in classifier_hits if _is_primary_smoke_classifier(name)]
        if primary_hits:
            return (
                "Smoke",
                f"primary smoke classifier threshold met; classifiers={', '.join(primary_hits)}; best_classifier={best_classifier}:{best_classifier_score:.3f}",
                ", ".join(primary_hits),
            )
        return (
            "Uncertain Smoke",
            f"support-only smoke classifier hit without primary confirmation; classifiers={', '.join(classifier_hits)}; best_classifier={best_classifier}:{best_classifier_score:.3f}",
            ", ".join(classifier_hits),
        )

    # Large plume recovery: do not let OpenCV alone declare Smoke, but allow it
    # to confirm a weak CLIP/plume signal that did not cross its own threshold.
    if localizer_hits and best_classifier_score >= config.SMOKE_CLASSIFIER_WEAK_SUPPORT_SCORE:
        triggered = [best_classifier] + localizer_hits
        return (
            "Smoke",
            f"weak smoke classifier supported by localizer; classifier={best_classifier}:{best_classifier_score:.3f}; localizers={', '.join(localizer_hits)}",
            ", ".join(triggered),
        )

    if localizer_hits:
        if not config.SMOKE_LOCALIZERS_SUPPORT_ONLY:
            return (
                "Smoke",
                f"localizer-only smoke evidence allowed by config; localizers={', '.join(localizer_hits)}; best_localizer={best_localizer}:{best_localizer_score:.3f}",
                ", ".join(localizer_hits),
            )
        return (
            "Uncertain Smoke",
            f"localizer-only smoke evidence; OpenCV/color detectors cannot classify Smoke by themselves; localizers={', '.join(localizer_hits)}; best_localizer={best_localizer}:{best_localizer_score:.3f}",
            ", ".join(localizer_hits),
        )

    if best_classifier_score >= config.SMOKE_UNCERTAIN_SCORE:
        return "Uncertain Smoke", f"weak smoke classifier evidence from {best_classifier}:{best_classifier_score:.3f}", best_classifier

    if best_localizer_score >= config.SMOKE_UNCERTAIN_SCORE:
        return (
            "Uncertain Smoke",
            f"weak localizer-only smoke evidence from {best_localizer}:{best_localizer_score:.3f}; OpenCV/color detectors cannot classify Smoke by themselves",
            best_localizer,
        )

    return "No Smoke", "no smoke classifier or supported smoke evidence met threshold", "None"
