"""
Weak-supervision utilities for priority training.

The scope says citizen-selected priority is unreliable, so if no validated
priority label exists we generate AI priority labels from a transparent rubric.
Those generated labels are then used to TRAIN a local priority model. At runtime,
the model predicts priority first, and mandatory zone constraints are enforced
second.
"""
from __future__ import annotations
from . import config


def count_terms(text: str, terms: list[str]) -> int:
    t = (text or "").lower()
    return sum(1 for term in terms if term in t)


def severity_features(text: str) -> dict:
    return {
        "critical_hits": count_terms(text, config.CRITICAL_TERMS),
        "high_hits": count_terms(text, config.HIGH_TERMS),
        "repeat_hits": count_terms(text, config.REPEAT_TERMS),
    }


def has_serious_content(text: str) -> bool:
    return severity_features(text)["critical_hits"] > 0


CATEGORY_BASE = {
    "Health & Food Safety": 35,
    "Hygiene & Sanitation": 30,
    "Pricing & Fraud": 22,
    "Licensing & Compliance": 20,
    "Product Quality": 16,
    "Service Quality": 8,
    "Other": 5,
}


def score_to_priority(score: int) -> str:
    if score >= 75:
        return "CRITICAL"
    if score >= 55:
        return "HIGH"
    if score >= 30:
        return "MEDIUM"
    return "LOW"


def apply_mandatory_zone_constraints(priority: str, zone: str, serious_content: bool) -> tuple[str, str | None]:
    """Return corrected priority and explanation if a constraint changed it."""
    zone = (zone or "UNKNOWN").upper()
    rank = config.PRIORITY_RANK.get(priority, 1)
    original = priority

    if zone == "RED":
        if serious_content:
            priority = "CRITICAL"
        else:
            priority = config.RANK_PRIORITY[max(rank, config.PRIORITY_RANK["HIGH"])]
    elif zone == "YELLOW":
        priority = config.RANK_PRIORITY[max(rank, config.PRIORITY_RANK["MEDIUM"])]
    elif zone == "GREEN":
        # As stated in the guide/scope: GREEN may be LOW or MEDIUM depending on content.
        priority = config.RANK_PRIORITY[min(rank, config.PRIORITY_RANK["MEDIUM"])]

    if priority != original:
        return priority, f"Mandatory zone rule changed {original} -> {priority} for {zone}."
    return priority, None


def weak_priority_label(text: str, category: str, establishment: dict | None) -> dict:
    """Generate a training label when no trustworthy priority label exists."""
    est = establishment or {}
    zone = (est.get("zone") or "UNKNOWN").upper()
    violations = int(est.get("violations", 0) or 0)
    open_complaints = int(est.get("open_complaints", 0) or 0)
    feats = severity_features(text)

    score = CATEGORY_BASE.get(category, 5)
    score += min(40, feats["critical_hits"] * 18)
    score += min(20, feats["high_hits"] * 8)
    score += 8 if feats["repeat_hits"] else 0
    score += min(15, violations * 3)
    score += min(10, open_complaints * 2)
    if zone == "RED":
        score += 15
    elif zone == "YELLOW":
        score += 8
    score = max(0, min(100, int(round(score))))

    model_label = score_to_priority(score)
    final_label, note = apply_mandatory_zone_constraints(model_label, zone, feats["critical_hits"] > 0)
    return {
        "score": score,
        "model_label": model_label,
        "final_label": final_label,
        "zone_note": note,
        **feats,
    }


def normalize_citizen_priority(value):
    if not value:
        return None
    v = str(value).strip().lower()
    aliases = {
        "low": "LOW", "normal": "LOW", "minor": "LOW",
        "medium": "MEDIUM", "moderate": "MEDIUM", "med": "MEDIUM",
        "high": "HIGH", "urgent": "HIGH", "important": "HIGH",
        "critical": "CRITICAL", "emergency": "CRITICAL", "very urgent": "CRITICAL",
    }
    return aliases.get(v)
