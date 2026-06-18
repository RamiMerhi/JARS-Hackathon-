"""
Final triage engine.

Runtime flow:
  trained category model -> trained priority model -> mandatory zone correction.

Only the Ministry's mandatory policy constraints remain rule-based. Category and
initial priority are model outputs learned locally from the CSV data.
"""
from __future__ import annotations

from . import config
from .labeler import normalize_citizen_priority, has_serious_content, apply_mandatory_zone_constraints

PRIORITY_RANK = config.PRIORITY_RANK
RANK_PRIORITY = config.RANK_PRIORITY


def recommend_action(priority: str, zone: str, category: str) -> str:
    if priority == "CRITICAL":
        return "Dispatch inspector immediately (same-day). Consider closure order."
    if priority == "HIGH":
        return "Schedule on-site inspection within 48 hours."
    if priority == "MEDIUM":
        return "Queue for routine inspection within 1-2 weeks."
    return "Log and monitor. Review if further complaints arrive."


def _priority_score_after_constraints(model_score: int, final_priority: str) -> int:
    """Keep triage_score model-driven but consistent with final priority floors."""
    floor = config.PRIORITY_SCORE_FLOOR.get(final_priority, 0)
    center = config.PRIORITY_SCORE_CENTER.get(final_priority, model_score)
    # If a mandatory rule escalates priority, lift score toward that band but do
    # not fabricate a perfect 100 unless model/context supports it.
    adjusted = max(int(model_score or center), floor)
    if final_priority == "CRITICAL":
        adjusted = max(adjusted, 80)
    return max(0, min(100, int(round(adjusted))))


def triage_one(
    complaint: dict,
    establishment: dict | None,
    category_prediction,
    priority_prediction=None,
) -> dict:
    """
    Build the API-facing triage record.

    category_prediction can be a string or model.Prediction.
    priority_prediction should be model.Prediction from the trained priority model.
    """
    text = f"{complaint.get('subject','')}. {complaint.get('message','')}".strip()
    zone = (establishment or {}).get("zone", "UNKNOWN") or "UNKNOWN"
    violations = int((establishment or {}).get("violations", 0) or 0)
    open_complaints = int((establishment or {}).get("open_complaints", 0) or 0)

    # Category prediction details
    if hasattr(category_prediction, "label"):
        category = category_prediction.label
        category_conf = float(category_prediction.confidence)
    else:
        category = str(category_prediction or "Other")
        category_conf = None

    # Priority prediction details
    if priority_prediction is not None and hasattr(priority_prediction, "label"):
        model_priority = priority_prediction.label
        model_conf = float(priority_prediction.confidence)
        model_score = int(priority_prediction.score or config.PRIORITY_SCORE_CENTER.get(model_priority, 15))
    else:
        # Last-resort fallback should rarely happen; included for robust API behavior.
        model_priority = "MEDIUM" if category in {"Health & Food Safety", "Hygiene & Sanitation"} else "LOW"
        model_conf = None
        model_score = config.PRIORITY_SCORE_CENTER.get(model_priority, 42)

    serious = has_serious_content(text)
    final_priority, zone_note = apply_mandatory_zone_constraints(model_priority, zone, serious)
    triage_score = _priority_score_after_constraints(model_score, final_priority)

    citizen = normalize_citizen_priority(complaint.get("citizen_priority"))
    mismatch = bool(citizen and citizen != final_priority)

    reasons = [
        f"Category model predicted '{category}'"
        + (f" with {category_conf:.0%} confidence." if category_conf is not None else "."),
        f"Priority model predicted '{model_priority}'"
        + (f" with {model_conf:.0%} confidence" if model_conf is not None else "")
        + f" and model score {model_score}/100.",
    ]

    if establishment:
        reasons.append(
            f"Matched establishment '{establishment.get('name')}' has zone {zone}, "
            f"{violations} prior violation(s), and {open_complaints} open complaint(s)."
        )
    else:
        reasons.append("No establishment match found; triage used complaint text only.")

    if zone_note:
        reasons.append(zone_note)
    if mismatch:
        direction = "under-reported" if PRIORITY_RANK[citizen] < PRIORITY_RANK[final_priority] else "over-reported"
        reasons.append(f"Citizen selected '{citizen}' but AI final priority is '{final_priority}' ({direction}).")

    return {
        "complaint_id": complaint.get("complaint_id"),
        "subject": complaint.get("subject", ""),
        "message": complaint.get("message", ""),
        "province": complaint.get("province") or None,
        "purchase_place": complaint.get("purchase_place") or None,
        "matched_establishment_name": (establishment or {}).get("name"),
        "establishment_zone": zone,
        "violations": violations,
        "open_complaints": open_complaints,
        "citizen_priority": citizen,
        "predicted_category": category,
        "triage_score": triage_score,
        "final_priority": final_priority,
        "priority_mismatch": mismatch,
        "recommended_action": recommend_action(final_priority, zone, category),
        "reasoning": " ".join(reasons),
        "status": complaint.get("status") or "New",
        # Extra backend/debug fields. Existing frontend can ignore these.
        "model_priority": model_priority,
        "model_confidence": model_conf,
        "category_confidence": category_conf,
    }
