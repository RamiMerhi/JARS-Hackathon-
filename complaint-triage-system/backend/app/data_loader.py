"""
CSV loading utilities.

The loader is intentionally tolerant of column-name variations so the same
backend can run on the provided hackathon CSVs or future Ministry exports.
"""
import glob
import pandas as pd
from . import config


COMPLAINT_ALIASES = {
    "complaint_id": ["complaint_id", "id", "complaintid", "ticket_id", "ref", "trn"],
    "subject": ["subject", "title", "complaint_subject", "summary", "topic"],
    "message": ["message", "description", "details", "body", "complaint",
                "complaint_text", "text", "content"],
    "province": ["province", "governorate", "region", "mohafaza", "area"],
    "purchase_place": ["purchase_place", "establishment_name", "place",
                       "establishment", "shop", "store", "merchant", "business",
                       "vendor"],
    # In the scope this is citizen-selected priority, not a trusted target label.
    "citizen_priority": ["citizen_priority", "priority", "user_priority",
                         "selected_priority", "urgency"],
    "category": ["category", "label", "complaint_category", "type", "class"],
    "status": ["status", "state", "complaint_status"],
}

# These are treated as true/validated priority labels if present. They are
# intentionally separate from `priority`, which the scope describes as citizen input.
TRUE_PRIORITY_COLUMNS = [
    "true_priority", "ai_priority", "validated_priority", "verified_priority",
    "final_priority", "priority_category", "triage_priority", "official_priority",
]

ESTABLISHMENT_ALIASES = {
    "establishment_id": ["establishment_id", "id", "estab_id", "est_id"],
    "name": ["name", "establishment_name", "business_name", "shop_name"],
    "zone": ["zone", "risk_zone", "color", "risk", "risk_level"],
    "province": ["province", "governorate", "region", "mohafaza", "area"],
    "sector": ["sector", "type", "category", "business_type"],
    "violations": ["violations", "violation_count", "num_violations", "past_violations"],
    "open_complaints": ["open_complaints", "open_cases", "active_complaints",
                        "pending_complaints", "complaints_open"],
}


def _find_file(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(str(config.DATA_DIR / pat)))
        hits = [h for h in hits if h.lower().endswith(".csv")]
        if hits:
            return hits[0]
    return None


def _normalize_columns(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for canonical, variants in aliases.items():
        for v in variants:
            if v in lower and lower[v] not in rename:
                rename[lower[v]] = canonical
                break
    return df.rename(columns=rename)


def _subject_from_message(msg: str) -> str:
    msg = (msg or "").strip()
    if not msg:
        return "Complaint"
    first = msg.replace("\n", " ").split(". ")[0]
    return (first[:70] + "…") if len(first) > 72 else first


def _normalize_priority(value):
    if value is None:
        return ""
    v = str(value).strip().upper()
    aliases = {
        "LOW": "LOW", "NORMAL": "LOW", "MINOR": "LOW",
        "MEDIUM": "MEDIUM", "MODERATE": "MEDIUM", "MED": "MEDIUM",
        "HIGH": "HIGH", "URGENT": "HIGH", "IMPORTANT": "HIGH",
        "CRITICAL": "CRITICAL", "EMERGENCY": "CRITICAL", "VERY URGENT": "CRITICAL",
    }
    return aliases.get(v, "")


def load_complaints() -> pd.DataFrame:
    path = _find_file(config.COMPLAINTS_PATTERNS)
    if not path:
        raise FileNotFoundError(
            f"No complaints CSV found in {config.DATA_DIR}. "
            "Place consumer_complaints(in).csv in the project data folder."
        )

    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    original_columns = list(raw.columns)
    df = _normalize_columns(raw.copy(), COMPLAINT_ALIASES)

    if "complaint_id" not in df.columns:
        df["complaint_id"] = ["C" + str(i + 1).zfill(4) for i in range(len(df))]
    for col in ["subject", "message", "province", "purchase_place",
                "citizen_priority", "category", "status"]:
        if col not in df.columns:
            df[col] = ""

    # Preserve a trustworthy priority target only if a clearly validated label exists.
    true_priority_col = None
    lower_original = {c.lower().strip(): c for c in original_columns}
    for candidate in TRUE_PRIORITY_COLUMNS:
        if candidate in lower_original:
            true_priority_col = lower_original[candidate]
            break
    if true_priority_col:
        df["true_priority"] = raw[true_priority_col].apply(_normalize_priority)
    else:
        df["true_priority"] = ""

    needs_subject = df["subject"].astype(str).str.strip() == ""
    df.loc[needs_subject, "subject"] = df.loc[needs_subject, "message"].apply(_subject_from_message)

    df["text"] = (df["subject"].fillna("") + ". " + df["message"].fillna("")).str.strip()
    df["citizen_priority"] = df["citizen_priority"].apply(_normalize_priority).replace("", None)
    df["status"] = df["status"].replace("", "New")
    df["category_display"] = df["category"].apply(
        lambda x: config.to_display(x) if str(x).strip() else ""
    )
    return df


def load_establishments() -> pd.DataFrame:
    path = _find_file(config.ESTABLISHMENTS_PATTERNS)
    if not path:
        raise FileNotFoundError(f"No establishments CSV found in {config.DATA_DIR}.")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = _normalize_columns(df, ESTABLISHMENT_ALIASES)

    if "name" not in df.columns:
        raise ValueError("Establishments CSV must contain a name column.")
    if "establishment_id" not in df.columns:
        df["establishment_id"] = ["EST-" + str(i + 1).zfill(4) for i in range(len(df))]
    for col in ["zone", "province", "sector"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["violations", "open_complaints"]:
        if col not in df.columns:
            df[col] = "0"
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Ensure IDs are stable and unique even if input has blanks or duplicates.
    ids = []
    seen = set()
    for i, raw_id in enumerate(df["establishment_id"].astype(str).tolist(), start=1):
        est_id = raw_id.strip() or f"EST-{i:04d}"
        if est_id in seen:
            est_id = f"{est_id}-{i:04d}"
        ids.append(est_id)
        seen.add(est_id)
    df["establishment_id"] = ids

    df["zone"] = df["zone"].astype(str).str.upper().str.strip()
    df["zone"] = df["zone"].replace({
        "G": "GREEN", "Y": "YELLOW", "R": "RED",
        "LOW": "GREEN", "MEDIUM": "YELLOW", "HIGH": "RED", "": "UNKNOWN",
    })
    return df


if __name__ == "__main__":
    c = load_complaints()
    e = load_establishments()
    print(f"Complaints: {len(c)} rows | category labels: "
          f"{(c['category_display'].str.strip() != '').mean():.0%} | "
          f"true priority labels: {(c['true_priority'].str.strip() != '').mean():.0%}")
    print(f"Establishments: {len(e)} rows | zones: {dict(e['zone'].value_counts())}")
