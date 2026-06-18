"""
Backend configuration for the MOET complaint triage engine.

The backend follows Option B from the Student Guide: local ML, no external LLM
API. ML learns complaint category and priority from local CSV data. The only
hard constraints left as rules are the Ministry's mandatory GREEN/YELLOW/RED
zone escalation policies.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
ROOT_DIR = BASE_DIR.parent                                 # repo root
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

COMPLAINTS_PATTERNS = ["consumer_complaints*", "*complaints*"]
ESTABLISHMENTS_PATTERNS = ["establishments*", "*establishment*"]

# New model artifacts. These do not conflict with older cached demo models.
CATEGORY_MODEL_PATH = MODEL_DIR / "category_tfidf_logreg.joblib"
PRIORITY_MODEL_PATH = MODEL_DIR / "priority_tfidf_logreg.joblib"
MODEL_META_PATH = MODEL_DIR / "model_metadata.json"

# Backward-compatible old path, not used by the new pipeline.
MODEL_PATH = MODEL_DIR / "tfidf_logreg.joblib"

# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------
RAW_TO_DISPLAY = {
    "food_safety": "Health & Food Safety",
    "health_food_safety": "Health & Food Safety",
    "health_&_food_safety": "Health & Food Safety",
    "health & food safety": "Health & Food Safety",
    "hygiene": "Hygiene & Sanitation",
    "hygiene_sanitation": "Hygiene & Sanitation",
    "hygiene & sanitation": "Hygiene & Sanitation",
    "price_fraud": "Pricing & Fraud",
    "pricing_fraud": "Pricing & Fraud",
    "pricing & fraud": "Pricing & Fraud",
    "consumer_fraud": "Pricing & Fraud",
    "licensing": "Licensing & Compliance",
    "licensing_compliance": "Licensing & Compliance",
    "licensing & compliance": "Licensing & Compliance",
    "service_quality": "Service Quality",
    "service complaints": "Service Quality",
    "service_complaints": "Service Quality",
    "product_quality": "Product Quality",
    "product quality": "Product Quality",
    "other": "Other",
}

CATEGORIES = [
    "Health & Food Safety",
    "Hygiene & Sanitation",
    "Pricing & Fraud",
    "Licensing & Compliance",
    "Product Quality",
    "Service Quality",
    "Other",
]

PRIORITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
PRIORITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
RANK_PRIORITY = {v: k for k, v in PRIORITY_RANK.items()}
PRIORITY_SCORE_CENTER = {"LOW": 15, "MEDIUM": 42, "HIGH": 67, "CRITICAL": 90}
PRIORITY_SCORE_FLOOR = {"LOW": 0, "MEDIUM": 30, "HIGH": 55, "CRITICAL": 75}


def to_display(raw_label: str) -> str:
    """Map raw dataset labels to dashboard-facing labels."""
    if raw_label is None:
        return "Other"
    raw = str(raw_label).strip()
    if not raw:
        return "Other"
    key = raw.lower().replace("&", "and")
    key = key.replace("/", "_").replace("-", "_").replace(" ", "_")
    key = "_".join(part for part in key.split("_") if part)
    return RAW_TO_DISPLAY.get(raw.lower(), RAW_TO_DISPLAY.get(key, raw))


# ---------------------------------------------------------------------------
# Severity signals. These are used to build weak training labels and structured
# ML features; they are not the final priority engine by themselves.
# ---------------------------------------------------------------------------
CRITICAL_TERMS = [
    "poison", "poisoning", "food poisoning", "expired meat", "expired chicken",
    "spoiled meat", "spoiled chicken", "rotten meat", "rotten chicken",
    "hospital", "hospitalized", "hospitalised", "emergency room", "sick",
    "ill", "illness", "vomit", "vomiting", "diarrhea", "diarrhoea",
    "unsafe food", "salmonella", "e coli", "death", "died", "child",
    "children", "baby", "infant", "elderly", "pregnant", "allergic reaction",
    "severe", "infection", "infected", "maggot", "maggots", "worms", "blood",
]

HIGH_TERMS = [
    "expired", "expiry", "rotten", "spoiled", "moldy", "mould", "mold",
    "contaminated", "contamination", "cockroach", "roach", "insect", "rat",
    "rats", "rodent", "mice", "fraud", "scam", "counterfeit", "fake",
    "price gouging", "gouging", "overcharge", "overcharged", "dangerous",
    "unsafe", "unhygienic", "dirty", "bacteria", "smell", "rancid", "sour",
]

REPEAT_TERMS = [
    "again", "repeated", "repeatedly", "still", "many times", "every time",
    "second time", "third time", "keeps happening", "multiple times", "as usual",
]
