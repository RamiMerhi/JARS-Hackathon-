"""
FastAPI application for the MOET complaint triage backend.

Frontend contract is preserved. Backend AI has been upgraded to:
- trained local category model
- trained local priority model / weak-supervised priority model
- mandatory zone policy correction layer
"""
from __future__ import annotations

import itertools
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config, data_loader, model as model_mod
from .matcher import EstablishmentMatcher
from .priority_engine import triage_one, PRIORITY_RANK
from .schemas import (
    Complaint,
    Establishment,
    Summary,
    ActionUpdate,
    TriageRequest,
    ModelInfo,
)

ALLOWED_STATUSES = {"New", "Assigned to Inspector", "Under Review", "Resolved"}

app = FastAPI(title="Consumer Complaint Triage API", version="3.0.0-local-ml")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {
    "complaints": {},          # complaint_id -> triaged complaint dict
    "establishments": {},      # establishment_id -> live establishment dict
    "matcher": None,
    "models": None,
}
_id_counter = itertools.count(1)


def _establishment_store(establishments_df):
    store = {}
    for _, r in establishments_df.iterrows():
        est_id = r.get("establishment_id")
        store[est_id] = {
            "establishment_id": est_id,
            "name": r["name"],
            "zone": r.get("zone", "UNKNOWN"),
            "province": r.get("province") or None,
            "violations": int(r.get("violations", 0) or 0),
            "open_complaints": int(r.get("open_complaints", 0) or 0),
        }
    return store


def _matched_live_est(purchase_place, province):
    """Match an establishment and return the live mutable record keyed by stable ID."""
    matcher: EstablishmentMatcher = STATE["matcher"]
    if matcher is None:
        return None
    matched = matcher.match(purchase_place or "", province)
    if not matched:
        return None
    return STATE["establishments"].get(matched.get("establishment_id"))


def _complaint_text(payload: dict) -> str:
    return f"{payload.get('subject','')}. {payload.get('message','')}".strip()


def _triage_payload(payload: dict, est: dict | None) -> dict:
    models = STATE["models"]
    text = _complaint_text(payload)
    category_pred = models.predict_category(text)
    priority_pred = models.predict_priority(text, category_pred.label, est, payload.get("province"))
    return triage_one(payload, est, category_pred, priority_pred)


def build_state(force_train: bool = False):
    complaints_df = data_loader.load_complaints()
    establishments_df = data_loader.load_establishments()

    matcher = EstablishmentMatcher(establishments_df)
    establishments = _establishment_store(establishments_df)
    models = model_mod.get_models(complaints_df, matcher, establishments, force_train=force_train)

    STATE["matcher"] = matcher
    STATE["establishments"] = establishments
    STATE["models"] = models

    triaged = {}
    for _, row in complaints_df.iterrows():
        c = row.to_dict()
        est = _matched_live_est(c.get("purchase_place", ""), c.get("province"))
        result = _triage_payload(c, est)
        result["_est_id"] = est["establishment_id"] if est else None
        result["_counts_open"] = False        # seeded complaints use CSV baseline
        triaged[result["complaint_id"]] = result
    STATE["complaints"] = triaged

    print(
        f"[api] seeded {len(triaged)} complaints, "
        f"{len(STATE['establishments'])} establishments. "
        f"Local ML ready for /api/triage."
    )


@app.on_event("startup")
def _startup():
    build_state(force_train=False)


def _sorted_complaints():
    return sorted(
        STATE["complaints"].values(),
        key=lambda c: (-PRIORITY_RANK.get(c["final_priority"], 0), -c["triage_score"]),
    )


@app.post("/api/triage", response_model=Complaint)
def triage_new_complaint(req: TriageRequest):
    new_id = f"USR-{next(_id_counter):04d}"
    est = _matched_live_est(req.purchase_place, req.province)
    if est is not None:
        est["open_complaints"] += 1

    payload = {
        "complaint_id": new_id,
        "subject": req.subject or "",
        "message": req.message,
        "province": req.province,
        "purchase_place": req.purchase_place,
        "citizen_priority": req.citizen_priority,
        "status": "New",
    }
    if not payload["subject"].strip():
        first = req.message.strip().split(". ")[0]
        payload["subject"] = (first[:70] + "…") if len(first) > 72 else first

    result = _triage_payload(payload, est)
    result["_est_id"] = est["establishment_id"] if est else None
    result["_counts_open"] = est is not None
    STATE["complaints"][new_id] = result
    return result


@app.get("/api/complaints", response_model=list[Complaint])
def get_complaints(
    final_priority: str | None = None,
    province: str | None = None,
    establishment_zone: str | None = None,
    predicted_category: str | None = None,
    status: str | None = None,
    priority_mismatch: bool | None = None,
    establishment: str | None = None,
):
    items = _sorted_complaints()
    est_q = (establishment or "").strip().lower()

    def keep(c):
        if final_priority and c["final_priority"] != final_priority:
            return False
        if province and (c["province"] or "") != province:
            return False
        if establishment_zone and (c["establishment_zone"] or "") != establishment_zone:
            return False
        if predicted_category and c["predicted_category"] != predicted_category:
            return False
        if status and c["status"] != status:
            return False
        if priority_mismatch is not None and c["priority_mismatch"] != priority_mismatch:
            return False
        if est_q and est_q not in (c.get("matched_establishment_name") or "").lower():
            return False
        return True

    return [c for c in items if keep(c)]


@app.get("/api/complaints/{complaint_id}", response_model=Complaint)
def get_complaint(complaint_id: str):
    c = STATE["complaints"].get(complaint_id)
    if not c:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return c


@app.get("/api/establishments", response_model=list[Establishment])
def get_establishments():
    return list(STATE["establishments"].values())


@app.get("/api/summary", response_model=Summary)
def get_summary():
    items = list(STATE["complaints"].values())
    return Summary(
        total_complaints=len(items),
        critical_complaints=sum(1 for c in items if c["final_priority"] == "CRITICAL"),
        high_priority_complaints=sum(1 for c in items if c["final_priority"] == "HIGH"),
        red_zone_complaints=sum(1 for c in items if c["establishment_zone"] == "RED"),
        priority_mismatches=sum(1 for c in items if c["priority_mismatch"]),
        unresolved_complaints=sum(1 for c in items if c["status"] != "Resolved"),
    )


@app.post("/api/complaints/{complaint_id}/action", response_model=Complaint)
def update_action(complaint_id: str, update: ActionUpdate):
    c = STATE["complaints"].get(complaint_id)
    if not c:
        raise HTTPException(status_code=404, detail="Complaint not found")
    if update.status not in ALLOWED_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status. Allowed: {sorted(ALLOWED_STATUSES)}")

    est = STATE["establishments"].get(c.get("_est_id")) if c.get("_est_id") else None
    was_open = c.get("_counts_open", False)
    now_open = update.status != "Resolved"
    if est is not None and c.get("_est_id"):
        if was_open and not now_open:
            est["open_complaints"] = max(0, est["open_complaints"] - 1)
            c["_counts_open"] = False
        elif (not was_open) and now_open and c["complaint_id"].startswith("USR-"):
            est["open_complaints"] += 1
            c["_counts_open"] = True

    c["status"] = update.status
    if est is not None:
        c["open_complaints"] = int(est.get("open_complaints", c.get("open_complaints", 0)) or 0)
    return c


@app.get("/api/model-info", response_model=ModelInfo)
def model_info():
    models = STATE.get("models")
    meta = models.metadata if models else {}
    cat = meta.get("category_model", {})
    pri = meta.get("priority_model", {})
    return ModelInfo(
        local_ml=True,
        external_llm_api_used=False,
        category_model_type=cat.get("model_type"),
        priority_model_type=pri.get("model_type"),
        training_rows=cat.get("training_rows"),
        categories=cat.get("classes", []),
        priority_classes=pri.get("classes", []),
        priority_label_source=pri.get("label_source"),
        category_metrics=cat.get("metrics", {}),
        priority_metrics=pri.get("metrics", {}),
        model_files_loaded={
            "category_model": config.CATEGORY_MODEL_PATH.exists(),
            "priority_model": config.PRIORITY_MODEL_PATH.exists(),
            "metadata": config.MODEL_META_PATH.exists(),
        },
        trained_vs_rule_based={
            "trained": [
                "complaint category classification",
                "initial AI priority prediction",
                "priority confidence/score from model probabilities",
            ],
            "rule_based_policy_constraints": meta.get("mandatory_rule_layer", [
                "YELLOW minimum MEDIUM",
                "RED minimum HIGH",
                "RED + serious content => CRITICAL",
                "GREEN LOW/MEDIUM constraint",
            ]),
            "note": "The backend does not use Claude/OpenAI/Gemini APIs. It uses local scikit-learn models and then applies mandatory Ministry zone rules.",
        },
    )


@app.get("/")
def root():
    return {"service": "Consumer Complaint Triage API", "docs": "/docs", "local_ml": True}
