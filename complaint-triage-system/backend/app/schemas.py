"""Pydantic models for API request/response validation."""
from typing import Optional, Any
from pydantic import BaseModel


class Complaint(BaseModel):
    complaint_id: str
    subject: str
    message: str
    province: Optional[str] = None
    purchase_place: Optional[str] = None
    matched_establishment_name: Optional[str] = None
    establishment_zone: Optional[str] = None
    violations: int = 0
    open_complaints: int = 0
    citizen_priority: Optional[str] = None
    predicted_category: str
    triage_score: int
    final_priority: str
    priority_mismatch: bool = False
    recommended_action: str
    reasoning: str
    status: str = "New"
    # Optional additions do not require frontend changes.
    model_priority: Optional[str] = None
    model_confidence: Optional[float] = None
    category_confidence: Optional[float] = None


class TriageRequest(BaseModel):
    subject: str = ""
    message: str
    province: Optional[str] = None
    purchase_place: Optional[str] = None
    citizen_priority: Optional[str] = None


class ActionUpdate(BaseModel):
    status: str


class Establishment(BaseModel):
    establishment_id: Optional[str] = None
    name: str
    zone: Optional[str] = None
    province: Optional[str] = None
    violations: int = 0
    open_complaints: int = 0


class Summary(BaseModel):
    total_complaints: int
    critical_complaints: int
    high_priority_complaints: int
    red_zone_complaints: int
    priority_mismatches: int
    unresolved_complaints: int


class ModelInfo(BaseModel):
    local_ml: bool
    external_llm_api_used: bool
    category_model_type: Optional[str] = None
    priority_model_type: Optional[str] = None
    training_rows: Optional[int] = None
    categories: list[str] = []
    priority_classes: list[str] = []
    priority_label_source: Optional[str] = None
    category_metrics: dict[str, Any] = {}
    priority_metrics: dict[str, Any] = {}
    model_files_loaded: dict[str, bool] = {}
    trained_vs_rule_based: dict[str, Any] = {}
