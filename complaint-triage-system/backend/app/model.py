"""
Local ML models for the MOET triage backend.

Implemented Option B:
1) Category model: supervised TF-IDF + SGD Logistic Classifier trained on complaint
   category labels from the provided CSV.
2) Priority model: trained locally. If a validated priority label exists, it is
   trained directly. If not, weak-supervision labels are generated from the
   triage rubric and establishment risk context, then a classifier is trained on
   those labels. This means runtime priority is model-predicted first, then
   mandatory Ministry zone rules are applied as a safety/policy correction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from . import config
from .labeler import weak_priority_label, severity_features


# ---------------------------------------------------------------------------
# Generic local text classifier wrapper
# ---------------------------------------------------------------------------

def _make_text_pipeline(min_df: int = 2, c: float = 6.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=0.95,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            tol=1e-3,
            alpha=max(1e-6, 1.0 / (c * 10000)),
            class_weight="balanced",
            random_state=42,
        )),
    ])


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    score: int | None = None


class LocalTextClassifier:
    def __init__(self, pipe: Pipeline, label_order: list[str] | None = None):
        self.pipe = pipe
        self.label_order = label_order or list(getattr(pipe.named_steps.get("clf"), "classes_", []))

    def predict(self, text: str) -> str:
        return self.predict_full(text).label

    def predict_full(self, text: str) -> Prediction:
        text = text or ""
        label = str(self.pipe.predict([text])[0])
        probabilities: dict[str, float] = {}
        confidence = 1.0
        if hasattr(self.pipe.named_steps.get("clf"), "predict_proba"):
            classes = list(self.pipe.named_steps["clf"].classes_)
            probs = self.pipe.predict_proba([text])[0]
            probabilities = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
            confidence = float(max(probs))
        return Prediction(label=label, confidence=confidence, probabilities=probabilities)

    def save(self, path: Path):
        joblib.dump({"pipe": self.pipe, "label_order": self.label_order}, path)

    @classmethod
    def load(cls, path: Path) -> "LocalTextClassifier":
        obj = joblib.load(path)
        if isinstance(obj, dict) and "pipe" in obj:
            return cls(obj["pipe"], obj.get("label_order"))
        # Backward compatibility with old direct-pipeline cache.
        return cls(obj)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def priority_feature_text(
    text: str,
    category: str,
    establishment: dict | None,
    province: str | None = None,
) -> str:
    """
    Build a single local-ML input string containing complaint text + structured
    context tokens. This keeps the model simple and hackathon-stable while still
    learning from establishment risk context.
    """
    est = establishment or {}
    zone = _safe_text(est.get("zone") or "UNKNOWN").upper()
    violations = int(est.get("violations", 0) or 0)
    open_complaints = int(est.get("open_complaints", 0) or 0)
    feats = severity_features(text)

    def bucket(value: int) -> str:
        if value <= 0:
            return "0"
        if value <= 2:
            return "1_2"
        if value <= 5:
            return "3_5"
        return "6_plus"

    tokens = [
        _safe_text(text),
        f"CATEGORY_{category.replace(' ', '_').replace('&', 'AND')}",
        f"ZONE_{zone}",
        f"PROVINCE_{_safe_text(province or est.get('province') or 'UNKNOWN').replace(' ', '_')}",
        f"VIOLATIONS_{bucket(violations)}",
        f"OPEN_COMPLAINTS_{bucket(open_complaints)}",
        f"CRITICAL_HITS_{bucket(feats['critical_hits'])}",
        f"HIGH_HITS_{bucket(feats['high_hits'])}",
        f"REPEAT_HITS_{bucket(feats['repeat_hits'])}",
    ]
    return " ".join(tokens)


class PriorityClassifier(LocalTextClassifier):
    def predict_priority(
        self,
        text: str,
        category: str,
        establishment: dict | None,
        province: str | None = None,
    ) -> Prediction:
        feature_text = priority_feature_text(text, category, establishment, province)
        pred = self.predict_full(feature_text)
        if pred.probabilities:
            expected = 0.0
            total = 0.0
            for label, prob in pred.probabilities.items():
                expected += config.PRIORITY_SCORE_CENTER.get(label, 15) * prob
                total += prob
            pred.score = int(round(expected / total)) if total else config.PRIORITY_SCORE_CENTER.get(pred.label, 15)
        else:
            pred.score = config.PRIORITY_SCORE_CENTER.get(pred.label, 15)
        return pred

    @classmethod
    def load(cls, path: Path) -> "PriorityClassifier":
        obj = joblib.load(path)
        if isinstance(obj, dict) and "pipe" in obj:
            return cls(obj["pipe"], obj.get("label_order"))
        return cls(obj)


# ---------------------------------------------------------------------------
# Training/evaluation
# ---------------------------------------------------------------------------

def _classification_metrics(texts: list[str], labels: list[str], min_df: int = 2) -> dict:
    labels_series = pd.Series(labels)
    if len(set(labels)) < 2 or len(labels) < 10:
        return {"available": False, "reason": "Not enough labeled samples/classes."}

    # Some small classes can break stratified splitting. Use simple split if needed.
    stratify = labels if labels_series.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=stratify
    )
    ev = _make_text_pipeline(min_df=min_df).fit(Xtr, ytr)
    pred = ev.predict(Xte)
    metrics = {
        "available": True,
        "holdout_accuracy": round(float(accuracy_score(yte, pred)), 4),
        "test_rows": len(yte),
        "train_rows": len(ytr),
        "classification_report": classification_report(yte, pred, zero_division=0),
    }
    # Cross-validation is intentionally skipped for hackathon speed; hold-out metrics are enough.
    return metrics


def _train_category(complaints: pd.DataFrame) -> tuple[LocalTextClassifier, dict]:
    texts = complaints["text"].fillna("").astype(str).tolist()
    has_labels = (complaints["category_display"].astype(str).str.strip() != "").mean() > 0.5
    if has_labels:
        labels = complaints["category_display"].apply(config.to_display).tolist()
        label_source = "supervised_csv_category"
    else:
        # Last-resort fallback if future CSV has no labels.
        from .labeler import CATEGORY_BASE
        labels = []
        for text in texts:
            lowered = text.lower()
            best = "Other"
            best_hits = 0
            for cat in CATEGORY_BASE:
                if cat == "Other":
                    continue
                hits = lowered.count(cat.split()[0].lower())
                if hits > best_hits:
                    best, best_hits = cat, hits
            labels.append(best)
        label_source = "weak_fallback_category"

    print(f"[model] training category classifier on {len(labels)} rows ({label_source}).")
    metrics = _classification_metrics(texts, labels, min_df=2)
    pipe = _make_text_pipeline(min_df=2, c=8.0).fit(texts, labels)
    clf = LocalTextClassifier(pipe)
    clf.save(config.CATEGORY_MODEL_PATH)
    meta = {
        "model_type": "TF-IDF + SGD Logistic Classifier",
        "label_source": label_source,
        "training_rows": len(labels),
        "classes": sorted(list(set(labels))),
        "metrics": metrics,
        "artifact": str(config.CATEGORY_MODEL_PATH),
    }
    return clf, meta


def _priority_labels(
    complaints: pd.DataFrame,
    matcher,
    establishments_by_id: dict[str, dict],
) -> tuple[list[str], list[str], str, list[int]]:
    texts: list[str] = []
    labels: list[str] = []
    weak_scores: list[int] = []

    has_true = (complaints.get("true_priority", pd.Series([], dtype=str)).astype(str).str.strip() != "").mean() > 0.5
    source = "direct_validated_priority" if has_true else "weak_supervision_generated_from_scope_rules"

    for _, row in complaints.iterrows():
        text = str(row.get("text", ""))
        category = config.to_display(row.get("category_display") or row.get("category") or "Other")
        matched = matcher.match(row.get("purchase_place", ""), row.get("province"))
        est = establishments_by_id.get(matched["establishment_id"]) if matched else None

        feature_text = priority_feature_text(text, category, est, row.get("province"))
        texts.append(feature_text)

        if has_true and str(row.get("true_priority", "")).strip():
            label = str(row.get("true_priority")).strip().upper()
            labels.append(label)
            weak_scores.append(config.PRIORITY_SCORE_CENTER.get(label, 15))
        else:
            weak = weak_priority_label(text, category, est)
            labels.append(weak["final_label"])
            weak_scores.append(weak["score"])

    return texts, labels, source, weak_scores


def _train_priority(
    complaints: pd.DataFrame,
    matcher,
    establishments_by_id: dict[str, dict],
) -> tuple[PriorityClassifier, dict]:
    texts, labels, label_source, weak_scores = _priority_labels(complaints, matcher, establishments_by_id)
    print(f"[model] training priority classifier on {len(labels)} rows ({label_source}).")

    # min_df=1 because structured tokens can be sparse in small datasets.
    metrics = _classification_metrics(texts, labels, min_df=1)
    pipe = _make_text_pipeline(min_df=1, c=6.0).fit(texts, labels)
    clf = PriorityClassifier(pipe)
    clf.save(config.PRIORITY_MODEL_PATH)
    meta = {
        "model_type": "TF-IDF + SGD Logistic Classifier over text + establishment feature tokens",
        "label_source": label_source,
        "training_rows": len(labels),
        "classes": sorted(list(set(labels)), key=lambda x: config.PRIORITY_RANK.get(x, 99)),
        "label_distribution": {str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
        "weak_score_mean": round(float(np.mean(weak_scores)), 2) if weak_scores else None,
        "metrics": metrics,
        "artifact": str(config.PRIORITY_MODEL_PATH),
    }
    return clf, meta


class TriageModels:
    def __init__(self, category_model: LocalTextClassifier, priority_model: PriorityClassifier, metadata: dict):
        self.category_model = category_model
        self.priority_model = priority_model
        self.metadata = metadata

    def predict_category(self, text: str) -> Prediction:
        return self.category_model.predict_full(text)

    def predict_priority(self, text: str, category: str, establishment: dict | None, province: str | None = None) -> Prediction:
        return self.priority_model.predict_priority(text, category, establishment, province)


def _save_metadata(metadata: dict):
    metadata = dict(metadata)
    metadata["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(config.MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _load_metadata() -> dict:
    if not config.MODEL_META_PATH.exists():
        return {}
    try:
        with open(config.MODEL_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def train_models(complaints: pd.DataFrame, matcher, establishments_by_id: dict[str, dict]) -> TriageModels:
    category_model, cat_meta = _train_category(complaints)
    priority_model, pri_meta = _train_priority(complaints, matcher, establishments_by_id)
    metadata = {
        "category_model": cat_meta,
        "priority_model": pri_meta,
        "local_ml": True,
        "external_llm_api_used": False,
        "mandatory_rule_layer": [
            "YELLOW minimum MEDIUM",
            "RED minimum HIGH",
            "RED + serious content => CRITICAL",
            "GREEN constrained to LOW/MEDIUM as provided in the guide",
        ],
    }
    _save_metadata(metadata)
    return TriageModels(category_model, priority_model, metadata)


def get_models(complaints: pd.DataFrame, matcher, establishments_by_id: dict[str, dict], force_train: bool = False) -> TriageModels:
    if not force_train and config.CATEGORY_MODEL_PATH.exists() and config.PRIORITY_MODEL_PATH.exists():
        try:
            print("[model] loading cached local ML models.")
            cat = LocalTextClassifier.load(config.CATEGORY_MODEL_PATH)
            pri = PriorityClassifier.load(config.PRIORITY_MODEL_PATH)
            meta = _load_metadata()
            return TriageModels(cat, pri, meta)
        except Exception as exc:  # noqa: BLE001
            print(f"[model] cached model load failed ({exc}); retraining.")
    return train_models(complaints, matcher, establishments_by_id)


# Backward-compatible helper used by older code paths.
def get_classifier(complaints: pd.DataFrame | None = None):
    if config.CATEGORY_MODEL_PATH.exists():
        return LocalTextClassifier.load(config.CATEGORY_MODEL_PATH)
    if complaints is None:
        from .data_loader import load_complaints
        complaints = load_complaints()
    clf, meta = _train_category(complaints)
    _save_metadata({"category_model": meta})
    return clf


if __name__ == "__main__":
    from .data_loader import load_complaints, load_establishments
    from .matcher import EstablishmentMatcher

    complaints_df = load_complaints()
    establishments_df = load_establishments()
    matcher = EstablishmentMatcher(establishments_df)
    establishments_by_id = {
        r["establishment_id"]: {
            "establishment_id": r["establishment_id"],
            "name": r["name"],
            "zone": r.get("zone", "UNKNOWN"),
            "province": r.get("province") or None,
            "violations": int(r.get("violations", 0) or 0),
            "open_complaints": int(r.get("open_complaints", 0) or 0),
        }
        for _, r in establishments_df.iterrows()
    }
    train_models(complaints_df, matcher, establishments_by_id)
