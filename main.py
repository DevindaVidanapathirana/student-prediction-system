"""
FastAPI service exposing classification (persona) and regression (engagement score)
models trained in model_test.ipynb.

Run locally:
    uvicorn main:app --reload

Dependencies:
    pip install fastapi uvicorn[standard] joblib pandas scikit-learn
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Student Engagement Service",
    description="Predict personas and engagement scores for students",
    version="1.0.0",
)

# Enable broad CORS for local testing/UI usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File locations (relative to repo root)
BASE_DIR = Path(__file__).resolve().parent
CLASSIFIER_PATH = BASE_DIR / "student_engagement_model.pkl"
CLASSIFIER_SCALER_PATH = BASE_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
REGRESSOR_PATH = BASE_DIR / "engagement_score_model.pkl"
REGRESSOR_SCALER_PATH = BASE_DIR / "engagement_score_scaler.pkl"

# Loaded artifacts
classifier = None
classifier_scaler = None
label_encoder = None
regressor = None
regressor_scaler = None


class StudentFeatures(BaseModel):
    """Input schema for engagement features."""

    login_frequency: float = Field(..., ge=0, description="Logins per week")
    session_duration: float = Field(..., ge=0, description="Avg session minutes")
    forum_participation: float = Field(..., ge=0, description="Forum posts/replies per week")
    assignment_access: float = Field(..., ge=0, description="Assignments opened per week")
    time_gap_avg: float = Field(..., ge=0, description="Avg days between logins")
    inactivity_days: float = Field(..., ge=0, description="Recent days inactive")

    def as_dataframe(self) -> pd.DataFrame:
        cols = [
            "login_frequency",
            "session_duration",
            "forum_participation",
            "assignment_access",
            "time_gap_avg",
            "inactivity_days",
        ]
        return pd.DataFrame([[getattr(self, c) for c in cols]], columns=cols)


def load_artifacts() -> None:
    """Load models and scalers from disk if present."""

    global classifier, classifier_scaler, label_encoder, regressor, regressor_scaler

    if CLASSIFIER_PATH.exists():
        classifier = joblib.load(CLASSIFIER_PATH)
    if CLASSIFIER_SCALER_PATH.exists():
        classifier_scaler = joblib.load(CLASSIFIER_SCALER_PATH)
    if LABEL_ENCODER_PATH.exists():
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    if REGRESSOR_PATH.exists():
        regressor = joblib.load(REGRESSOR_PATH)
    if REGRESSOR_SCALER_PATH.exists():
        regressor_scaler = joblib.load(REGRESSOR_SCALER_PATH)


load_artifacts()


def ensure_classifier() -> None:
    if classifier is None or classifier_scaler is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Classification artifacts missing. Re-train or place pkl files next to main.py.")


def ensure_regressor() -> None:
    if regressor is None or regressor_scaler is None:
        raise HTTPException(status_code=503, detail="Regression artifacts missing. Re-train or place pkl files next to main.py.")


def engagement_action(score: float) -> str:
    if score >= 80:
        return "No action"
    if score >= 50:
        return "Light nudges"
    return "Immediate intervention"


@app.get("/health")
def health() -> Dict[str, bool]:
    """Lightweight health check."""

    return {
        "ok": True,
        "classifier_loaded": classifier is not None,
        "regressor_loaded": regressor is not None,
    }


@app.post("/persona")
def predict_persona(features: StudentFeatures) -> Dict[str, object]:
    """Predict persona label and probabilities."""

    ensure_classifier()
    df = features.as_dataframe()
    X_scaled = classifier_scaler.transform(df)

    pred_idx = classifier.predict(X_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    prob_values = classifier.predict_proba(X_scaled)[0]
    classes = list(label_encoder.classes_)

    return {
        "persona": pred_label,
        "probabilities": {cls: float(prob_values[i]) for i, cls in enumerate(classes)},
    }


@app.post("/score")
def predict_score(features: StudentFeatures) -> Dict[str, float | str]:
    """Predict engagement score and recommended action."""

    ensure_regressor()
    df = features.as_dataframe()
    X_scaled = regressor_scaler.transform(df)
    score = float(regressor.predict(X_scaled)[0])

    return {
        "engagement_score": round(score, 2),
        "action": engagement_action(score),
    }


@app.post("/predict")
def predict_all(features: StudentFeatures) -> Dict[str, object]:
    """Predict both persona and engagement score."""

    persona_part = {}
    score_part = {}

    # Run classification if available
    if classifier is not None and classifier_scaler is not None and label_encoder is not None:
        df = features.as_dataframe()
        X_scaled_cls = classifier_scaler.transform(df)
        pred_idx = classifier.predict(X_scaled_cls)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        prob_values = classifier.predict_proba(X_scaled_cls)[0]
        classes = list(label_encoder.classes_)
        persona_part = {
            "persona": pred_label,
            "probabilities": {cls: float(prob_values[i]) for i, cls in enumerate(classes)},
        }

    # Run regression if available
    if regressor is not None and regressor_scaler is not None:
        df_reg = features.as_dataframe()
        X_scaled_reg = regressor_scaler.transform(df_reg)
        score = float(regressor.predict(X_scaled_reg)[0])
        score_part = {
            "engagement_score": round(score, 2),
            "action": engagement_action(score),
        }

    if not persona_part and not score_part:
        raise HTTPException(status_code=503, detail="No models loaded. Place pickle files in the project root.")

    return {**persona_part, **score_part}


# Convenience root endpoint
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Student Engagement API. See /docs for Swagger UI."}
