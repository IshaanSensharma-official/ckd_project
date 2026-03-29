"""
predict.py
──────────
Inference module — loads any saved model and runs a single-patient
prediction. Used by the Streamlit app.
"""

import os
import joblib
import numpy as np
from src.preprocess import (
    load_scaler_and_metadata,
    preprocess_input,
    flag_clinical_abnormalities,
    MODELS_DIR
)

# Available model file names → display names
MODEL_FILES = {
    "XGBoost"          : "XGBoost.pkl",
    "Gradient Boosting": "Gradient_Boosting.pkl",
    "Random Forest"    : "Random_Forest.pkl",
    "AdaBoost"         : "AdaBoost.pkl",
    "SVM"              : "SVM.pkl",
}


def list_available_models(models_dir: str = MODELS_DIR) -> list[str]:
    """Return model display names whose .pkl files exist on disk."""
    available = []
    for name, fname in MODEL_FILES.items():
        if os.path.exists(os.path.join(models_dir, fname)):
            available.append(name)
    return available


def load_model(model_name: str, models_dir: str = MODELS_DIR):
    fname = MODEL_FILES.get(model_name)
    if fname is None:
        raise ValueError(f"Unknown model '{model_name}'")
    path = os.path.join(models_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run `python -m src.train` first."
        )
    return joblib.load(path)


def get_best_model_name(models_dir: str = MODELS_DIR) -> str:
    path = os.path.join(models_dir, "best_model_name.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return "XGBoost"   # sensible fallback


def predict_patient(patient_dict: dict, model_name: str = None,
                    models_dir: str = MODELS_DIR) -> dict:
    """
    Full inference pipeline for a single patient.

    Parameters
    ----------
    patient_dict : dict   – raw feature values keyed by column name
    model_name   : str    – one of MODEL_FILES keys; if None, uses best model

    Returns
    -------
    dict with keys:
        prediction      – 0 (No CKD) or 1 (CKD)
        label           – "CKD Detected" / "No CKD Detected"
        confidence      – probability for the predicted class (0–1)
        ckd_probability – raw CKD (class=1) probability
        model_used      – name of model that ran inference
        clinical_flags  – list of abnormal lab findings
        risk_level      – "High" / "Moderate" / "Low"
    """
    scaler, feature_cols = load_scaler_and_metadata(models_dir)

    if model_name is None:
        model_name = get_best_model_name(models_dir)

    model = load_model(model_name, models_dir)

    # Scale input
    X = preprocess_input(patient_dict, scaler, feature_cols)

    # Predict
    prediction   = int(model.predict(X)[0])
    proba        = model.predict_proba(X)[0]         # [P(No CKD), P(CKD)]
    ckd_prob     = float(proba[1])
    confidence   = float(proba[prediction])

    # Clinical flags
    flags = flag_clinical_abnormalities(patient_dict)

    # Risk level based on CKD probability
    if ckd_prob >= 0.75:
        risk_level = "High"
    elif ckd_prob >= 0.45:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "prediction"     : prediction,
        "label"          : "CKD Detected" if prediction == 1 else "No CKD Detected",
        "confidence"     : round(confidence * 100, 2),
        "ckd_probability": round(ckd_prob * 100, 2),
        "model_used"     : model_name,
        "clinical_flags" : flags,
        "risk_level"     : risk_level,
    }
