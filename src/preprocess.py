"""
preprocess.py
─────────────
Handles all data loading, cleaning, and feature engineering for the
CKD detection pipeline. Exports the scaler and feature column list
so the Streamlit app can reuse them at inference time.
"""




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TARGET       = "Diagnosis"
DROP_COLS    = ["PatientID", "DoctorInCharge"]

DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "FINAL CKD DATASET.xlsx")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")

# Clinical thresholds for the UI report (ISN Atlas 2023)
CLINICAL_THRESHOLDS = {
    "GFR"             : ("<",  60,   "mL/min/1.73m²", "Reduced kidney function"),
    "SerumCreatinine" : (">",  1.2,  "mg/dL",         "Elevated creatinine"),
    "BUNLevels"       : (">",  20,   "mg/dL",         "Elevated blood urea nitrogen"),
    "ProteinInUrine"  : (">",  0.3,  "g/day",         "Pathological proteinuria"),
    "ACR"             : (">=", 30,   "mg/g",          "Increased albuminuria"),
    "HbA1c"           : (">=", 6.5,  "%",             "Diabetic range"),
    "HemoglobinLevels": ("<",  12,   "g/dL",          "Anemia of CKD"),
    "BMI"             : (">=", 30,   "kg/m²",         "Obesity (CKD risk factor)"),
}


def load_and_preprocess(data_path: str = DATA_PATH):
    """
    Load the Excel dataset, drop non-predictive columns, scale features.

    Returns
    -------
    X_train, X_test, y_train, y_test  – numpy arrays
    scaler                             – fitted StandardScaler
    feature_cols                       – list of feature column names
    df_raw                             – original DataFrame (for patient info)
    """
    df_raw = pd.read_excel(data_path)

    # Validate required target column
    if TARGET not in df_raw.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")

    # Drop non-predictive identifiers
    df = df_raw.drop(columns=DROP_COLS, errors="ignore")

    # Separate features and target
    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols].values
    y = df[TARGET].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"[INFO] Dataset loaded  →  {len(df_raw)} patients, {len(feature_cols)} features")
    print(f"[INFO] CKD (1): {(y == 1).sum()}  |  No CKD (0): {(y == 0).sum()}")
    print(f"[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}")

    return X_train, X_test, y_train, y_test, scaler, feature_cols, df_raw


def save_scaler_and_metadata(scaler, feature_cols, models_dir: str = MODELS_DIR):
    """Persist the scaler and feature list so the app can use them."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler,       os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_cols.pkl"))
    print(f"[INFO] Scaler & feature list saved to '{models_dir}/'")


def load_scaler_and_metadata(models_dir: str = MODELS_DIR):
    """Load the scaler and feature list for inference."""
    scaler       = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(models_dir, "feature_cols.pkl"))
    return scaler, feature_cols


def preprocess_input(patient_dict: dict, scaler, feature_cols) -> np.ndarray:
    """
    Convert a raw patient dict (from the Streamlit form) into a
    scaled numpy array ready for model.predict().

    Parameters
    ----------
    patient_dict : dict  – {feature_name: value}
    scaler       : fitted StandardScaler
    feature_cols : list  – ordered feature names

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    row = np.array([[patient_dict[col] for col in feature_cols]], dtype=float)
    return scaler.transform(row)


def flag_clinical_abnormalities(patient_dict: dict) -> list[dict]:
    """
    Return a list of flagged abnormal lab values with clinical context.

    Returns
    -------
    list of {"feature", "value", "threshold", "unit", "message"}
    """
    flags = []
    for feature, (op, threshold, unit, message) in CLINICAL_THRESHOLDS.items():
        value = patient_dict.get(feature)
        if value is None:
            continue
        triggered = (
            (op == "<"  and value <  threshold) or
            (op == ">"  and value >  threshold) or
            (op == ">=" and value >= threshold) or
            (op == "<=" and value <= threshold)
        )
        if triggered:
            flags.append({
                "feature"  : feature,
                "value"    : value,
                "threshold": f"{op} {threshold} {unit}",
                "message"  : message,
            })
    return flags
