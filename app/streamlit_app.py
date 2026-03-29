"""
app/streamlit_app.py
────────────────────
Streamlit UI for Early Chronic Kidney Disease Detection.
Users fill in patient lab values, pick a model, and receive
a prediction with confidence, clinical flags, and a risk badge.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import joblib
import pandas as pd

from src.predict import (
    predict_patient,
    list_available_models,
    get_best_model_name,
    MODELS_DIR,
)
from src.preprocess import CLINICAL_THRESHOLDS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CKD Early Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f7f9fc; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }

    /* Risk badge */
    .badge-high     { background:#e74c3c; color:white; border-radius:20px; padding:6px 18px; font-weight:700; font-size:1rem; }
    .badge-moderate { background:#f39c12; color:white; border-radius:20px; padding:6px 18px; font-weight:700; font-size:1rem; }
    .badge-low      { background:#27ae60; color:white; border-radius:20px; padding:6px 18px; font-weight:700; font-size:1rem; }

    /* Result banner */
    .result-ckd    { background:#fde8e8; border-left:5px solid #e74c3c; padding:16px; border-radius:8px; }
    .result-nocd   { background:#e8f8f1; border-left:5px solid #27ae60; padding:16px; border-radius:8px; }

    /* Section headers */
    .section-head  { font-size:1.05rem; font-weight:700; color:#2c3e50; margin-bottom:6px; }

    /* Clinical flag row */
    .flag-row { background:#fff8e1; border-radius:6px; padding:8px 12px; margin:4px 0;
                border-left:4px solid #f39c12; font-size:0.9rem; }

    /* Divider */
    hr { border: none; border-top: 1px solid #e0e0e0; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_artefacts():
    available   = list_available_models(MODELS_DIR)
    best_name   = get_best_model_name(MODELS_DIR)
    results_df  = joblib.load(os.path.join(MODELS_DIR, "results_df.pkl"))
    return available, best_name, results_df

available_models, best_model_name, results_df = load_artefacts()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/kidney.png", width=72)
    st.title("CKD Detector")
    st.caption("Early Chronic Kidney Disease Detection powered by ML")
    st.markdown("---")

    st.subheader("⚙️ Model Selection")
    model_choice = st.selectbox(
        "Choose a classifier",
        options=available_models,
        index=available_models.index(best_model_name) if best_model_name in available_models else 0,
        help="All models are pre-trained on 1 659 patients. Best by F1 is highlighted."
    )

    st.markdown(f"🏆 **Best model (F1):** `{best_model_name}`")
    st.markdown("---")

    # Model scorecard
    st.subheader("📊 Model Leaderboard")
    display_df = results_df.copy().reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df["Model"] = display_df["Model"].apply(
        lambda m: f"⭐ {m}" if m == best_model_name else m
    )
    st.dataframe(
        display_df[["Model", "F1 Score", "Accuracy", "ROC-AUC"]],
        use_container_width=True,
        hide_index=False,
    )

    st.markdown("---")
    st.caption("⚠️ This tool is for research / educational use only.\nNot a substitute for medical diagnosis.")


# ── Main content ──────────────────────────────────────────────────────────────
st.title("🫁 Early Chronic Kidney Disease Detection")
st.markdown(
    "Enter the patient's clinical measurements below. "
    "The model will predict whether the patient shows signs of **Early CKD**."
)
st.markdown("---")

# ── Patient Input Form ────────────────────────────────────────────────────────
st.subheader("📋 Patient Lab Values")

with st.form("patient_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-head">Demographics</div>', unsafe_allow_html=True)
        age       = st.number_input("Age (years)",   min_value=1,   max_value=120, value=55)
        gender    = st.selectbox("Gender",           options=["Female (0)", "Male (1)"], index=1)
        bmi       = st.number_input("BMI (kg/m²)",   min_value=10.0, max_value=60.0, value=27.0, step=0.1,
                                    help=">= 30 → Obesity (CKD risk factor)")
        ethnicity = st.selectbox("Ethnicity",
                        options=["African American (0)", "Asian (1)", "Caucasian (2)", "Other (3)"],
                        index=0)

        st.markdown('<div class="section-head" style="margin-top:14px">Risk Factors</div>', unsafe_allow_html=True)
        smoking     = st.selectbox("Smoking",                        options=["No (0)", "Yes (1)"], index=0)
        family_hist = st.selectbox("Family History — Kidney Disease",options=["No (0)", "Yes (1)"], index=0)
        prev_aki    = st.selectbox("Previous Acute Kidney Injury",   options=["No (0)", "Yes (1)"], index=0)
        uti         = st.selectbox("Urinary Tract Infections",       options=["No (0)", "Yes (1)"], index=0)

    with c2:
        st.markdown('<div class="section-head">Kidney Function</div>', unsafe_allow_html=True)
        gfr       = st.number_input("GFR (mL/min/1.73m²)",            min_value=0.0,  max_value=200.0, value=75.0, step=0.1,
                                    help="< 60 → Reduced kidney function")
        creatinine= st.number_input("Serum Creatinine (mg/dL)",        min_value=0.1,  max_value=15.0,  value=1.0, step=0.01,
                                    help="> 1.2 → Elevated")
        bun       = st.number_input("BUN Levels (mg/dL)",              min_value=1.0,  max_value=150.0, value=15.0, step=0.1,
                                    help="> 20 → Elevated")
        protein   = st.number_input("Protein in Urine (g/day)",        min_value=0.0,  max_value=10.0,  value=0.1, step=0.01,
                                    help="> 0.3 → Pathological proteinuria")
        acr       = st.number_input("ACR — Albumin:Creatinine (mg/g)", min_value=0.0,  max_value=500.0, value=20.0, step=0.1,
                                    help=">= 30 → Moderately increased albuminuria")

    with c3:
        st.markdown('<div class="section-head">Blood Markers</div>', unsafe_allow_html=True)
        hba1c     = st.number_input("HbA1c (%)",                       min_value=3.0,  max_value=15.0, value=5.5, step=0.1,
                                    help=">= 6.5 → Diabetic range")
        hemo      = st.number_input("Hemoglobin (g/dL)",               min_value=3.0,  max_value=20.0, value=13.5, step=0.1,
                                    help="< 12 → Anemia of CKD")
        sodium    = st.number_input("Serum Sodium (mEq/L)",            min_value=100.0, max_value=170.0, value=138.0, step=0.1,
                                    help="Normal: 136–145 mEq/L")
        phosphorus= st.number_input("Serum Phosphorus (mg/dL)",        min_value=0.5,  max_value=10.0,  value=3.5, step=0.1,
                                    help="Normal: 2.5–4.5 mg/dL; elevated in CKD")

    submitted = st.form_submit_button("🔍 Run Diagnosis", use_container_width=True, type="primary")


# ── Process & Display Result ──────────────────────────────────────────────────
if submitted:
    with st.spinner("Running inference…"):
        # Parse encoded dropdowns
        gender_val    = int(gender.split("(")[1].replace(")", ""))
        ethnicity_val = int(ethnicity.split("(")[1].replace(")", ""))
        smoking_val   = int(smoking.split("(")[1].replace(")", ""))
        fhist_val     = int(family_hist.split("(")[1].replace(")", ""))
        aki_val       = int(prev_aki.split("(")[1].replace(")", ""))
        uti_val       = int(uti.split("(")[1].replace(")", ""))

        patient = {
            "Age"                        : age,
            "Gender"                     : gender_val,
            "Ethnicity"                  : ethnicity_val,
            "BMI"                        : bmi,
            "Smoking"                    : smoking_val,
            "FamilyHistoryKidneyDisease" : fhist_val,
            "PreviousAcuteKidneyInjury"  : aki_val,
            "UrinaryTractInfections"     : uti_val,
            "HbA1c"                      : hba1c,
            "SerumCreatinine"            : creatinine,
            "BUNLevels"                  : bun,
            "GFR"                        : gfr,
            "ProteinInUrine"             : protein,
            "ACR"                        : acr,
            "SerumElectrolytesSodium"    : sodium,
            "SerumElectrolytesPhosphorus": phosphorus,
            "HemoglobinLevels"           : hemo,
        }

        try:
            result = predict_patient(patient, model_name=model_choice)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

    st.markdown("---")
    st.subheader("📣 Diagnosis Result")

    # ── Result banner
    if result["prediction"] == 1:
        st.markdown(
            f"""<div class="result-ckd">
            <h2 style="margin:0;color:#c0392b">⚠️ {result['label']}</h2>
            <p style="margin:4px 0 0">The model flagged this patient as likely to have <strong>Chronic Kidney Disease</strong>.
            Please consult a nephrologist for further evaluation.</p>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""<div class="result-nocd">
            <h2 style="margin:0;color:#1e8449">✅ {result['label']}</h2>
            <p style="margin:4px 0 0">The model found <strong>no significant indicators</strong> of Chronic Kidney Disease
            based on the provided values. Routine monitoring is still recommended.</p>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CKD Probability",  f"{result['ckd_probability']}%")
    m2.metric("Model Confidence", f"{result['confidence']}%")

    risk = result["risk_level"]
    badge_class = {"High": "badge-high", "Moderate": "badge-moderate", "Low": "badge-low"}[risk]
    m3.markdown(
        f"**Risk Level**<br><span class='{badge_class}'>{risk}</span>",
        unsafe_allow_html=True
    )
    m4.metric("Model Used", result["model_used"])

    # ── Clinical flags
    st.markdown("---")
    flags = result["clinical_flags"]
    if flags:
        st.subheader(f"🚩 {len(flags)} Clinical Abnormalit{'y' if len(flags)==1 else 'ies'} Detected")
        for f in flags:
            st.markdown(
                f"<div class='flag-row'>"
                f"<strong>{f['feature']}</strong>: {f['value']} "
                f"<span style='color:#888'>(threshold {f['threshold']})</span> — "
                f"{f['message']}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.success("✅ No clinical thresholds breached — all measured values are within reference ranges.")

    # ── Disclaimer
    st.markdown("---")
    st.caption(
        "⚠️ **Medical Disclaimer:** This prediction is generated by a machine learning model "
        "trained on a research dataset. It is intended for educational and research purposes only. "
        "It does **not** constitute medical advice and should never replace a qualified physician's evaluation."
    )


# ── Info section at the bottom (when no submission yet) ───────────────────────
if not submitted:
    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("ℹ️ About This Tool")
        st.markdown("""
This app uses five machine-learning classifiers trained on **1,659 patient records**
to detect early Chronic Kidney Disease.

**Models included:**
- 🟢 XGBoost
- 🟢 Gradient Boosting
- 🟢 Random Forest
- 🟢 AdaBoost
- 🟢 SVM (RBF kernel)

Features used span demographics, kidney-function markers, blood panel results,
and risk-factor indicators.
        """)

    with col_b:
        st.subheader("🔬 Key Clinical Markers")
        ref_data = {
            "Marker"    : ["GFR", "Serum Creatinine", "BUN", "Protein in Urine", "ACR", "HbA1c", "Hemoglobin", "BMI"],
            "Threshold" : ["< 60", "> 1.2", "> 20", "> 0.3", "≥ 30", "≥ 6.5", "< 12", "≥ 30"],
            "Unit"      : ["mL/min/1.73m²", "mg/dL", "mg/dL", "g/day", "mg/g", "%", "g/dL", "kg/m²"],
            "Significance": [
                "Reduced kidney function",
                "Elevated creatinine",
                "Elevated BUN",
                "Pathological proteinuria",
                "Increased albuminuria",
                "Diabetic range",
                "Anemia of CKD",
                "Obesity risk factor",
            ]
        }
        st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)
