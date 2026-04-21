# 🫁 Early Chronic Kidney Disease Detection

An end-to-end Machine Learning project for early CKD detection, featuring a full training pipeline and a Streamlit-powered diagnostic web app.

---








## 📁 Project Structure

```
ckd_project/
│
├── data/
│   └── FINAL CKD DATASET.xlsx       # 1,659 patient records
│
├── models/                           # Auto-generated after training
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   ├── best_model_name.pkl
│   ├── results_df.pkl
│   ├── XGBoost.pkl
│   ├── Gradient_Boosting.pkl
│   ├── Random_Forest.pkl
│   ├── AdaBoost.pkl
│   └── SVM.pkl
│
├── outputs/                          # Auto-generated plots
│   ├── performance_metrics.png
│   ├── roc_auc_curves.png
│   ├── confusion_matrices.png
│   └── comparison_table.png
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                 # Data loading, scaling, clinical flags
│   ├── train.py                      # Training pipeline + model persistence
│   └── predict.py                    # Inference module
│
├── app/
│   └── streamlit_app.py              # Streamlit UI
│
├── notebooks/                        # (Optional) EDA notebooks
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone / unzip the project

```bash
cd ckd_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train all models (run once)

```bash
python -m src.train
```

This will:
- Load and preprocess the dataset
- Train 5 classifiers (XGBoost, Gradient Boosting, Random Forest, AdaBoost, SVM)
- Save all `.pkl` model files to `models/`
- Generate performance plots in `outputs/`

### 4. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 🤖 Models

| Model | F1 Score | Accuracy | ROC-AUC |
|---|---|---|---|
| AdaBoost | 96.35% | 93.07% | 71.66% |
| Random Forest | 95.89% | 92.17% | 74.30% |
| SVM | 95.76% | 91.87% | 68.17% |
| XGBoost | 95.69% | 91.87% | 76.16% |
| Gradient Boosting | 95.51% | 91.57% | 76.43% |

---

## 🔬 Features Used (17)

| Category | Features |
|---|---|
| Demographics | Age, Gender, Ethnicity, BMI |
| Kidney Function | GFR, Serum Creatinine, BUN, Protein in Urine, ACR |
| Blood Markers | HbA1c, Hemoglobin, Cholesterol, Triglycerides, Fasting Blood Sugar |
| Blood Pressure | Systolic BP, Diastolic BP |
| Risk Factors | Smoking, Family History, Previous AKI, UTIs |

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. It does not constitute medical advice and should not be used for clinical diagnosis.
