"""
train.py
────────
Trains all five classifiers, evaluates them, generates performance plots,
and saves every model as a .pkl file under models/.

Run this ONCE to produce the artifacts needed by the Streamlit app:

    python -m src.train          (from the ckd_project/ root)
    — or —
    python src/train.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)

from src.preprocess import (
    load_and_preprocess, save_scaler_and_metadata,
    DATA_PATH, MODELS_DIR
)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
RANDOM_STATE = 42


def build_models():
    return {
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=RANDOM_STATE
        ),
    }


def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba, {
        "Accuracy" : round(accuracy_score(y_test, y_pred)                           * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0)         * 100, 2),
        "Recall"   : round(recall_score(y_test, y_pred, zero_division=0)            * 100, 2),
        "F1 Score" : round(f1_score(y_test, y_pred, zero_division=0)                * 100, 2),
        "ROC-AUC"  : round(roc_auc_score(y_test, y_proba)                           * 100, 2),
    }


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _save_metrics_bar_charts(results, model_names, outputs_dir):
    METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    BAR_COLORS  = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("CKD Detection — Per-Model Performance Metrics",
                 fontsize=16, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        vals = [results[name][m] for m in METRIC_KEYS]
        bars = ax.bar(METRIC_KEYS, vals, color=BAR_COLORS, width=0.55,
                      edgecolor="white", linewidth=0.8)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Score (%)")
        ax.tick_params(axis="x", rotation=20)
        ax.axhline(y=90, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    axes[-1].set_visible(False)
    plt.tight_layout(pad=2.5)
    path = os.path.join(outputs_dir, "performance_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Saved: {path}")


def _save_roc_curves(results, proba_scores, y_test, model_names, outputs_dir):
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(model_names):
        fpr, tpr, _ = roc_curve(y_test, proba_scores[name])
        ax.plot(fpr, tpr, color=COLORS[i], lw=2,
                label=f"{name}  (AUC = {results[name]['ROC-AUC']:.2f}%)")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (50%)")
    ax.set(xlim=[0, 1], ylim=[0, 1.05],
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — All Models")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outputs_dir, "roc_auc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Saved: {path}")


def _save_confusion_matrices(predictions, y_test, model_names, outputs_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Confusion Matrices — All Models", fontsize=15, fontweight="bold")
    axes = axes.flatten()
    for idx, name in enumerate(model_names):
        cm = confusion_matrix(y_test, predictions[name])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["No CKD", "CKD"],
                    yticklabels=["No CKD", "CKD"],
                    linewidths=0.5, linecolor="gray")
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].set(ylabel="Actual", xlabel="Predicted")
    axes[-1].set_visible(False)
    plt.tight_layout(pad=2.5)
    path = os.path.join(outputs_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Saved: {path}")


def _save_comparison_table(results_df, outputs_dir):
    METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    col_labels = ["Rank", "Model"] + METRIC_KEYS
    table_data = [
        [str(i), row["Model"]] + [f"{row[m]:.2f}%" for m in METRIC_KEYS]
        for i, row in results_df.iterrows()
    ]
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.2, 2.0)
    for col_idx in range(len(col_labels)):
        tbl[0, col_idx].set_facecolor("#2c3e50")
        tbl[0, col_idx].set_text_props(color="white", fontweight="bold")
    for row_idx in range(1, len(table_data) + 1):
        bg = "#d5f5e3" if row_idx == 1 else ("#f2f3f4" if row_idx % 2 == 0 else "white")
        for col_idx in range(len(col_labels)):
            tbl[row_idx, col_idx].set_facecolor(bg)
    ax.set_title("Model Comparison — Ranked by F1 Score",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(outputs_dir, "comparison_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Saved: {path}")


# ── Main training pipeline ────────────────────────────────────────────────────

def train_and_save(data_path=DATA_PATH, models_dir=MODELS_DIR, outputs_dir=OUTPUTS_DIR):
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print("=" * 65)
    print("  CHRONIC KIDNEY DISEASE — ML Training Pipeline")
    print("=" * 65)

    # 1. Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_cols, _ = \
        load_and_preprocess(data_path)
    save_scaler_and_metadata(scaler, feature_cols, models_dir)

    # 2. Train
    print("\n" + "─" * 65)
    print("  TRAINING MODELS")
    print("─" * 65)

    models       = build_models()
    results      = {}
    predictions  = {}
    proba_scores = {}

    for name, model in models.items():
        print(f"\n  ► {name} …", end=" ", flush=True)
        model.fit(X_train, y_train)
        print("done.")

        y_pred, y_proba, metrics = evaluate(model, X_test, y_test)
        results[name]      = metrics
        predictions[name]  = y_pred
        proba_scores[name] = y_proba

        print(f"    Acc={metrics['Accuracy']:.2f}%  "
              f"Prec={metrics['Precision']:.2f}%  "
              f"Rec={metrics['Recall']:.2f}%  "
              f"F1={metrics['F1 Score']:.2f}%  "
              f"AUC={metrics['ROC-AUC']:.2f}%")

        # Save model
        model_path = os.path.join(models_dir, f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        print(f"    💾  Saved → {model_path}")

    # 3. Results summary
    model_names = list(models.keys())
    results_df  = pd.DataFrame(results).T.reset_index()
    results_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    results_df = results_df.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    results_df.index += 1

    best_model = results_df.iloc[0]["Model"]
    print(f"\n  🏆  Best model by F1: {best_model}")
    print("\n" + results_df.to_string())

    # Save best model name for the app
    joblib.dump(best_model, os.path.join(models_dir, "best_model_name.pkl"))
    joblib.dump(results_df, os.path.join(models_dir, "results_df.pkl"))

    # 4. Plots
    print("\n" + "─" * 65)
    print("  GENERATING PLOTS")
    print("─" * 65)

    _save_metrics_bar_charts(results, model_names, outputs_dir)
    _save_roc_curves(results, proba_scores, y_test, model_names, outputs_dir)
    _save_confusion_matrices(predictions, y_test, model_names, outputs_dir)
    _save_comparison_table(results_df, outputs_dir)

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"\n  Models saved to  : {models_dir}/")
    print(f"  Plots saved to   : {outputs_dir}/\n")

    return results_df, best_model


if __name__ == "__main__":
    train_and_save()
