"""End-to-end reproducible pipeline.

Running this module will:
    1. Load the raw dataset from ``data/raw/``.
    2. Clean it and build a leakage-safe, model-ready train/test split.
    3. Train and compare baseline classifiers and regressors.
    4. Persist the champion classifier and regressor to ``models/``.
    5. Persist evaluation metrics/results to ``reports/``.

Usage
-----
    python -m src.pipeline
"""
import json
from pathlib import Path

import joblib

from src.data_loader import load_dataset
from src.evaluate import evaluate_classifier, evaluate_regressor
from src.preprocessing import build_processed_dataset
from src.train import (
    select_champion,
    train_and_compare_classifiers,
    train_and_compare_regressors,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def run_pipeline() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw dataset...")
    df = load_dataset()

    print("Building leakage-safe, model-ready dataset...")
    data = build_processed_dataset(df)

    print("Training and comparing classification models (SMOTE-balanced)...")
    clf_results, clf_models = train_and_compare_classifiers(data)
    champion_clf_name, champion_clf = select_champion(clf_results, clf_models, "Macro-F1")
    print(f"Champion classifier: {champion_clf_name}")

    print("Training and comparing regression models...")
    reg_results, reg_models = train_and_compare_regressors(data)
    champion_reg_name, champion_reg = select_champion(reg_results, reg_models, "R2")
    print(f"Champion regressor: {champion_reg_name}")

    clf_eval = evaluate_classifier(champion_clf, data)
    reg_eval = evaluate_regressor(champion_reg, data)

    # Persist model artifacts.
    joblib.dump(champion_clf, MODELS_DIR / f"champion_classifier_{champion_clf_name.replace(' ', '_')}.pkl")
    joblib.dump(champion_reg, MODELS_DIR / f"champion_regressor_{champion_reg_name.replace(' ', '_')}.pkl")
    joblib.dump(data.scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(data.label_encoder, MODELS_DIR / "label_encoder.pkl")

    # Persist reports.
    clf_results.to_csv(REPORTS_DIR / "classification_model_comparison.csv", index=False)
    reg_results.to_csv(REPORTS_DIR / "regression_model_comparison.csv", index=False)

    metrics = {
        "champion_classifier": champion_clf_name,
        "champion_regressor": champion_reg_name,
        "classification_evaluation": clf_eval,
        "regression_evaluation": reg_eval,
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Artifacts saved to '{MODELS_DIR}' and reports saved to '{REPORTS_DIR}'.")


if __name__ == "__main__":
    run_pipeline()
