"""Model training and comparison utilities.

Trains a suite of classifiers to predict ``performance_category`` (Low /
Medium / High) and a suite of regressors to predict ``final_score``,
selecting a "champion" model of each type based on held-out test performance.
"""
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.preprocessing import ProcessedData

RANDOM_STATE = 42


def get_classification_models() -> Dict[str, object]:
    """Return the suite of baseline classifiers evaluated in the capstone."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }


def get_regression_models() -> Dict[str, object]:
    """Return the suite of baseline regressors evaluated in the capstone."""
    return {
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }


def balance_with_smote(X_train: pd.DataFrame, y_train: np.ndarray, random_state: int = RANDOM_STATE):
    """Oversample the minority classes of the training set with SMOTE."""
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    return smote.fit_resample(X_train, y_train)


def train_and_compare_classifiers(
    data: ProcessedData,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Train every candidate classifier on SMOTE-balanced data and score it on the test set."""
    X_train_bal, y_train_bal = balance_with_smote(data.X_train, data.y_train)

    results = []
    trained_models = {}
    for name, model in get_classification_models().items():
        start = time.time()
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(data.X_test)
        y_proba = model.predict_proba(data.X_test)

        acc = accuracy_score(data.y_test, y_pred)
        f1_macro = f1_score(data.y_test, y_pred, average="macro")
        f1_weighted = f1_score(data.y_test, y_pred, average="weighted")
        precision_macro = precision_score(data.y_test, y_pred, average="macro")
        recall_macro = recall_score(data.y_test, y_pred, average="macro")

        low_idx = list(data.label_encoder.classes_).index("Low")
        recall_low = recall_score(data.y_test, y_pred, labels=[low_idx], average="macro")
        try:
            auc_ovr = roc_auc_score(data.y_test, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            auc_ovr = np.nan

        results.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Macro-F1": round(f1_macro, 4),
                "Weighted-F1": round(f1_weighted, 4),
                "Macro-Precision": round(precision_macro, 4),
                "Macro-Recall": round(recall_macro, 4),
                "Recall (Low)": round(recall_low, 4),
                "ROC-AUC (OvR)": round(auc_ovr, 4) if not np.isnan(auc_ovr) else np.nan,
                "Train Time (s)": round(time.time() - start, 2),
            }
        )
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("Macro-F1", ascending=False).reset_index(drop=True)
    return results_df, trained_models


def train_and_compare_regressors(data: ProcessedData) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Train every candidate regressor and score it on the test set."""
    results = []
    trained_models = {}
    for name, model in get_regression_models().items():
        start = time.time()
        model.fit(data.X_train, data.y_reg_train)
        y_pred = model.predict(data.X_test)

        r2 = r2_score(data.y_reg_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(data.y_reg_test, y_pred)))
        mae = mean_absolute_error(data.y_reg_test, y_pred)

        results.append(
            {
                "Model": name,
                "R2": round(r2, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "Train Time (s)": round(time.time() - start, 2),
            }
        )
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    return results_df, trained_models


def select_champion(results_df: pd.DataFrame, trained_models: Dict[str, object], metric: str):
    """Return the (name, model) pair with the best value for ``metric``."""
    champion_name = results_df.sort_values(metric, ascending=False).iloc[0]["Model"]
    return champion_name, trained_models[champion_name]
