"""Model evaluation utilities."""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing import ProcessedData


def evaluate_classifier(model, data: ProcessedData) -> Dict:
    """Return a structured evaluation report for a trained classifier."""
    y_pred = model.predict(data.X_test)
    class_names = list(data.label_encoder.classes_)

    report = classification_report(
        data.y_test, y_pred, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(data.y_test, y_pred).tolist()

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": class_names,
    }


def evaluate_regressor(model, data: ProcessedData) -> Dict:
    """Return R2/RMSE/MAE for a trained regressor."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(data.X_test)
    return {
        "r2": float(r2_score(data.y_reg_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(data.y_reg_test, y_pred))),
        "mae": float(mean_absolute_error(data.y_reg_test, y_pred)),
    }
