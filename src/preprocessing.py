"""Preprocessing and feature engineering utilities.

This module reproduces, as clean and testable functions, the leakage-safe
preprocessing pipeline developed in the capstone notebook
(``notebooks/Gerric_Gomez_Capstone_Project_Pillar_5_Steps_1to5.ipynb``):

    Raw Data -> Cleaning -> Train/Test Split -> Outlier Capping
             -> Feature Engineering -> Encoding -> Scaling -> Model-Ready

All fitting (bounds, encoders, scalers) is done on the training split only and
then applied to the test split, to avoid data leakage.
"""
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Columns that must never be used as predictors.
LEAKAGE_COLUMNS = ["passed"]
ID_COLUMNS = ["student_id"]

# Categorical columns and how they are encoded.
GRADE_ORDER = {
    "10th": 1,
    "11th": 2,
    "12th": 3,
    "1st Year": 4,
    "2nd Year": 5,
    "3rd Year": 6,
}
ONE_HOT_COLUMNS = ["gender", "ai_tools_used", "ai_usage_purpose"]

CLASS_TARGET = "performance_category"
REG_TARGET = "final_score"


@dataclass
class ProcessedData:
    """Container for the fully processed, model-ready dataset."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    y_reg_train: np.ndarray
    y_reg_test: np.ndarray
    label_encoder: LabelEncoder
    scaler: StandardScaler
    outlier_bounds: Dict[str, Tuple[float, float]]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage/identifier columns, duplicates, and fill missing categoricals."""
    df_clean = df.copy()
    df_clean = df_clean.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df_clean.columns])
    df_clean = df_clean.drop(columns=[c for c in ID_COLUMNS if c in df_clean.columns])
    df_clean = df_clean.drop_duplicates()
    for col in ["ai_tools_used", "ai_usage_purpose"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("None")
    return df_clean


def split_dataset(
    df_clean: pd.DataFrame, test_size: float = 0.20, random_state: int = 42
):
    """Stratified train/test split performed BEFORE any preprocessing."""
    X = df_clean.drop(columns=[CLASS_TARGET, REG_TARGET])
    y_class = df_clean[CLASS_TARGET]
    y_reg = df_clean[REG_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=test_size, stratify=y_class, random_state=random_state
    )
    y_reg_train = y_reg.loc[X_train.index]
    y_reg_test = y_reg.loc[X_test.index]
    return X_train, X_test, y_train, y_test, y_reg_train, y_reg_test


def compute_outlier_bounds(
    df: pd.DataFrame, cols, k: float = 1.5
) -> Dict[str, Tuple[float, float]]:
    """Compute IQR-based winsorization bounds, fit on training data only."""
    bounds = {}
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - k * iqr, q3 + k * iqr)
    return bounds


def apply_outlier_bounds(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Clip numeric columns to the pre-computed IQR bounds."""
    df_capped = df.copy()
    for col, (lo, hi) in bounds.items():
        df_capped[col] = df_capped[col].clip(lo, hi)
    return df_capped


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 10 domain-driven engineered features."""
    df = df.copy()

    df["ai_risk_index"] = (df["ai_dependency_score"] * df["ai_generated_content_percentage"]) / 100
    df["ethics_dependency_gap"] = df["ai_ethics_score"] - df["ai_dependency_score"]
    df["study_efficiency"] = df["concept_understanding_score"] / (df["study_hours_per_day"] + 0.1)
    df["distraction_ratio"] = df["social_media_hours"] / (df["study_hours_per_day"] + 0.1)
    df["adequate_sleep"] = (df["sleep_hours"] >= 7).astype(int)
    df["engagement_score"] = (
        df["attendance_percentage"] / 10 + df["class_participation_score"] + df["study_consistency_index"]
    ) / 3
    df["heavy_ai_user"] = (
        (df["ai_dependency_score"] >= 7) & (df["ai_usage_time_minutes"] >= 90)
    ).astype(int)
    df["minutes_per_prompt"] = df["ai_usage_time_minutes"] / (df["ai_prompts_per_week"] + 1)
    df["academic_momentum"] = df["last_exam_score"] * (1 + df["improvement_rate"] / 100)
    df["prior_performance"] = (df["last_exam_score"] + df["assignment_scores_avg"]) / 2
    return df


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Ordinal-encode ``grade_level`` and one-hot encode the remaining categoricals."""

    def _encode(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["grade_level"] = df["grade_level"].map(GRADE_ORDER)
        return df

    X_train_enc = _encode(X_train)
    X_test_enc = _encode(X_test)

    cols = [c for c in ONE_HOT_COLUMNS if c in X_train_enc.columns]
    X_train_enc = pd.get_dummies(X_train_enc, columns=cols)
    X_test_enc = pd.get_dummies(X_test_enc, columns=cols)

    # Align columns between train and test (fit on train, transform test).
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
    return X_train_enc, X_test_enc


def scale_features(X_train_enc: pd.DataFrame, X_test_enc: pd.DataFrame):
    """Standardize numeric features, fitting the scaler on the training split only."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_enc), columns=X_train_enc.columns, index=X_train_enc.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_enc), columns=X_test_enc.columns, index=X_test_enc.index
    )
    return X_train_scaled, X_test_scaled, scaler


def build_processed_dataset(df: pd.DataFrame, random_state: int = 42) -> ProcessedData:
    """Run the full leakage-safe preprocessing pipeline end-to-end."""
    df_clean = clean_dataset(df)
    X_train, X_test, y_train, y_test, y_reg_train, y_reg_test = split_dataset(
        df_clean, random_state=random_state
    )

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    bounds = compute_outlier_bounds(X_train, numeric_features)
    X_train_capped = apply_outlier_bounds(X_train, bounds)
    X_test_capped = apply_outlier_bounds(X_test, bounds)

    X_train_fe = engineer_features(X_train_capped)
    X_test_fe = engineer_features(X_test_capped)

    X_train_enc, X_test_enc = encode_categoricals(X_train_fe, X_test_fe)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_enc, X_test_enc)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    return ProcessedData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train_enc,
        y_test=y_test_enc,
        y_reg_train=y_reg_train.to_numpy(),
        y_reg_test=y_reg_test.to_numpy(),
        label_encoder=label_encoder,
        scaler=scaler,
        outlier_bounds=bounds,
    )
