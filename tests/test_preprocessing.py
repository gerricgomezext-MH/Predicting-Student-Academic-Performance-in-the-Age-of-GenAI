"""Unit tests for the preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_processed_dataset,
    clean_dataset,
    engineer_features,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "student_id": range(1, 21),
            "age": np.random.randint(15, 25, 20),
            "gender": ["Female", "Male"] * 10,
            "grade_level": ["10th", "1st Year"] * 10,
            "study_hours_per_day": np.random.uniform(0.5, 6, 20),
            "uses_ai": np.random.randint(0, 2, 20),
            "ai_usage_time_minutes": np.random.randint(0, 200, 20),
            "ai_tools_used": ["ChatGPT", None] * 10,
            "ai_usage_purpose": ["Homework", None] * 10,
            "ai_dependency_score": np.random.randint(0, 11, 20),
            "ai_generated_content_percentage": np.random.randint(0, 101, 20),
            "ai_prompts_per_week": np.random.randint(0, 120, 20),
            "ai_ethics_score": np.random.randint(0, 11, 20),
            "last_exam_score": np.random.randint(0, 101, 20),
            "assignment_scores_avg": np.random.uniform(0, 100, 20),
            "attendance_percentage": np.random.uniform(0, 100, 20),
            "concept_understanding_score": np.random.randint(0, 11, 20),
            "study_consistency_index": np.random.uniform(0, 10, 20),
            "improvement_rate": np.random.uniform(-20, 20, 20),
            "sleep_hours": np.random.uniform(4, 10, 20),
            "social_media_hours": np.random.uniform(0, 6, 20),
            "tutoring_hours": np.random.uniform(0, 5, 20),
            "class_participation_score": np.random.randint(0, 11, 20),
            "final_score": np.random.uniform(0, 100, 20),
            "passed": np.random.randint(0, 2, 20),
            "performance_category": (["Low", "Medium", "High"] * 7)[:20],
        }
    )


def test_clean_dataset_drops_leakage_and_id_columns(sample_df):
    cleaned = clean_dataset(sample_df)
    assert "passed" not in cleaned.columns
    assert "student_id" not in cleaned.columns
    assert cleaned["ai_tools_used"].isnull().sum() == 0
    assert cleaned["ai_usage_purpose"].isnull().sum() == 0


def test_engineer_features_adds_expected_columns(sample_df):
    cleaned = clean_dataset(sample_df)
    engineered = engineer_features(cleaned)
    expected_new_cols = {
        "ai_risk_index",
        "ethics_dependency_gap",
        "study_efficiency",
        "distraction_ratio",
        "adequate_sleep",
        "engagement_score",
        "heavy_ai_user",
        "minutes_per_prompt",
        "academic_momentum",
        "prior_performance",
    }
    assert expected_new_cols.issubset(set(engineered.columns))


def test_build_processed_dataset_shapes_align(sample_df):
    data = build_processed_dataset(sample_df, random_state=0)
    assert data.X_train.shape[1] == data.X_test.shape[1]
    assert len(data.y_train) == data.X_train.shape[0]
    assert len(data.y_test) == data.X_test.shape[0]
    assert set(data.label_encoder.classes_) == {"Low", "Medium", "High"}
