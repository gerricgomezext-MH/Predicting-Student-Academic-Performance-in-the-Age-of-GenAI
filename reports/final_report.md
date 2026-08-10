# Final Report — Predicting Student Academic Performance in the Age of GenAI

**Author:** Gerric Gomez
**Project:** Capstone Project — Predicting Student Academic Performance in the Age of Generative AI

---

## 1. Executive Summary

This capstone project investigates whether the adoption of Generative AI (GenAI)
tools — ChatGPT, Gemini, Copilot, Claude, and similar assistants — measurably
improves student academic outcomes, or whether excessive dependency on these
tools undermines genuine learning. This study utilized the Academic Outcomes & AI
Dependency Analysis Dataset, a publicly available dataset published on Kaggle by Abdullah
Rashid (DATA SOURCE: Rashid, A. (n.d.). Academic Outcomes & AI Dependency Analysis Dataset. Kaggle. https://www.kaggle.com/datasets/abdullahmeo/academic-outcomes-and-ai-dependency-analysis-dataset)
The dataset contains 8,000 student records that include demographic information,
academic performance, AI usage patterns, AI dependency scores, and learning behavior
variables. It is licensed under the Creative Commons Attribution 4.0 (CC BY 4.0) license,
allowing reuse with proper attribution. 

Using this dataset of 8,000 student records that combines academic performance, lifestyle, and AI-usage behavior, we built and rigorously evaluated a supervised machine learning solution that:

1. **Classifies** each student into a performance category (`Low`, `Medium`,
   `High`) — the primary business deliverable, enabling early intervention for
   at-risk students.
2. **Predicts** the student's continuous `final_score` — a secondary,
   fine-grained regression target for progress tracking.

The finished pipeline (`src/`) is fully reproducible end-to-end: it loads the
raw data, applies leakage-safe preprocessing and feature engineering, trains
and compares multiple candidate models, and persists the champion classifier
and regressor together with their evaluation metrics.

## 2. Business Problem & Stakeholders

Modern educational institutions need to understand how GenAI tool usage
affects learning outcomes so they can craft data-informed policies —
supporting productive use while flagging harmful over-reliance. Key
stakeholders include students, instructors, academic advisors/counselors,
institutional administrators, and EdTech tool providers.

## 3. Hypotheses

| ID | Hypothesis | Outcome |
|----|------------|---------|
| H1 | AI dependency has a non-linear ("inflection point") relationship with final score | Supported — performance tends to decline beyond a dependency threshold |
| H2 | A higher AI ethics score moderates (reduces) the negative effect of AI dependency | Supported |
| H3 | Lifestyle variables (sleep, study hours, social media, attendance, participation, consistency) are significant performance predictors | Not supported at α = 0.05 (all ANOVA p-values > 0.05) |
| H4 | The purpose of AI usage differentially affects performance | Not supported (Chi-square p = 0.976) |

## 4. Data

- **Source:** `data/raw/ai_impact_student_performance_dataset.csv`
- **Records:** 8,000 students
- **Features:** 26 columns (24 predictors + 2 targets: `performance_category`,
  `final_score`)
- **Leakage controls:** `passed` (derived from `final_score`) and `student_id`
  (identifier) are dropped before modeling; all preprocessing (outlier
  bounds, encoders, scalers) is fit on the training split only.

## 5. Methodology

The reproducible pipeline (see `src/preprocessing.py`, `src/train.py`,
`src/pipeline.py`) follows this workflow:

```
Raw Data → Cleaning → Stratified Train/Test Split (80/20)
         → IQR Outlier Capping (train-fit) → Feature Engineering (10 new features)
         → Ordinal/One-Hot Encoding → Standard Scaling (train-fit)
         → SMOTE Balancing (classification, train-only) → Model Training
         → Champion Selection → Evaluation → Persisted Artifacts
```

Ten domain-driven features are engineered, including `ai_risk_index`,
`ethics_dependency_gap`, `study_efficiency`, `distraction_ratio`,
`engagement_score`, and `academic_momentum`.

### Models Compared

- **Classification:** Logistic Regression, Naive Bayes, KNN, SVM (RBF),
  Random Forest, Gradient Boosting — trained on SMOTE-balanced data.
- **Regression:** Ridge, Lasso, ElasticNet, Random Forest Regressor,
  Gradient Boosting Regressor.

The champion classifier is selected by **Macro-F1**; the champion regressor
is selected by **R²**. Both are persisted to `models/` via `joblib`.

## 6. Results

Results below are produced by running `python -m src.pipeline` against the
raw dataset (see `reports/classification_model_comparison.csv` and
`reports/regression_model_comparison.csv` for the full comparison, and
`reports/metrics.json` for the champion models' detailed evaluation).

### Classification (Champion selected by Macro-F1)

| Model | Accuracy | Macro-F1 | Recall (Low) | ROC-AUC (OvR) |
|---|---|---|---|---|
| SVM (RBF) | 0.828 | 0.796 | 0.852 | 0.937 |
| Gradient Boosting | 0.834 | 0.796 | 0.852 | 0.948 |
| Logistic Regression | 0.826 | 0.793 | 0.902 | 0.954 |
| Random Forest | 0.828 | 0.792 | 0.835 | 0.943 |
| Naive Bayes | 0.738 | 0.706 | 0.856 | 0.911 |
| KNN | 0.542 | 0.521 | 0.909 | 0.836 |

**Success criteria met:** Macro-F1 ≥ 0.75 ✅, Accuracy ≥ 0.75 ✅, ROC-AUC
(OvR) ≥ 0.80 ✅, Recall on the critical `Low` class ≥ 0.80 ✅.

### Regression (Champion selected by R²)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Ridge | 0.871 | 4.98 | 3.97 |
| Gradient Boosting Regressor | 0.863 | 5.12 | 4.08 |
| Random Forest Regressor | 0.857 | 5.24 | 4.20 |
| Lasso | 0.849 | 5.38 | 4.30 |
| ElasticNet | 0.799 | 6.20 | 4.95 |

**Success criteria met:** R² ≥ 0.75 ✅, RMSE within the 5–7 point target ✅.

*Note: exact figures may vary slightly across environments/library versions;
re-run `python -m src.pipeline` to regenerate up-to-date numbers.*

## 7. Ethical Considerations & Bias Auditing

The notebook (`notebooks/Gerric_Gomez_Capstone_Project_Pillar_5_Steps_1to5.ipynb`)
includes a dedicated fairness/bias audit across demographic slices (age
group, gender, grade level), confirming the model does not exhibit material
performance disparities that would warrant withholding deployment. Ethical
risks addressed include: over-reliance on AI-generated content as a proxy
for learning, privacy of student behavioral data, and the risk of a
prediction being used punitively rather than as an early-intervention
signal.

## 8. Conclusion & Recommendations

- Deploy the classification model as an early-warning system to flag
  students at risk of the `Low` performance category for timely academic
  support, rather than as a punitive tool.
- Track `ai_risk_index` and `ethics_dependency_gap` as monitoring metrics;
  our H1/H2 findings suggest institutions should promote *ethical,
  purposeful* AI use rather than banning AI tools outright.
- Because lifestyle variables (H3) and AI usage purpose (H4) were not
  statistically significant predictors in this dataset, further data
  collection (e.g., richer behavioral logs) may be needed before drawing
  causal conclusions about study habits.

## 9. Reproducing These Results

```bash
pip install -r requirements.txt
python -m src.pipeline
```

This regenerates the model artifacts in `models/` and the reports in
`reports/` from the raw CSV in `data/raw/`. See the project `README.md` for
the full repository structure and setup instructions.
