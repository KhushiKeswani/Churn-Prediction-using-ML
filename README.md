# Churn Prediction using Machine Learning

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Notebook](https://img.shields.io/badge/notebook-Jupyter-orange.svg)](https://github.com/KhushiKeswani/Churn-Prediction-using-ML)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)]

A polished, end-to-end churn prediction project that demonstrates how to turn raw customer data into business value using machine learning. The repository contains data-loading patterns, EDA, feature engineering, model training and evaluation, interpretability analysis (SHAP), and guidance for deploying a production-ready inference service.

Why this project

- Predicting churn helps retain high-value customers and increases lifetime value.
- The pipeline shows reproducible experiments and business-focused evaluation (Precision@K, AUC, cost-sensitive analysis).
- Interpretability artifacts help stakeholders understand why the model makes decisions.

Highlights

- End-to-end pipeline: data ingestion → EDA → features → modeling → evaluation → export
- Multiple model baselines and a tuned gradient-boosted model (XGBoost/LightGBM)
- Interpretability with SHAP and feature importance
- Exportable inference wrapper (joblib/ONNX) and deployment notes

Repository layout

- notebooks/            — Jupyter notebooks (EDA, experiments, explainability)
- src/                  — Reusable modules: data, features, models, inference
- data/                 — (Optional) data samples or pointers to data stores
- models/               — Serialized model artifacts and vectorizers
- reports/              — Evaluation reports, model card, figures
- requirements.txt      — Python dependencies
- README.md             — Project overview (this file)

Project flow (fast read)

1. Data ingestion: load customer-level snapshots and transaction logs.
2. EDA: analyze churn rate, class imbalance, missingness, and correlations.
3. Preprocessing: impute, encode, and scale while preventing leakage.
4. Feature engineering: tenure, recency/frequency, aggregated usage, billing signals.
5. Model training: baseline (Logistic Regression), tree ensembles (RF, XGBoost/LightGBM).
6. Validation: stratified K-fold CV, calibration, and threshold tuning for Precision@K.
7. Explainability: SHAP for global and local explanations.
8. Export: model + preprocessing pipeline, inference wrapper, and simple API.

Data expectations

The pipeline expects a tabular dataset where each row is a customer snapshot and the target column is `churn` (0 = retained, 1 = churned). Common columns:

- customer_id
- signup_date
- last_active_date
- tenure
- monthly_spend
- num_logins
- support_tickets
- contract_type
- churn

If your dataset is private, place a sample CSV in `data/` and update `src/data_loader.py` accordingly.

Feature engineering examples

- Time-based: tenure_days, days_since_last_activity
- Aggregates: avg_sessions_per_month, sessions_std, month_over_month_change
- Billing: overdue_count, avg_payment_amount
- Engagement: feature_adoption_flags, support_ticket_rate

Models & experiments

Recommended pipelines (implemented in notebooks and src):

- Baseline: Logistic Regression + simple imputation and one-hot encoding
- Tree-based: Random Forest for robustness
- Gradient-boosted: XGBoost or LightGBM for highest tabular performance

Best practices included

- Stratified K-Fold cross-validation to keep class ratios consistent
- Hyperparameter tuning (Optuna/Grid/RandomizedSearch)
- Probability calibration where downstream decisions depend on accurate probabilities
- Threshold tuning for business goals (Precision@K, minimize expected retention cost)

Evaluation: sample results (hold-out test set)

| Model | AUC  | Accuracy | Precision | Recall | F1   |
|-------|------|----------|-----------|--------|------|
| XGBoost | 0.88 | 0.86     | 0.78      | 0.72   | 0.75 |
| Random Forest | 0.85 | 0.84     | 0.75      | 0.68   | 0.71 |
| Logistic Regression | 0.80 | 0.80     | 0.70      | 0.60   | 0.65 |

Notes on metrics

- AUC evaluates ranking quality; Precision@K and recall at business thresholds are more actionable.
- Convert model outputs into expected business impact using per-customer lifetime value and retention cost.

Interpretability

- SHAP summary plots and dependence plots are provided in `notebooks/`.
- Feature importance and permutation importance help prioritize product fixes.
- Model card in `reports/` describes intended use, limitations, and performance by segment.

Quickstart

1. Clone the repo

   git clone https://github.com/KhushiKeswani/Churn-Prediction-using-ML.git
   cd Churn-Prediction-using-ML

2. Create virtual environment and install

   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt

3. Reproduce experiments

- Run the notebooks in order (start with `notebooks/01_data_exploration.ipynb`).
- Or run the training script:

   python src/train.py --config configs/train_config.yaml

Deployment guidance

- Export the trained pipeline with `joblib.dump(pipeline, 'models/pipeline.joblib')`.
- Provide `src/inference.py` for loading pipeline and scoring a batch or single customer.
- Wrap `src/inference.py` with FastAPI for a lightweight prediction API.

Reproducibility

- Pin dependencies in `requirements.txt` and set seeds in training code.
- Use experiment tracking (MLflow) to log parameters, metrics, and artifacts.

Contributing

Contributions are welcome. Suggested areas:
- Add unit tests for preprocessing and feature modules
- Improve CI to run linting and core tests
- Add more evaluation by customer segment and fairness checks

License

This project is available under the MIT License. See LICENSE for details.

Acknowledgements

This repo follows industry-standard patterns for ML projects. If you adapt or include third-party datasets/code, please provide attribution in the notebooks or reports.
