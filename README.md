<<<<<<< HEAD
# Customer Churn Prediction — Streamlit

A simple Streamlit interface for the Customer Churn Prediction ML project.

## Model

The app follows the original notebook workflow:

- Drops `customerID`
- Converts `TotalCharges` to numeric
- Label-encodes categorical features
- Splits data using `test_size=0.2, random_state=42`
- Applies SMOTE with `random_state=42` to the training set
- Trains `RandomForestClassifier(random_state=42)`

The model is trained once per Streamlit instance and cached with `st.cache_resource`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push `app.py`, `requirements.txt`, and `.streamlit/config.toml` to GitHub, then deploy the repository from Streamlit Community Cloud.
=======

>>>>>>> 34fc16dd9d46958c40af06ab3356274f795a31c0
