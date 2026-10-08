import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/SohelRaja/Customer-Churn-Analysis/"
    "master/Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df = df.dropna(subset=["TotalCharges"]).copy()
    return df

@st.cache_resource
def train_model():
    df = load_data()

    encoders = {}

    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder

    churn_encoder = LabelEncoder()
    df["Churn"] = churn_encoder.fit_transform(df["Churn"])

    X = df[FEATURE_COLUMNS]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    smote = SMOTE(random_state=42)

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=1,
        random_state=42
    )

    model.fit(X_train_smote, y_train_smote)

    return model, encoders, churn_encoder

st.title("Customer Churn Prediction")

st.caption(
    "Optimized Random Forest + SMOTE | Telco Customer Churn"
)

with st.spinner("Loading the trained model..."):
    model, encoders, churn_encoder = train_model()

st.markdown(
    """
    Predict whether a telecom customer is likely to churn based on
    their account and service information.
    """
)

st.divider()

with st.form("churn_form"):
    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

    with col2:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=0.01
        )

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0,
            step=0.01
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    )

if submitted:
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([input_data])

    for column, encoder in encoders.items():
        input_df[column] = encoder.transform(input_df[column])

    input_df = input_df[FEATURE_COLUMNS]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    churn_class = churn_encoder.transform(["Yes"])[0]
    churn_probability = probabilities[churn_class]

    st.divider()
    st.subheader("Prediction")

    if prediction == churn_class:
        st.error("⚠️ Customer is likely to CHURN")

        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

        st.warning(
            "This customer shows characteristics associated "
            "with a higher likelihood of leaving the service."
        )
    else:
        st.success("✅ Customer is likely to STAY")

        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

        st.info(
            "This customer is currently predicted to remain "
            "with the service."
        )

    st.progress(float(churn_probability))

with st.expander("About this model"):
    st.write(
        "The model uses categorical Label Encoding, an 80/20 "
        "train-test split, SMOTE, and an optimized Random Forest "
        "selected through GridSearchCV with 5-fold Stratified "
        "Cross-Validation."
    )

    st.write(
        "Selected Random Forest parameters: "
        "n_estimators=400, max_depth=8, min_samples_leaf=1."
    )

    st.write(
        "F1 score is used as the primary model selection metric "
        "because identifying customers likely to churn is the "
        "main objective."
    )

