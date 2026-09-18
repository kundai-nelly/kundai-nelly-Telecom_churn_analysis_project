from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Path is relative to this file so the app runs from any clone location
MODEL_PATH = Path(__file__).resolve().parent / "churn_production_model.pkl"

# Loads the model
model_artifact = joblib.load(MODEL_PATH)
pipeline = model_artifact["pipeline"]
threshold = model_artifact["threshold"]

st.title("📉 Customer Churn Prediction")
st.write("Enter the customer's details below:")

# Inputs required fields
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)

if st.button("Predict"):
    data = {
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
    }
    df = pd.DataFrame(data)
    proba = pipeline.predict_proba(df)[0, 1]
    pred = int(proba >= threshold)
    decision = "⚠️ LIKELY TO CHURN" if pred == 1 else "✅ LIKELY TO STAY"
    st.subheader(f"Prediction: {decision}")
    st.write(f"Probability of Churn: {proba:.2%}")
