import streamlit as st
import pandas as pd
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import feature_engineering

# -------------------------------
# Load Model Files
# -------------------------------
model = pickle.load(open("models/fraud_model.pkl", "rb"))
encoder = pickle.load(open("models/encoder.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Online Fraud Detection System")
st.write("Enter transaction details to check if it's fraudulent.")

# -------------------------------
# User Input
# -------------------------------
step = st.number_input("Step", min_value=1, value=1)

txn_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
)

amount = st.number_input("Amount", min_value=0.0, value=1000.0)

oldbalanceOrg = st.number_input("Old Sender Balance", value=0.0)
newbalanceOrig = st.number_input("New Sender Balance", value=0.0)

oldbalanceDest = st.number_input("Old Receiver Balance", value=0.0)
newbalanceDest = st.number_input("New Receiver Balance", value=0.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Fraud"):

    data = {
        "step": step,
        "type": txn_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": 0
    }

    df = pd.DataFrame([data])

    # Feature engineering
    df = feature_engineering(df)

    # Encode
    df["type"] = encoder.transform(df["type"])

    # Drop unused
    df = df.drop(["nameOrig", "nameDest"], axis=1, errors="ignore")

    # Scale
    df = scaler.transform(df)

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    risk = round(probability * 100, 2)

    # -------------------------------
    # Output
    # -------------------------------
    st.subheader("Result")

    if risk >= 60:
        st.error(f"Fraud Detected! Risk: {risk}%")
    elif risk >= 30:
        st.warning(f"Suspicious Transaction! Risk: {risk}%")
    else:
        st.success(f"Safe Transaction. Risk: {risk}%")