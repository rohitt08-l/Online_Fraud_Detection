from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

from src.preprocessing import feature_engineering

app = Flask(__name__)
@app.route("/")
def home():
    return "Fraud Detection API is running 🚀"
# -------------------------------
# Load Model Files
# -------------------------------
model = pickle.load(open("models/fraud_model.pkl", "rb"))
encoder = pickle.load(open("models/encoder.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))


# -------------------------------
# Preprocess Input
# -------------------------------
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Feature engineering
    df = feature_engineering(df)

    # Encode
    df["type"] = encoder.transform(df["type"])

    # Drop unused
    df = df.drop(["nameOrig", "nameDest"], axis=1, errors="ignore")

    # Scale
    df = scaler.transform(df)

    return df


# -------------------------------
# API Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        processed = preprocess_input(data)

        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

        risk = round(probability * 100, 2)

        if risk >= 60:
            result = "Fraud"
        elif risk >= 30:
            result = "Risk"
        else:
            result = "Safe"

        return jsonify({
            "prediction": result,
            "risk": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)