import sys
import os

# ✅ Fix import path (VERY IMPORTANT)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
import pandas as pd
import pickle

from src.preprocessing import feature_engineering

app = Flask(__name__)

@app.route("/")
def home():
    return "Fraud Detection API is running 🚀"


# -------------------------------
# Load Model Files (Robust Path)
# -------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

model_path = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
encoder_path = os.path.join(BASE_DIR, "models", "encoder.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
encoder = pickle.load(open(encoder_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))


# -------------------------------
# Preprocess Input
# -------------------------------
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Feature engineering
    df = feature_engineering(df)

    # Encode
    df["type"] = encoder.transform(df["type"])

    # Drop unused columns
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

        if not data:
            return jsonify({"error": "No input data provided"}), 400

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
        return jsonify({"error": str(e)}), 500


# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)