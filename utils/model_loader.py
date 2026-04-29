import os
import subprocess

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model not found. Training model now...")

        subprocess.run(["python", "src/model.py"], check=True)

        print("Model training completed!")
    else:
        print("Model already exists.")