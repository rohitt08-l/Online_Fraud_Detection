# 💳 Online Fraud Detection System

## 🚀 Overview

This project is an **end-to-end Machine Learning system** for detecting fraudulent financial transactions.

It includes:

* 📊 Exploratory Data Analysis (EDA)
* 🤖 Machine Learning Model (Random Forest + SMOTE)
* 🌐 Flask API for real-time prediction
* 🎨 Streamlit UI for interactive usage

---

## 🎯 Objective

To classify transactions into:

* ✅ Safe
* ⚠️ Risk
* 🚨 Fraud

and provide a **risk percentage score**.

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Flask (API)
* Streamlit (UI)
* Matplotlib, Seaborn

---

## 📂 Project Structure

```id="0k2j8g"
ONLINE_FRAUD_DETECTION/
│
├── app/
│   ├── __init__.py
│   ├── app.py                  # Flask API
│   └── streamlit_app.py        # Streamlit UI
│
├── data/
│   └── PS_20174392719_1491204439457_log.csv
│
├── models/
│   ├── fraud_model.pkl
│   ├── encoder.pkl
│   └── scaler.pkl
│
├── notebook/                   # (Optional notebooks)
│
├── reports/                    # Generated EDA reports
│
├── src/
│   ├── __init__.py
│   ├── eda.py                  # EDA script
│   ├── model.py                # Model training script
│   └── preprocessing.py        # Preprocessing pipeline
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 🔹 1. Clone Repository

```bash id="rq9ydl"
git clone https://github.com/rohitt08-l/Online_Fraud_Detection.git
```

---

### 🔹 2. Create Virtual Environment

#### Using Conda:

```bash id="t3v9q6"
conda create -n fraud_env python=3.10
conda activate fraud_env
```

#### OR using venv:

```bash id="y8v5m1"
python -m venv venv
venv\Scripts\activate
```

---

### 🔹 3. Install Requirements

```bash id="k3mj8w"
pip install -r requirements.txt
```

---

## 📊 Dataset

Download dataset from drive:

👉 https://www.kaggle.com/datasets/ealaxi/paysim1

Place it in:

```id="z3mx4n"
data/PS_20174392719_1491204439457_log.csv
```

---

## 🤖 Model Training

Run:

```bash id="5c7n9p"
python src/model.py
```

### ✔ This will:

* Perform feature engineering
* Encode categorical data
* Handle imbalance using SMOTE
* Train Random Forest model
* Save model in `/models/`

---

## 🌐 Run Flask API

```bash id="p4c2y8"
python app/app.py
```

### 🔗 Base URL:

```id="2mvx9j"
http://127.0.0.1:5000/
```

---

### 🔹 API Endpoint

```id="9n8x3k"
POST /predict
```

### 📥 Sample Request

```json id="x8v2k1"
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 10000,
  "oldbalanceOrg": 50000,
  "newbalanceOrig": 40000,
  "oldbalanceDest": 0,
  "newbalanceDest": 10000,
  "isFlaggedFraud": 0
}
```

### 📤 Sample Response

```json id="s7d4m2"
{
  "prediction": "Risk",
  "risk": 45.23
}
```

---

## 🎨 Run Streamlit UI

```bash id="f2v6b9"
streamlit run app/streamlit_app.py
```

### ✔ Features:

* User-friendly interface
* Real-time fraud prediction
* Risk visualization

---

## 📊 Run EDA (Exploratory Data Analysis)

```bash id="m9x2k4"
python src/eda.py
```

### ✔ Output:

* Fraud distribution
* Transaction analysis
* Correlation heatmap
* Saved in `/reports/`

---


## 🚀 Future Improvements

* Deploy using Docker
* Add FastAPI version
* Improve model using XGBoost
* Add real-time streaming data
* Build dashboard (Power BI / Streamlit advanced)

---

## 👨‍💻 Author

**Rohit Patil**
AIML Engineer | ML Enthusiast

---

## ⭐ Conclusion

This project demonstrates a **complete ML pipeline**:
EDA → Preprocessing → Model Training → API → UI

It is suitable for:

* Internships
* ML Engineer roles
* Portfolio projects
