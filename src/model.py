import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from preprocessing import load_data, feature_engineering, preprocess_data, split_features, scale_data


def main():
    path = "data/PS_20174392719_1491204439457_log.csv"

    print("Loading data...")
    df = load_data(path)

    print("Feature engineering...")
    df = feature_engineering(df)

    print("Preprocessing...")
    df, encoder = preprocess_data(df)

    print("Splitting features...")
    X, y = split_features(df)

    print("Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Applying SMOTE on training data...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("Scaling...")
    X_train, scaler = scale_data(X_train)
    X_test = scaler.transform(X_test)

    print("Training model...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Saving model...")

    # Create model folder
    import os
    os.makedirs("models", exist_ok=True)

    pickle.dump(model, open("models/fraud_model.pkl", "wb"))
    pickle.dump(encoder, open("models/encoder.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))

    print("Model saved successfully")


if __name__ == "__main__":
    main()