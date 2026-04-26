import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)

    # Use sampling for speed
    df = df.sample(n=200000, random_state=42)

    return df


def feature_engineering(df):
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    return df


def preprocess_data(df):
    # Drop useless columns
    df = df.drop(["nameOrig", "nameDest"], axis=1)

    # Encode categorical
    encoder = LabelEncoder()
    df["type"] = encoder.fit_transform(df["type"])

    return df, encoder


def split_features(df):
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    return X, y


def handle_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    return X_res, y_res


def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def train_test(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    path = "data/PS_20174392719_1491204439457_log.csv"

    print("loading data...")
    df = load_data(path)

    print("Feature engineering...")
    df = feature_engineering(df)

    print("Preprocessing...")
    df, encoder = preprocess_data(df)

    print("Splitting features...")
    X, y = split_features(df)

    print("Handling imbalance (SMOTE)...")
    X, y = handle_imbalance(X, y)

    print("Scaling...")
    X, scaler = scale_data(X)

    print("Train-test split...")
    X_train, X_test, y_train, y_test = train_test(X, y)

    print("Preprocessing completed")

    return X_train, X_test, y_train, y_test, encoder, scaler


if __name__ == "__main__":
    main()