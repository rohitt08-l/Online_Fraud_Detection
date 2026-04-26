import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# -------------------------------
# CONFIG (VERY IMPORTANT)
# -------------------------------
SAMPLE_SIZE = 100000   # change to None for full dataset
RANDOM_STATE = 42

# -------------------------------
# Load Dataset
# -------------------------------
def load_data(path):
    try:
        print("Loading dataset...")
        df = pd.read_csv(path)

        print(f"Full dataset loaded: {df.shape}")

        # Sampling for speed
        if SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
            print(f"Sampled dataset used: {df.shape}")

        return df

    except Exception as e:
        print("Error loading dataset:", e)
        return None


# -------------------------------
# Basic Info
# -------------------------------
def basic_info(df):
    print("\n🔹 Shape of dataset:", df.shape)
    print("\n🔹 Dataset Info:")
    print(df.info())
    print("\n🔹 Statistical Summary:")
    print(df.describe())


# -------------------------------
# Missing Values
# -------------------------------
def check_missing(df):
    print("\n🔹 Missing Values:")
    print(df.isnull().sum())


# -------------------------------
# Fraud Distribution
# -------------------------------
def fraud_distribution(df):
    print("\n🔹 Fraud Distribution:")
    print(df['isFraud'].value_counts(normalize=True))

    sns.countplot(x='isFraud', data=df)
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.savefig("reports/fraud_distribution.png")
    plt.close()


# -------------------------------
# Transaction Type Analysis
# -------------------------------
def transaction_type_analysis(df):
    print("\n🔹 Transaction Types:")
    print(df['type'].value_counts())

    sns.countplot(x='type', data=df)
    plt.title("Transaction Types")
    plt.xticks(rotation=45)
    plt.savefig("reports/transaction_types.png")
    plt.close()


# -------------------------------
# Fraud by Transaction Type
# -------------------------------
def fraud_by_type(df):
    fraud_type = df.groupby('type')['isFraud'].sum()

    print("\n🔹 Fraud by Transaction Type:")
    print(fraud_type)

    fraud_type.plot(kind='bar', title="Fraud by Transaction Type")
    plt.savefig("reports/fraud_by_type.png")
    plt.close()


# -------------------------------
# Amount Distribution (Optimized)
# -------------------------------
def amount_distribution(df):
    plt.figure(figsize=(10, 5))

    # sample again for heavy plotting
    sample_amount = df['amount'].sample(min(50000, len(df)))

    sns.histplot(sample_amount, bins=50)
    plt.title("Transaction Amount Distribution")
    plt.savefig("reports/amount_distribution.png")
    plt.close()


# -------------------------------
# Correlation Heatmap (Optimized)
# -------------------------------
def correlation_heatmap(df):
    print("\nGenerating correlation heatmap (optimized)...")

    numeric_df = df.select_dtypes(include=['number'])

    # reduce size for correlation
    sample_df = numeric_df.sample(min(50000, len(numeric_df)))

    plt.figure(figsize=(10, 6))
    sns.heatmap(sample_df.corr(), cmap='coolwarm')
    plt.title("Feature Correlation")
    plt.savefig("reports/correlation_heatmap.png")
    plt.close()


# -------------------------------
# Main Function
# -------------------------------
def main():
    data_path = os.path.join("data", "PS_20174392719_1491204439457_log.csv")

    df = load_data(data_path)

    if df is not None:
        os.makedirs("reports", exist_ok=True)

        basic_info(df)
        check_missing(df)
        fraud_distribution(df)
        transaction_type_analysis(df)
        fraud_by_type(df)
        amount_distribution(df)
        correlation_heatmap(df)

        print("\nEDA Completed Successfully")

        # Important insights (auto print)
        fraud_ratio = df['isFraud'].mean() * 100
        print(f"\nFraud Percentage: {fraud_ratio:.4f}%")

    else:
        print("Failed to run EDA")


if __name__ == "__main__":
    main()