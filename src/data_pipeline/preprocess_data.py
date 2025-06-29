# src/data_pipeline/preprocess_data.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

RAW_PATH = "data/raw/parkinsons.csv"
PROCESSED_DIR = "data/processed"
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "parkinsons_clean.csv")

def preprocess():
    print("📥 Loading data...")
    df = pd.read_csv(RAW_PATH)
    print("Raw data shape:", df.shape)

    # View column names for debugging
    print("Column names in dataset:")
    print(df.columns.tolist())

    # Drop ID/name/doctor columns if they exist
    drop_cols = ["PatientID", "DoctorInCharge"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Split features & target (Diagnosis = target label)
    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Combine back into a DataFrame
    df_clean = pd.DataFrame(X_scaled, columns=X.columns)
    df_clean["Diagnosis"] = y.reset_index(drop=True)

    # Save processed CSV
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Cleaned data saved to: {PROCESSED_PATH}")
    print("Processed shape:", df_clean.shape)

if __name__ == "__main__":
    preprocess()
