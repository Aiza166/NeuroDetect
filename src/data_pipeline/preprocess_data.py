# src/data_pipeline/preprocess_data.py
"""Load the vendored raw CSV, split, and standardise WITHOUT leaking test
statistics into the training fold.

This module is the single source of truth for the train/test split so that
every downstream script (train_model.py, predict.py, analysis/) sees exactly
the same partition.
"""

import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
RAW_PATH = "data/raw/parkinsons.csv"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

TARGET = "Diagnosis"
ID_COLS = ["PatientID", "DoctorInCharge"]

# UPDRS and MoCA are clinical assessment instruments FOR Parkinson's, and
# FunctionalAssessment is scored alongside them. A clinician's score is
# recorded downstream of the diagnosis, so these three columns partly encode
# the label. This is a property of the dataset, not a bug to be fixed -- see
# ANALYSIS.md. We keep both feature sets available so the effect stays visible.
LEAKY_COLS = ["UPDRS", "MoCA", "FunctionalAssessment"]

ARMS = {
    "ALL_FEATURES_LEAKY": [],
    "HONEST": LEAKY_COLS,
}


def load_raw():
    """Raw features and target, with identifier columns dropped."""
    df = pd.read_csv(RAW_PATH)
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def build_splits(exclude=(), legacy_leaky_scaler=False):
    """Split first, then scale. Returns (X_train, X_test, y_train, y_test,
    scaler, feature_names) with X_* already standardised.

    WHY THE ORDER MATTERS -- DO NOT REORDER THIS:
    `scaler.fit()` computes the mean and standard deviation of each column.
    If it is fitted on the full frame BEFORE the split (i.e.
    `fit_transform(X)` then split), those statistics are computed partly from
    test rows and are then baked into the training features. The model has
    thereby seen a summary of the test set during training, and the reported
    test score is no longer a clean held-out estimate. That is a preprocessing
    leak, and it is what the previous version of this file did.
    Correct order: split -> fit on TRAIN ONLY -> transform train and test with
    those same train-derived statistics.

    `legacy_leaky_scaler=True` deliberately reproduces the old, leaky order.
    It exists only so train_model.py can measure how much the bug was worth.
    Never use it to produce a shipped model.
    """
    X, y = load_raw()
    X = X.drop(columns=[c for c in exclude if c in X.columns])
    feature_names = X.columns.tolist()

    if legacy_leaky_scaler:
        # WRONG ON PURPOSE -- for measurement only. See docstring.
        leaked = StandardScaler().fit(X)
        X_scaled = pd.DataFrame(leaked.transform(X), columns=feature_names)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=SEED, stratify=y)
        return X_train, X_test, y_train, y_test, leaked, feature_names

    # Correct order: split BEFORE the scaler ever sees the data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    scaler = StandardScaler().fit(X_train)          # TRAIN ONLY
    X_train = pd.DataFrame(scaler.transform(X_train), columns=feature_names)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    return X_train, X_test, y_train, y_test, scaler, feature_names


def preprocess():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_raw, y_raw = load_raw()
    print(f"Raw data shape (ids dropped): {X_raw.shape[0]} x {X_raw.shape[1] + 1}")
    print(f"Class balance: {dict(y_raw.value_counts().sort_index())}")

    for arm, exclude in ARMS.items():
        X_train, X_test, y_train, y_test, scaler, feats = build_splits(exclude)

        train = X_train.copy()
        train[TARGET] = y_train.reset_index(drop=True)
        test = X_test.copy()
        test[TARGET] = y_test.reset_index(drop=True)

        train_path = os.path.join(PROCESSED_DIR, f"train_{arm}.csv")
        test_path = os.path.join(PROCESSED_DIR, f"test_{arm}.csv")
        scaler_path = os.path.join(MODELS_DIR, f"scaler_{arm}.pkl")

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        joblib.dump({"scaler": scaler, "features": feats}, scaler_path)

        print(f"\n[{arm}] {len(feats)} features"
              + (f" (excluded {exclude})" if exclude else ""))
        print(f"  train {train.shape} -> {train_path}")
        print(f"  test  {test.shape} -> {test_path}")
        print(f"  scaler (fitted on train only) -> {scaler_path}")


if __name__ == "__main__":
    preprocess()
