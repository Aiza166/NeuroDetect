"""Evaluate saved model artifacts on the held-out test set.

    python src/models/predict.py

Previously this script hardcoded UCI Parkinson's *voice* features
('MDVP:Fo(Hz)', 'Jitter:DDP', ...). Those 22 columns do not exist in this
dataset, which has 32 clinical/demographic features, so the script could only
ever raise on `scaler.transform`. Rewritten to do what the README always said
it did: run inference and report metrics.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib  # noqa: E402
from tensorflow.keras.models import load_model  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)

from src.data_pipeline.preprocess_data import ARMS, build_splits  # noqa: E402

THRESHOLD = 0.50

print("NOTE: synthetic data, pipeline demonstration only. Not a screening tool.")

for arm, exclude in ARMS.items():
    model_path = f"models/parkinsons_model_{arm}.keras"
    scaler_path = f"models/scaler_{arm}.pkl"
    if not os.path.exists(model_path):
        print(f"\n{arm}: {model_path} not found -- run src/models/train_model.py")
        continue

    # Rebuild the same split the model was trained on; the persisted scaler is
    # the authority on column order.
    _, X_test, _, y_test, _, _ = build_splits(exclude)
    features = joblib.load(scaler_path)["features"]
    X_test = X_test[features]

    prob = load_model(model_path).predict(X_test, verbose=0).ravel()
    pred = (prob >= THRESHOLD).astype(int)

    print(f"\n--- {arm} ({len(features)} features, n={len(y_test)}) ---")
    for name, val in [
        ("accuracy", accuracy_score(y_test, pred)),
        ("precision", precision_score(y_test, pred, zero_division=0)),
        ("recall", recall_score(y_test, pred, zero_division=0)),
        ("f1", f1_score(y_test, pred, zero_division=0)),
        ("roc_auc", roc_auc_score(y_test, prob)),
    ]:
        print(f"  {name:10s} {val:.4f}")
    (tn, fp), (fn, tp) = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
    print(f"  confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    if arm == "ALL_FEATURES_LEAKY":
        print("  ^ inflated by target leakage (UPDRS/MoCA/FunctionalAssessment).")

print(f"\nMajority-class baseline accuracy: 0.6200")
