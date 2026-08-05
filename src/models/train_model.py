"""Train and persist both model arms.

Two models, identical architecture and seed, differing only in feature set:

  ALL_FEATURES_LEAKY -- all 32 features, including UPDRS / MoCA /
                        FunctionalAssessment. Retained as a documented
                        baseline. NOT the model to use. See ANALYSIS.md.
  HONEST             -- those three clinical assessment scores excluded.

Nothing here is tuned. Architecture and hyperparameters are unchanged from the
original version of this file; the only corrections are the scaler order (see
preprocess_data.py) and a stratified split. If a metric gets worse, it gets
reported.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", "42")

import joblib  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402
from tensorflow.keras.layers import Dense, Dropout, Input  # noqa: E402
from tensorflow.keras.models import Sequential  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)

from src.data_pipeline.preprocess_data import ARMS, MODELS_DIR, build_splits  # noqa: E402

SEED = 42
THRESHOLD = 0.50


def build_model(n_features):
    """64 relu -> dropout .3 -> 32 relu -> dropout .2 -> 1 sigmoid.
    Identical for every arm. Do not tune per-arm -- the comparison depends on
    this being held constant."""
    tf.keras.utils.set_random_seed(SEED)
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def evaluate(model, X_test, y_test):
    prob = model.predict(X_test, verbose=0).ravel()
    pred = (prob >= THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    }


def report(name, m):
    print(f"\n--- {name} ---")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k:10s} {m[k]:.4f}")
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    print("  confusion matrix (rows=actual 0/1, cols=pred 0/1):")
    print(f"      TN={tn:4d}  FP={fp:4d}")
    print(f"      FN={fn:4d}  TP={tp:4d}")


def train_arm(arm, exclude, legacy_leaky_scaler=False, persist=True):
    X_train, X_test, y_train, y_test, scaler, feats = build_splits(
        exclude, legacy_leaky_scaler=legacy_leaky_scaler)

    model = build_model(len(feats))
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5,
                                       restore_best_weights=True)],
              verbose=0)

    metrics = evaluate(model, X_test, y_test)
    metrics.update({"arm": arm, "n_features": len(feats), "features": feats})

    if persist:
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"parkinsons_model_{arm}.keras")
        scaler_path = os.path.join(MODELS_DIR, f"scaler_{arm}.pkl")
        model.save(model_path)
        joblib.dump({"scaler": scaler, "features": feats}, scaler_path)
        metrics["model_path"] = model_path
        metrics["scaler_path"] = scaler_path
        print(f"  saved -> {model_path}")
        print(f"  saved -> {scaler_path}")
    return metrics


def main():
    print(f"TensorFlow {tf.__version__}  seed={SEED}  threshold={THRESHOLD}")

    results = {}
    for arm, exclude in ARMS.items():
        results[arm] = train_arm(arm, exclude)
        report(arm, results[arm])

    # How much was the preprocessing bug actually worth? Train the leaky-scaler
    # order once, for measurement only. These models are NOT persisted.
    print("\n" + "=" * 70)
    print("SCALER-ORDER EFFECT (measurement only, not persisted)")
    print("=" * 70)
    legacy = {}
    for arm, exclude in ARMS.items():
        legacy[arm] = train_arm(arm, exclude, legacy_leaky_scaler=True,
                                persist=False)
        report(f"{arm} [pre-split scaler, the old bug]", legacy[arm])

    print("\n" + "=" * 70)
    print("DELTA: correct scaler order minus old leaky order")
    print("=" * 70)
    for arm in ARMS:
        print(f"\n  {arm}")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            a, b = legacy[arm][k], results[arm][k]
            print(f"    {k:10s} leaky {a:.4f} -> correct {b:.4f}  ({b - a:+.4f})")

    out = os.path.join(MODELS_DIR, "metrics.json")
    with open(out, "w") as f:
        json.dump({"tensorflow": tf.__version__, "seed": SEED,
                   "threshold": THRESHOLD, "correct_scaler_order": results,
                   "legacy_leaky_scaler_order": legacy}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
