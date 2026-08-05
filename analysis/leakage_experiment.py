"""Target-leakage experiment: identical model, two feature sets.

Model A: all 32 features.
Model B: all features EXCEPT UPDRS, MoCA, FunctionalAssessment.

Everything else is held constant: same seed, same split indices, same
architecture, same hyperparameters, same 0.50 decision threshold.
Nothing is tuned in either arm -- the point is an honest comparison, not a
good score.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

SEED = 42
RAW = "data/raw/parkinsons.csv"
TARGET = "Diagnosis"
ID_COLS = ["PatientID", "DoctorInCharge"]
LEAKY = ["UPDRS", "MoCA", "FunctionalAssessment"]
MOTOR = ["Tremor", "Rigidity", "Bradykinesia", "PosturalInstability",
         "SpeechProblems"]

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

df = pd.read_csv(RAW)
y = df[TARGET].values
X_all = df.drop(columns=[c for c in ID_COLS if c in df.columns] + [TARGET])

# One split, reused by both arms, so the two models see identical patients.
idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)

BACKEND = None
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    BACKEND = "keras"
except Exception:
    from sklearn.neural_network import MLPClassifier
    BACKEND = "sklearn-mlp"


def build_and_fit(Xtr, ytr, n_features):
    """Same architecture in both arms: 64 relu -> drop .3 -> 32 relu ->
    drop .2 -> 1 sigmoid, adam, binary crossentropy, 50 epochs max,
    batch 32, early stopping on 20% validation split."""
    if BACKEND == "keras":
        tf.keras.utils.set_random_seed(SEED)
        m = Sequential([
            Input(shape=(n_features,)),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ])
        m.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
        m.fit(Xtr, ytr, validation_split=0.2, epochs=50, batch_size=32,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5,
                                       restore_best_weights=True)],
              verbose=0)
        return lambda Z: m.predict(Z, verbose=0).ravel()
    m = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                      solver="adam", batch_size=32, max_iter=50,
                      early_stopping=True, validation_fraction=0.2,
                      n_iter_no_change=5, random_state=SEED)
    m.fit(Xtr, ytr)
    return lambda Z: m.predict_proba(Z)[:, 1]


def run(name, cols):
    Xs = X_all[cols].values
    # Scaler fit on TRAIN ONLY -- avoids the leakage in the repo's
    # preprocess_data.py, which scales the whole frame before splitting.
    sc = StandardScaler().fit(Xs[idx_train])
    Xtr, Xte = sc.transform(Xs[idx_train]), sc.transform(Xs[idx_test])
    ytr, yte = y[idx_train], y[idx_test]

    predict = build_and_fit(Xtr, ytr, len(cols))
    prob = predict(Xte)
    pred = (prob >= 0.50).astype(int)

    cm = confusion_matrix(yte, pred, labels=[0, 1])
    r = {
        "model": name,
        "n_features": len(cols),
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "roc_auc": roc_auc_score(yte, prob),
        "confusion_matrix": cm.tolist(),
    }
    print(f"\n--- {name} ({len(cols)} features) ---")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k:10s} {r[k]:.4f}")
    tn, fp, fn, tp = cm.ravel()
    print(f"  confusion matrix (rows=actual 0/1, cols=pred 0/1):")
    print(f"      TN={tn:4d}  FP={fp:4d}")
    print(f"      FN={fn:4d}  TP={tp:4d}")
    return r


print(f"backend: {BACKEND}")
print(f"train n={len(idx_train)}  test n={len(idx_test)}  "
      f"test positive rate={y[idx_test].mean():.4f}")
print(f"majority-class baseline accuracy on test: "
      f"{max(y[idx_test].mean(), 1 - y[idx_test].mean()):.4f}")

results = [
    run("A: all features", list(X_all.columns)),
    run("B: clinical scores removed",
        [c for c in X_all.columns if c not in LEAKY]),
    # Supplementary arm: the motor-symptom flags are also observations of
    # Parkinson's, not antecedent risk factors, so they are arguably leaky too.
    run("C: scores + motor signs removed (supplementary)",
        [c for c in X_all.columns if c not in LEAKY + MOTOR]),
]

print("\n" + "=" * 70)
print("DELTA (B - A)")
print("=" * 70)
for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    a, b = results[0][k], results[1][k]
    print(f"  {k:10s} {a:.4f} -> {b:.4f}   ({b - a:+.4f})")

os.makedirs("analysis", exist_ok=True)
with open("analysis/results.json", "w") as f:
    json.dump({"backend": BACKEND, "seed": SEED, "results": results}, f, indent=2)
print("\nwrote analysis/results.json")
