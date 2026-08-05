# Model artifacts

**Use `parkinsons_model_HONEST.keras`.** It is the only model here that is not
reading the answer off the exam paper.

Neither model is a screening tool. Both are trained on synthetic data — see
[`../ANALYSIS.md`](../ANALYSIS.md) §5.

## Files

| File | What it is |
|---|---|
| `parkinsons_model_HONEST.keras` | **The model to use.** 29 features; UPDRS, MoCA and FunctionalAssessment excluded. |
| `scaler_HONEST.pkl` | `StandardScaler` for the above, **fitted on the training fold only**. Pickled as `{"scaler": ..., "features": [...]}` so column order can be validated at inference. |
| `parkinsons_model_ALL_FEATURES_LEAKY.keras` | **Documented baseline. Do not use.** All 32 features. Its apparent accuracy comes from target leakage. |
| `scaler_ALL_FEATURES_LEAKY.pkl` | Scaler for the baseline, same train-only fit and same dict structure. |
| `metrics.json` | Full metrics for both arms, both scaler orders, machine-readable. |

Regenerate everything with:

```bash
python src/models/train_model.py
```

## Why the baseline is named `_LEAKY`

UPDRS and MoCA are clinical assessment instruments *for Parkinson's disease*.
A clinician records those scores because they are assessing a patient they have
already diagnosed. Using them as input features means the model is not
predicting a diagnosis — it is restating one. The name is deliberately
unpleasant so nobody loads it by accident.

## Metrics

Held-out test set, n = 421, 62.00% positive.
**Majority-class baseline accuracy: 0.6200** — read every accuracy against this,
not against 0.50.

| Model | Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| `HONEST` | 29 | 0.6912 | 0.7148 | 0.8352 | 0.7703 | 0.7368 |
| `ALL_FEATURES_LEAKY` | 32 | 0.8504 | 0.8694 | 0.8927 | 0.8809 | 0.9121 |

Confusion matrices (rows = actual 0/1, columns = predicted 0/1):

```text
HONEST                          ALL_FEATURES_LEAKY
         pred 0   pred 1                 pred 0   pred 1
act 0        73       87        act 0       125       35
act 1        43      218        act 1        28      233
```

The honest model reaches 45.6% specificity (73/160) — it is wrong more often
than right on healthy patients — and its accuracy of 0.6912 sits about 7 points
above the 0.6200 majority baseline. That is the real amount of signal in this
dataset once the clinical assessment scores are removed.

## Effect of fixing the preprocessing leak

The previous `preprocess_data.py` called `scaler.fit_transform(X)` *before*
splitting, leaking test-set means and standard deviations into training. That is
fixed. The measured cost of the bug, same seed and architecture:

| Model | Metric | Old (pre-split scaler) | Fixed (train-only) | Δ |
|---|---|---|---|---|
| `ALL_FEATURES_LEAKY` | accuracy | 0.8527 | 0.8504 | **−0.0024** |
| | precision | 0.8727 | 0.8694 | −0.0033 |
| | recall | 0.8927 | 0.8927 | ±0.0000 |
| | F1 | 0.8826 | 0.8809 | −0.0017 |
| | ROC-AUC | 0.9126 | 0.9121 | −0.0005 |
| `HONEST` | accuracy | 0.6888 | 0.6912 | **+0.0024** |
| | precision | 0.7138 | 0.7148 | +0.0009 |
| | recall | 0.8314 | 0.8352 | +0.0038 |
| | F1 | 0.7681 | 0.7703 | +0.0022 |
| | ROC-AUC | 0.7383 | 0.7368 | −0.0015 |

**The fix barely moved anything — no metric shifted by more than 0.0033, and the
direction is not even consistent.** That is the expected result for
standardisation on 2,105 rows: the train fold's mean and standard deviation are
already very close to the full dataset's, so the leaked information was almost
worthless. The bug was a correctness problem, not a scoring problem. Reported
here rather than quietly dropped, because a fix that does nothing is still worth
knowing about.

These numbers also match the earlier `analysis/leakage_experiment.py` run
exactly (0.8504 / 0.9121 and 0.6912 / 0.7368), because that script already fitted
its scaler on the training fold only. The old committed `.h5` artifact was the
only thing carrying the bug, and it has been deleted.

## Format note

Artifacts are saved as `.keras` (the native Keras 3 format), not the legacy
`.h5` used previously. Loading is unchanged: `load_model(path)`.

## Regeneration provenance

TensorFlow 2.21.0, seed 42, decision threshold 0.50, stratified 80/20 split.
Architecture: `Dense(64, relu)` → `Dropout(0.3)` → `Dense(32, relu)` →
`Dropout(0.2)` → `Dense(1, sigmoid)`, Adam, binary cross-entropy, max 50 epochs,
batch 32, early stopping on `val_loss` with patience 5. Identical across both
arms; nothing was tuned per-arm.
