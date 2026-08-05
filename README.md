# 🧠 NeuroDetect – Parkinson’s Disease Detection

An end-to-end machine learning pipeline — data preparation, training,
evaluation, and inference — demonstrated on a Parkinson's disease
classification task.

> ## ⚠️ Disclaimer
>
> **This is a pipeline and methodology demonstration on synthetic data. It is
> not a diagnostic tool, not a screening tool, and not for clinical use.**
>
> The dataset is computationally generated, not collected from real patients
> ([evidence](ANALYSIS.md)). Most of the apparent predictive accuracy comes
> from target leakage: the clinical assessment scores used as inputs already
> encode the diagnosis. Both issues are documented rather than hidden — that
> documentation is the point of this repository.
>
> No output of this project should inform any medical decision.


## 📁 Dataset

- **Target**: `Diagnosis` (binary: 0 or 1) — 2,105 rows × 35 columns, no missing values
- **Class balance**: 1,304 positive (61.95%) / 801 negative (38.05%) → majority-class baseline accuracy **0.6195**
- **Features**: 32, after dropping `PatientID` and `DoctorInCharge`. Age, Gender, Sleep Quality, BMI, Family History, UPDRS, MoCA, Functional Assessment, and more.

### ⚠️ Source availability

The Kaggle dataset originally cited in this README
(`nidaguler/parkinsons-dataset`) **is no longer available — the URL returns
HTTP 404.** The raw CSV is therefore **vendored in-repo at
[`data/raw/parkinsons.csv`](data/raw/parkinsons.csv)**, which is the
authoritative copy for reproducing everything below. Do not expect
`src/data_pipeline/download_data.py` to reproduce it.

Note that `download_data.py` actually references a *different* slug,
`rabieelkharoua/parkinsons-disease-dataset-analysis`, which is still live.
That publisher distributes **explicitly synthetic** datasets, which is
consistent with the synthetic-data findings below.

### ⚠️ This dataset is synthetic

Statistical screening ([`analysis/eda_leakage.py`](analysis/eda_leakage.py))
shows this is generated data, not real patient records:

| Check | Finding |
|---|---|
| Uniform distributions | **12 of 15** continuous features are statistically indistinguishable from `Uniform(min, max)` (KS test, p > 0.05) |
| Excess kurtosis | **every** continuous feature sits at ≈ −1.18 to −1.24; the exact value for a uniform distribution is −1.2 |
| Inter-feature correlation | max \|r\| between any two features = **0.076**, mean \|r\| = **0.018** — columns were sampled independently. Real clinical data has correlated blocks (BMI↔BP↔cholesterol) |
| Clinical score granularity | `UPDRS`, `MoCA`, `FunctionalAssessment` each have **2,105 unique float values across 2,105 rows**. Real UPDRS and MoCA are integer-scored instruments and cannot take values like `29.181289291248678` |
| Range fit | UPDRS spans `[0.03, 198.95]`, MoCA `[0.02, 29.97]` — each fills its nominal instrument range almost exactly end to end |
| Exact duplicate rows | 0 (features-only: 0); `PatientID` is a perfectly contiguous integer range |

Zero duplicates plus zero missing values plus uniform marginals plus
uncorrelated columns is the signature of a generator, not of collected
clinical data. **Treat all metrics below as a pipeline demonstration, not as
evidence of clinical validity.**


## 🎯 Target leakage: UPDRS / MoCA / Functional Assessment

UPDRS and MoCA are clinical assessment instruments *for Parkinson's* — a
clinician's score is downstream of the diagnosis, so using them as features
is close to using the label. To quantify this, two models were trained with
**identical architecture, identical hyperparameters, the same `random_state=42`
split, and the same 0.50 decision threshold**. Nothing was tuned in either arm.

### Feature association with the target (ranked, top 10 by mutual information)

| # | Feature | Pearson r | Mutual info |
|---|---|---|---|
| 1 | **UPDRS** | +0.3980 | 0.1453 |
| 2 | **FunctionalAssessment** | −0.2250 | 0.0591 |
| 3 | Rigidity | +0.1856 | 0.0437 |
| 4 | **MoCA** | −0.1731 | 0.0344 |
| 5 | Tremor | +0.2744 | 0.0295 |
| 6 | Constipation | +0.0253 | 0.0119 |
| 7 | Ethnicity | −0.0051 | 0.0114 |
| 8 | Bradykinesia | +0.1840 | 0.0097 |
| 9 | SpeechProblems | −0.0122 | 0.0096 |
| 10 | Stroke | +0.0281 | 0.0087 |

The top 5 are the three clinical scores plus two motor signs. **17 of 32
features have a mutual information of exactly 0.0000** with the target —
including Age, Gender, BMI, all four cholesterol measures, both blood
pressures, and family history. Full ranking in
[`analysis/eda_leakage.py`](analysis/eda_leakage.py) output.

### Held-out test metrics (n = 421, 62.0% positive)

Baseline to beat: **0.6200** (predict the majority class every time).

| Model | Features | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **A** — all features | 32 | **0.8504** | 0.8694 | 0.8927 | 0.8809 | **0.9121** |
| **B** — UPDRS, MoCA, FunctionalAssessment removed | 29 | **0.6912** | 0.7148 | 0.8352 | 0.7703 | **0.7368** |
| *C* — B, also without motor signs *(supplementary)* | 24 | *0.6081* | *0.6171* | *0.9693* | *0.7541* | ***0.5302*** |

Delta, A → B: accuracy **−0.1591**, precision −0.1546, recall −0.0575,
F1 −0.1106, ROC-AUC **−0.1753**.

#### Confusion matrices (rows = actual 0/1, columns = predicted 0/1)

```text
Model A (all features)          Model B (scores removed)        Model C (supplementary)
        pred 0  pred 1                  pred 0  pred 1                  pred 0  pred 1
act 0      125      35          act 0       73      87          act 0        3     157
act 1       28     233          act 1       43     218          act 1        8     253
```

### Interpretation

**Your leakage concern is confirmed.** Three of 32 features carry most of the
model's discriminative power. Removing them costs 17.5 ROC-AUC points and
leaves accuracy at 0.6912 against a 0.6200 majority baseline — the honest,
non-clinical-assessment signal in this dataset is thin.

The supplementary Model C is the sharper result: with the motor-symptom flags
(Tremor, Rigidity, Bradykinesia, PosturalInstability, SpeechProblems) also
removed, **ROC-AUC collapses to 0.5302 — indistinguishable from random
guessing**, and the model degenerates into near-always predicting the positive
class (3 true negatives out of 160). Every bit of predictive signal in this
dataset lives in features that are *observations of Parkinson's disease
itself*, not antecedent risk factors. There is no demographic or
metabolic-risk signal to learn.

Reproduce with:

```bash
python analysis/eda_leakage.py
```

```bash
python analysis/leakage_experiment.py
```


## 🛠️ Preprocessing

Order matters here, and it is enforced in
[`src/data_pipeline/preprocess_data.py`](src/data_pipeline/preprocess_data.py):

1. Drop identifier columns (`PatientID`, `DoctorInCharge`). No missing values
   exist to clean — the dataset has none.
2. **Split first**, 80/20, `stratify=y`, `random_state=42`.
3. **Then** standardise: `StandardScaler` is fitted on the **training fold
   only** and applied to test with those train-derived statistics.
4. Persist the scaler fitted on train alone, bundled with its column order.

> **Why step 2 precedes step 3.** An earlier version of this file called
> `scaler.fit_transform(X)` on the whole frame *before* splitting. That
> computes each column's mean and standard deviation partly from test rows and
> bakes them into the training features — a preprocessing leak that makes the
> reported test score not a clean held-out estimate. It is fixed. Measured
> impact of the fix was under 0.004 on every metric
> ([details](models/README.md)), but correctness does not depend on the bug
> being expensive.


## 🤖 Model

- **Algorithm**: Deep Neural Network (Keras Sequential)
- **Framework**: TensorFlow 2.21 + Keras
- **Architecture**: `Dense(64, relu)` → `Dropout(0.3)` → `Dense(32, relu)` →
  `Dropout(0.2)` → `Dense(1, sigmoid)`
- **Training**: Adam, binary cross-entropy, max 50 epochs, batch 32, early
  stopping on `val_loss` (patience 5), seed 42, decision threshold 0.50
- **Output**: Probability of the positive class
- **Artifacts** — two arms, identical architecture, differing only in features:
  - `models/parkinsons_model_HONEST.keras` + `scaler_HONEST.pkl` —
    **the model to use.** 29 features; the three clinical assessment scores excluded.
  - `models/parkinsons_model_ALL_FEATURES_LEAKY.keras` + `scaler_ALL_FEATURES_LEAKY.pkl` —
    **documented baseline, do not use.** All 32 features; accuracy inflated by leakage.

See [`models/README.md`](models/README.md) for which file to use and why.


## 🧪 Prediction Script

[`src/predict_single.py`](src/predict_single.py) runs single-patient inference.
It defaults to the honest model and prints a synthetic-data warning on every
run; selecting the leaky baseline prints an explicit explanation of where its
apparent accuracy comes from.

```bash
python src/predict_single.py                 # honest model, dataset medians
python src/predict_single.py --interactive   # prompt for each feature
python src/predict_single.py --model leaky   # documented baseline, warns loudly
```

[`src/models/predict.py`](src/models/predict.py) evaluates both saved arms on
the held-out test set.


## 📎 Run Locally

```bash
git clone https://github.com/Aiza166/NeuroDetect.git
```

```bash
cd NeuroDetect && pip install -r requirements.txt
```

Reproduce the full pipeline in order:

```bash
python src/data_pipeline/preprocess_data.py
```

```bash
python src/models/train_model.py
```

```bash
python src/models/predict.py
```

Reproduce the audit:

```bash
python analysis/eda_leakage.py
```

```bash
python analysis/leakage_experiment.py
```


## 📂 Project Structure

```bash
NeuroDetect/
├── README.md                       # Project overview and documentation
├── ANALYSIS.md                     # Target-leakage and synthetic-data audit
├── LICENSE                         # MIT for code; dataset excluded, see notice
├── requirements.txt                # Pinned Python dependencies
│
├── analysis/
│   ├── eda_leakage.py              # Shape, balance, correlation/MI, synthetic screens
│   ├── leakage_experiment.py       # Three-arm controlled leakage experiment
│   └── results.json                # Machine-readable experiment metrics
│
├── data/
│   ├── raw/
│   │   └── parkinsons.csv          # Vendored dataset; upstream source is dead
│   └── processed/
│       ├── train_HONEST.csv        # 29 features, scaled on train stats only
│       ├── test_HONEST.csv
│       ├── train_ALL_FEATURES_LEAKY.csv   # 32 features
│       ├── test_ALL_FEATURES_LEAKY.csv
│       └── parkinsons_clean.csv    # DEPRECATED: scaled before splitting (leaky).
│                                   # Kept only so the notebook still loads.
│
├── disease-predictor/
│   └── Parkinson_Detection.ipynb   # Inference notebook (leaky baseline; see banner)
│
├── models/
│   ├── README.md                   # Which artifact to use and why
│   ├── parkinsons_model_HONEST.keras          # USE THIS ONE
│   ├── scaler_HONEST.pkl                      # {"scaler", "features"}
│   ├── parkinsons_model_ALL_FEATURES_LEAKY.keras  # Baseline, do not use
│   ├── scaler_ALL_FEATURES_LEAKY.pkl
│   └── metrics.json                # Metrics for both arms, both scaler orders
│
└── src/
    ├── predict_single.py           # Single-patient inference CLI
    │
    ├── data_pipeline/
    │   ├── download_data.py        # (Optional) upstream fetch; not needed
    │   └── preprocess_data.py      # Split → fit scaler on train → transform
    │
    └── models/
        ├── train_model.py          # Trains and persists both arms
        └── predict.py              # Evaluates both arms on the held-out test set
```



## 📦 Requirements

Verified on Python 3.13.5. These are the versions that actually reproduce the
numbers in this README and in `ANALYSIS.md`:

```bash
pandas==2.3.0
numpy==2.3.1
scikit-learn==1.7.0
scipy==1.16.0
tensorflow-cpu==2.21.0
joblib==1.5.1
```
Install with:
```bash
pip install -r requirements.txt
```


## 🔍 Future Enhancements

- Add audio-based detection via voice samples (e.g. tremor in speech)
- Create a web frontend using Streamlit
- Dockerize the app and add CI/CD for production deployment

